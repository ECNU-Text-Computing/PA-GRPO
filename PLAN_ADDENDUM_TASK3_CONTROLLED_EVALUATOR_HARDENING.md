# Permutation-GRPO 项目补充说明：Task 3 Controlled Evaluator Hardening

> **用途**：本文件补充 `PLAN.md` 与现有各项 addendum，用于固定 Task 3 的目标、已完成内容、正式实验使用规范、验收标准，以及已补齐的测试证据记录。
>
> **Task 3 定位**：
>
> \[
> \boxed{
> \text{Task 3 只负责“如何严格评测模型输出”，不修改训练逻辑}
> }
> \]
>
> 它解决的是：
>
> ```text
> 模型生成结果
> → 判断任务是 A/B Judge 还是 A/B/C/D MCQ
> → 严格解析最终答案
> → 计算 Accuracy / Consistency / CA
> ```
>
> 不负责：
>
> ```text
> reward
> advantage
> GRPO trainer
> Task 2B identity
> rollout pairing
> EIS / PA / ALC
> ```

---

# 1. Task 3 为什么需要做

PA-GRPO 官方 evaluator 主要面向原论文复现，但直接用于 controlled experiments 存在两个问题。

## 1.1 官方 option detector 较宽松

官方 `evaluation/evaluate_models.py` 中的 option detector：

- 主要基于 prompt 内容做启发式判断；
- 只看第一条 prompt；
- 对 parquet 中常见的 `numpy.ndarray` prompt container 支持不足；
- 无法可靠判断时默认返回 4 options。

这会导致 A/B Judge 数据被误判成 A/B/C/D MCQ。

---

## 1.2 官方 direct parser 较宽松

官方 direct parser 不仅接受：

```text
A
B
```

还会从更长文本中寻找合法字母。

例如类似：

```text
invalid
```

的字符串可能因为包含字母 `A` 而被解释成：

```text
A
```

这对官方复现可以保留，但不适合作为本项目的 controlled evaluation contract。

---

# 2. Task 3 的核心设计

Task 3 不修改官方 evaluator 语义，而是增加两条显式路径：

```text
official
controlled
```

命令行参数：

```bash
--evaluation_contract official|controlled
```

默认：

```text
official
```

因此：

\[
\boxed{
\text{旧命令不加新参数时，仍保持官方复现行为}
}
\]

正式 controlled experiments 必须显式使用：

```bash
--evaluation_contract controlled
```

---

# 3. Controlled option detector

新增独立模块：

```text
permstudy/controlled_evaluator.py
```

controlled detector 支持 prompt container：

```text
list
tuple
numpy.ndarray
```

并且不是只看第一行，而是扫描整个 parquet。

---

## 3.1 Whole-dataset contract

controlled evaluator 要求整个数据集的 option contract 一致。

允许：

```text
全部为 A/B Judge
```

或者：

```text
全部为 A/B/C/D MCQ
```

如果出现：

```text
部分 2-option
部分 4-option
```

或者某一行无法可靠判断，则：

\[
\boxed{\text{fail fast}}
\]

不能静默默认成四选项。

---

## 3.2 不依赖普通正文里的 C / D

controlled detector 不允许仅因为 candidate response 正文里出现：

```text
C
D
candidate C
candidate D
```

就把 A/B Judge 判断成 MCQ。

它必须从：

```text
明确输出约束
或
完整结构化选项
```

中推断 option contract。

---

## 3.3 显式 override

Task 3 新增：

```bash
--num_options auto|2|4
```

默认：

```text
auto
```

如果 benchmark prompt 本身不能可靠自动识别，可以显式指定：

```bash
--num_options 2
```

或者：

```bash
--num_options 4
```

显式 override 优先于自动检测。

---

# 4. Controlled answer parser

## 4.1 Direct 模式

controlled `direct` 模式只接受：

\[
\boxed{\text{strip 后的单个合法字母}}
\]

A/B Judge：

```text
"A"      -> A
" B "    -> B
```

以下全部无效：

```text
"A."
"Answer: A"
"I choose A"
"I think the answer is A"
"invalid"
```

返回：

```python
None
```

MCQ 同理，只是合法集合扩展为：

```text
A / B / C / D
```

---

## 4.2 Think / Thinking 模式

controlled `think` / `thinking` 模式要求：

```xml
<answer>X</answer>
```

并且必须满足：

\[
\boxed{\text{恰好一个合法 answer tag}}
\]

合法示例：

```xml
<think>...</think>
<answer>A</answer>
```

或：

```xml
<thinking>...</thinking>
<answer>D</answer>
```

以下情况全部无效：

```text
没有 <answer>
重复 <answer>
冲突答案
未闭合标签
Judge 输出 C/D
<answer>A.</answer>
```

返回：

```python
None
```

---

# 5. answer_probability 只能作为 diagnostic

evaluator 中仍然可以保存：

```text
answer_probability
answer_debug_info
```

用于调试模型输出概率。

但 controlled contract 下：

\[
\boxed{
\text{extracted_answer 和 accuracy 只能由 strict parser 决定}
}
\]

例如：

```text
model response:
"I think the answer might be A"

P(A)=0.99
```

controlled evaluator 必须得到：

```text
extracted_answer = None
```

不能因为概率很高而把它改成 `A`。

---

# 6. Batch 与 Sequential 必须共享同一 contract

当前 evaluator 有两条推理路径：

```text
vLLM batch
Transformers sequential
```

Task 3 要求二者统一通过同一个 contract dispatcher：

```text
extract_answer_for_contract(...)
```

不得：

```text
batch 自己一套 parser
sequential 自己一套 parser
```

因此：

\[
\boxed{
\text{推理后端不同，不得改变答案解析语义}
}
\]

---

# 7. Result metadata 必须记录 contract

正式输出 JSON 必须写入：

```json
{
  "evaluation_contract": "controlled",
  "num_options": 2
}
```

summary 中也必须保留：

```text
evaluation_contract
num_options
```

这样任何实验结果都可以追溯：

```text
是 official 评测
还是 controlled 评测
```

避免后续表格混用两种 evaluator contract。

---

# 8. Official 路径必须保持原行为

Task 3 的重要边界是：

\[
\boxed{
\text{controlled hardening 不能“顺手修复” official reproduction}
}
\]

默认：

```bash
--evaluation_contract official
```

仍使用原来的：

```text
detect_num_options_from_file()
extract_answer_from_response()
```

因此官方已有的宽松行为仍被保留。

这不是 bug 修复，而是：

```text
official reproduction path
与
controlled research path
```

显式隔离。

---

# 9. Task 3 当前测试覆盖

当前新增：

```text
tests_permstudy/test_controlled_evaluator.py
```

测试覆盖至少包括：

```text
list / tuple / numpy.ndarray prompt
A/B Judge detection
A/B/C/D MCQ detection
incidental C/D 不污染 Judge detection
dataset mixed contract rejection
ambiguous row rejection
explicit --num_options override
strict direct parsing
strict answer-tag parsing
official behavior preservation
CLI default official
batch / sequential 共用 controlled contract
answer_probability 不影响最终答案
```

并且测试包含真实仓库 Judge parquet prompt，而不只是人工 fixture。

---

# 10. Task 3 的完成定义

Task 3 功能完成必须满足：

```text
[x] 默认 official 路径行为不变
[x] controlled detector 支持 list/tuple/ndarray
[x] controlled detector 扫描整个 dataset
[x] mixed contract fail fast
[x] ambiguous prompt fail fast
[x] --num_options auto|2|4 可用
[x] direct 只接受裸字母
[x] think/thinking 只接受唯一合法 <answer>X</answer>
[x] answer_probability 不影响 extracted_answer
[x] batch/sequential 使用同一 contract dispatch
[x] output metadata 记录 evaluation_contract
[x] reward parser 未修改
[x] Task 2B identity 未修改
[x] trainer / advantage 未修改
```

当前仓库代码层面已经满足以上设计目标。

---

# 11. Task 3 测试证据 artifact（已补齐）

当前仓库没有 CI workflow / status 自动证明：

```text
具体运行了哪些 pytest
最终多少 tests passed
使用的环境是什么
```

这不是 Task 3 核心代码问题，也不阻塞进入下一阶段。为了和 Task 1.5、Task 2A 的 audit 记录保持一致，仓库已经补充：

```text
artifacts/evaluator/task3/
├── README.md
├── pytest_output.txt
└── environment.txt
```

其中：

### `README.md`

实际记录：

```text
Task 3 purpose
commit SHA
test command
result summary
known limitations
```

### `pytest_output.txt`

保存 Windows 与 WSL2 实际执行：

```bash
pytest tests_permstudy/test_controlled_evaluator.py -q
```

的 pytest 输出与退出状态。被测 evaluator commit 为：

```text
8b9cfff57d4668d1fcaa7f95039dce6782dcd621
```

两套环境的实际结果均为：

```text
42 passed
```

### `environment.txt`

至少记录：

```text
OS
Python version
pytest version
pandas version
numpy version
pyarrow version
torch version
```

如果 evaluator tests 在 WSL/Linux 执行，也记录：

```text
WSL distro
```

---

# 12. Task 3 artifact 的性质

该 artifact 只作为：

\[
\boxed{\text{测试证据 / reproducibility record}}
\]

不允许因为补 artifact 再修改：

```text
controlled parser semantics
official parser
reward parser
trainer
Task 2B identity
advantage
```

如果 pytest 全部通过：

\[
\boxed{\text{Task 3 最终状态 = PASS}}
\]

如果失败，只修复与 controlled evaluator contract 直接相关的问题，不扩展 Task 3 范围。

---

# 13. Task 3 不负责训练数据 verifier

必须区分：

```text
Controlled evaluator
```

和：

```text
Training-data outcome verifier
```

Task 3 负责：

```text
Judge 模型训练完成以后
解析 JudgeBench / ReasoningJudgeBench 输出
```

而未来训练集构造中的：

```text
MATH response 正确性验证
ReClor candidate correctness
```

属于：

```text
training-data pipeline
```

不能直接把 Task 3 的 answer parser 当作 MATH verifier 使用。

---

# 14. Task 3 与 D_sensitive 的关系

未来构造：

```text
D_sensitive
```

时需要：

```text
base judge
→ AB / BA inference
→ strict answer parsing
→ semantic mapping
→ 判断是否 permutation flip
```

这里应该复用 Task 3 已固定的：

```text
controlled Judge answer contract
```

从而避免 official 宽松 parser 对 sensitivity filtering 产生污染。

但：

\[
\boxed{
\text{Task 3 本身不负责生成 }D_{sensitive}
}
\]

它只提供后续 filtering 所需要的严格解析基础。

---

# 15. 当前项目进度中的位置

当前状态：

```text
Task 1A   PASS
Task 1.5  PASS
Task 2A   PASS
Task 2B   PASS
Task 3    PASS（Windows + WSL2: 42 tests passed）
```

下一阶段仍按原计划：

```text
Task 3
↓
真实 vLLM / GPU / GRPO smoke
↓
EIS / PA / ALC implementation
↓
Pilot
↓
正式 controlled experiments
```

训练数据 pipeline 可以并行开发，但不能绕过后续正式 smoke 和实验验收。

---

# 16. 最终执行约束

后续正式评测必须遵循：

\[
\boxed{
\text{官方复现使用 official contract}
}
\]

\[
\boxed{
\text{本项目 controlled experiments 使用 controlled contract}
}
\]

\[
\boxed{
\text{不得混用 official 与 controlled 指标进入同一主结果表}
}
\]

\[
\boxed{
\text{controlled answer 必须由 strict parser 唯一决定}
}
\]

\[
\boxed{
\text{Task 3 不修改训练逻辑}
}
\]

本文件作为后续 evaluator 使用与审查的固定规范。
