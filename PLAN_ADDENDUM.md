# Permutation-GRPO 项目补充计划：Paper–Code Fidelity Audit 与稳健分组实现

> **用途**：本文件是 `Permutation_GRPO_Project_Plan.md` 的补充说明，重点处理在阅读 PA-GRPO 官方代码后发现的论文—代码一致性问题，以及正式训练前必须完成的 CPU 单测与工程加固。
>
> **研究范围补充**：Pairwise 主线的数据单位、`original_question_id` / `pair_id` 契约及 MCQ 可选扩展边界见 [`PLAN_ADDENDUM_PAIRWISE_SCOPE_AND_MCQ_EXTENSION.md`](PLAN_ADDENDUM_PAIRWISE_SCOPE_AND_MCQ_EXTENSION.md)。
>
> **适用上游**：`ECNU-Text-Computing/PA-GRPO`
>
> **固定官方 commit**：`0ee9abd903cb4ac4945f1176e943d20436470096`
>
> **优先级**：本补充计划中的“正式训练前检查项”高于原计划中直接进入 AutoDL 训练 smoke 的步骤；若与原计划冲突，以本文件为准。
>
> **原则**：不要先假设官方代码就是论文公式的唯一 ground truth。区分：
>
> 1. **PA-paper-faithful**：严格按论文公式定义实现；
> 2. **PA-official-code**：严格复现当前官方仓库行为；
> 3. **controlled implementation**：为 EIS / PA / ALC 公平比较而使用的统一实现。
>
> 三者必须可独立切换、可测试、可记录。

---

# 0. 本补充计划要解决什么

在检查 PA-GRPO 官方仓库后，需要在正式 GPU 训练前确认四件事：

1. **论文中的低方差 gate 是否被官方代码完整实现。**
2. **Consistency reward 的 AB/BA rollout 配对是否依赖隐式 batch 顺序。**
3. **官方 PA advantage 路径是否会因为 response length 不同而改变最终 advantage。**
4. **后续自建 MATH/ReClor 数据不能依赖 `index // 2` 等隐式约定，必须显式保存分组信息。**

其中：

- 第 1、3 点属于 **paper–code fidelity**；
- 第 2、4 点属于 **grouping / pairing robustness**；
- 第 4 点也是第 2 点的工程加固方案。

这些检查**不阻塞当前 Windows 的 Transformers inference + scorer smoke**，但必须在任何正式 GRPO 训练前完成。

---

# 1. 当前阶段任务重新编号

## Task 1A — Windows inference / scorer smoke

继续按原计划执行，不修改 verl 核心代码。

验收链路：

```text
完整官方源码 checkout
    ↓
构造 2–4 个 Judge smoke samples
    ↓
evaluation/evaluate_models.py --use_transformers
    ↓
生成官方 JSON 结果
    ↓
evaluation/compute_metrics_judge.py
    ↓
正常输出 Accuracy / Consistency / Consistent Accuracy
```

### Task 1A 验收标准

必须留下：

```text
artifacts/smoke/task1a/
├── command.txt
├── env.txt
├── smoke_judge_think.parquet
├── eval_output.json
├── metrics.txt
└── README.md
```

`README.md` 至少记录：

- commit；
- Python / PyTorch / Transformers 版本；
- 使用 CPU 还是本地 GPU；
- 模型名称；
- 命令；
- 是否成功产生 Acc / Con / CA；
- 已知 Windows 限制。

**Task 1A 不要求 verl、vLLM、Ray、GRPO 训练可运行。**

---

# 2. 新增 Task 1.5 — Paper–Code Fidelity Audit

> 在进入 AutoDL Linux 的训练 smoke 之前，必须完成。

目标不是“找官方代码 bug”，而是明确：

```text
论文数学定义
vs
官方当前代码行为
vs
本项目受控实现
```

到底是否一致。

Task 1.5 应完全可以在 CPU 上运行，不使用大模型，不消耗 GPU 训练预算。

---

# 3. Audit A：低方差 sigma gate

## 3.1 论文定义

PA-GRPO 论文中的 cross-permutation advantage 形式为：

\[
A_i =
\begin{cases}
0, & \sigma_G < \delta \\
\dfrac{r_i-\mu_G}{\sigma_G+\epsilon}, & \text{otherwise}
\end{cases}
\]

核心含义：

```text
如果整个 permutation group 的 reward 方差过小，
则认为相对比较信号不可靠，直接令该 group advantage = 0。
```

这和单纯防止除零不同。

---

## 3.2 当前官方代码需要验证的行为

目前检查到：

```text
verl/trainer/ppo/ray_trainer.py
```

中的 group baseline 路径会：

```python
stds = sqrt(clamp(var, min=1e-6))
adv = (returns - mean) / std
adv = clamp(adv, -5, 5)
```

因此当前实现看起来更接近：

\[
\sigma \leftarrow \max(\sigma,\epsilon)
\]

而不是显式：

\[
\sigma < \delta \Rightarrow A=0
\]

Codex 不得只根据阅读结论修改代码，先用单测固定行为。

---

## 3.3 必须新增的函数

建议在：

```text
permstudy/advantages.py
```

实现纯函数：

```python
def global_advantage_paper(
    rewards,
    group_ids,
    *,
    eps=1e-6,
    sigma_gate_threshold=None,
):
    ...
```

要求：

- 输入必须是 **response-level scalar rewards**；
- 不依赖 token mask；
- 不依赖 trainer；
- `sigma_gate_threshold=None` 时关闭 gate；
- 当提供 threshold 时，严格执行论文 gate；
- 返回：
  - `advantage`
  - `group_mean`
  - `group_std`
  - `gate_mask`

另实现：

```python
def global_advantage_official_compatible(...):
    ...
```

用于模拟当前官方代码的 baseline 行为。

---

## 3.4 单元测试

新增：

```text
tests_permstudy/test_pa_sigma_gate.py
```

至少覆盖：

### Case A：完全相同 reward

```python
rewards = [1.0, 1.0, 1.0, 1.0]
```

期望：

```text
paper gate: A = 0
official-compatible: A = 0
```

因为 numerator 本身为 0。

### Case B：几乎相同 reward

例如：

```python
rewards = [1.0000, 1.0000, 1.0001, 1.0000]
```

必须证明：

- 不带 gate 时可能得到非零 standardized advantage；
- 带 gate 且 `sigma < delta` 时全部为 0。

### Case C：正常方差

```python
rewards = [1.0, 1.0, -1.0, -1.0]
```

当 `sigma > delta`：

```text
paper gated == normal standardized advantage
```

---

## 3.5 重要规则

在没有确认论文或作者代码给出具体 `delta` 之前：

```text
禁止随意写死一个“论文默认 delta”。
```

配置中使用：

```yaml
method:
  sigma_gate:
    enabled: false
    threshold: null
    source: "unspecified"
```

后续若从论文正文、附录、作者 issue 或代码中找到可信值，再更新并记录来源。

---

# 4. Audit B：官方 PA advantage 是否受 response length 影响

## 4.1 为什么要测

论文定义使用 response-level scalar reward：

\[
r_i \rightarrow \mu_G,\sigma_G \rightarrow A_i
\]

理论上如果两个 response 的最终 scalar reward 相同，仅仅输出 token 数不同，不应改变这一数学定义下的 group-relative advantage。

但当前官方训练路径大致经过：

```text
scalar reward
→ compute_grpo_outcome_advantage
→ broadcast 到 response token
→ returns
→ ray_trainer.py 再压回 scalar
→ group mean/std
```

如果压回 scalar 使用对 padded token 维度直接 `mean()`，就可能引入长度比例。

这个问题必须通过数值实验确认，不能只靠代码阅读下结论。

---

## 4.2 新增 CPU fidelity harness

新增：

```text
scripts_permstudy/audit/audit_official_advantage.py
```

功能：

1. 构造 synthetic `token_level_rewards`；
2. 构造不同 `response_mask`；
3. 调用官方当前函数；
4. 调用 `permstudy/advantages.py` 的 paper-faithful 实现；
5. 输出两者 difference。

输出：

```text
artifacts/audit/advantage_length/
├── equal_length.json
├── unequal_length.json
└── summary.md
```

---

## 4.3 必测实验

### Experiment B1：等长 response

固定：

```python
scalar_rewards = [1.0, 1.0, -1.0, -1.0]
lengths = [8, 8, 8, 8]
```

要求比较：

\[
A_{\text{paper}}
\]

和：

\[
A_{\text{official-code}}
\]

若不一致，必须先定位原因，再进入正式训练。

---

### Experiment B2：仅改变 response length

固定相同 scalar rewards：

```python
scalar_rewards = [1.0, 1.0, -1.0, -1.0]
```

改变长度：

```python
lengths = [2, 8, 4, 10]
```

要求：

```text
paper-faithful advantage 不因长度变化而变化。
```

检查官方代码输出是否变化。

### 判定

若：

```text
official(equal-length) != official(unequal-length)
```

且 reward/group 都未改变，则记录为：

```text
Observed implementation-level response-length dependence
```

不要直接在 README 写“官方代码 bug”。

更严谨表述：

> The current official implementation path exhibits response-length dependence that is absent from the scalar-reward formulation in the paper.

---

## 4.4 如果确认存在 length dependence，如何处理

不要直接覆盖官方实现。

同时保留：

```yaml
method:
  pa_advantage_impl: official
```

和：

```yaml
method:
  pa_advantage_impl: paper
```

主研究中建议：

- `Mode A / PA sanity reproduction`：优先保留 `official`；
- `Mode B / controlled comparison`：优先使用 `paper` 的 response-level scalar 实现；
- 如果两种实现结果差异明显，单独作为 appendix / analysis 报告。

---

# 5. Audit C：Consistency reward 的 rollout 配对

## 5.1 论文希望表达的逻辑

对于 pairwise Judge：

```text
Permutation 0: AB
Permutation 1: BA
```

每个 permutation 采样 N 个 rollout。

Consistency reward 比较的是：

\[
(AB, i) \leftrightarrow (BA, i)
\]

但真正比较的是 semantic winner，而不是表面字母。

例如：

```text
AB 输出 A → semantic candidate = y1
BA 输出 B → semantic candidate = y1
```

应判 consistency = true。

---

## 5.2 当前官方实现

当前 reward 代码会分别收集：

```python
group[pair_id][0] = [...]
group[pair_id][1] = [...]
```

然后按照两个列表中的**出现顺序**：

```python
idxs0[t] <-> idxs1[t]
```

配对。

这意味着 rollout identity 主要由顺序隐式维护。

这不等于“论文逻辑没实现”：

```text
论文逻辑已经实现；
但 rollout slot 没有被显式编码为身份字段。
```

需要确认 batch reorder 是否会改变这种隐式配对的统计行为。

---

# 6. 改进方案：显式 Group Identity Schema

这是本补充计划最重要的工程加固。

今后本项目生成的所有 Judge training samples / rollouts，统一使用以下概念：

```text
pair_id            # 原始 Judge pair 的唯一 ID
permutation_id     # 0=AB, 1=BA
rollout_slot       # 同一 permutation 下第几个 rollout
original_question_id
```

不要把概念混在一个 `index` 里。

---

## 6.1 训练数据 schema

预处理后的 dataset 每一条 permutation 输入至少保存：

```json
{
  "pair_id": "math_000123_pair_00",
  "original_question_id": "math_000123",
  "permutation_id": 0,
  "question": "...",
  "candidate_a": "...",
  "candidate_b": "...",
  "gold_surface": "A",
  "gold_semantic": "pos",
  "extra_info": {
    "pair_id": "math_000123_pair_00",
    "original_question_id": "math_000123",
    "permutation_id": 0
  }
}
```

BA 版本：

```json
{
  "pair_id": "math_000123_pair_00",
  "original_question_id": "math_000123",
  "permutation_id": 1,
  "candidate_a": "negative response",
  "candidate_b": "positive response",
  "gold_surface": "B",
  "gold_semantic": "pos"
}
```

---

## 6.2 rollout_slot 何时产生

`rollout_slot` 不属于静态 dataset 字段，因为它是在：

```text
rollout.n = N
```

展开之后才产生。

在生成 rollout batch 后必须显式生成：

```python
rollout_slot = 0, 1, ..., N-1
```

并且和对应 sample 一起随 DataProto reorder / repeat / union 传播。

原则：

```text
任何 batch reorder 都必须同步 reorder rollout_slot。
```

---

## 6.3 新 consistency key

以后 consistency pairing 不通过“列表第 t 个”猜，而通过：

```python
consistency_key = (pair_id, rollout_slot)
```

在同一个 key 下应找到：

```text
permutation_id = 0
permutation_id = 1
```

两条 response。

伪代码：

```python
pairs = defaultdict(dict)

for sample in batch:
    key = (sample.pair_id, sample.rollout_slot)
    pairs[key][sample.permutation_id] = sample

for key, item in pairs.items():
    if 0 in item and 1 in item:
        reward_consistency(item[0], item[1])
```

---

# 7. Consistency pairing 单测

新增：

```text
tests_permstudy/test_consistency_pairing_reorder.py
```

## Case C1：正常顺序

输入：

```text
AB0 AB1 AB2
BA0 BA1 BA2
```

配对必须为：

```text
AB0 ↔ BA0
AB1 ↔ BA1
AB2 ↔ BA2
```

---

## Case C2：随机 reorder

例如输入：

```text
BA2 AB0 BA0 AB2 AB1 BA1
```

只要显式 ID 保留，输出仍必须：

```text
AB0 ↔ BA0
AB1 ↔ BA1
AB2 ↔ BA2
```

---

## Case C3：缺失某个 rollout

例如：

```text
AB0 AB1 AB2
BA0 BA2
```

必须：

- 正确配 `slot=0` 和 `slot=2`；
- `AB1` 标记 `unpaired_consistency_sample`；
- 不允许悄悄变成 `AB1 ↔ BA2`。

---

## Case C4：duplicate key

如果出现两条：

```text
(pair_id=7, permutation=0, slot=1)
```

必须报错，不能 silently overwrite。

---

# 8. 官方顺序配对是否真的有问题：做对照，不预设结论

新增：

```text
scripts_permstudy/audit/audit_consistency_pairing.py
```

比较两个实现：

```text
official_order_pairing
explicit_id_pairing
```

在以下条件下测试：

1. 原始顺序；
2. 完全随机 shuffle；
3. 按 response length 排序；
4. 模拟 verl `_balance_batch` 重排。

记录：

```text
pairing_match_rate
consistency_reward_match_rate
mean_reward_difference
```

如果结果完全一致，则说明官方隐式顺序在当前单卡设置下足够稳健。

如果不一致，则明确记录不一致发生在哪种 reorder。

**不要提前把它定性为官方 bug。**

---

# 9. 统一 ID 规则

后续主项目不允许使用以下逻辑作为唯一身份来源：

```python
pair_id = index // 2
permutation = index % 2
```

它们可以作为 legacy fallback，但不能作为 controlled experiments 的主实现。

建议统一：

```python
@dataclass(frozen=True)
class PermutationSampleId:
    pair_id: str
    permutation_id: int

@dataclass(frozen=True)
class RolloutId:
    pair_id: str
    permutation_id: int
    rollout_slot: int
```

新增：

```text
permstudy/ids.py
```

所有 grouping / reward / diagnostics 统一调用这里。

---

# 10. 对原项目配置的补充

在原 config 上新增：

```yaml
method:
  pa_advantage_impl: paper       # paper | official
  use_sigma_gate: false
  sigma_gate_threshold: null

grouping:
  identity_mode: explicit        # explicit | legacy_index
  require_pair_id: true
  require_permutation_id: true
  require_rollout_slot_for_consistency: true
  error_on_duplicate_rollout_key: true
  error_on_unpaired_consistency: false

audit:
  enabled: true
  compare_official_vs_paper_advantage: true
  compare_order_vs_explicit_pairing: true
```

### controlled comparison 规定

正式主表必须使用：

```yaml
grouping.identity_mode: explicit
method.pa_advantage_impl: paper
```

除非后续 Audit 证明 paper / official 完全等价。

### PA 官方复现规定

sanity reproduction 使用：

```yaml
grouping.identity_mode: legacy_index
method.pa_advantage_impl: official
```

尽量不改作者行为。

---

# 11. 新增诊断指标

除了原计划中的：

```text
Acc
Consistency
Consistent Accuracy
SCR
LRR
LSR
Delta_mu
```

再增加：

## 11.1 advantage fidelity

```text
adv_l1_diff_official_vs_paper
adv_linf_diff_official_vs_paper
adv_sign_disagreement_rate
```

其中：

\[
\text{sign disagreement}
=
P(\operatorname{sign}(A_{official})\neq
\operatorname{sign}(A_{paper}))
\]

如果连更新方向都出现差异，必须单独报告。

---

## 11.2 grouping fidelity

```text
consistency_pairing_match_rate
num_unpaired_rollouts
num_duplicate_rollout_keys
```

正式实验要求：

```text
num_duplicate_rollout_keys = 0
```

controlled implementation 中：

```text
num_unpaired_rollouts = 0
```

除非明确做故障测试。

---

# 12. Task 1.5 验收标准

只有以下全部通过，才能进入 AutoDL training smoke。

## 必须通过

```text
[ ] paper global advantage 纯函数测试通过
[ ] sigma gate synthetic test 通过
[ ] equal-length paper/code 对照完成
[ ] unequal-length paper/code 对照完成
[ ] 是否存在 length dependence 有明确结论
[ ] explicit ID grouping test 通过
[ ] reorder 后 consistency pairing test 通过
[ ] duplicate / missing slot 行为有明确处理
[ ] audit 报告生成
```

产物：

```text
artifacts/audit/task1_5/
├── PA_PAPER_CODE_AUDIT.md
├── sigma_gate_results.json
├── advantage_equal_length.json
├── advantage_unequal_length.json
├── consistency_pairing_reorder.json
└── environment.txt
```

---

# 13. `PA_PAPER_CODE_AUDIT.md` 必须包含的结构

Codex 应自动生成并填写：

```markdown
# PA-GRPO Paper–Code Fidelity Audit

## Upstream
- repo:
- commit:
- date checked:

## A. Sigma Gate
### Paper definition
...
### Official code behavior
...
### Observed numerical behavior
...
### Decision for project
...

## B. Response-Length Dependence
### Synthetic setup
...
### Equal-length results
...
### Unequal-length results
...
### Conclusion
...

## C. Consistency Pairing
### Official behavior
...
### Explicit-ID behavior
...
### Reorder tests
...
### Conclusion
...

## D. Controlled-Experiment Implementation
- advantage implementation:
- grouping identity:
- consistency pairing:
- legacy fallback:

## Open ambiguities
...
```

所有结论必须分成：

```text
Paper states
Official code does
Our test observes
Project decision
```

禁止把推断写成作者明确结论。

---

# 14. Task 2 前的 AutoDL Linux smoke

Task 1.5 完成后，才进入 Linux。

第一轮不要直接跑 8B 完整训练。

建议：

```text
AutoDL Linux
→ 拉取同一 commit
→ Python 3.12 / compatible CUDA
→ 安装 requirements-lock.txt
→ verl import
→ vLLM import
→ 官方 8B 模型只加载检查
→ 极小 batch / 极少 step training smoke
```

训练 smoke 的目标是：

```text
确认 pipeline 能跑，而不是看指标提升。
```

推荐只跑：

```text
train samples: 8–32 original pairs
rollout.n: 2
max_response_length: 128–256
steps: 1–5
```

如果官方脚本不能直接这样配置，使用命令行 override，不要复制 Trainer。

---

# 15. 正式实验时的三种 PA 版本

为了避免后续名字混乱，项目里统一命名：

## PA-Official

```text
官方当前仓库行为
```

用于 sanity reproduction。

---

## PA-Paper

```text
论文定义的 response-level global advantage
+ 论文定义的 sigma gate（仅在 threshold 有可靠依据时启用）
+ consistency reward
```

用于 paper-faithful 对照。

如果 `delta` 无可靠值，则拆成：

```text
PA-Paper-NoGate
PA-Paper-Gated(delta=..., source=...)
```

绝不把自选 delta 写成论文默认。

---

## PA-Controlled

```text
与 EIS/ALC 共用相同：
backbone
data
base reward
rollout budget
LoRA
optimizer
prompt
evaluation
```

用于主研究结论。

主论文式表格不应混用 PA-Official 与 PA-Controlled。

---

# 16. 对原计划中 ALC 的影响

ALC 公式暂时不变：

\[
A_{ALC}
=
A_g+\lambda_{local}A_l
\]

其中：

\[
d=
\frac{|\mu_{AB}-\mu_{BA}|}
{\sigma_G+\epsilon}
\]

\[
\lambda_{local}
=
clip(d,0,1)
\]

但必须确保：

```text
A_g 使用 paper-faithful response-level scalar reward 计算。
A_l 也使用 response-level scalar reward 计算。
```

不要让 ALC 的主实验继承未经确认的 token-length dependence。

同时：

```text
mu_AB / mu_BA / sigma_G
```

应基于 **base reward，不含 consistency reward**，和原计划一致。

---

# 17. Codex 实施约束

Codex 执行本补充计划时必须遵守：

1. **先写测试，再改 Trainer。**
2. 不允许为了“让测试通过”修改官方 vendored verl。
3. 官方实现保持可调用，作为 reference path。
4. 新研究实现尽量放在：
   ```text
   permstudy/
   ```
5. `ray_trainer.py` 只做最小 dispatch patch。
6. 每次改变 advantage / reward / grouping 行为，都要能通过 config 开关恢复官方行为。
7. 不允许隐藏 fallback。
8. 任何“论文没写清楚”的参数必须标记来源，不能猜成论文默认。
9. 所有正式结果记录：
   ```text
   git commit
   config
   seed
   model
   dataset manifest
   upstream PA commit
   ```
10. 如果发现新的 paper–code discrepancy，先加入 audit，不要直接修。

---

# 18. 推荐 commit 顺序

建议 Codex 按以下粒度提交：

```text
chore: pin PA-GRPO upstream commit and document environment

test: add scalar global-advantage fidelity cases

test: add sigma-gate synthetic cases

audit: compare official and paper advantage under unequal lengths

feat: add explicit permutation and rollout identity schema

test: verify consistency pairing survives batch reorder

feat: add paper-faithful PA advantage implementation

feat: add configurable official/paper advantage dispatch

docs: add PA paper-code fidelity audit report
```

之后才进入：

```text
feat: implement EIS global-local advantage
feat: implement ALC
```

不要在一个 commit 里同时改 grouping、reward、advantage 和 trainer。

---

# 19. 立即给 Codex 的下一条任务

可以直接复制下面这段：

> 阅读根目录的 `Permutation_GRPO_Project_Plan.md` 和本补充计划。当前不要进入 AutoDL 训练，也不要修改官方 verl 核心逻辑。先完成 Task 1A；若 Task 1A 已通过，则开始 Task 1.5 Paper–Code Fidelity Audit。固定上游 commit 为 `0ee9abd903cb4ac4945f1176e943d20436470096`。先实现独立于 Trainer 的 response-level scalar advantage 纯函数和 synthetic CPU tests，再对照官方 `compute_grpo_outcome_advantage` / `ray_trainer.py` 路径检查：(1) sigma gate 是否存在 paper/code 差异；(2) 仅改变 response length 时官方最终 group advantage 是否变化；(3) consistency reward 的顺序配对在 batch reorder 后是否与显式 `(pair_id, rollout_slot)` 配对一致。不要预设官方代码有 bug，所有结论写成 `Paper states / Official code does / Our test observes / Project decision`。完成后生成 `artifacts/audit/task1_5/PA_PAPER_CODE_AUDIT.md`，在我确认 audit 结果前不要开始正式 GRPO 训练。

---

# 20. 本补充计划的完成定义

这份补充计划完成后，我们应明确知道：

\[
\boxed{
\text{PA 论文公式、官方实现、我们的受控实现到底哪里相同，哪里不同}
}
\]

并保证后续 EIS vs PA vs ALC 比较不会被以下隐性因素污染：

```text
response length
batch reorder
index 编码假设
未声明的 sigma gate 行为
隐式 rollout pairing
```

最终目标不是“修官方代码”，而是得到一套：

\[
\boxed{
\text{可复现、可解释、可公平消融的 permutation-aware GRPO 实验框架}
}
\]

这将作为后续 EIS、PA、ALC 主实验的可靠基础。
