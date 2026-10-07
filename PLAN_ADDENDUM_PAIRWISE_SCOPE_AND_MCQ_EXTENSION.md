# Permutation-GRPO 项目补充说明：Pairwise 主线约束与 MCQ 可选扩展

> **用途**：本文件补充 `PLAN.md` 与 `PLAN_ADDENDUM.md`，用于明确当前主实验的数据单位、`original_question_id` / `pair_id` 的关系，以及 MCQ（P>2 permutation）是否需要并入当前 Task 2B 和主研究路线。
>
> **结论先行**：
>
> 1. 当前主线继续保持 **Pairwise Judge, P=2**，不因 MCQ 讨论而修改 Task 2B 主体实现。
> 2. controlled main experiments 中，**每个被保留进入训练的原始 question 恰好对应一个固定 positive-negative Judge pair**；无法形成有效正负 pair 的问题直接过滤。
> 3. MCQ 的 P>2 permutation setting 作为 **Optional Extension**，不进入当前 EIS–PA–ALC 主实验。
> 4. 若未来扩展到 MCQ，需要重新定义多 subgroup imbalance、consistency grouping 与 identity schema；不能直接把当前二元 AB/BA 逻辑机械复用。

---

# 1. 当前主研究范围不变

本项目当前核心研究问题仍然是：

> 在统一 backbone、训练数据、基础 reward、rollout budget 下，比较 Vanilla / Global-only / EIS / PA / ALC，分析何时 global cross-permutation advantage 已经足够、何时 local correction 仍然必要。

主实验继续限定在：

```text
Pairwise Judge
P = 2
Permutation = AB / BA
```

即一个固定 Judge pair：

```text
candidate A
candidate B
```

构造两个等价输入：

```text
perm 0: AB
perm 1: BA
```

每个 permutation 再独立采样 N 个 rollout：

\[
AB_0,\ldots,AB_{N-1}
\]

\[
BA_0,\ldots,BA_{N-1}
\]

因此每个原始 Judge instance 的 rollout budget 为：

\[
2N
\]

当前主计划中的：

```yaml
permutation:
  num_permutations: 2
  semantic_mapping: pairwise_ab_ba
```

继续保持不变。

---

# 2. 明确主训练数据单位：一个 question 只保留一个固定 Judge pair

当前 `PLAN.md` 的 J4R-style reasoning pair dataset 定义为：

```text
question
+ response_pos
+ response_neg
```

训练时构造：

```text
AB: A = pos, B = neg
BA: A = neg, B = pos
```

为避免 `original_question_id` 与 `pair_id` 的歧义，主实验新增如下约束：

\[
\boxed{
\text{每个进入训练的 original_question_id 恰好对应一个固定 positive-negative pair}
}
\]

也就是说，主数据流程明确为：

```text
原始问题 q
    ↓
生成 K 个 candidate responses
    ↓
划分 correct / incorrect
    ↓
从中选择 1 个 response_pos
    ↓
选择 1 个 response_neg
    ↓
构造唯一 Judge pair
    ↓
AB / BA
    ↓
每个 permutation 采 N 次
```

而不是：

```text
q
├── pair_0 = (pos_0, neg_0)
├── pair_1 = (pos_1, neg_1)
└── pair_2 = ...
```

这种“一题多 pair”方案不进入当前 controlled main experiments。

---

# 3. original_question_id 与 pair_id 的关系

在当前主实验约束下，对每个被保留进入训练的数据问题：

```text
1 question
↔
1 Judge pair
```

因此主数据中：

\[
original\_question\_id
\]

与：

\[
pair\_id
\]

是一一对应的。

controlled 数据必须同时保存两者，因为语义不同：

```text
original_question_id
```

表示原始 MATH / ReClor question 的身份。

```text
pair_id
```

表示进入 Judge 训练的固定 positive-negative pair 身份。

推荐 schema：

```json
{
  "original_question_id": "math_000123",
  "pair_id": "math_000123_pair_00",
  "response_pos": "...",
  "response_neg": "...",
  "permutation_id": 0
}
```

对应 BA 样本：

```json
{
  "original_question_id": "math_000123",
  "pair_id": "math_000123_pair_00",
  "response_pos": "...",
  "response_neg": "...",
  "permutation_id": 1
}
```

虽然当前两者是一一对应，但保留两个字段有利于以后扩展，而不需要现在改变训练逻辑。controlled 数据预处理必须验证：

```text
original_question_id -> 唯一 pair_id
pair_id -> 恰好两个 permutation_id：0、1
```

Task 2B 中用 `original_question_id` 补位缺失 `pair_id` 的行为只作为兼容路径；正式 controlled 数据不得依赖该 fallback。

---

# 4. Task 2B 不需要因为 MCQ 而返工

Task 2B 当前 explicit identity 为：

```python
RolloutId(
    pair_id,
    permutation_id,
    rollout_slot
)
```

并且当前主线限定：

```text
permutation_id ∈ {0, 1}
```

分别表示：

```text
0 = AB
1 = BA
```

这与当前主研究完全一致。

因此：

\[
\boxed{
\text{Task 2B 不需要为了未来 MCQ 支持改成任意 P}
}
\]

当前 Task 2B 的目标仍然只是确保：

```text
dataset
→ DataProto
→ repeat(n)
→ rollout
→ reorder / balance
→ reward
→ consistency pairing
```

全过程中：

```text
pair_id
permutation_id
rollout_slot
```

不丢失、不错位。

Consistency pairing 继续使用：

\[
(pair\_id,\ rollout\_slot)
\]

并在同一个 key 下寻找：

```text
permutation_id = 0
permutation_id = 1
```

即：

\[
AB_i \leftrightarrow BA_i
\]

---

# 5. 为什么 MCQ 不应现在并入主实验

PA-GRPO 的 MCQ setting 在高层上与 Pairwise Judge 使用相同思想：

\[
\text{一个原始实例}
\rightarrow
P\text{ 个等价 permutation}
\rightarrow
\text{每个 permutation }N\text{ 个 rollout}
\]

Pairwise Judge 是：

\[
P=2
\]

而 PA 官方 MCQ setting 使用多个选项排列。当前仓库中的官方训练数据每题使用：

\[
P=5
\]

当前 4-option MCQ evaluation 路径则执行 24 个全排列，即：

\[
P=24
\]

因此本文讨论的 MCQ 扩展统一写作 \(P>2\)；具体训练和评测阶段的 \(P\) 必须在 run manifest 中分别记录。

这意味着 MCQ 并不只是“多几个 permutation”，而会同时改变：

```text
semantic mapping
consistency reward
grouping structure
subgroup imbalance definition
实验 benchmark
```

如果现在把 MCQ 并入主线，会把当前清晰的研究问题从：

> 为什么 pairwise Judge 中 global-only 在某些情况下不够？local correction 什么时候必要？

扩展成：

> 方法能否从 P=2 泛化到任意 P>2 permutation group？

这是一个新的研究问题，不应与当前主线混在一起。

---

# 6. MCQ 与 Pairwise Judge 的统一抽象

虽然当前不做 MCQ 主实验，但可以保留统一抽象：

\[
q
\rightarrow
\{p_1,\ldots,p_P\}
\rightarrow
\text{每个 }p_t\text{ 采样 }N\text{ 次}
\]

rollout identity 可统一理解为：

\[
(instance\_id,\ permutation\_id,\ rollout\_slot)
\]

其中：

### Pairwise Judge

```text
instance_id = pair_id
P = 2
permutation_id ∈ {0, 1}
```

### MCQ

```text
instance_id = original_question_id
P > 2
permutation_id ∈ {0, ..., P-1}
```

或者直接保存 permutation string，例如：

```text
ABCD
BCDA
DCAB
...
```

这个统一抽象只作为未来设计依据，不要求当前 Task 2B 实现任意 P。

---

# 7. MCQ 下 consistency reward 不再是二元配对

当前 Pairwise Judge：

```text
AB_i
BA_i
```

映射回 semantic candidate 后判断二者是否一致。

即：

\[
AB_i \leftrightarrow BA_i
\]

而 MCQ 的 P>2 setting 应变成：

```text
slot i:
perm_0 rollout_i
perm_1 rollout_i
...
perm_{P-1} rollout_i
```

先把各 permutation 下的表面选项映射回 canonical semantic answer，再在：

\[
\{a_i^{(1)},a_i^{(2)},\ldots,a_i^{(P)}\}
\]

上计算一致性或 majority / unique-mode reward。

因此当前：

```python
pair_consistency_rollouts()
```

这种二元 pairing 逻辑不能直接作为 MCQ 的最终实现。

未来若做 MCQ，应新增独立的：

```text
multi_permutation_grouping
```

而不是修改 Pairwise 主路径。

---

# 8. ALC 在 MCQ 下也需要重新定义

当前 ALC 的 subgroup imbalance 定义针对两个 permutation：

\[
d=
\frac{
|\mu_{AB}-\mu_{BA}|
}{
\sigma_G+\epsilon
}
\]

然后：

\[
\lambda_{local}=\operatorname{clip}(d,0,1)
\]

\[
A_{ALC}=A_g+\lambda_{local}A_l
\]

这个定义天然是 P=2 的。

如果未来扩展到 MCQ 的 P>2，需要重新定义多 subgroup imbalance，例如可考虑：

\[
d_{multi}
=
\frac{
\operatorname{Std}(\mu_1,\mu_2,\ldots,\mu_P)
}{
\sigma_G+\epsilon
}
\]

或者其他 permutation-subgroup dispersion 指标。

但这属于未来方法扩展，当前项目不提前固定公式。

因此必须避免：

> 直接把 Pairwise ALC 公式原样应用到 MCQ，并声称已经自然泛化。

---

# 9. 对当前计划的具体修改

当前 `PLAN.md` / `PLAN_ADDENDUM.md` 主体不需要重写。

只补充以下两条规范。

## 9.1 主实验数据约束

新增：

> **Controlled main experiment data constraint**：每个被保留进入训练的 `original_question_id` 恰好对应一个固定 positive-negative Judge pair；无法形成有效正负 pair 的问题直接过滤。该 pair 再构造 AB / BA 两个 permutation。当前主实验不使用“一题多 pair”训练数据。

因此主线中：

\[
original\_question\_id
\leftrightarrow
pair\_id
\]

是一一对应关系。

---

## 9.2 MCQ 定位

新增：

> **Optional Extension — Multi-Permutation MCQ**：PA-GRPO 的 MCQ P>2 setting 可作为主实验完成后的扩展，用于检验方法从 Pairwise P=2 到多 permutation group 的泛化能力。该扩展不属于当前 EIS–PA–ALC 主实验，并需要单独定义 multi-permutation identity、consistency grouping 与 subgroup imbalance 指标。

---

# 10. 当前项目执行顺序不变

本补充说明不改变现有执行顺序。

当前仍然是：

```text
Task 1A
Windows evaluation smoke
    ↓
Task 1.5
Paper–Code Fidelity Audit
    ↓
Task 2A
Linux official advantage fidelity
    ↓
Task 2B
Explicit rollout identity Trainer integration
    ↓
Controlled evaluator hardening
    ↓
AutoDL real vLLM / GRPO smoke
    ↓
EIS / PA / ALC implementation
    ↓
1.5B Pilot
    ↓
1K ablation
    ↓
7B main experiment
```

MCQ 不插入上述主线。

---

# 11. 主研究最终保持的实验边界

当前主实验最终仍然是：

```text
Task:
Pairwise LLM-as-a-Judge

Data:
MATH + ReClor reasoning pairs

Permutation:
AB / BA

Methods:
Vanilla GRPO
Global-only
Global-Gated
EIS
PA
ALC
ALC+Con

Evaluation:
JudgeBench
ReasoningJudgeBench

Analysis:
Permutation subgroup imbalance
Global/local conflict
Local rescue
EIS–Global gain
ALC behavior
```

MCQ：

```text
Optional only
```

只有在主实验完成且时间、算力允许时，再考虑：

```text
P > 2
multi-permutation consistency
multi-subgroup ALC
MCQ benchmark
```

---

# 12. 最终决策

\[
\boxed{
\text{当前项目继续以 Pairwise Judge, P=2 为唯一主线}
}
\]

\[
\boxed{
\text{一个原始 question 在 controlled main experiments 中只对应一个固定 Judge pair}
}
\]

\[
\boxed{
\text{Task 2B 不因 MCQ 讨论返工}
}
\]

\[
\boxed{
\text{MCQ 作为后续可选的 P>2 泛化实验}
}
\]

这既保持了当前研究问题的可解释性，也为后续扩展预留了清晰接口。
