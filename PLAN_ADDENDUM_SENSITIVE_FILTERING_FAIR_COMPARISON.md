# Permutation-GRPO 项目补充修改：Permutation-Sensitive Filtering 的公平比较协议

> **用途**：本文件补充 `PLAN.md`、`PLAN_ADDENDUM.md` 以及 `PLAN_ADDENDUM_PAIRWISE_SCOPE_AND_MCQ_EXTENSION.md`，用于修正此前过于粗略的 `D_all vs D_sensitive` 设计，明确 permutation-sensitive filtering 的公平比较方式。
>
> **核心原则**：
>
> \[
> \boxed{\text{主算法比较固定训练数据；过滤策略比较固定数据规模与训练预算}}
> \]
>
> 之后任何正式实验、脚本、配置、结果表和结论，均按本文件执行。

---

# 1. 为什么原先的 `D_all vs D_sensitive` 不能直接作为 filtering effect 结论

定义：

\[
D_{all}
\]

为所有满足主训练数据构造条件的 Judge pairs：

```text
MATH / ReClor question
→ 生成多个 candidate responses
→ outcome verification
→ 同时存在 correct 与 incorrect response
→ 每题保留一个固定 pos-neg pair
```

然后使用 base judge 在 AB / BA 两种 permutation 下进行推理，只保留语义选择发生变化的 pair，得到：

\[
D_{sensitive}\subset D_{all}
\]

例如：

```text
AB:
A = pos
B = neg
model → A

BA:
A = neg
B = pos
model → A
```

表面标签虽然同为 `A`，但 semantic choice 从 `pos` 变成 `neg`，因此该 pair 被定义为 permutation-sensitive。

问题在于：

\[
|D_{sensitive}| < |D_{all}|
\]

如果直接比较：

\[
D_{all}
\quad vs \quad
D_{sensitive}
\]

则同时改变了：

```text
1. 数据类型 / sample selection
2. 数据规模
3. 通常也会间接改变总 optimization steps / rollout count
```

因此最终性能差异无法归因于 filtering 本身。

---

# 2. 主实验中的算法比较：统一使用 `D_all`

本项目的核心研究问题仍然是：

> 在相同训练数据、相同 backbone、相同 reward、相同 rollout budget 下，比较不同 advantage / consistency 设计的效果。

因此主表中的：

```text
Vanilla GRPO
Global-only
Global-Gated
EIS
PA-Controlled
ALC
ALC+Con
```

必须统一训练在：

\[
\boxed{D_{all}}
\]

上。

即：

```text
同一批 original_question_id
同一批 Judge pair
同一批 AB / BA permutations
同一 train split
```

这样主表回答的才是：

\[
\boxed{\text{algorithm effect}}
\]

---

# 3. `D_eligible`、`D_sensitive` 与 `D_consistent` 的构造方式

在 `D_all` 构造完成后，使用固定 base judge 对每个 pair 分别执行 AB / BA，并通过 Task 3 的 strict controlled parser 解析输出：

```text
D_all
  ↓
固定 base judge 对每个 pair 跑 AB / BA
  ↓
strict controlled parser
  ↓
├─ AB=None 或 BA=None
│      → undetermined
│      → 不进入 sensitivity 分析
│
└─ AB、BA 均有效
       → D_eligible
          ├─ semantic(AB) != semantic(BA)
          │      → D_sensitive
          └─ semantic(AB) == semantic(BA)
                 → D_consistent
```

因此：

\[
D_{sensitive}\subset D_{eligible}\subseteq D_{all}
\]

`None`、`C`、`D` 或其他非法 Judge 输出不进入 surface-to-semantic truth table，统一标记为：

```text
undetermined
```

## 3.1 固定 surface-to-semantic truth table

映射必须作为单一 truth table 实现并由单测锁定，不能散落在 filtering 代码中：

| permutation | surface=A | surface=B |
|---|---|---|
| AB | pos | neg |
| BA | neg | pos |

由此得到：

```text
AB=A, BA=B → pos / pos → consistent
AB=B, BA=A → neg / neg → consistent

AB=A, BA=A → pos / neg → sensitive
AB=B, BA=B → neg / pos → sensitive
```

## 3.2 Filtering inference 必须确定性

正式 sensitivity filtering 固定：

```text
temperature = 0
top_p = 1.0
固定 prompt template / prompt contract
固定 base judge model checkpoint / revision
固定 max_new_tokens
controlled parser
```

从而 sensitivity 状态是以下输入的确定性结果：

\[
(pair,\ base\ judge,\ prompt\ contract)
\rightarrow
sensitive/consistent/undetermined
\]

必须记录：

```text
original_question_id
pair_id
AB surface prediction
BA surface prediction
AB semantic prediction
BA semantic prediction
sensitivity status: sensitive / consistent / undetermined
base judge model
base judge checkpoint / revision
temperature
top_p
max_new_tokens
prompt template / contract revision
parser contract
```

确保 sensitivity filtering 完全可复现。

---

# 4. Filtering Effect 的正式比较：必须等规模

设：

\[
M=|D_{sensitive}|
\]

从 `D_eligible` 中随机抽取相同数量的样本，形成：

\[
D_{random}^{(M)}
\]

正式 filtering ablation 比较：

\[
\boxed{
D_{sensitive}^{(M)}
\quad vs \quad
D_{random}^{(M)}
}
\]

且：

\[
|D_{sensitive}^{(M)}|
=
|D_{random}^{(M)}|
=
M
\]

这样才能隔离：

\[
\boxed{\text{sample selection / filtering effect}}
\]

---

# 5. Random baseline 不只抽一次

不能只生成一个随机子集。

至少构造：

\[
D_{random,1}^{(M)},
D_{random,2}^{(M)},
D_{random,3}^{(M)}
\]

建议至少 3 个 random subset seeds：

```text
subset_seed=1
subset_seed=2
subset_seed=3
```

正式报告：

```text
Sensitive_M

Random_M_subset_seed1
Random_M_subset_seed2
Random_M_subset_seed3
Random_M_mean ± std
```

比较：

\[
D_{sensitive}^{(M)}
\quad vs \quad
\operatorname{Mean}
\left(
D_{random,1}^{(M)},
D_{random,2}^{(M)},
D_{random,3}^{(M)}
\right)
\]

如果算力允许，可增加 random subset seed，但 3 个为最低建议。

---

# 6. Random subset 的抽样原则

随机抽样必须从：

\[
D_{eligible}
\]

中进行。

不得从 `D_all` 直接抽样，否则 random baseline 可能包含 base judge 无法有效解析的样本，而 `D_sensitive` 天然全部可解析。

random baseline **允许抽到 sensitive 样本**。正式定义为：

\[
D_{random}^{(M)}
\sim
\operatorname{SampleWithoutReplacement}(D_{eligible}, M)
\]

它回答的是：

> 专门选择 sensitive 数据，是否优于从正常 eligible pool 中随机选择同样多的数据？

不得把 random baseline 改成只从 `D_consistent` 抽样；`D_sensitive vs D_consistent` 回答的是另一项研究问题，不能代替 filtering-effect baseline。

抽样必须：

```text
without replacement
以 original_question_id 为唯一抽样 / 去重单位
固定 subset_seed
保存 sampled original_question_id 与 pair_id manifest
```

即使当前主数据满足 `1 question ↔ 1 pair`，正式实现仍以 `original_question_id` 为去重单位，避免未来 schema 扩展后出现同题泄漏。

如果后续发现 MATH / ReClor 比例、回答长度或题目难度在 `D_sensitive` 与随机集之间差异过大，可增加 matched random baseline，例如匹配：

```text
source: MATH / ReClor
response length bin
question difficulty proxy
```

但第一轮正式协议保持：

\[
\boxed{\text{equal-size deterministic random subset}}
\]

---

# 7. Filtering 比较时训练预算必须一致

\[
D_{sensitive}^{(M)}
\]

和：

\[
D_{random}^{(M)}
\]

必须固定相同：

```text
backbone
initial checkpoint
training algorithm
base reward
consistency setting
rollout.n
batch size
learning rate
KL config
LoRA config
max prompt length
max response length
epochs
optimizer
固定 training_seed
evaluation protocol
```

并重点保证：

\[
\boxed{\text{optimization steps 一致}}
\]

以及：

\[
\boxed{\text{总 rollout 数一致}}
\]

若未来动态 batching 导致 token budget 差异明显，额外记录：

```text
total generated tokens
total valid response tokens
```

但第一层公平性以：

```text
same number of pairs
same epochs
same optimization steps
same rollout.n
```

为硬要求。

## 7.1 `subset_seed` 与 `training_seed` 必须分离

```text
subset_seed
→ 只决定从 D_eligible 中抽到哪 M 个 original_question_id

training_seed
→ 决定 LoRA 初始化、数据顺序、rollout sampling 等训练随机性
```

第一版最低协议：

```text
subset_seed = 1 / 2 / 3
training_seed = 42（所有 filtering runs 固定相同）
```

在这个设计下，Random runs 的 `mean ± std` 只描述：

\[
\boxed{\text{random subset selection variability}}
\]

不能写成 training variance。

如果后续算力允许，可升级为二维设计：

```yaml
subset_seed: [1, 2, 3]
training_seed: [41, 42, 43]
```

此时共需 9 个 Random 训练 run，并分别报告 subset-selection 与 training variability。

---

# 8. Filtering Effect 与 Sample Efficiency 必须分开报告

## 8.1 Filtering Effect

回答：

> 在相同数据规模和训练预算下，选择 permutation-sensitive 样本是否优于随机选择样本？

比较：

\[
\boxed{
D_{sensitive}^{(M)}
\quad vs \quad
D_{random}^{(M)}
}
\]

这是严格的：

\[
\boxed{\text{sample selection effect}}
\]

## 8.2 Sample Efficiency

回答：

> 只训练 sensitive subset，能否用更少数据达到接近甚至超过完整数据集的效果？

比较：

\[
\boxed{
D_{sensitive}^{(M)}
\quad vs \quad
D_{all}^{(N)}
}
\]

其中：

\[
M<N
\]

例如：

```text
D_all       = 3000 pairs
D_sensitive = 800 pairs
```

如果：

```text
D_all 3000       → 61.0
D_sensitive 800  → 60.5
```

正确结论是：

> sensitive filtering 具有较高 sample efficiency，使用更少训练样本获得了接近完整数据集的表现。

不能写成：

> sensitive filtering 在严格公平比较下优于 D_all。

因为两边数据规模不同。

---

# 9. 推荐实验结构

Phase 6 的 filtering 分析修改为：

```text
Step 1
构造 D_all

Step 2
固定 base judge
对 D_all 全部 pair 跑 AB / BA
使用 deterministic decoding + strict controlled parser

Step 3
生成 sensitivity manifest，并划分：
undetermined / D_eligible

Step 4
从 D_eligible 得到：
D_sensitive / D_consistent
M = |D_sensitive|

Step 5
从 D_eligible 按 original_question_id 无放回构造：
D_random_M_subset_seed1
D_random_M_subset_seed2
D_random_M_subset_seed3

Step 6
选择一个固定训练方法
固定同一个 training_seed
分别训练：
D_sensitive_M
D_random_M_subset_seed1
D_random_M_subset_seed2
D_random_M_subset_seed3

Step 7
比较 filtering effect

Step 8
额外训练 / 复用 D_all_N 结果
比较 sample efficiency
```

---

# 10. Filtering 分析使用哪个训练方法

第一版只选 **一个固定方法** 做数据筛选实验。

优先：

```text
PA-Controlled
```

因为 permutation-sensitive filtering 本身来自 PA-GRPO 的设计动机。

如果算力允许，再补：

```text
EIS
或
ALC
```

作为附加分析。

第一版不需要对所有方法都重复 Sensitive vs Random。

正式最低要求：

\[
\boxed{\text{one fixed method + equal-size filtering comparison}}
\]

---

# 11. 防止数据泄漏

`D_eligible`、`D_sensitive`、`D_consistent` 与 `D_random` 都只能从：

```text
training split
```

中构造。

不得使用：

```text
JudgeBench
ReasoningJudgeBench
validation benchmark
test benchmark
```

来决定一个训练 pair 是否 sensitive。

Sensitivity detection 只使用：

```text
training pair 的 AB / BA predictions
```

评测集只用于最终 evaluation。

---

# 12. Manifest 与可复现性要求

必须保存：

```text
artifacts/manifests/d_all.jsonl
artifacts/manifests/d_eligible.jsonl
artifacts/manifests/d_sensitive.jsonl
artifacts/manifests/d_consistent.jsonl
artifacts/manifests/d_undetermined.jsonl
artifacts/manifests/d_random_M_subset_seed1.jsonl
artifacts/manifests/d_random_M_subset_seed2.jsonl
artifacts/manifests/d_random_M_subset_seed3.jsonl
```

每个 manifest 至少记录：

```text
pair_id
original_question_id
source
sensitivity flag
sensitivity status
subset_seed
```

`sensitivity manifest` 额外记录：

```text
AB surface prediction
BA surface prediction
AB semantic prediction
BA semantic prediction
base judge
base judge checkpoint / revision
prompt template / contract revision
parser contract
temperature
top_p
max_new_tokens
```

所有训练 run metadata 必须写入：

```text
dataset_manifest
num_pairs
num_permutations
rollouts_per_permutation
optimization_steps
training_algorithm
subset_seed
training_seed
```

---

# 13. 正式结果表建议

## 13.1 Filtering Effect

| Dataset | #Pairs | Method | Acc | Consistency | CA |
|---|---:|---|---:|---:|---:|
| Sensitive | M | PA-Controlled | ... | ... | ... |
| Random subset seed 1 | M | PA-Controlled | ... | ... | ... |
| Random subset seed 2 | M | PA-Controlled | ... | ... | ... |
| Random subset seed 3 | M | PA-Controlled | ... | ... | ... |
| Random mean ± std | M | PA-Controlled | ... | ... | ... |

第一版所有行固定同一个 `training_seed`；表中的 `mean ± std` 仅表示 random subset selection variability。

## 13.2 Sample Efficiency

| Dataset | #Pairs | Relative Data | Acc | Consistency | CA |
|---|---:|---:|---:|---:|---:|
| D_all | N | 100% | ... | ... | ... |
| D_sensitive | M | M/N | ... | ... | ... |

两张表的结论不能混在一起。

---

# 14. 允许的结论与禁止的结论

如果：

\[
D_{sensitive}^{(M)}
>
D_{random}^{(M)}
\]

可以写：

> 在等规模、等训练预算条件下，permutation-sensitive filtering 提高了训练样本的信息密度或 permutation-robustness learning efficiency。

如果：

\[
D_{sensitive}^{(M)}
\approx
D_{all}^{(N)}
\]

且：

\[
M\ll N
\]

可以写：

> sensitive subset 展现出更高的 sample efficiency。

不能仅根据不同规模的：

\[
D_{sensitive}^{(M)}
\quad vs \quad
D_{all}^{(N)}
\]

直接归因 filtering 本身。

---

# 15. 对原项目计划的修改

原计划中任何类似：

```text
D_all vs D_sensitive
```

且用于 **filtering effect** 的描述，统一替换为：

\[
\boxed{
D_{sensitive}^{(M)}
\quad vs \quad
D_{random-from-eligible}^{(M)}
}
\]

其中：

\[
M=|D_{sensitive}|
\]

并至少使用 3 个 random subset seeds。

其中 random baseline 从 `D_eligible` 无放回抽样，允许自然抽到部分 sensitive 样本，并以 `original_question_id` 为唯一抽样 / 去重单位。

原来的：

\[
D_{all}^{(N)}
\quad vs \quad
D_{sensitive}^{(M)}
\]

保留，但重新定义为：

\[
\boxed{\text{sample-efficiency analysis}}
\]

不再作为 strict filtering ablation。

---

# 16. 当前执行顺序中的位置

本修改不影响当前 Task 2B、controlled evaluator hardening、真实 vLLM smoke、EIS/ALC 实现等前置工作。

Filtering 分析仍然放在主实验之后：

```text
Task 2B
↓
Controlled evaluator hardening
↓
AutoDL vLLM / GRPO smoke
↓
EIS / PA / ALC implementation
↓
1.5B Pilot
↓
1K ablation
↓
7B main experiment
↓
Permutation-sensitive filtering analysis
    ├── Sensitive_M vs Random-from-eligible_M
    └── Sensitive_M vs All_N
```

当前不需要提前实现 filtering training pipeline，只需在数据 schema 和 manifest 中保留后续所需字段。

---

# 17. 最终执行约束

后续实现与实验遵循以下硬约束：

\[
\boxed{
\text{主算法比较：所有方法统一使用 } D_{all}
}
\]

\[
\boxed{
\text{Filtering effect：Sensitive}_M
\text{ vs Random-from-eligible}_M
}
\]

\[
\boxed{
\text{任一 permutation strict parse 无效} \Rightarrow \text{undetermined，不进入 sensitivity 分析}
}
\]

\[
\boxed{
\text{Random baseline 从 }D_{eligible}\text{ 按 original_question_id 无放回抽样}
}
\]

\[
\boxed{
\text{至少 3 个 random subset seeds}
}
\]

\[
\boxed{
\text{第一版固定同一个 training_seed；mean ± std 只表示 subset-selection variability}
}
\]

\[
\boxed{
\text{Filtering inference 使用固定 base judge、固定 prompt contract 与 deterministic decoding}
}
\]

\[
\boxed{
\text{Filtering comparison 必须等数据规模、等 rollout budget、等 optimization steps}
}
\]

\[
\boxed{
\text{Sensitive}_M
\text{ vs All}_N
\text{ 只解释为 sample efficiency}
}
\]

\[
\boxed{
\text{评测集不得参与 sensitivity filtering}
}
\]

本文件优先于此前计划中任何不够严格的 `D_all vs D_sensitive` 表述。
