# EIS-GRPO × PA-GRPO 统一复现、机制分析与改进项目计划书

> 项目定位：LLM 后训练 / GRPO / LLM-as-a-Judge / permutation robustness
>
> 建议周期：4 周完成主交付，5–6 周完成双种子与扩展实验
>
> 推荐主模型：Qwen2.5-7B-Instruct + LoRA
>
> Pilot 模型：Qwen2.5-1.5B-Instruct
>
> 推荐基础代码：PA-GRPO 官方仓库（基于 verl），在其上最小侵入地加入 EIS 与自定义方法
>
> Pairwise 主线与 MCQ 扩展边界见 [`PLAN_ADDENDUM_PAIRWISE_SCOPE_AND_MCQ_EXTENSION.md`](PLAN_ADDENDUM_PAIRWISE_SCOPE_AND_MCQ_EXTENSION.md)。其中“一题一个固定 Judge pair”及 MCQ 仅作为可选扩展的约束适用于 controlled main experiments。

---

## 0. 项目摘要

本项目不以“精确复现两篇论文的全部绝对数值”为目标，而以**在统一实验条件下复现核心机制、解释两篇方法的经验差异，并提出一个可验证的改进**为目标。

核心研究对象：

1. **EIS-GRPO / J4R**：将同一 Judge 样本的两个顺序 `AB` / `BA` 视为 equivalent initial states；其核心 advantage 为：

   \[
   A_{EIS}=A_{global}+A_{local}
   \]

   其中 global 跨两个 permutation 统一标准化，local 在各 permutation 内分别标准化。

2. **PA-GRPO**：将不同 permutation 组成一个 Permutation Group；核心组件是：
   - Cross-Permutation Advantage：跨 permutation 统一算 advantage；
   - Consistency-Aware Reward：直接奖励 permutation 后仍选择相同 semantic candidate；
   - 低方差 gate：当 permutation-group reward 方差过小时令 advantage 为 0，避免放大噪声。

项目要回答的主要问题：

- **RQ1**：在统一 backbone、训练数据、基础 reward、rollout budget 下，EIS 的 `Global+Local` 与 PA 的 `Global-only + consistency reward` 各自贡献是什么？
- **RQ2**：为什么 J4R 中 global-only 几乎失效，而 PA 的 cross-permutation global advantage 在其设置下可以产生收益？
- **RQ3**：不同 permutation subgroup 的难度不平衡，是否决定了 local advantage 的必要性？
- **RQ4（改进）**：能否让 local advantage 的权重随 subgroup imbalance 自适应变化，而不是像 EIS 一样始终固定加入？

最终主改进暂定名：**Adaptive Local Correction (ALC)**。

---

# 1. 论文事实与本项目需要保留的关键机制

## 1.1 EIS-GRPO / J4R

J4R 在 pairwise Judge 场景中使用两个等价状态：

- permutation 1：`A=y1, B=y2`
- permutation 2：`A=y2, B=y1`

假设 `y1` 是更优回答，则正确表面标签分别是 `A` 与 `B`，但 semantic winner 始终是 `y1`。

原论文的 advantage：

\[
A^{(i,l)}=
\frac{R^{(i,l)}-\bar R_{global}}{\sigma_{global}}
+
\frac{R^{(i,l)}-\bar R_l}{\sigma_l}
\]

即：

\[
A_{EIS}=A_g+A_l
\]

J4R 的 Judge reward：

\[
R=R_j+R_f
\]

其中判断正确 `R_j=1`，错误 `R_j=0`；格式正确 `R_f=+0.5`，格式错误 `R_f=-0.5`。

原论文使用 `G=32`，即两个顺序各 16 个 rollout。训练 pair 来自 MATH + ReClor，通过多个模型生成 correct / incorrect responses，最终约 10K pair。

论文附录 D.1 对 global-only 的失败给出明确解释：当两个 subgroup 难度差异较大时，弱 subgroup 内相对优秀的 response 可能因为 global baseline 被强 subgroup 拉高而得不到足够正向 advantage；local advantage 可以修复这种“弱组优秀样本被低估”的问题。

### 注意：不要写错误的 L=1 单测

原 EIS 公式在 `L=1` 时：

\[
A_g=A_l=A_{GRPO}
\]

所以：

\[
A_{EIS}=2A_{GRPO}
\]

**并不会严格退化成 GRPO。** 因此之前“L=1 必须等于 GRPO”的想法不能作为单元测试。正确测试应该是：`L=1` 时 global 与 local 数值一致，EIS 等于二者之和。

---

## 1.2 PA-GRPO

PA-GRPO 将同一原始实例的多个 permutation 放入同一个 Permutation Group：

\[
G(x)=\{p^{(t)}\}_{t=1}^{P}
\]

每个 permutation 采样 `N` 个输出，因此一个原始实例总共有 `P*N` 个 rollout。

Pairwise Judge 中：

\[
P=2\quad (AB, BA)
\]

论文训练设置中每个 prompt variant 采 `N=8`，即每个 pair 总共 16 个 rollout。

### PA 基础 reward

论文 preliminary reward：

\[
r_{pre}=r_{acc}+r_{len}+r_{fmt}
\]

官方论文实现：
- correctness：`+1/-1`
- length regularization：`±0.1`
- format regularization：`±0.3`

### PA consistency reward

Judge 中，将两个 permutation 的第 `i` 个 rollout 对齐：

\[
r_{con}^{(1,i)}=r_{con}^{(2,i)}=
\begin{cases}
+1,&z^{(1,i)}=z^{(2,i)}\\
-1,&z^{(1,i)}\neq z^{(2,i)}
\end{cases}
\]

这里比较的是 **semantic choice**，不是表面的 A/B 字母。

例如：
- AB 输出 A -> semantic `y1`
- BA 输出 B -> semantic `y1`

这两个是“一致”。

PA 默认 consistency 系数论文选择 `lambda=1.0`。

### PA cross-permutation advantage

将两个 permutation 的全部 `P*N` rewards 放在一起：

\[
\mu_G=\frac{1}{PN}\sum_{t,i}r^{(t,i)}
\]

\[
\sigma_G=Std(\{r^{(t,i)}\})
\]

\[
A_{PA}^{(t,i)}=
\begin{cases}
0,&\sigma_G<\delta\\
\frac{r^{(t,i)}-\mu_G}{\sigma_G+\epsilon},&otherwise
\end{cases}
\]

这与 EIS 的 `A_global` 是同一类信号，但 PA 不额外加入 `A_local`。

### PA 的数据筛选是重要 confound

PA 在训练前使用 base model 对不同 permutation 做推理，并**只保留 permutation 预测不一致的样本**，以强化 selection-bias 学习信号。

因此 EIS 与 PA 的原论文结果不能直接拿来归因于 advantage 公式，因为至少同时存在：

- advantage 形式不同；
- consistency reward 不同；
- sigma gate 不同；
- reward 设计不同；
- 数据源不同；
- PA 有 permutation-sensitive filtering；
- backbone 与训练配置不同。

本项目的价值就在于控制这些变量。

---

# 2. 明确的项目范围

## 2.1 必须完成

1. 跑通 PA-GRPO 官方训练与评测框架。
2. 在同一框架内实现：
   - vanilla/local GRPO
   - global-only
   - EIS = global + local
   - global + sigma gate
   - PA full = global + gate + consistency reward
   - ALC（本项目改进）
3. 在统一训练条件下做核心消融。
4. 在 JudgeBench + ReasoningJudgeBench 上统一评测。
5. 分析 subgroup imbalance 与 local advantage 价值的关系。
6. 用 7B 级模型做最终 scale-up 验证。

## 2.2 暂不作为主线

以下内容只有主线完成后再做：

- SFT / DPO 大规模 baseline；
- 3/4 candidate listwise Judge；
- online hard-example mining；
- 多 backbone 全量复现；
- 完整 paper-scale 10K × G=32 复现 J4R；
- 全参数训练。

这些不应阻塞主项目。

---

# 3. 最终研究问题与假设

## RQ1：Global / Local / Consistency 各自作用是什么？

控制其他变量，仅改变：

- local advantage 是否存在；
- consistency reward 是否存在；
- sigma gate 是否存在。

主指标：Acc / Consistency / Consistent Accuracy。

---

## RQ2：Global-only 什么时候会失败？

假设：

> 当 `AB` 和 `BA` 两个 subgroup 的平均表现差异很大时，global baseline 会压低弱 subgroup 内优秀 response 的正向 advantage，local advantage 的收益更大。

定义 subgroup imbalance：

\[
\Delta_\mu=|\mu_{AB}-\mu_{BA}|
\]

更推荐使用标准化版本：

\[
d=\frac{|\mu_{AB}-\mu_{BA}|}{\sigma_G+\epsilon}
\]

注意：用于分析/自适应权重的 `mu` 优先来自 **base reward（不含 consistency reward）**，避免形成“用 consistency reward 自己定义 imbalance 再证明 consistency 有效”的循环。

---

## RQ3：Local advantage 的“救援效应”能否被直接观察？

除整体 Acc/Consistency 外，必须记录：

### sign conflict rate

\[
SCR=P(sign(A_g)\neq sign(A_l))
\]

表示 global 与 local 对某 rollout 的更新方向是否冲突。

### local rescue rate

定义弱 subgroup 内：

\[
LRR=P(A_g\le0\ \text{且}\ A_l>0)
\]

这正对应 J4R 附录 D.1 的“global 低估弱 subgroup 优秀样本”的现象。

### local suppression rate

\[
LSR=P(A_g>0\ \text{且}\ A_l<0)
\]

用于检测 global 是否在强 subgroup 中错误鼓励局部差 response。

分析目标：按 `d` 或 `Delta_mu` 分桶，观察 `SCR/LRR/LSR` 是否随 imbalance 增长。

---

# 4. 本项目改进：Adaptive Local Correction (ALC)

## 4.1 动机

EIS 固定：

\[
A=A_g+A_l
\]

PA 主要：

\[
A=A_g
\]

本项目提出：local correction 不一定对所有样本都同等必要。

当两个 permutation subgroup 难度接近时，global signal 可能已足够；当 subgroup 差异较大时，再增强 local correction。

---

## 4.2 第一版公式

计算 base-reward subgroup means：

\[
\mu_{AB},\mu_{BA}
\]

定义：

\[
d=\frac{|\mu_{AB}-\mu_{BA}|}{\sigma_G+\epsilon}
\]

无额外超参的第一版：

\[
\lambda_{local}=clip(d,0,1)
\]

最终：

\[
A_{ALC}=A_g+\lambda_{local}A_l
\]

解释：

- `d≈0`：两个 subgroup 很平衡 -> 主要依赖 global；
- `d` 大：存在明显 permutation difficulty imbalance -> 加强 local correction；
- `lambda<=1`：避免 local signal 无限放大。

### 为什么先不用 sigmoid

不要第一版就引入 `k`、`tau` 等额外超参。先验证核心假设；若有效，再做：

\[
\lambda=\sigma(k(d-\tau))
\]

作为附加 ablation。

---

## 4.3 ALC 与 consistency reward 的关系

先拆开验证：

- `ALC-noCon`：只测试 adaptive local 是否有效；
- `ALC+Con`：再加入 PA consistency reward，作为最终候选方法。

**lambda 的计算必须使用不含 `r_con` 的 base reward。**

否则会出现循环依赖：consistency reward 改变 subgroup means，再控制 local 权重，难以解释结果。

---

## 4.4 Advantage magnitude confound

EIS 的 `A_g + A_l` 会改变 advantage 的整体尺度；ALC 的尺度也随 `lambda` 变化。

因此必须记录：

- `adv_global_std`
- `adv_local_std`
- `adv_final_std`
- `adv_final_rms`
- actor gradient norm

如果发现某方法仅仅因为 advantage 尺度明显更大而更新更激进，需要增加一个 **norm-matched ablation**，但不要一开始就修改原论文公式。

推荐 norm-matched 只作为分析项：

\[
A' = A / (Std(A)+\epsilon)
\]

主表仍使用原始公式。

---

# 5. 代码基础与仓库策略

## 5.1 基础仓库

以官方 PA-GRPO 为底座：

`https://github.com/ECNU-Text-Computing/PA-GRPO`

原因：

- 已基于 verl；
- 已实现 permutation grouping；
- 已有 Judge / MCQ 脚本；
- 已有 evaluation；
- 已有预构造训练 parquet；
- README 明确核心改动集中在 `verl/trainer/ppo/ray_trainer.py`。

### 第一条工程规则

**不要直接大范围重写 vendored verl。**

目标是把研究逻辑抽离到独立模块，仅对 `ray_trainer.py` 做最小 dispatch patch。

---

## 5.2 推荐目录

在 PA-GRPO fork 中新增：

```text
PA-GRPO/
├── permstudy/
│   ├── __init__.py
│   ├── grouping.py
│   ├── advantages.py
│   ├── rewards.py
│   ├── adaptive.py
│   ├── parsing.py
│   ├── diagnostics.py
│   └── metrics.py
│
├── configs_permstudy/
│   ├── methods/
│   │   ├── grpo_local.yaml
│   │   ├── global_only.yaml
│   │   ├── global_gated.yaml
│   │   ├── eis.yaml
│   │   ├── pa_full.yaml
│   │   ├── eis_con.yaml
│   │   ├── alc.yaml
│   │   └── alc_con.yaml
│   ├── model/
│   │   ├── qwen25_15b_lora.yaml
│   │   └── qwen25_7b_lora.yaml
│   └── data/
│       ├── pilot.yaml
│       └── reasoning_main.yaml
│
├── scripts_permstudy/
│   ├── data/
│   │   ├── build_reasoning_pairs.py
│   │   ├── generate_candidates.py
│   │   ├── verify_candidates.py
│   │   ├── build_permutations.py
│   │   └── filter_sensitive.py
│   ├── train/
│   │   ├── run_pilot.sh
│   │   ├── run_small_ablation.sh
│   │   └── run_main_7b.sh
│   ├── eval/
│   │   ├── eval_judgebench.sh
│   │   ├── eval_rjb.sh
│   │   └── aggregate_metrics.py
│   └── analysis/
│       ├── analyze_advantage_conflict.py
│       ├── analyze_imbalance_bins.py
│       └── make_main_tables.py
│
├── tests_permstudy/
│   ├── test_semantic_mapping.py
│   ├── test_consistency_reward.py
│   ├── test_advantage_global.py
│   ├── test_advantage_local.py
│   ├── test_advantage_eis.py
│   ├── test_advantage_pa.py
│   ├── test_advantage_alc.py
│   ├── test_rollout_budget.py
│   └── test_end_to_end_smoke.py
│
└── artifacts/
    ├── manifests/
    ├── metrics/
    ├── tables/
    └── figures/
```

---

# 6. 统一配置接口

所有实验必须通过 config 切换，而不是复制多个 Trainer。

建议配置：

```yaml
method:
  advantage_mode: local        # local | global | eis | adaptive
  use_consistency_reward: false
  consistency_coef: 1.0
  use_sigma_gate: false
  sigma_gate_threshold: null   # 从 PA 官方代码读取，不猜

  adaptive:
    enabled: false
    imbalance_source: base_reward
    imbalance_mode: normalized_mean_gap
    lambda_mode: clipped_linear
    lambda_max: 1.0

permutation:
  num_permutations: 2
  rollouts_per_permutation: 8
  semantic_mapping: pairwise_ab_ba

reward:
  base_scheme: unified_j4r
  judge_correct: 1.0
  judge_wrong: 0.0
  format_correct: 0.5
  format_wrong: -0.5

logging:
  log_group_diagnostics: true
  log_advantage_diagnostics: true
```

### 重要：两种实验模式

必须区分：

#### Mode A：paper-faithful sanity reproduction

- PA 使用 PA 原始 reward / config；
- EIS 尽量使用 J4R 原始 reward / formula；
- 目标：验证实现方向正确，不要求统一比较。

#### Mode B：controlled comparison

所有方法统一：

- backbone
- data
- base reward
- prompt
- P/N
- LoRA
- LR
- KL
- train epochs
- rollout budget
- eval harness

主结论必须来自 Mode B。

不能把 Mode A 的两篇论文数字直接当成机制比较证据。

---

# 7. 数据计划

## 7.1 V0：先使用 PA 官方预构造数据跑通系统

目的：

- 不让数据生成阻塞代码验证；
- 先确认 verl / vLLM / LoRA / grouping / evaluation 全链路可运行。

只用于工程 smoke test 和 PA sanity reproduction。

---

## 7.2 主训练数据：J4R-style reasoning pair dataset

来源：

- MATH train
- ReClor train

目标规模：

- Pilot：200 pairs
- Small：1,000 pairs
- Main：2,000–3,000 pairs

不追求论文 10K。

### 数据单位

每条原始 pair：

```json
{
  "sample_id": "...",
  "source": "MATH|ReClor",
  "question_id": "...",
  "question": "...",
  "response_pos": "...",
  "response_neg": "...",
  "gold_semantic": "pos",
  "pos_generator": "...",
  "neg_generator": "...",
  "pos_len": 0,
  "neg_len": 0,
  "metadata": {}
}
```

训练时在线或预处理构造：

- `perm_id=AB`: `A=pos, B=neg, correct_surface=A`
- `perm_id=BA`: `A=neg, B=pos, correct_surface=B`

semantic winner 始终是 `pos`。

---

## 7.3 Candidate 生成

第一版建议：

- generator：Qwen2.5-7B-Instruct
- temperature：1.0
- 每题 K=8
- max completion：先 512，pilot 后根据真实长度调整

对每题：

1. 生成 K 个 reasoning response；
2. 自动提取最终答案；
3. 分成 correct / incorrect；
4. 若没有同时存在 correct 和 incorrect，则该题暂时丢弃；
5. 从正负集合中构造 pair。

为了减少明显 length cue，本项目可优先配对长度差较小的正负 response。注意：这是本项目的数据设计选择，不要写成 J4R 训练集的已知规则。

---

## 7.4 正确性验证

### ReClor

最终答案为离散选项，直接 canonicalize 后精确比较。

### MATH

必须实现独立 verifier，不允许只做裸字符串比较。

最低要求：

- 清除 `\\boxed{}` / `Answer:` 等 wrapper；
- 规范空格、LaTeX；
- 分数 / 小数等价处理；
- 常见数值表达 canonicalization；
- 能使用可靠 math verifier 库则优先使用；
- 随机抽查至少 100 个自动判定样本。

如果自动 verifier 在 MATH 上误判率明显，则第一版主训练可先提高 ReClor 占比，MATH 作为第二阶段加入。

---

## 7.5 Split 规则

**按原始 question_id 切分，而不是按 pair 切分。**

禁止同一道题的不同 candidate pair 同时出现在 train/validation。

JudgeBench 与 ReasoningJudgeBench 永远 eval-only，不得进入任何：

- 数据生成训练源；
- hard-sample filter source；
- reward tuning；
- threshold tuning。

---

# 8. Hard / sensitive 数据子集

为了研究 PA 数据筛选这个 confound，需要构造两套训练视图：

## D_all

所有有效 J4R-style pairs。

## D_sensitive

用**训练前 base model**在 AB/BA 上各做一次或少量固定推理。

若 semantic decision 不一致：

```text
AB -> pos
BA -> neg
```

则标记 `permutation_sensitive=true`。

主分析：

- Global-only 在 D_all vs D_sensitive 上的差异；
- EIS 在 D_all vs D_sensitive 上的差异；
- 判断 PA 的 data filtering 是否解释了部分结果差异。

不需要所有方法都双倍跑；优先只比较 `Global-only / EIS / ALC`。

---

# 9. 六个核心方法定义

## M0 Base

无训练，仅统一评测。

---

## M1 Vanilla GRPO / Local-only

AB/BA 分别独立标准化：

\[
A=A_l
\]

无 consistency reward，无 sigma gate。

---

## M2 Global-only

AB+BA 全部 rollout 一起标准化：

\[
A=A_g
\]

无 consistency reward，无 gate。

这是 J4R global-only ablation 的统一版。

---

## M3 Global-Gated

\[
A=A_g
\]

但当 `sigma_global < threshold`：

\[
A=0
\]

无 consistency reward。

目的：单独隔离 PA sigma gate。

---

## M4 EIS

\[
A=A_g+A_l
\]

无 consistency reward，无 gate。

---

## M5 PA-Full

统一实验中：

- base reward 与其他方法一致；
- 加 `r_con`；
- global advantage；
- sigma gate。

即：

\[
r=r_{base}+\lambda_{con}r_{con}
\]

\[
A=A_g(r)
\]

---

## M6 ALC（主改进）

\[
A=A_g+\lambda_{local}A_l
\]

其中：

\[
\lambda_{local}=clip(\frac{|\mu_{AB}-\mu_{BA}|}{\sigma_G+\epsilon},0,1)
\]

第一版不加 `r_con`。

---

## M7 ALC+Con（最终候选）

在 M6 基础上加 consistency reward。

`lambda_local` 始终由不含 consistency reward 的 base reward 计算。

---

# 10. 核心消融矩阵

## 10.1 小模型完整矩阵（1.5B）

全部跑：

| ID | Advantage | Con Reward | Gate | 目的 |
|---|---|---:|---:|---|
| M1 | Local | × | × | Vanilla GRPO |
| M2 | Global | × | × | J4R global-only |
| M3 | Global | × | ✓ | sigma gate 贡献 |
| M4 | Global+Local | × | × | EIS |
| M5 | Global | ✓ | ✓ | PA-Full |
| M6 | Adaptive | × | × | ALC 本体 |
| M7 | Adaptive | ✓ | 可先× | 最终组合 |

Small 阶段 1 seed 即可，用来筛掉无意义方法。

---

## 10.2 7B 主实验

至少训练：

- M1 Vanilla GRPO
- M2 Global-only
- M4 EIS
- M5 PA-Full
- M6 ALC

如果预算允许再加入：

- M3 Global-Gated
- M7 ALC+Con

主方法尽量 2 seeds；若预算不足，先全部 1 seed，再对表现最关键的 3 个方法补第二 seed。

目标不是一开始跑 8 格 × 3 seeds。

---

# 11. 训练配置建议

## 11.1 Pilot 模型

`Qwen2.5-1.5B-Instruct`

原因：

- 足够真实；
- verl 有单卡 H100 GRPO-LoRA 配置经验；
- 便于快速跑完完整 ablation。

Pilot：

```text
P = 2
N = 4 first smoke
随后 N = 8
pairs = 200 -> 1000
LoRA r = 16 or 32
max_completion = 256 first
```

---

## 11.2 主模型

首选：

`Qwen2.5-7B-Instruct + LoRA`

理由：

- J4R 本身有 Qwen2.5-7B-Instruct 初始化版本；
- Qwen 获取方便；
- verl 当前设备调优文档已有 `Qwen2.5-7B GRPO-LoRA 1×H100` 示例；
- PA 官方仓库 README 提供 qwen judge 脚本入口。

建议：

```text
LoRA rank: 32
LoRA alpha: 64
all linear layers if official PA path already稳定
optimizer: AdamW
lr: 1e-5 first reference
P = 2
N = 8 per permutation
max epochs = 2
KL beta: start 0.001 if following PA infrastructure
```

但统一比较时，所有方法必须共享这些训练超参。

### 不要一上来换很多超参

第一版以 PA 官方配置作为工程起点；若 Qwen2.5-7B 表现异常，再做最小调参。

---

# 12. Rollout budget 统一规则

所有方法按**原始 pair**计算预算：

\[
B_{rollout}=P\times N
\]

主实验：

\[
P=2,N=8\Rightarrow16\text{ completions / pair}
\]

Vanilla GRPO 也必须产生相同总数：

- AB 8 个
- BA 8 个

不能让 EIS/PA 比 GRPO 多采样。

每个 run manifest 记录：

- original pairs seen
- total permutation prompts
- total rollout completions
- total generated tokens
- optimizer steps

公平性必须按这些数字检查，而不是只看 epoch。

---

# 13. 必须写的单元测试

GPU 正式训练前，以下 tests 全部通过。

## T1 semantic mapping

给定：

```text
AB: A=pos, B=neg
BA: A=neg, B=pos
```

验证：

- AB 表面 A -> semantic pos
- BA 表面 B -> semantic pos

---

## T2 consistency reward

验证：

```text
AB=A(pos), BA=B(pos) -> +1
AB=A(pos), BA=A(neg) -> -1
```

parse failure 必须显式定义，不允许 silent pass。

建议第一版：任意一侧 parse failure -> `r_con=-1`，同时单独计 `parse_fail`；若 PA 官方代码语义不同，以官方实现为准并记录。

---

## T3 global advantage

人工 rewards，验证 global mean/std 与 z-score。

---

## T4 local advantage

每个 subgroup 内独立 mean≈0，std≈1（非 zero-std 时）。

---

## T5 J4R D.1 数值回归测试

使用论文示例：

- subgroup A：12×1.5 + 4×1.0
- subgroup B：12×0.0 + 4×1.0

验证：

Local：
- A: 1.5 -> 约 0.577
- A: 1.0 -> 约 -1.732
- B: 1.0 -> 约 1.732
- B: 0.0 -> 约 -0.577

Global：
- 1.5 -> 约 1.044
- 1.0 -> 约 0.285
- 0.0 -> 约 -1.234

EIS = Global + Local。

这个 test 非常重要，它能证明实现确实对应论文解释。

---

## T6 L=1 行为

验证：

```text
global == local == GRPO z-score
EIS == global + local == 2 * GRPO z-score
```

不要测试“EIS==GRPO”。

---

## T7 sigma gate

构造：

```text
rewards=[1,1,1,1,...]
```

确认：

```text
sigma < delta -> all advantage = 0
```

---

## T8 ALC

构造：

- balanced groups -> lambda 接近 0
- highly imbalanced groups -> lambda 接近 1

验证 final advantage 数值。

---

## T9 rollout budget

给定 batch 中原始 pair 数 `B`：

所有方法总 completion 数必须是：

\[
B\times P\times N
\]

---

## T10 end-to-end smoke

使用极小模型 / 2–4 pairs：

- 完成 rollout
- reward
- advantage
- loss
- backward
- optimizer step

确认：

- LoRA 参数变化；
- frozen base 参数不变化；
- 无 NaN / inf。

---

# 14. Evaluation Harness

## 14.1 Benchmark

主评测：

1. JudgeBench
2. ReasoningJudgeBench

两者不参与训练。

---

## 14.2 统一推理协议

所有模型：

- 同一个 Judge system prompt；
- 同一个 output parser；
- temperature=0 主评测；
- AB 和 BA 都跑；
- 相同 max generation length；
- 不同方法不得使用不同 prompt。

如果为了 paper-faithful reproduction 需要原论文 prompt，则单独标记为 reproduction mode，不得与 controlled table 混表。

---

## 14.3 指标

### Accuracy

展开 permutation 后的平均正确率。

### Consistency

AB 与 BA 是否选择相同 semantic response。

### Consistent Accuracy (CA)

只有 AB 与 BA 两边都正确才算该原始 pair 正确。

### Position preference

记录：

\[
P(first\ position)
\]

以及：

\[
|P(first)-0.5|
\]

### Format Validity

合法输出比例。

### Length

平均 completion tokens。

---

# 15. 训练诊断日志

每个训练 step 至少记录：

```text
reward/base_mean
reward/base_std
reward/total_mean
reward/consistency_mean

subgroup/mu_ab
subgroup/mu_ba
subgroup/delta_mu
subgroup/normalized_gap
subgroup/std_ab
subgroup/std_ba
subgroup/std_global
subgroup/zero_std_local_frac
subgroup/zero_std_global_frac

adv/global_mean
adv/global_std
adv/local_mean
adv/local_std
adv/final_mean
adv/final_std
adv/final_rms
adv/sign_conflict_rate
adv/local_rescue_rate
adv/local_suppression_rate

policy/kl
policy/entropy
policy/clip_fraction
policy/grad_norm

output/format_valid_rate
output/avg_length
```

所有关键研究结论应能从日志自动生成，而不是训练结束后手工猜。

---

# 16. 关键机制分析

## 16.1 Imbalance bin analysis

按 `normalized_gap=d` 把 validation/eval rollout 分为 4 桶：

- Q1: 最平衡
- Q2
- Q3
- Q4: 最不平衡

分别统计：

- Global-only accuracy
- EIS accuracy
- ALC accuracy
- sign conflict rate
- local rescue rate

核心图：

```text
x-axis: subgroup imbalance d
left y-axis: EIS - Global performance gain
right y-axis: local rescue rate
```

如果两者随 d 一起升高，则强支持 ALC 动机。

---

## 16.2 Data-filtering analysis

比较：

```text
D_all
vs
D_sensitive
```

优先方法：

- Global-only
- EIS
- ALC

问题：

> PA 的 global advantage 是否因为训练数据本来就选了 permutation-sensitive samples 才更有效？

如果 Global-only 在 D_sensitive 上明显比 D_all 强，这是非常有价值的结果，即使 ALC 最终没有赢。

---

# 17. 实验阶段与验收标准

## Phase 0：环境与官方代码（1–2 天）

任务：

1. Fork PA-GRPO。
2. 固定上游 commit hash。
3. 按官方 requirements 创建环境。
4. 跑一个官方最小 inference。
5. 跑 evaluation scorer。
6. 确认 local modified verl 被正确 import，而不是 pip verl。

验收：

- 能生成合法 Judge output；
- scorer 能输出 metrics；
- 记录 environment lock 与 GPU 信息。

---

## Phase 1：统一 advantage/reward 层（2–3 天）

任务：

- 抽离 advantages.py
- rewards.py
- semantic mapping
- config dispatch
- diagnostics
- 完成 T1–T9

验收：

- 全部 CPU 单测通过；
- J4R D.1 数值回归通过；
- 不改变官方 PA 模式结果。

---

## Phase 2：200-pair 小 smoke（1–2 天）

模型：1.5B

方法：

- M1
- M2
- M4
- M5
- M6

配置：

```text
pairs=200
P=2
N=4
1 epoch
```

验收：

- 所有方法 loss 正常；
- 无 NaN；
- consistency / reward 有变化；
- 日志能导出所有诊断。

Kill switch：

如果连 200-pair smoke 都无法让 reward 或 consistency 出现任何可观察变化，先检查数据、reward、prompt、parser，不允许直接扩大模型。

---

## Phase 3：1.5B 完整消融（3–5 天）

数据：1K pairs

配置：

```text
P=2
N=8
max 2 epochs
```

跑 M1–M7，1 seed。

验收：

- 至少出现 GRPO 相对 Base 的学习；
- Global/EIS/PA 之间存在可解释差异；
- 能完成 imbalance 与 rescue analysis。

如果 1.5B 完全没有 Judge 能力，不要硬解释算法差异。可升到 3B 做中间验证，或直接在 7B 做少量 pilot。

---

## Phase 4：7B pilot 与算力测速（半天–1 天）

只跑：

```text
200 pairs
P=2
N=4 or 8
M1
```

必须记录：

- peak VRAM
- rollout tokens/s
- train tokens/s
- step time
- generated tokens/hour
- GPU utilization
- 是否 OOM

用真实速度估算整个项目成本。

**不要在测速前预付大批 GPU 时长。**

---

## Phase 5：7B 主实验（约 1–2 周，取决于算力）

主方法：

- M1 GRPO
- M2 Global
- M4 EIS
- M5 PA-Full
- M6 ALC

数据：2K–3K pairs

P=2，N=8。

先 1 seed 全跑。

然后补：

- 最关键的 3 个方法第二 seed；
- 如果预算足够，再补第三 seed。

每个 checkpoint 定期做小 validation，而不是训练完才发现崩了。

---

## Phase 6：confound 实验（2–4 天）

只选最关键方法跑：

- Global
- EIS
- ALC

比较：

- D_all
- D_sensitive

用于解释 PA data filtering 的影响。

---

## Phase 7：最终分析与交付（2–3 天）

自动生成：

- Main table
- Ablation table
- Imbalance-bin plot
- Rescue-rate plot
- Reward/consistency curves
- Position-bias plot
- Cost table

README 写清：

- 问题
- 两篇论文差异
- 控制变量
- 结果
- 结论
- 失败/负结果
- 复现命令

---

# 18. 推荐 4 周时间表

## Week 1：工程基座

Day 1：
- fork PA-GRPO
- 环境安装
- 官方 inference/eval 跑通

Day 2：
- 阅读 PA 核心代码路径
- 标出 grouping / reward / advantage / train loop
- 写 CODE_MAP.md

Day 3：
- 新建 permstudy 模块
- 实现 semantic mapping / base reward / global/local advantage

Day 4：
- EIS / PA / sigma gate / consistency reward
- 写 T1–T9

Day 5：
- J4R D.1 regression test
- end-to-end smoke

Day 6–7：
- 200-pair 数据
- 1.5B smoke
- 修所有 logging/parser 问题

---

## Week 2：小模型完整实验 + 数据主线

- 构建 1K–2K J4R-style pairs
- 完整跑 1.5B M1–M7
- 做 advantage conflict / rescue 分析
- 构建 D_sensitive
- 决定 7B 最终只保留哪些方法

Week 2 结束时必须能回答：

> 这个项目有没有真正的学习信号？

如果没有，暂停 scale-up。

---

## Week 3：7B 主实验

- 7B pilot
- 定预算
- 跑 M1/M2/M4/M5/M6
- 途中定期 eval
- 保存所有 manifests/checkpoints

---

## Week 4：ALC + confound + 总结

- ALC 主结果
- D_all vs D_sensitive
- 补第二 seed
- 最终表格/图
- README
- 简历描述
- 技术报告

如果时间不足，优先顺序：

```text
统一复现 > Global/EIS/PA核心对照 > ALC > 数据筛选分析 > 第二seed > 其他扩展
```

---

# 19. 算力与预算

## 19.1 Rollout 数量

主实验若：

```text
pairs=2000
P=2
N=8
```

每 epoch completion 数：

\[
2000\times2\times8=32000
\]

若平均 256 output tokens：

\[
\approx8.2M\ output\ tokens/epoch
\]

2 epochs：约 16.4M output tokens / run。

这解释了为什么 rollout 是主要成本。

---

## 19.2 GPU 选择

公开 verl device-tuning 文档显示 `Qwen2.5-7B GRPO-LoRA` 存在单张 H100 的参考配置，因此 80GB 级卡值得 pilot；但**H100 的可运行配置不等于 A800/H800 必定按同一参数无修改运行**。

推荐顺序：

1. 1.5B 调试：便宜 GPU / 本地可行则本地；
2. 7B pilot：80GB 单卡；
3. 根据实际 throughput 决定 A800-80G 还是 H800-80G。

AutoDL 当前页面价格（2026-10 查询）：

- A800-80GB：约 ¥5.59 / h
- H800-80GB：约 ¥9.98 / h

价格会变化，以租用时页面为准。

### 成本公式

不要提前假设 20/40 小时。用 pilot 计算：

```text
estimated_run_hours = total_expected_rollout_tokens / measured_rollout_tokens_per_hour
                      + optimization_overhead

cost_per_run = estimated_run_hours * gpu_price_per_hour
```

当 7B pilot 完成后，把真实数字写入 `COST_ESTIMATE.md`。

### 预算 Kill switch

若估计：

- 单个 7B run > 40h，或
- 单个 run > ¥300

优先缩：

1. pairs 3000 -> 2000 -> 1000
2. max completion
3. N 8 -> 4（仅在确实需要时）
4. epochs 2 -> 1

不要先砍 benchmark。

---

# 20. Run Manifest：每次实验必须保存

每次 run 自动生成：

```json
{
  "git_commit": "...",
  "method": "eis",
  "seed": 42,
  "model_name": "Qwen2.5-7B-Instruct",
  "model_revision": "...",
  "dataset_name": "reasoning_main",
  "dataset_hash": "...",
  "num_pairs": 2000,
  "num_permutations": 2,
  "rollouts_per_perm": 8,
  "reward_config": {},
  "advantage_config": {},
  "lora_config": {},
  "optimizer_config": {},
  "gpu": "...",
  "start_time": "...",
  "end_time": "...",
  "peak_vram_gb": 0,
  "generated_tokens": 0,
  "wandb_run_id": "..."
}
```

没有 manifest 的实验不能进入最终主表。

---

# 21. Codex 协作规则

以下规则建议原样交给 Codex。

## 21.1 总原则

1. 先读 `PLAN.md`、`CODE_MAP.md` 和现有 tests，再改代码。
2. 不得为了“让实验跑起来”静默改变公式或 reward。
3. 遇到论文/代码不一致时，停止并报告，不要自行猜。
4. 所有新算法必须通过 config 开关切换。
5. 不复制 5 份 trainer；共用一个 code path。
6. 训练前先加测试；复杂 bug 优先写最小复现。
7. expensive GPU run 前必须输出：
   - method config
   - rollout budget
   - dataset hash
   - model revision
   - estimated hours
8. eval 数据不可用于 train / threshold tune。
9. parse failure 不可静默忽略，必须计数。
10. 任何指标变化都要检查是否伴随 output length / format 变化。

---

## 21.2 Commit 建议

按语义拆 commit：

```text
chore: pin upstream PA-GRPO environment
feat: add semantic permutation mapping
feat: add unified reward interface
feat: add global/local/eis advantage estimators
test: add J4R D.1 regression test
feat: add consistency reward and sigma gate
feat: add adaptive local correction
feat: add unified diagnostics logger
feat: add JudgeBench/RJB evaluation wrapper
exp: add small-model ablation configs
analysis: add imbalance and rescue analysis
```

不要把“大量重构 + 新算法 + 训练脚本”塞一个 commit。

---

# 22. 需要 Codex 首先输出的 CODE_MAP.md

在任何核心代码修改前，让 Codex阅读 PA-GRPO repo 并产出：

```text
1. train entrypoint
2. dataset loading path
3. permutation grouping path
4. rollout generation path
5. reward computation path
6. semantic mapping path
7. advantage computation path
8. PPO/GRPO loss path
9. LoRA injection path
10. checkpoint saving path
11. evaluation entrypoint
12. metrics computation path
```

对每一项写：

- 文件
- 函数/类
- 输入输出
- 是否需要修改

在 CODE_MAP 没确认前，不开始大改 `ray_trainer.py`。

---

# 23. 主结果表预定义

最终表格提前规定格式，避免训练结束才决定指标。

| Method | Acc JB | Con JB | CA JB | Acc RJB | Con RJB | CA RJB | First-pos Bias | Valid% | Avg Len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | | | | | | | | | |
| GRPO | | | | | | | | | |
| Global | | | | | | | | | |
| EIS | | | | | | | | | |
| PA | | | | | | | | | |
| ALC | | | | | | | | | |
| ALC+Con | | | | | | | | | |

机制表：

| Method | Δμ | SCR | LRR | LSR | Final Adv Std | Grad Norm | Train Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global | | | | | | | |
| EIS | | | | | | | |
| ALC | | | | | | | |

---

# 24. 何时认为项目“成功”

项目成功不要求 ALC 一定 SOTA。

满足以下任一高级结果都算成功：

### 成功类型 A：ALC 有效

ALC 在同 rollout budget 下优于或接近 EIS/PA，并展示 imbalance-aware 的机制证据。

### 成功类型 B：解释 discrepancy

发现 Global-only 的有效性主要取决于 D_sensitive / subgroup imbalance；给出实证解释两篇论文结果差异。

### 成功类型 C：负结果但机制清楚

ALC 没有提升，但明确证明：

- local rescue 与 imbalance 无强相关；或
- consistency reward 才是主要增益来源；或
- 数据筛选主导结果。

只要实验控制严谨、结论可复现，这依然是有价值的研究型项目。

---

# 25. Kill Switch

## KS1：Base Judge 太弱

若 1.5B 输出大量格式错误、接近随机且训练无响应：

- 1.5B 只保留单元测试；
- 转 3B 或 7B pilot；
- 不在 1.5B 结果上做机制结论。

## KS2：Reward 全相同

若大量 group `sigma=0`：

- 检查任务过易/过难；
- 检查 pair 构造；
- 提高 hard-pair 比例；
- 不靠随意加 noise 解决。

## KS3：训练只学会格式

若 Valid% 大涨但 Acc/CA 不涨：

- 分离 format reward；
- 检查 reward scale；
- 减少 format reward 权重的 sensitivity ablation。

## KS4：方法差异只来自 advantage magnitude

若 EIS grad norm 显著更大：

- 补 norm-matched ablation；
- 主结论中明确说明。

## KS5：7B 成本过高

按优先级缩减：

```text
第二seed -> 非核心method -> dataset size -> epoch -> N
```

核心 M1/M2/M4/M5/M6 至少留 1 seed。

---

# 26. 最终交付物

仓库必须包含：

1. `README.md`
2. `PLAN.md`
3. `CODE_MAP.md`
4. 环境 lock
5. 数据生成脚本
6. 统一 method configs
7. 单元测试
8. eval harness
9. 所有主实验 manifests
10. 自动生成主表脚本
11. 至少 3 张机制分析图
12. `REPORT.md` 研究报告
13. `COST_ESTIMATE.md`
14. `RESULTS.md`

---

# 27. README 最终叙事模板

不要写：

> I reproduced J4R and tuned alpha.

建议写：

> We study two closely related permutation-aware GRPO paradigms for robust LLM-as-a-Judge training: EIS-GRPO and PA-GRPO. Under a unified backbone, dataset, reward, rollout budget and evaluation protocol, we disentangle cross-permutation advantage, local subgroup correction, consistency reward, variance gating and permutation-sensitive data filtering. We further propose Adaptive Local Correction (ALC), which activates local advantage according to cross-permutation subgroup imbalance, and analyze when global-only credit assignment is sufficient versus when local correction is necessary.

---

# 28. 简历表述模板（结果出来后再填数字）

> **Permutation-Aware GRPO for Robust LLM-as-Judge**：基于 verl/LoRA 统一复现 EIS-GRPO 与 PA-GRPO，在相同 backbone、训练数据与 rollout budget 下实现 Global/Local Advantage、Consistency Reward 与 variance gate 的因子消融；设计 subgroup-imbalance-aware Adaptive Local Correction，并在 JudgeBench / ReasoningJudgeBench 上分析位置一致性与 consistent accuracy。通过 rollout-level advantage 诊断定位 global credit assignment 在高 permutation imbalance 样本上的失效模式，将 Consistency 从 xx% 提升至 xx% / 或在相同效果下降低 xx% rollout 成本。

最终数字出来前，不提前写“主要贡献源是 X”。

---

# 29. 当前应立即执行的第一批任务

交给 Codex 后，按以下顺序执行，不得跳步：

### Task 1
Fork/clone PA-GRPO，固定 commit，创建环境，跑通官方最小 inference + judge metric。

### Task 2
产出 `CODE_MAP.md`，不修改核心 trainer。

### Task 3
建立 `permstudy/advantages.py` 与 tests，实现 local/global/EIS，并用 J4R D.1 做数值回归。

### Task 4
实现 consistency reward、sigma gate、ALC，全部 CPU tests。

### Task 5
跑 2–4 pair end-to-end tiny smoke。

### Task 6
准备 200-pair pilot 数据，1.5B 跑 M1/M2/M4/M5/M6。

### Task 7
结果确认后再开始 1K small ablation 与 7B pilot。

**在 Task 6 之前不要租长时间 80GB GPU。**

---

# 30. 参考来源

- J4R: *Learning to Judge with Equivalent Initial State Group Relative Policy Optimization*, arXiv:2505.13346v3.
- PA-GRPO: *Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO*, ACL 2026 Long Paper, https://aclanthology.org/2026.acl-long.1621/
- PA-GRPO official code: https://github.com/ECNU-Text-Computing/PA-GRPO
- verl project / device tuning: https://github.com/verl-project/verl
- AutoDL: https://www.autodl.com/

---

## 最后原则

本项目最重要的不是把 GPU 跑满，而是保持：

```text
一个变量一个变量地改
每个公式有单测
每个实验有 manifest
每个结论有诊断证据
```

如果最终只能完成一半，优先保证：

```text
GRPO / Global / EIS / PA / ALC
+ JudgeBench / RJB
+ imbalance / rescue analysis
```

这已经足够形成一个完整、可解释、适合算法岗面试展示的 LLM 后训练项目。
