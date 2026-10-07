# Codex 下一阶段任务说明：Task 2A — Linux Fidelity Confirmation + Trainer Integration

## 背景

项目仓库：

```text
https://github.com/heziss666/PA-GRPO.git
```

官方上游固定 commit：

```text
0ee9abd903cb4ac4945f1176e943d20436470096
```

当前已完成：

```text
Task 1A: Windows inference / scorer smoke
Task 1.5: Paper–Code Fidelity Audit（纯函数 + CPU 数值审计）
```

请先完整阅读：

```text
PLAN.md
PLAN_ADDENDUM.md
CODE_MAP.md
artifacts/smoke/task1a/README.md
artifacts/audit/task1_5/PA_PAPER_CODE_AUDIT.md
```

然后再执行下面的 Task 2A。

---

# 1. 先明确 Task 1A 与 Task 1.5 “完成”的含义

## Task 1A 已完成

Task 1A 只验证：

```text
parquet
→ Transformers inference
→ 官方 response parser
→ 官方 Judge scorer
→ Accuracy / Consistency / Consistent Accuracy
```

当前 Windows 已真实跑通这一链路。

它**不代表**：

```text
verl 已跑通
vLLM 已跑通
Ray 已跑通
GRPO 已训练
PA-GRPO 已复现
```

所以不要把 Task 1A 的完成状态扩大解释。

---

## Task 1.5 已完成

Task 1.5 只表示：

```text
论文公式
vs
官方源码行为
vs
我们自己的纯函数实现
```

已经通过 CPU synthetic tests 得到了明确的代码层结论。

目前已完成：

1. sigma gate 数值测试；
2. response-length dependence 数值测试；
3. consistency pairing reorder 测试；
4. 显式 ID 纯函数实现：

```text
pair_id
permutation_id
rollout_slot
```

但以下事情**尚未完成**：

```text
这些逻辑尚未接入真实 verl Trainer
rollout_slot 尚未贯穿 DataProto / repeat / rollout / reorder / reward
官方 Ray/verl trainer 尚未在 Linux 中做 end-to-end fidelity confirmation
```

---

# 2. Sigma Gate：不要误解为“已经确定用 max，不用 threshold”

这是目前最重要的 paper–code discrepancy。

## 论文定义

论文中的 PA-GRPO advantage 包含：

\[
A_i =
\begin{cases}
0, & \sigma_G < \delta \\
\dfrac{r_i-\mu_G}{\sigma_G+\epsilon}, & \text{otherwise}
\end{cases}
\]

也就是说：

```text
当 group reward std 太小时，
该 group 的 advantage 直接置零。
```

这是一个 signal-quality gate。

---

## 当前官方代码行为

当前官方 `ray_trainer.py` 路径中观察到的是类似：

```python
variance = clamp_min(variance, 1e-6)
std = sqrt(variance)
adv = (x - mean) / std
adv = clamp(adv, -5, 5)
```

它等价于：

```text
对 sigma 做 floor / max，防止除零或数值爆炸
```

而不是：

```text
sigma < delta -> A = 0
```

因此当前结论必须写成：

```text
Paper: threshold gate
Official code: variance floor / clipping
```

两者不等价。

---

## 当前项目决定

因为目前没有找到可信来源给出论文中的具体 `delta` 数值：

```text
不要猜 delta
不要把某个自选值写成论文默认
```

当前：

```python
sigma_gate_threshold = None
```

只表示：

```text
暂不启用 paper gate，因为缺少可靠 threshold provenance
```

不表示：

```text
论文不需要 gate
```

---

# 3. 为什么 response length 还需要在 Linux 再确认

当前 CPU audit 已经观察到：

固定 reward：

\[
[1, 1, -1, -1]
\]

等长 response：

```text
lengths = [8, 8, 8, 8]
```

得到近似：

```text
paper = [1, 1, -1, -1]
official-compatible = [1, 1, -1, -1]
```

只改变 response length：

```text
lengths = [2, 8, 4, 10]
```

paper 实现仍为：

```text
[1, 1, -1, -1]
```

而 official-compatible helper 得到：

```text
[0.4472, 1.3416, -0.4472, -1.3416]
```

说明按照当前源码数学路径：

```text
official implementation exhibits response-length dependence
```

但是目前这个结论来自：

```text
我们根据官方源码重实现的
global_advantage_official_compatible()
```

它还不是：

```text
真实 DataProto
→ 官方 compute_grpo_outcome_advantage
→ 官方 ray_trainer
→ 最终 advantages
```

因此 Linux 环境中的下一步不是重新研究这个问题，而是：

```text
直接调用官方 verl / Ray 路径，
确认真实 trainer 是否产生同样的数值结果。
```

如果结果一致：

```text
response-length dependence 正式确认
```

如果不一致：

```text
说明我们的 official-compatible helper 漏掉了某个真实 trainer 细节
```

---

# 4. Task 2A：Linux Fidelity Confirmation

## 目标

在 Linux 环境中完成：

```text
官方 verl / Ray import
→ synthetic tensor
→ 官方真实 advantage path
→ 与当前 audit 数值逐项对比
```

这个阶段不需要完整 8B 训练，不追求 benchmark 指标。

---

## 4.1 环境

优先使用与官方仓库兼容的 Linux 环境。

记录：

```text
OS
Python
PyTorch
CUDA
Ray
verl
vLLM
GPU
git commit
```

生成：

```text
artifacts/linux_fidelity/environment.txt
```

---

## 4.2 必做测试 A：Equal-Length

固定：

```python
rewards = [1.0, 1.0, -1.0, -1.0]
lengths = [8, 8, 8, 8]
```

要求直接经过官方真实路径得到最终 advantage。

比较：

```text
official-real
official-compatible
paper
```

输出：

```text
artifacts/linux_fidelity/advantage_equal_length.json
```

---

## 4.3 必做测试 B：Unequal-Length

固定相同 reward：

```python
rewards = [1.0, 1.0, -1.0, -1.0]
lengths = [2, 8, 4, 10]
```

再次比较：

```text
official-real
official-compatible
paper
```

重点检查：

```text
official-real 是否接近
[0.4472, 1.3416, -0.4472, -1.3416]
```

不要在测试之前硬编码“必须等于”，而应以实际值为准，并输出差异。

生成：

```text
artifacts/linux_fidelity/advantage_unequal_length.json
```

---

## 4.4 输出指标

至少记录：

```text
L1 difference
Linf difference
sign disagreement rate
per-sample values
response lengths
group ids
```

---

# 5. Task 2B：显式 rollout identity 接入真实 Trainer

当前纯函数已经有：

```python
RolloutId(
    pair_id,
    permutation_id,
    rollout_slot
)
```

但它还没有接入真实训练流。

目标是让：

```text
pair_id
permutation_id
rollout_slot
```

能够贯穿：

```text
dataset
→ DataProto
→ repeat(n)
→ rollout generation
→ balance/reorder
→ reward
→ consistency pairing
```

---

## 5.1 原则

任何 reorder 都必须同步 reorder：

```text
pair_id
permutation_id
rollout_slot
```

不能再依赖：

```python
pair_id = index // 2
```

或：

```python
idxs0[t] <-> idxs1[t]
```

作为 controlled experiment 的唯一依据。

---

## 5.2 controlled mode

controlled mode 中 consistency pairing 必须使用：

```python
key = (pair_id, rollout_slot)
```

并且同一个 key 下应找到：

```text
permutation_id = 0
permutation_id = 1
```

如果缺失：

```text
记录 unpaired
```

如果 duplicate：

```text
直接报错
```

---

## 5.3 official mode

必须保留官方行为。

也就是说通过配置可以切换：

```yaml
grouping:
  identity_mode: legacy_index
```

和：

```yaml
grouping:
  identity_mode: explicit
```

不要破坏官方 reproduction path。

---

# 6. 不要马上实现 EIS / ALC

在完成下面几项之前：

```text
[ ] Linux 官方真实 advantage path 对照完成
[ ] response-length dependence 最终结论确认
[ ] pair_id / permutation_id / rollout_slot 接入真实 DataProto
[ ] reorder 后 consistency pairing end-to-end 测试通过
[ ] official mode 可恢复
```

不要开始：

```text
EIS-GRPO
ALC
大规模 7B/8B 训练
```

---

# 7. 需要顺便修正的 evaluator 问题

Task 1A 暴露出两个问题：

## 7.1 direct parser 过宽松

当前官方 direct parser 可能把类似：

```text
invalid
```

中的字母误解析成合法答案。

正式 controlled evaluation 需要：

```text
strict parser
```

但 official reproduction 仍保留官方 parser。

---

## 7.2 Judge A/B 被识别成 4-option

Task 1A 中 A/B pairwise Judge 被 evaluator 记录为：

```text
num_options = 4
```

这对 smoke 不阻塞，但 formal evaluation 前必须修正。

要求区分：

```text
official evaluator
controlled evaluator
```

---

# 8. 推荐代码结构

不要大量改 vendored `verl/`。

优先新增：

```text
permstudy/
  trainer_integration.py
  rollout_identity.py
  parsing.py
  config.py

tests_permstudy/
  test_linux_official_advantage.py
  test_rollout_identity_propagation.py
  test_end_to_end_consistency_pairing.py
  test_strict_parser.py
```

如果必须改：

```text
verl/trainer/ppo/ray_trainer.py
```

只做最小 dispatch patch。

---

# 9. Codex 工作规则

必须遵守：

1. 先读计划与 audit。
2. 先写测试，再改 Trainer。
3. 不要把 paper / official / controlled 三种行为混成一个函数。
4. 不要猜论文没给出的 `delta`。
5. 不要把 `official-compatible helper` 当作真实 trainer 结果。
6. 不要为了测试方便直接删掉官方逻辑。
7. 所有新增行为都必须有 config 开关。
8. 所有 Linux fidelity 结果写入 artifacts。
9. 每次正式运行记录 commit、config、环境和 seed。
10. 在我确认 Linux fidelity 结果前，不开始正式 GRPO 实验。

---

# 10. 建议 commit 顺序

```text
test: add direct Linux official advantage fidelity harness

audit: confirm equal and unequal length behavior in verl

feat: propagate explicit rollout identity through training batch

test: verify explicit consistency pairing after trainer reorder

feat: add controlled strict judge parser

test: distinguish pairwise judge option count from MCQ

docs: record Linux fidelity confirmation
```

暂时不要提交：

```text
feat: implement EIS
feat: implement ALC
```

---

# 11. 最终输出文档

Task 2A/2B 完成后生成：

```text
artifacts/linux_fidelity/LINUX_FIDELITY_REPORT.md
```

必须按照：

```text
Paper states
Official code does
Official Linux run observes
Project controlled implementation does
Project decision
```

五段式记录。

---

# 12. 直接执行指令

请从这里开始：

> 先完整阅读 `PLAN.md`、`PLAN_ADDENDUM.md`、`CODE_MAP.md`、`artifacts/smoke/task1a/README.md` 和 `artifacts/audit/task1_5/PA_PAPER_CODE_AUDIT.md`。不要开始 EIS、ALC 或大规模训练。第一步在 Linux 环境里直接调用官方 verl/Ray 路径，对 Task 1.5 中 equal-length 与 unequal-length synthetic case 做 end-to-end fidelity confirmation，比较 official-real / official-compatible / paper 三套 advantage。确认后，再把显式 `pair_id / permutation_id / rollout_slot` 接入真实 DataProto/rollout/reorder/reward 流程，并保证 official legacy behavior 可通过 config 恢复。任何论文未给出的 sigma gate threshold 不允许猜测为论文默认。完成后输出 `artifacts/linux_fidelity/LINUX_FIDELITY_REPORT.md`，在我确认报告前不要进入 EIS/ALC 正式实现。

---

# 13. 执行前追加约束（2026-10-05）

以下约束属于 Task 2A/2B 的强制验收条件：

1. Task 2A 的 official-real harness 必须显式证明
   `apply_group_baseline_from_returns()` 被实际调用，并确认 stdout/stderr
   中不存在 `fallback to original GRPO`、`data.meta_info` 中存在
   `pair_baseline_metrics`。仅凭最终存在 `advantages` 不构成通过。调用证明使用
   测试内临时 monkeypatch/wrap sentinel，不为此修改生产代码。
2. Task 2B 在接入 Trainer 前必须先增加 rollout-slot propagation 单测。对于
   `repeat(n, interleave=True)`，两个原始 permutation 输入展开后的 slot 必须严格为
   `[0, 1, ..., n-1, 0, 1, ..., n-1]`，并额外验证实际 rollout output 顺序与该
   identity 展开顺序一致；不能只根据 `repeat()` 的实现推断 backend 顺序。
3. evaluator 的严格 parser 与 pairwise option detector 只进入 controlled path；
   原始 `evaluation/evaluate_models.py` 保持不变，继续作为 PA-Official reproduction
   路径。
4. WSL2 仅作为 Task 2A 的优先 Linux 环境。先验证 Python、Ray、verl、CUDA/GPU
   可见性；若导入官方路径要求修改官方依赖或源码，立即停止 WSL 兼容改造并转到
   AutoDL Linux。不得在 WSL 内安装 Linux NVIDIA kernel driver 覆盖宿主机提供的
   WSL GPU driver，也不为本阶段强行安装或修改 vLLM。

流程上，Task 2A 的 Linux fidelity harness/结果与 Task 2B 的 Trainer integration
必须分开提交，使 Linux fidelity 基准点可以独立复现和回退。`LINUX_FIDELITY_REPORT.md`
提交并由用户确认前，不进入 EIS/ALC。
