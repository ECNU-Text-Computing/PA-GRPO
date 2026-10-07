# PA-GRPO / permstudy code map

本文档是 `Permutation_GRPO_Project_Plan.md` Task 2 的只读代码审计结果。它描述当前
PA-GRPO 仓库的真实调用路径，并规定后续 `permstudy` 实现应接在哪里。Task 2 本身没有
修改 `verl` trainer、reward 或 evaluation 代码。

## 0. 审计边界

- PA-GRPO commit：`0ee9abd903cb4ac4945f1176e943d20436470096`
- PA-GRPO 分支：`codex/permstudy`
- README 声明的 verl 基线：`0eb50ec4a33cda97e05ed8caab9c7f17a30c05a9`
- 与该 verl 基线比较并忽略行尾差异后，vendored `verl/` 中的语义改动集中在：
  - `verl/trainer/ppo/ray_trainer.py`：154 insertions / 12 deletions；
  - `verl/trainer/constants_ppo.py`：增加 `RAY_DEBUG=1`。
- Windows 本地只验证 Transformers/CPU inference、parser 和 scorer；Ray、verl、vLLM、
  FSDP 与正式 GRPO 训练必须在 Linux GPU 环境验证。

下文“修改结论”的含义：

- **不改**：沿用 verl/PA-GRPO 当前实现。
- **新增模块**：在 `permstudy/` 或 `scripts_permstudy/` 实现，不改 vendored verl。
- **最小接线**：纯函数与测试先放 `permstudy/`，之后只在一个明确入口做 config dispatch；
  接线前必须单独 review。

## 1. 端到端主链路

```text
scripts/run_*.sh
  -> python -m verl.trainer.main_ppo (Hydra)
  -> TaskRunner.run
       -> RLHFDataset + DataLoader
       -> BatchRewardManager(custom compute_score)
       -> RayPPOTrainer.init_workers
       -> RayPPOTrainer.fit
            -> assign uid to each permutation prompt
            -> repeat each prompt N times
            -> ActorRolloutRefWorker.generate_sequences
                 -> vLLMRollout.generate_sequences
            -> BatchRewardManager: decoded response -> scalar outcome reward
            -> compute_advantage: reward -> token-shaped advantage
            -> DataParallelPPOActor.update_policy
                 -> compute_policy_loss_vanilla
            -> optional checkpoint

evaluation/evaluate_models.py
  -> per-sample result JSON
  -> compute_metrics_judge.py or compute_metrics_mcq.py
```

对 base DataLoader batch 大小 `B`、每个原始实例的 permutation 数 `P`、每个 prompt 的
rollout 数 `N`：

- DataLoader 输入：`B` 条 permutation prompt；
- generation 输入：`B * N`，由 trainer `repeat(..., interleave=True)` 产生；
- 一个完整 permutation group 应包含 `P * N` 条 completion；
- token-level reward / advantage 的形状为 `[B * N, response_length]`；
- 完整 group 只有在同一原始实例的全部 `P` 条 prompt 落在同一个 DataLoader batch 时才成立。

## 2. 训练数据契约

`RLHFDataset` 默认要求/透传的关键字段如下：

| 字段 | 位置 | 用途 |
|---|---|---|
| `prompt` | top-level | chat message list；dataset 将其渲染和 tokenize |
| `data_source` | top-level | reward routing / logging |
| `ability` | top-level | metadata，当前 PA 主链路不依赖 |
| `reward_model.ground_truth` | top-level nested object | reward 的 surface gold label |
| `extra_info.index` | nested object | 当前 dataset 同时复制为 top-level `index` |
| `extra_info.original_question_id` | Judge | trainer 当前优先用它构造 group uid |
| `extra_info.original_index` | MCQ | trainer/reward/metric 的原始实例 group key |
| `extra_info.permutation` | nested object | Judge 为 `0/1`；MCQ 为如 `ABCD` 的字符串 |

对仓库内实际 parquet 的抽查结果：

- `chatbot_arena_raw_2perm_thinking.parquet`：12,152 行，6,076 个 group，每组严格
  2 行且当前文件中相邻；gold label 随 AB/BA 从 `A` 翻为 `B`。
- `mmlu_raw_5perm_thinking.parquet`：29,295 行，5,859 个 group，每组严格 5 行；训练集
  使用 5 个 permutation，而不是 evaluation 代码所描述的 24 个全排列。
- 官方脚本的 `data.shuffle=false` 保留文件顺序；但 MCQ `train_batch_size=32` 不能被
  `P=5` 整除，因此 batch 边界会拆开 permutation group。

后续 `permstudy` 数据应显式提供稳定字段：

```text
group_id       # 稳定、无碰撞的原始实例 ID
perm_id        # AB / BA 或规范化 permutation ID
permutation    # surface position -> canonical candidate 的映射
index          # 仅作为 row id，不再承担 group_id 语义
```

rollout 对齐还应显式生成 `rollout_id in [0, N)`；不能永久依赖“各 permutation 中第 t 次
出现就是同一个 rollout column”的隐式顺序。

## 3. 十二条代码路径

### 3.1 Train entrypoint

**文件 / 函数**

- `scripts/run_judge_qwen.sh`、`run_judge_llama.sh`、`run_mcq_qwen.sh`、`run_mcq_llama.sh`
- `verl/trainer/main_ppo.py::main`
- `verl/trainer/main_ppo.py::run_ppo`
- `verl/trainer/main_ppo.py::TaskRunner.run`

**输入**

- Hydra config；官方脚本设置 model path、parquet、`adv_estimator=grpo`、LoRA、
  `rollout.n=8`、batch reward manager 和 custom reward path。

**输出**

- Ray workers、train/val datasets、reward managers 和 `RayPPOTrainer`；最终调用
  `trainer.init_workers()` 与 `trainer.fit()`。

**修改结论：不改入口；新增 wrapper/config。** 后续使用统一的
`configs_permstudy/` 与 `scripts_permstudy/train/` 覆盖配置，不复制 `main_ppo.py`。

### 3.2 Dataset loading path

**文件 / 函数**

- `verl/trainer/main_ppo.py::create_rl_dataset`
- `verl/utils/dataset/rl_dataset.py::RLHFDataset`
- `RLHFDataset::_read_files_and_tokenize`、`_build_messages`、`__getitem__`
- `verl/utils/dataset/rl_dataset.py::collate_fn`

**输入**

- 一个或多个 parquet；默认类是 `RLHFDataset`，也支持 config 指定 custom class。
- `prompt`、tokenizer/processor、最大 prompt 长度和 truncation 规则。

**输出**

- tensor fields：`input_ids`、`attention_mask`、`position_ids`；
- non-tensor fields：`raw_prompt_ids` 以及 parquet 中的 `data_source`、`reward_model`、
  `extra_info` 等；
- `extra_info.index` 被复制到 top-level `index`。

**修改结论：不改 dataset core；新增数据构建和 schema validation。** 在进入训练前验证
每个 group 的 `P`、唯一 `perm_id`、gold semantic/surface 映射以及 batch sampler 的 group
完整性。只有 group-aware sampler 无法用现有 sampler 扩展时，才考虑 custom dataset/sampler。

### 3.3 Permutation grouping path

**文件 / 函数**

- `verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit`（当前约 1116 行）
- `my_reward/judge_{qwen,llama}.py::compute_score`
- `my_reward/mcq_{qwen,llama}.py::compute_score`

**输入**

- DataLoader batch 的 `extra_info`；reward 还接收 repeat 后的 completion 顺序。

**输出**

- trainer 的 `batch.non_tensor_batch["uid"]`；
- reward 内部的 `group[pair_id][permutation] -> row indices`。

**当前优先级**

```text
trainer uid:
  original_question_id -> hash(value) % 1e9
  else original_index
  else index // 2
  else local batch position // 2

Judge reward pair_id:
  int(extra_info.index) // 2

MCQ reward group_id:
  extra_info.original_index
```

**修改结论：新增 `permstudy/grouping.py`，之后做一个最小 trainer 接线。** 必须统一
trainer、reward、diagnostics、evaluation 的 group key；禁止 Python `hash()`、`i // 2` 和
“相邻两行必成对”作为正式路径。group-aware batching 必须保证 group 不跨 batch。

### 3.4 Rollout generation path

**文件 / 函数**

- `RayPPOTrainer::_get_gen_batch`
- `RayPPOTrainer::fit`
- `verl/workers/fsdp_workers.py::ActorRolloutRefWorker._build_rollout`
- `ActorRolloutRefWorker::rollout_mode`、`init_model`、`generate_sequences`
- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py::vLLMRollout.generate_sequences`

**输入**

- tokenized prompt `DataProto`；trainer 先以 `interleave=True` 将每个 prompt 重复 `N` 次。
- vLLM sampling config；`vLLMRollout` 内部强制 `SamplingParams.n=1`，因为 repetition 已由
  trainer 完成。

**输出**

- `responses`、拼接后的 `input_ids`、`attention_mask`、`position_ids`，以及可选
  `rollout_log_probs`；trainer 再与 repeat 后的原 batch union。

**修改结论：不改 rollout engine。** 新增 `test_rollout_budget.py` 验证 `B * P * N`、
group completeness 和 rollout alignment。任何算法差异都不应改变总 rollout budget。

### 3.5 Reward computation path

**文件 / 函数**

- `verl/trainer/ppo/reward.py::get_custom_reward_fn`
- `verl/trainer/ppo/reward.py::load_reward_manager`
- `verl/trainer/ppo/reward.py::compute_reward`
- `verl/workers/reward_manager/batch.py::BatchRewardManager.verify`
- `BatchRewardManager::__call__`
- `my_reward/*.py::compute_score`

**输入**

- decoded response strings、`reward_model.ground_truth`、`data_source`、`extra_info`；
- custom reward path/name 和 reward kwargs。

**输出**

- 每条 completion 一个 scalar reward；manager 将其放在最后一个有效 response token，形成
  `[batch, response_length]` 的 `reward_tensor`；同时写 `data.batch["acc"]`。
- `compute_reward` 返回 `(reward_tensor, reward_extra_info)`。

**当前 PA reward**

- Judge：正确性 `+1/-1`、格式 `+0.3/-0.3`、长度分段 reward，再加 pairwise
  consistency `+lambda/-lambda`。
- Judge 遇到无法 semantic mapping 的答案时跳过 consistency，而不是给 `-lambda`。
- MCQ：先把 surface letter 映射到 canonical candidate，再按跨 permutation unique mode
  给 consistency reward。

**修改结论：新增 `permstudy/rewards.py`，不改 reward manager。** controlled comparison
使用计划书统一 base reward。若要记录分量，custom reward 应返回 manager 已支持的
`list[dict(score=..., ...)]` 语义；当前 `my_reward` 的全局
`{"reward_tensor", "reward_extra_info"}` 返回格式不能直接被 `BatchRewardManager.verify`
消费，因此当前默认 `return_dict=False` 时分量不会沿 trainer 主链路回传。

### 3.6 Semantic mapping path

**文件 / 函数**

- Judge：`my_reward/judge_{qwen,llama}.py::CONSIST_MAP` 与 `compute_score`
- MCQ：`my_reward/mcq_{qwen,llama}.py::map_to_canonical`
- evaluation：`compute_metrics_mcq.py::get_original_answer_content`
- Judge metrics：`compute_metrics_judge.py::calculate_consistency`

**输入**

- surface prediction（A/B/C/D）与 permutation metadata。

**输出**

- canonical/semantic candidate，或 Judge AB/BA 下的 mirror-label consistency verdict。

**修改结论：新增一个共享 semantic mapping 模块。** 训练 reward、metrics 和单元测试必须
调用同一个纯函数；Judge 不再只用 `A <-> B` 的隐式规则，明确表达
`(surface_label, permutation) -> semantic candidate`。

### 3.7 Advantage computation path

**文件 / 函数**

- `verl/trainer/ppo/ray_trainer.py::compute_advantage`
- `verl/trainer/ppo/ray_trainer.py::apply_group_baseline_from_returns`（PA 新增）
- `verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`
- `verl/trainer/ppo/core_algos.py::register_adv_est` / `get_adv_estimator_fn`

**输入**

- `token_level_rewards`、`response_mask`、`uid`、algorithm config。

**输出**

- `data.batch["advantages"]` 与 `data.batch["returns"]`，形状通常为
  `[batch, response_length]`。

**当前 PA 行为**

1. 先调用 upstream GRPO，但强制 `norm_adv_by_std_in_grpo=False`；
2. upstream 返回按 uid 去均值后的 token-shaped `returns`；
3. PA helper 将多维 `returns` 对序列维取普通 mean，再按同一 uid 用 population variance
   标准化、clamp 到 `[-5, 5]`；
4. scalar advantage 展开到有效 response tokens；
5. 异常时打印 warning 并回退到未标准化的 upstream result。

**修改结论：新增 `permstudy/advantages.py` 纯函数和 CPU tests，之后最小接线。**
`local/global/EIS/gated/ALC` 不应继续堆进 `ray_trainer.py`。接线层需要把 raw scalar
base/total reward、`group_id`、`perm_id`、`rollout_id` 和 mask 显式传入。

当前 PA 路径在作为实验基线前必须数值复核：它对已去均值、mask 后的 token-shaped
`returns` 再取普通 sequence mean，因此 completion 长度可能影响 scalar；config 的
`norm_adv_by_std_in_grpo` 在该分支不控制最终标准化；实现也没有计划书中的显式
`sigma < delta -> 0` gate。

### 3.8 PPO / GRPO loss path

**文件 / 函数**

- `verl/workers/actor/dp_actor.py::DataParallelPPOActor.update_policy`
- `verl/trainer/ppo/core_algos.py::get_policy_loss_fn`
- `verl/trainer/ppo/core_algos.py::compute_policy_loss_vanilla`

**输入**

- `responses`、`response_mask`、`old_log_probs`、current `log_prob`、`advantages`；可选
  reference log-prob、entropy、rollout correction weights。

**输出**

- clipped PPO policy loss、clip fraction、approximate KL 等 metrics；backward、gradient
  clipping 和 optimizer step 由 actor 执行。

**修改结论：不改 loss。** GRPO/EIS/PA/ALC 的差异只应体现在送入同一 loss 的
advantage/reward。这样方法比较不混入不同 actor objective。

### 3.9 LoRA injection path

**文件 / 函数**

- `verl/trainer/config/model/hf_model.yaml`
- `verl/workers/fsdp_workers.py::ActorRolloutRefWorker._build_model_optimizer`
- `ActorRolloutRefWorker::rollout_mode`
- `vLLMRollout::__init__`

**输入**

- `model.lora_rank`、`lora_alpha`、`target_modules`、`exclude_modules`，或已有
  `lora_adapter_path`。官方脚本为 rank 32、alpha 64、`all-linear`。

**输出**

- PEFT-wrapped actor；optimizer 只更新 trainable parameters；rollout mode 将 base/adapter
  weights 同步到 vLLM，并用 `LoRARequest` 推理。

**修改结论：不改。** Linux smoke 必须验证 LoRA 参数变化、frozen base 不变、无
NaN/Inf，并验证一次 update 后 vLLM 使用新 adapter 权重。

### 3.10 Checkpoint saving path

**文件 / 函数**

- `RayPPOTrainer::_save_checkpoint` / `_load_checkpoint`
- `ActorRolloutRefWorker::save_checkpoint` / `load_checkpoint`
- `verl/utils/checkpoint/fsdp_checkpoint_manager.py::FSDPCheckpointManager`

**输入**

- `trainer.default_local_dir`、`global_steps`、retention config 和 actor checkpoint contents。

**输出**

- `global_step_<n>/actor/` 下的 FSDP model/optimizer/extra-state shards；
- `actor/huggingface/` 下的 tokenizer/config，按配置可含 full HF weights；
- LoRA 时额外保存 `actor/lora_adapter/adapter_model.safetensors` 和 config；
- trainer 另存 `data.pt` 与 `latest_checkpointed_iteration.txt`。

**修改结论：不改。** 需要在统一 run config 中显式设置正数 `trainer.save_freq`；官方
PA shell scripts 未设置它，而默认值是 `-1`，当前 trainer 在这种情况下连最后一步也不会
自动保存。正式评测前增加 save/resume/LoRA-load smoke。

### 3.11 Evaluation entrypoint

**文件 / 函数**

- `evaluation/evaluate_models.py::main`
- `detect_checkpoint_type`
- `VLLMModelWrapper` / `TransformersModelWrapper`
- `evaluate_dataset_batch` / `evaluate_dataset_sequential`
- `extract_answer_from_response`

**输入**

- base model、LoRA checkpoint、merged HF model，或 checkpoint directory；
- parquet、mode、backend、decoding config。

**输出**

- 每条样本的 response、extracted answer、gold、correctness、probability 和原 metadata；
- streaming JSON result 与 summary。

**修改结论：保留 backend/checkpoint loader；新增共享 parser 和统一 eval wrapper。**
Windows 已跑通 Transformers/CPU smoke，Linux 再验证 vLLM/LoRA。当前 `direct` parser 会
逐字符寻找候选字母，例如字符串 `invalid` 会因包含 `a` 被解析为 `A`；正式指标前必须改为
严格、可测试的解析规则并报告 valid rate。

### 3.12 Metrics computation path

**文件 / 函数**

- Judge：`evaluation/compute_metrics_judge.py::process_single_file`、
  `group_by_pairs`、`calculate_accuracy`、`calculate_consistency`、
  `calculate_consistent_and_correct`、`calculate_rstd`、`calculate_ckld`
- MCQ：`evaluation/compute_metrics_mcq.py` 中对应的 `*_24perm` 函数

**输入**

- `evaluate_models.py` 产生的 result JSON。

**输出**

- Acc、Consistency、Consistent Accuracy、RStd、CKLD；可输出文本、JSON 和 Excel 汇总。

**修改结论：新增 `permstudy/metrics.py` 作为唯一指标定义，旧 CLI 可作为兼容 wrapper。**
必须用稳定 `group_id` 和共享 semantic mapping。当前 Judge grouping 会在 question-id 配对率
很低时才整体 fallback 到 `original_index`，且 consistency 主要用“两个 surface prediction
不相等”判断；空值或异常 label 可能被误计。新实现需显式要求 `{AB, BA}` 完整、预测有效，
并独立报告 invalid/incomplete group。

## 4. 后续允许的最小改动面

Task 3/4 应优先新增：

```text
permstudy/
  grouping.py       # stable group/perm/rollout identity and validation
  semantic.py       # surface -> semantic mapping
  advantages.py     # local/global/EIS/gate/ALC pure functions
  rewards.py        # unified base + optional consistency reward
  parsing.py        # strict response parser
  diagnostics.py    # imbalance, SCR, LRR, LSR, advantage scale
  metrics.py        # shared Acc/Con/CA definitions

tests_permstudy/
  ...               # plan T1-T9 and end-to-end contract tests
```

在这些纯函数和 CPU 单测通过前，不修改 `ray_trainer.py`。之后若必须接线，只允许一个薄层：

```text
RayPPOTrainer.fit
  reward tensor + explicit metadata
    -> permstudy method dispatcher
    -> token-shaped advantages
    -> unchanged actor update / PPO loss
```

不修改 vLLM rollout、LoRA injection、PPO loss 或 FSDP checkpoint 实现。

## 5. 必须先处理或用测试锁定的风险

1. **group identity 不一致**：trainer、Judge reward、MCQ reward 和 metrics 使用不同 fallback。
2. **不稳定 hash**：`hash(original_question_id) % 1e9` 跨进程/运行不稳定且存在碰撞。
3. **group 跨 batch**：尤其官方 MCQ 的 `P=5, batch=32` 已确认会拆组。
4. **rollout 对齐是隐式的**：consistency reward 依赖每个 permutation 内的出现顺序。
5. **PA advantage 输入需复核**：当前从去均值且 mask 后的 token tensor 取普通 mean，可能
   引入 response-length 因子；且没有显式 variance threshold gate。
6. **异常被宽泛回退**：PA advantage 失败时回退到另一个公式，可能让实验方法静默变化。
7. **reward 分量未进入主日志**：当前 custom reward 默认只返回 tensor。
8. **parser 过宽松**：会把自然语言中的候选字母误当最终答案。
9. **Judge consistency 判定过宽**：`pred1 != pred2` 不等价于严格 semantic consistency。
10. **默认不保存 checkpoint**：`save_freq=-1` 必须在正式 run config 覆盖。

以上风险中，1-9 都应先由 CPU 单元测试给出明确 contract；不能靠训练结果反推实现是否正确。
