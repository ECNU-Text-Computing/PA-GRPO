# Task 4 Real vLLM / GPU / GRPO Smoke Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that the Task 2B explicit rollout identity and PA-GRPO training path complete real vLLM generation, reward pairing, GRPO advantage computation, LoRA/FSDP2 actor updates, checkpointing, and a minimal controlled evaluation on one Linux GPU.

**Architecture:** Build a deterministic eight-pair fixture from the existing Qwen pairwise parquet, run exactly two synchronous vLLM/GRPO optimization steps with explicit identity enabled, then verify logs, metrics, rollout evidence, and checkpoint contents with a standalone evidence validator. The smoke reuses the current trainer and reward implementation; it may add fixture/launcher/verifier scripts, but it must not alter trainer, reward, advantage, Task 2B identity, or Task 3 parser semantics merely to make the smoke pass.

**Tech Stack:** Linux, Python 3.12, CUDA 12.4-compatible driver/runtime, PyTorch 2.6.0, Ray 2.49.0, vLLM 0.8.5, local vendored verl, FSDP2, PEFT LoRA, Qwen3-8B, pandas/pyarrow, safetensors, pytest.

**Spec:** `PLAN.md`, `Codex_Task2A_Linux_Fidelity_Trainer_Integration.md`, `artifacts/linux_fidelity/LINUX_FIDELITY_REPORT.md`, `PLAN_ADDENDUM_TASK3_CONTROLLED_EVALUATOR_HARDENING.md`, and the approved Task 4 scope recorded on 2026-10-07.

## Global Constraints

- Branch: `codex/task4-real-vllm-grpo-smoke`.
- Base: `main@c281f6627805056007def47bfb2b24aca430d498`.
- Do not branch from or modify `codex/data-pipeline-plan`.
- Task 4 does not depend on new MATH/ReClor generation or the data-pipeline branch.
- Primary PASS model: `Qwen/Qwen3-8B` or a byte-identical local snapshot with recorded revision/hash.
- A smaller model may diagnose environment failures, but it cannot produce Task 4 PASS.
- Source data: `dataset/train/chatbot_arena_raw_2perm_thinking.parquet` from the pinned repository commit.
- Tiny data: exactly 8 complete pairwise examples = 16 rows, each with permutations `0` and `1`.
- Hardware: one Linux NVIDIA GPU with at least 80 GB VRAM.
- Rollout backend: synchronous vLLM only; `rollout.name=vllm`, `rollout.mode=sync`.
- Identity: `grouping.identity_mode=explicit`; legacy index fallback is forbidden.
- Training: FSDP2 + LoRA, `rollout.n=2`, `max_response_length=256`, exactly 2 optimization steps.
- `trainer.balance_batch=true` remains enabled; the fixture order must make the resulting reorder observable.
- No silent fallback to original GRPO is allowed.
- Do not install a Linux NVIDIA kernel driver over the host/provider driver.
- Do not weaken assertions or change production semantics to accommodate an environment failure.
- Do not start EIS/ALC or formal training until this plan is reviewed and Task 4 is PASS.

---

## 1. Scope

### In scope

```text
Linux GPU preflight
→ deterministic tiny pairwise fixture
→ local vendored verl import
→ synchronous vLLM rollout
→ DataProto explicit identity propagation
→ balance_batch reorder
→ reward pairing by (pair_id, rollout_slot)
→ PA pair-baseline GRPO advantage
→ FSDP2 + LoRA backward / optimizer update
→ two optimization steps
→ checkpoint save and LoRA merge
→ minimal Task 3 controlled-evaluator sanity
→ compact reproducibility artifact
```

### Out of scope

```text
new MATH/ReClor data generation
data-quality or benchmark claims
EIS / ALC implementation
large-scale convergence
accuracy improvement claims
multi-GPU, async rollout, SGLang, or REMAX
official evaluator changes
reward/parser/trainer semantic changes
```

Task 4 answers only:

\[
\boxed{\text{Does the already-reviewed controlled path complete real GPU training end to end?}}
\]

---

## 2. Fixed Smoke Design

### 2.1 Model

Primary model:

```text
Qwen/Qwen3-8B
```

Record all of:

```text
model path
Hugging Face revision or local snapshot hash
config.json SHA256
tokenizer_config.json SHA256
```

If Qwen3-8B cannot be obtained, stop and report the model-access blocker. A run on Qwen2.5-1.5B may validate package installation, but must be labeled `DIAGNOSTIC_ONLY`.

### 2.2 Tiny dataset

Input:

```text
dataset/train/chatbot_arena_raw_2perm_thinking.parquet
```

Selection contract:

- Group by `extra_info.original_question_id`.
- Require exactly one row with `permutation=0` and one with `permutation=1`.
- Require mirrored ground truth: permutation 0/1 labels must describe the same semantic winner.
- Tokenize both permutations with the selected Qwen3-8B tokenizer.
- Reject a pair if either prompt exceeds 4096 tokens.
- Select 8 complete pairs deterministically.
- Choose pairs across prompt-length quantiles and arrange them in alternating long/short order so `balance_batch` produces a non-identity reorder.
- Preserve each pair's two rows adjacent in the saved fixture.
- Write 16 rows and a manifest containing source hash, selected IDs, permutations, row indices, token lengths, labels, and fixture hash.

The fixture is a smoke input, not a new training-data product and not part of the data-pipeline branch.

### 2.3 Training configuration

| Setting | Value |
|---|---|
| train rows | 16 |
| complete pairs | 8 |
| `data.train_batch_size` | 16 |
| `data.shuffle` | `false` |
| `data.max_prompt_length` | 4096 |
| `data.max_response_length` | 256 |
| `rollout.n` | 2 |
| generated responses per step | 32 |
| optimization steps | 2 |
| GPUs | 1 |
| rollout backend | vLLM sync |
| actor strategy | FSDP2 |
| LoRA rank / alpha | 32 / 64 |
| LoRA targets | `all-linear` |
| balance batch | `true` |
| checkpoint frequency | every step |
| validation during training | disabled |
| resume | disabled |

Training rollouts remain stochastic (`temperature=1.0`, `top_p=1.0`), because GRPO needs distinct samples. This does not conflict with the deterministic decoding requirement for future sensitivity filtering; that is a separate workflow.

### 2.4 Initial memory-safe settings

```text
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.50
actor_rollout_ref.rollout.enforce_eager=true
actor_rollout_ref.rollout.max_num_seqs=32
actor_rollout_ref.rollout.max_model_len=4352
actor_rollout_ref.rollout.max_num_batched_tokens=8192
actor_rollout_ref.model.enable_gradient_checkpointing=true
actor_rollout_ref.rollout.layered_summon=true
```

If an OOM occurs, only memory/resource knobs may change in the retry. Every retry must have its own command/config and log. A retry must not change identity, reward, advantage, sample count, rollout count, or PASS assertions.

---

## 3. Required Evidence Hooks

### 3.1 vLLM and generation

Evidence:

- vLLM engine initialization in the run log.
- Two non-empty rollout JSONL files, one per step.
- Exactly 32 generated rows per step.
- No backend prompt-order or rollout-identity alignment exception.

### 3.2 Explicit identity and reorder

The existing production path is fail-closed:

```text
attach_permutation_identity
→ repeat_for_rollout
→ snapshot_rollout_identity
→ vLLM generate_sequences
→ validate_generation_identity
→ balance_batch / DataProto.reorder
→ reward metadata merge
```

Evidence:

- Every expected `(pair_id, rollout_slot)` has exactly one `PAIR_CHECK` joining permutation 0 and 1.
- No `PAIR_UNPAIRED` line.
- No duplicate rollout-key exception.
- `reward/consistency_unpaired_rate = 0` where emitted.
- Reward log source-index order differs from canonical interleaved input order for at least one step, proving a real reorder occurred before successful explicit pairing.

The reward implementation writes to a persistent project-level log. The launcher must record that file's byte offset immediately before training, and the verifier must inspect only bytes appended after that offset. Historical `PAIR_CHECK` lines are never admissible evidence for the current run.

### 3.3 Reward and PA-GRPO advantage

Evidence:

- Finite reward component values.
- Presence of all three metrics:

```text
pair_baseline/mean_of_pair_means
pair_baseline/mean_of_pair_stds
pair_baseline/std_of_pair_stds
```

- Run log does not contain `fallback to original GRPO`.
- Advantages and returns are finite; no NaN/Inf warning.

The metrics-plus-no-warning rule proves the PA pair-baseline branch ran. Merely observing an `advantages` tensor is insufficient because the fallback also produces one.

Trainer TensorBoard evidence must be isolated per run by exporting `TENSORBOARD_DIR="$TASK4_RUN_DIR/tensorboard"`. The hard-coded reward TensorBoard directory is diagnostic only and is not used for PASS because it may contain events from earlier runs.

### 3.4 Actor update and LoRA

Evidence for both optimization steps:

```text
actor/pg_loss: finite
actor/grad_norm: finite and > 0
actor/kl_loss: finite
training/global_step: 1 then 2
```

After checkpoint merge, load `adapter_model.safetensors` and assert:

- at least one `lora_B` tensor exists;
- every tensor is finite;
- at least one `lora_B` tensor has `abs().max() > 0`.

PEFT LoRA-B starts at zero, so a non-zero saved LoRA-B tensor plus finite non-zero gradient evidence proves an optimizer update occurred.

### 3.5 Checkpoint and controlled eval

Evidence:

- `global_step_1` and `global_step_2` checkpoint directories exist.
- `latest_checkpointed_iteration.txt` contains `2`.
- The step-2 actor checkpoint merges successfully with `python -m verl.model_merger merge`.
- The merged LoRA adapter is loadable by `evaluation/evaluate_models.py`.
- A four-sample evaluation finishes with:

```text
--evaluation_contract controlled
--num_options 2
--mode thinking
```

No accuracy or valid-answer-rate threshold applies to this two-step smoke. The eval gate is process completion plus correct controlled metadata.

---

## 4. PASS / FAIL Contract

### PASS requires all of the following

- GPU/environment preflight succeeds.
- The exact model revision and source/fixture hashes are recorded.
- Tiny fixture contains exactly 8 complete pairs / 16 rows.
- Training exits with code 0 after exactly 2 optimization steps.
- Real synchronous vLLM generation occurs for both steps.
- Explicit identity survives generation and a proven non-identity balance reorder.
- All 16 `(pair_id, rollout_slot)` groups per step pair permutation 0 with 1 exactly once.
- `consistency_unpaired_rate` is zero and no duplicate identity is observed.
- PA pair-baseline metrics exist and no original-GRPO fallback warning appears.
- Reward, advantage, KL, policy loss, and grad norm evidence is finite.
- `actor/grad_norm > 0` for both steps.
- Step-2 checkpoint exists, merges, and contains a non-zero finite LoRA-B tensor.
- Minimal controlled evaluation completes and records `evaluation_contract=controlled`, `num_options=2`.
- Required compact artifacts are saved; model weights and full checkpoints are not committed.

### FAIL conditions

- Any traceback, CUDA OOM after the documented retry ladder, NCCL/Ray worker failure, or non-zero exit.
- vLLM is bypassed, mocked, or replaced with Transformers/HF rollout.
- `legacy_index` is used.
- Identity/order validation fails or is disabled.
- Missing, duplicated, or unpaired `(pair_id, rollout_slot)` groups.
- `fallback to original GRPO` appears.
- Pair-baseline metrics are missing.
- No actor update, zero/non-finite grad norm, or missing/non-updated LoRA checkpoint.
- The run completes fewer than 2 optimization steps.
- Evidence is reconstructed from expectations rather than captured from the actual run.

An environment or model-access failure is `BLOCKED`, not PASS and not evidence that trainer logic is wrong.

---

## 5. Evidence Artifact

Create after the real GPU run:

```text
artifacts/gpu_smoke/task4/
├── README.md
├── environment.txt
├── git_state.txt
├── model_manifest.json
├── dataset_manifest.json
├── resolved_config.yaml
├── command.sh
├── train.log
├── reward_log_window.json
├── metrics_summary.json
├── identity_pairing_summary.json
├── checkpoint_manifest.json
├── lora_update.json
├── controlled_eval_summary.json
└── checksums.txt
```

Do not commit:

```text
model weights
full checkpoints
Ray session directories
TensorBoard event files
full rollout dumps
credentials or provider paths containing secrets
```

`README.md` must distinguish actual facts from inferences and list every retry/config change.

---

## 6. Implementation Plan

### Task 1: Add the deterministic tiny-fixture builder

**Files:**
- Create: `scripts_permstudy/task4/build_tiny_pairwise_smoke.py`
- Test: `tests_permstudy/test_task4_tiny_pairwise_smoke.py`

**Interfaces:**
- Consumes: source parquet path, Qwen tokenizer path, pair count, max prompt length, output parquet path, manifest path.
- Produces: 16-row parquet and JSON manifest with `pair_id`, `permutation_id`, source row index, prompt-token length, ground truth, source SHA256, and fixture SHA256.

- [ ] **Step 1: Write fixture tests**

Test exact pair completeness, invalid/missing permutations, overlong-pair exclusion, deterministic selection, alternating prompt-length ordering, mirrored labels, and manifest hashes.

```python
assert len(fixture) == 16
assert fixture["extra_info"].map(lambda x: x["original_question_id"]).nunique() == 8
assert all(sorted(group.map(lambda x: x["permutation"])) == [0, 1] for _, group in groups)
assert manifest["pair_count"] == 8
assert manifest["row_count"] == 16
```

- [ ] **Step 2: Run tests and observe failure**

```bash
pytest tests_permstudy/test_task4_tiny_pairwise_smoke.py -q
```

Expected: FAIL because the builder does not exist.

- [ ] **Step 3: Implement the builder minimally**

Command contract:

```bash
python scripts_permstudy/task4/build_tiny_pairwise_smoke.py \
  --source dataset/train/chatbot_arena_raw_2perm_thinking.parquet \
  --model-path /path/to/Qwen3-8B \
  --pair-count 8 \
  --max-prompt-length 4096 \
  --output /path/to/task4_tiny_8pairs.parquet \
  --manifest /path/to/dataset_manifest.json
```

- [ ] **Step 4: Run focused and existing tests**

```bash
pytest tests_permstudy/test_task4_tiny_pairwise_smoke.py -q
pytest tests_permstudy/test_rollout_identity_propagation.py \
       tests_permstudy/test_end_to_end_consistency_pairing.py -q
```

- [ ] **Step 5: Commit**

```bash
git add scripts_permstudy/task4/build_tiny_pairwise_smoke.py \
        tests_permstudy/test_task4_tiny_pairwise_smoke.py
git commit -m "test: add deterministic task4 pairwise fixture"
```

### Task 2: Add a foreground, fail-fast GPU launcher

**Files:**
- Create: `scripts/run_task4_real_gpu_smoke.sh`
- Test: `tests_permstudy/test_task4_smoke_launcher.py`

**Interfaces:**
- Consumes: `MODEL_PATH`, `TINY_DATASET`, `TASK4_RUN_ID`, `TASK4_RUN_DIR`.
- Produces: foreground exit code, captured command, resolved Hydra config, rollout JSONL, run log, step checkpoints.

- [ ] **Step 1: Write launcher contract tests**

Assert the launcher contains the fixed settings, runs in the foreground with `set -euo pipefail`, records environment/config and the reward-log start offset before training, assigns a unique TensorBoard directory, does not use `nohup`, and opts into explicit identity.

```python
assert "grouping.identity_mode=explicit" in launcher
assert "actor_rollout_ref.rollout.name=vllm" in launcher
assert "actor_rollout_ref.rollout.mode=sync" in launcher
assert "trainer.total_training_steps=2" in launcher
assert "trainer.save_freq=1" in launcher
assert 'TENSORBOARD_DIR="$TASK4_RUN_DIR/tensorboard"' in launcher
assert "nohup" not in launcher
```

- [ ] **Step 2: Run the test and observe failure**

```bash
pytest tests_permstudy/test_task4_smoke_launcher.py -q
```

- [ ] **Step 3: Implement the launcher**

The trainer invocation must include:

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  grouping.identity_mode=explicit \
  data.train_files="$TINY_DATASET" \
  data.val_files="$TINY_DATASET" \
  data.train_batch_size=16 \
  data.max_prompt_length=4096 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=false \
  data.truncation=error \
  data.shuffle=false \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.tokenizer_path="$MODEL_PATH" \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=64 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.rollout.max_model_len=4352 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  algorithm.use_kl_in_reward=false \
  trainer.balance_batch=true \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=2 \
  trainer.total_training_steps=2 \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=false \
  trainer.resume_mode=disable \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name=task4_real_gpu_smoke \
  trainer.experiment_name="$TASK4_RUN_ID" \
  trainer.rollout_data_dir="$TASK4_RUN_DIR/rollouts" \
  trainer.default_local_dir="$TASK4_RUN_DIR/checkpoints" \
  reward_model.reward_manager=batch \
  custom_reward_function.path="$PROJECT_ROOT/my_reward/judge_qwen.py" \
  custom_reward_function.name=compute_score
```

Before this invocation, the launcher must:

1. Require a non-empty `TASK4_RUN_ID` and a newly created empty `TASK4_RUN_DIR`.
2. Export `TENSORBOARD_DIR="$TASK4_RUN_DIR/tensorboard"`.
3. Record the current byte size (or zero if absent) and absolute path of `logs/judge_qwen3_8b.log` in `reward_log_window.json`.
4. Refuse to reuse a run directory containing any prior evidence, then capture its own stdout/stderr to `train.log` while preserving the foreground trainer exit code with `set -o pipefail` and `PIPESTATUS`.

- [ ] **Step 4: Validate Hydra composition without GPU training**

Use the repository config composition path to assert every override resolves and `grouping.identity_mode == "explicit"`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_task4_real_gpu_smoke.sh tests_permstudy/test_task4_smoke_launcher.py
git commit -m "chore: add task4 real gpu smoke launcher"
```

### Task 3: Add the evidence verifier

**Files:**
- Create: `scripts_permstudy/task4/verify_real_gpu_smoke.py`
- Test: `tests_permstudy/test_task4_smoke_verifier.py`

**Interfaces:**
- Consumes: run directory, dataset manifest, train log, reward-log window metadata, rollout directory, checkpoint directory, merged adapter, controlled-eval JSON.
- Produces: `metrics_summary.json`, `identity_pairing_summary.json`, `checkpoint_manifest.json`, `lora_update.json`, and a non-zero exit on any failed gate.

- [ ] **Step 1: Write failing verifier tests with synthetic artifacts**

Cover success plus one test for each hard failure: fallback warning, missing pair metric, unpaired/duplicate key, missing step, zero/non-finite grad norm, absent checkpoint, zero LoRA-B tensor, wrong evaluation contract, and reconstructed rather than observed row counts.

```python
assert summary["optimization_steps"] == [1, 2]
assert summary["fallback_detected"] is False
assert summary["identity"]["unpaired_count"] == 0
assert summary["identity"]["reorder_observed"] is True
assert summary["lora"]["nonzero_lora_b"] is True
```

- [ ] **Step 2: Run tests and observe failure**

```bash
pytest tests_permstudy/test_task4_smoke_verifier.py -q
```

- [ ] **Step 3: Implement strict parsing and validation**

The verifier must parse actual files, reject missing fields, check all floats with `math.isfinite`, and never substitute expected defaults for absent evidence. It must seek to the recorded reward-log byte offset and parse only the suffix appended by this run; file truncation, rotation, path mismatch, or an end offset not greater than the start offset is a hard failure.

- [ ] **Step 4: Run focused plus full project tests**

```bash
pytest tests_permstudy/test_task4_smoke_verifier.py -q
pytest tests_permstudy -q
```

- [ ] **Step 5: Commit**

```bash
git add scripts_permstudy/task4/verify_real_gpu_smoke.py \
        tests_permstudy/test_task4_smoke_verifier.py
git commit -m "test: verify task4 gpu smoke evidence"
```

### Task 4: Execute the AutoDL preflight and real smoke

**Files:**
- Create after execution: files beneath the external `$TASK4_RUN_DIR`.
- Do not commit provider caches, models, or checkpoints.

- [ ] **Step 1: Record immutable preflight evidence**

```bash
nvidia-smi
python --version
python -c "import torch, ray, vllm, verl; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), ray.__version__, vllm.__version__, verl.__file__)"
git rev-parse HEAD
git status --porcelain
pip freeze
```

Require the imported `verl.__file__` to resolve inside this repository checkout.

- [ ] **Step 2: Build and inspect the tiny fixture**

Run Task 1's builder, inspect the manifest, and independently assert 8 IDs × 2 permutations before training.

- [ ] **Step 3: Run the foreground launcher once**

```bash
MODEL_PATH=/path/to/Qwen3-8B \
TINY_DATASET=/path/to/task4_tiny_8pairs.parquet \
TASK4_RUN_ID=task4_run_001 \
TASK4_RUN_DIR=/path/to/task4_run_001 \
bash scripts/run_task4_real_gpu_smoke.sh
```

Do not detach the process. The launcher owns `train.log` capture and must return the trainer's exact exit code.

- [ ] **Step 4: Apply the documented memory retry ladder only if needed**

Retry order:

1. Lower vLLM `gpu_memory_utilization` from `0.50` to `0.40`.
2. Enable actor parameter/optimizer offload.
3. Lower `max_num_batched_tokens` while keeping 16 rows, `n=2`, and 2 steps fixed.

Each retry uses a new run directory. If all three fail, mark Task 4 FAIL/BLOCKED and stop; do not patch production trainer code during the GPU session.

### Task 5: Verify checkpoint, LoRA update, and controlled eval

**Files:**
- Read: step-2 actor checkpoint.
- Create externally: merged adapter and eval output.

- [ ] **Step 1: Merge the final checkpoint**

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$TASK4_RUN_DIR/checkpoints/global_step_2/actor" \
  --target_dir "$TASK4_RUN_DIR/checkpoints/global_step_2/actor/hf"
```

- [ ] **Step 2: Normalize the merged adapter config**

`verl/model_merger/base_model_merger.py` writes `"lora_alpha": 0` into the merged PEFT config,
while training uses `lora_rank=32` and `lora_alpha=64`. PEFT scales a LoRA update by `alpha / r`,
so an uncorrected adapter would scale the trained delta by `0 / 32 = 0` and the controlled
evaluation would not see the trained adapter. This step is mandatory and must run before the
evaluation:

```bash
python scripts_permstudy/task4/normalize_merged_adapter.py \
  --adapter-dir "$TASK4_RUN_DIR/checkpoints/global_step_2/actor/hf/lora_adapter" \
  --expected-rank 32 \
  --expected-alpha 64
```

It refuses a rank mismatch without rewriting anything, corrects `lora_alpha` to the training
value, and records the before/after SHA256 of `adapter_config.json` in
`adapter_normalization.json`.

- [ ] **Step 3: Run minimal controlled evaluation**

```bash
python evaluation/evaluate_models.py \
  --model_path "$TASK4_RUN_DIR/checkpoints/global_step_2/actor/hf" \
  --base_model_path "$MODEL_PATH" \
  --model_type qwen3 \
  --mode thinking \
  --dataset_files "$TINY_DATASET" \
  --max_samples 4 \
  --output_dir "$TASK4_RUN_DIR/controlled_eval" \
  --evaluation_contract controlled \
  --num_options 2 \
  --temperature 0 \
  --top_p 1 \
  --max_new_tokens 256
```

- [ ] **Step 4: Run the strict verifier once, after the evaluation exists**

```bash
python scripts_permstudy/task4/verify_real_gpu_smoke.py \
  --run-dir "$TASK4_RUN_DIR" \
  --dataset-manifest "$TASK4_RUN_DIR/dataset_manifest.json"
```

The verifier checks the merged adapter, the controlled-evaluation tree and the reward-log window
together, and it requires `adapter_normalization.json` with an `after_sha256` matching the current
`adapter_config.json`. It therefore cannot run before Step 3: with no `controlled_eval/**/summary.json`
it fails with an evidence error. Require controlled metadata but no accuracy threshold.

Execution order for this task is exactly:

```text
merge -> normalize_merged_adapter -> controlled evaluation -> final strict verifier
```

### Task 6: Curate and review the compact evidence artifact

**Files:**
- Create: `artifacts/gpu_smoke/task4/*` listed in Section 5.
- Modify: `Codex_Task4_Real_vLLM_GRPO_Smoke.md` only to append the reviewed execution result and final status.

- [ ] **Step 1: Copy only compact, redacted evidence**

Use actual captured outputs. Do not infer test counts, metric values, hashes, or checkpoint state.

- [ ] **Step 2: Run final verification**

```bash
pytest tests_permstudy -q
python scripts_permstudy/task4/verify_real_gpu_smoke.py \
  --run-dir "$TASK4_RUN_DIR" \
  --dataset-manifest "$TASK4_RUN_DIR/dataset_manifest.json"
git diff --check
```

- [ ] **Step 3: Request code/evidence review**

The reviewer must independently inspect the actual run evidence and explicitly confirm every PASS gate.

- [ ] **Step 4: Commit Task 4 evidence separately**

```bash
git add artifacts/gpu_smoke/task4 Codex_Task4_Real_vLLM_GRPO_Smoke.md
git commit -m "audit: record task4 real gpu smoke"
```

Task 4 is not complete until this evidence commit is reviewed. Planning or launcher implementation alone is not PASS.

---

## 7. Review Gate Before GPU Use

This commit contains the execution/design plan only. Do not implement the scripts or start AutoDL until the user reviews and approves:

- the Qwen3-8B model choice;
- 8 pairs / 16 rows;
- 2 optimization steps;
- initial memory settings and retry ladder;
- PASS/FAIL requirements;
- artifact contents.

After approval, execute Tasks 1–3 locally with TDD, review those scripts, then proceed to Tasks 4–6 on AutoDL.
