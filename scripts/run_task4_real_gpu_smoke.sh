#!/usr/bin/env bash
#
# Foreground, fail-fast launcher for the Task 4 real vLLM / GPU / GRPO smoke.
#
# This script is executed on the Linux GPU host (AutoDL). It never detaches:
# it owns stdout/stderr capture and returns the trainer's exact exit code.
#
# Every precondition -- including the reward-log sizing probe and the Hydra
# composition -- is checked BEFORE the run directory is created, so a failed
# precondition leaves no half-built run directory that would refuse a retry.
# The run directory is absolutised against PROJECT_ROOT before any check, so a
# relative TASK4_RUN_DIR can never be validated in one directory and used in
# another. REWARD_LOG is never taken from the environment.
#
# Exit codes: 2 = precondition/capture failure, otherwise the trainer's status.
#
# Required environment:
#   MODEL_PATH     path or snapshot of Qwen/Qwen3-8B
#   TINY_DATASET   the 16-row fixture produced by
#                  scripts_permstudy/task4/build_tiny_pairwise_smoke.py
#   TASK4_RUN_ID   unique identifier for this run
#   TASK4_RUN_DIR  absent or provably empty directory this run may own
#
# Optional environment:
#   PROJECT_ROOT   repository root (defaults to the parent of this script); it must
#                  contain my_reward/judge_qwen.py, because the production reward derives
#                  its log path from that module's location and reads no environment
#                  variable, so the reward log is not overridable.
#
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${TINY_DATASET:?TINY_DATASET is required}"
: "${TASK4_RUN_ID:?TASK4_RUN_ID is required}"
: "${TASK4_RUN_DIR:?TASK4_RUN_DIR is required}"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# The trainer needs CWD = repository root, and every path below must mean the same thing
# for the checks and for the writes, so normalise before anything else runs.
cd "$PROJECT_ROOT"
case "$TASK4_RUN_DIR" in
  /*) ;;
  *) TASK4_RUN_DIR="$PROJECT_ROOT/$TASK4_RUN_DIR" ;;
esac

# ---------------------------------------------------------------------------
# Precondition 1: the run directory must be absent, or prove itself empty.
# This guard is fail-closed: being unable to list the directory is treated as
# "may hold prior evidence", never as "empty". A partially written run directory
# would otherwise let stale checkpoints or rollouts be read as this run's
# evidence.
# ---------------------------------------------------------------------------
if [ -e "$TASK4_RUN_DIR" ]; then
  if [ ! -d "$TASK4_RUN_DIR" ]; then
    echo "refusing to run: TASK4_RUN_DIR exists and is not a directory: $TASK4_RUN_DIR" >&2
    exit 2
  fi
  if ! entries="$(ls -A "$TASK4_RUN_DIR")"; then
    echo "refusing to run: cannot list TASK4_RUN_DIR to prove it is empty: $TASK4_RUN_DIR" >&2
    exit 2
  fi
  if [ -n "$entries" ]; then
    echo "refusing to reuse TASK4_RUN_DIR with prior evidence: $TASK4_RUN_DIR" >&2
    exit 2
  fi
fi

# ---------------------------------------------------------------------------
# Precondition 2: PROJECT_ROOT must be the root the production reward belongs to.
#
# `my_reward/judge_qwen.py:28-32` derives its log path from its own module
# location (`dirname(dirname(__file__))/logs/judge_qwen3_8b.log`) and never reads
# the environment, so there is no legitimate way to point the reward at another
# file. Requiring the module here makes PROJECT_ROOT provably that root and makes
# the windowed path provably the reward's own LOG_PATH.
# ---------------------------------------------------------------------------
if [ ! -f "$PROJECT_ROOT/my_reward/judge_qwen.py" ]; then
  echo "refusing to run: PROJECT_ROOT does not contain my_reward/judge_qwen.py, so the reward's log path cannot be derived: $PROJECT_ROOT" >&2
  exit 2
fi
REWARD_LOG="$PROJECT_ROOT/logs/judge_qwen3_8b.log"

# ---------------------------------------------------------------------------
# Precondition 3: the reward log's directory must exist and the file must be
# writable BEFORE the trainer starts.
#
# `safe_log` swallows its own errors (`my_reward/judge_qwen.py:140-146`) and its
# parent directory is otherwise created only by `get_tb_writer()`
# (`:166-173`), which runs at the END of the first reward call. On a clean
# checkout that loses the whole first step's BASE/PAIR_CHECK evidence silently,
# and the verifier requires one reward block per step.
# ---------------------------------------------------------------------------
if ! mkdir -p "$(dirname "$REWARD_LOG")"; then
  echo "refusing to run: cannot create the reward log directory: $(dirname "$REWARD_LOG")" >&2
  exit 2
fi
if ! touch "$REWARD_LOG"; then
  echo "refusing to run: cannot create the reward log: $REWARD_LOG" >&2
  exit 2
fi
if [ ! -f "$REWARD_LOG" ]; then
  echo "refusing to run: reward log is not a regular file: $REWARD_LOG" >&2
  exit 2
fi
if [ ! -r "$REWARD_LOG" ]; then
  echo "refusing to run: reward log is not readable: $REWARD_LOG" >&2
  exit 2
fi
if [ ! -w "$REWARD_LOG" ]; then
  echo "refusing to run: reward log is not writable, so the run could not record its evidence: $REWARD_LOG" >&2
  exit 2
fi
if ! REWARD_LOG_START_BYTES="$(wc -c < "$REWARD_LOG" | tr -d '[:space:]')"; then
  echo "cannot size the reward log: $REWARD_LOG" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Fixed smoke configuration. Every value here is part of the PASS contract.
# ---------------------------------------------------------------------------
TRAINER_OVERRIDES=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=false
  grouping.identity_mode=explicit
  data.train_files="$TINY_DATASET"
  data.val_files="$TINY_DATASET"
  data.train_batch_size=16
  data.max_prompt_length=4096
  data.max_response_length=256
  data.filter_overlong_prompts=false
  data.truncation=error
  data.shuffle=false
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.tokenizer_path="$MODEL_PATH"
  actor_rollout_ref.model.lora_rank=32
  actor_rollout_ref.model.lora_alpha=64
  actor_rollout_ref.model.target_modules=all-linear
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.actor.optim.lr=1e-5
  actor_rollout_ref.actor.ppo_mini_batch_size=16
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_kl_loss=true
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=sync
  actor_rollout_ref.rollout.n=2
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50
  actor_rollout_ref.rollout.enforce_eager=true
  actor_rollout_ref.rollout.max_num_seqs=32
  actor_rollout_ref.rollout.max_model_len=4352
  actor_rollout_ref.rollout.max_num_batched_tokens=8192
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.layered_summon=true
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.fsdp_config.param_offload=true
  trainer.balance_batch=true
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.total_epochs=2
  trainer.total_training_steps=2
  trainer.save_freq=1
  trainer.test_freq=-1
  trainer.val_before_train=false
  trainer.resume_mode=disable
  trainer.logger='["console","tensorboard"]'
  trainer.project_name=task4_real_gpu_smoke
  trainer.experiment_name="$TASK4_RUN_ID"
  trainer.rollout_data_dir="$TASK4_RUN_DIR/rollouts"
  trainer.default_local_dir="$TASK4_RUN_DIR/checkpoints"
  reward_model.reward_manager=batch
  custom_reward_function.path="$PROJECT_ROOT/my_reward/judge_qwen.py"
  custom_reward_function.name=compute_score
)

# ---------------------------------------------------------------------------
# Precondition 3: the overrides must compose against the repository config, and
# that composed result is itself required evidence. Nothing is written into the
# run directory until this succeeds.
# ---------------------------------------------------------------------------
RESOLVED_CONFIG_TMP="$(mktemp)"
if ! python -c '
import os, sys
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config"), version_base=None):
    cfg = compose(config_name="ppo_trainer", overrides=sys.argv[1:])
sys.stdout.write(OmegaConf.to_yaml(cfg))
' "${TRAINER_OVERRIDES[@]}" > "$RESOLVED_CONFIG_TMP"; then
  rm -f "$RESOLVED_CONFIG_TMP"
  echo "refusing to run: trainer overrides do not compose against verl/trainer/config" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Every precondition passed: the run directory may now be created and owned.
# ---------------------------------------------------------------------------
mkdir -p "$TASK4_RUN_DIR"
mv "$RESOLVED_CONFIG_TMP" "$TASK4_RUN_DIR/resolved_config.yaml"

# A per-run TensorBoard directory; the reward implementation's hard-coded
# directory may hold events from earlier runs and is diagnostic only.
export TENSORBOARD_DIR="$TASK4_RUN_DIR/tensorboard"
mkdir -p "$TENSORBOARD_DIR"

printf '{"reward_log_path": "%s", "start_offset_bytes": %s}\n' \
  "$REWARD_LOG" "$REWARD_LOG_START_BYTES" > "$TASK4_RUN_DIR/reward_log_window.json"

# ---------------------------------------------------------------------------
# Record the environment and Git state actually used by this run.
#
# Provenance capture must never be able to abort the run before the trainer
# starts: a hostile or unusual environment (missing git, git safe.directory
# refusal, absent pip) has to be recorded honestly instead of masking the
# trainer's own exit status.
# ---------------------------------------------------------------------------
{
  printf 'task4_run_id=%s\n' "$TASK4_RUN_ID"
  printf 'task4_run_dir=%s\n' "$TASK4_RUN_DIR"
  printf 'model_path=%s\n' "$MODEL_PATH"
  printf 'tiny_dataset=%s\n' "$TINY_DATASET"
  printf 'tensorboard_dir=%s\n' "$TENSORBOARD_DIR"
  printf 'reward_log=%s\n' "$REWARD_LOG"
  printf 'reward_log_start_offset_bytes=%s\n' "$REWARD_LOG_START_BYTES"
  printf 'python=%s\n' "$(python -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo unavailable)"
  pip freeze 2>/dev/null || printf 'pip_freeze=unavailable\n'
} > "$TASK4_RUN_DIR/environment.txt"

{
  git rev-parse HEAD 2>&1 || printf 'git_rev_parse=unavailable\n'
  printf -- '--- git status --porcelain ---\n'
  git status --porcelain 2>&1 || printf 'git_status=unavailable\n'
} > "$TASK4_RUN_DIR/git_state.txt"

# Record the exact command the launcher is about to run, from the directory it
# actually runs in.
command_line="cd '$PROJECT_ROOT' && python -m verl.trainer.main_ppo"
for override in "${TRAINER_OVERRIDES[@]}"; do
  command_line+=" $(printf '%q' "$override")"
done
printf '%s\n' "$command_line" > "$TASK4_RUN_DIR/command.sh"

# ---------------------------------------------------------------------------
# Run in the foreground, capture output, and preserve the trainer's exit code.
# ---------------------------------------------------------------------------
TRAINER_STATUS=0
TEE_STATUS=0
set +e
set -o pipefail
python -m verl.trainer.main_ppo "${TRAINER_OVERRIDES[@]}" 2>&1 | tee "$TASK4_RUN_DIR/train.log"
# PIPESTATUS is reset by every subsequent simple command, so snapshot the whole array
# immediately. Reading ${PIPESTATUS[0]} first would make ${PIPESTATUS[1]} unbound under
# `set -u` and kill the script on every run, including a clean one.
pipeline_status=("${PIPESTATUS[@]}")
set -e
TRAINER_STATUS=${pipeline_status[0]:-0}
TEE_STATUS=${pipeline_status[1]:-0}

# A partially written or missing train.log means the capture failed, and train.log
# is a PASS-gating artifact: report that instead of the trainer's clean status.
if [ "$TEE_STATUS" -ne 0 ] || [ ! -s "$TASK4_RUN_DIR/train.log" ]; then
  echo "trainer output was not captured to ${TASK4_RUN_DIR}/train.log (trainer exit status was ${TRAINER_STATUS}, tee exit status was ${TEE_STATUS})" >&2
  exit 2
fi

exit "$TRAINER_STATUS"
