"""Tests for the Task 4 foreground, fail-fast real-GPU smoke launcher.

The launcher is a Linux shell script executed on AutoDL, so these tests pin its
contract three ways: textual guarantees for the fixed PASS settings, textual
guarantees that every precondition is executed fail-closed, and a real Hydra
composition proving every override the launcher passes actually resolves.
"""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_task4_real_gpu_smoke.sh"

OVERRIDE_BLOCK = re.compile(r"TRAINER_OVERRIDES=\(\n(.*?)\n\)\n", re.S)
TRAINER_INVOCATION = 'python -m verl.trainer.main_ppo "${TRAINER_OVERRIDES[@]}"'

# Values only ever set at runtime on the GPU host; substituted while emulating bash.
RUNTIME_VALUES = {
    "TASK4_RUN_DIR": "/tmp/task4_run_001",
    "TASK4_RUN_ID": "task4_run_001",
    "TINY_DATASET": "/tmp/task4_tiny_8pairs.parquet",
    "MODEL_PATH": "/tmp/Qwen3-8B",
    "PROJECT_ROOT": REPO_ROOT.as_posix(),
}
_VARIABLE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


def bash_word(line: str) -> str:
    """Emulate bash quote removal plus expansion for one array element.

    The launcher relies on real bash semantics: `data.train_files="$TINY_DATASET"`
    contributes one argv element with no quotes, while
    `trainer.logger='["console","tensorboard"]'` keeps its inner double quotes.
    """
    out: list[str] = []
    index = 0
    in_single = False
    in_double = False
    while index < len(line):
        char = line[index]
        if in_single:
            if char == "'":
                in_single = False
            else:
                out.append(char)
            index += 1
            continue
        if char == "'" and not in_double:
            in_single = True
            index += 1
            continue
        if char == '"':
            in_double = not in_double
            index += 1
            continue
        match = _VARIABLE.match(line, index)
        if match:
            name = match.group(1)
            assert name in RUNTIME_VALUES, f"unknown variable in override: {name}"
            out.append(RUNTIME_VALUES[name])
            index = match.end()
            continue
        out.append(char)
        index += 1
    assert not in_single and not in_double, f"unbalanced quotes in override: {line}"
    return "".join(out).strip()

EXPECTED_OVERRIDES = {
    "algorithm.adv_estimator=grpo",
    "algorithm.use_kl_in_reward=false",
    "grouping.identity_mode=explicit",
    "data.train_files=/tmp/task4_tiny_8pairs.parquet",
    "data.val_files=/tmp/task4_tiny_8pairs.parquet",
    "data.train_batch_size=16",
    "data.max_prompt_length=4096",
    "data.max_response_length=256",
    "data.filter_overlong_prompts=false",
    "data.truncation=error",
    "data.shuffle=false",
    "actor_rollout_ref.model.path=/tmp/Qwen3-8B",
    "actor_rollout_ref.model.tokenizer_path=/tmp/Qwen3-8B",
    "actor_rollout_ref.model.lora_rank=32",
    "actor_rollout_ref.model.lora_alpha=64",
    "actor_rollout_ref.model.target_modules=all-linear",
    "actor_rollout_ref.model.enable_gradient_checkpointing=true",
    "actor_rollout_ref.actor.strategy=fsdp2",
    "actor_rollout_ref.actor.optim.lr=1e-5",
    "actor_rollout_ref.actor.ppo_mini_batch_size=16",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.actor.use_kl_loss=true",
    "actor_rollout_ref.actor.kl_loss_coef=0.001",
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
    "actor_rollout_ref.rollout.name=vllm",
    "actor_rollout_ref.rollout.mode=sync",
    "actor_rollout_ref.rollout.n=2",
    "actor_rollout_ref.rollout.temperature=1.0",
    "actor_rollout_ref.rollout.top_p=1.0",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.50",
    "actor_rollout_ref.rollout.enforce_eager=true",
    "actor_rollout_ref.rollout.max_num_seqs=32",
    "actor_rollout_ref.rollout.max_model_len=4352",
    "actor_rollout_ref.rollout.max_num_batched_tokens=8192",
    "actor_rollout_ref.rollout.load_format=safetensors",
    "actor_rollout_ref.rollout.layered_summon=true",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.ref.fsdp_config.param_offload=true",
    "trainer.balance_batch=true",
    "trainer.n_gpus_per_node=1",
    "trainer.nnodes=1",
    "trainer.total_epochs=2",
    "trainer.total_training_steps=2",
    "trainer.save_freq=1",
    "trainer.test_freq=-1",
    "trainer.val_before_train=false",
    "trainer.resume_mode=disable",
    'trainer.logger=["console","tensorboard"]',
    "trainer.project_name=task4_real_gpu_smoke",
    "trainer.experiment_name=task4_run_001",
    "trainer.rollout_data_dir=/tmp/task4_run_001/rollouts",
    "trainer.default_local_dir=/tmp/task4_run_001/checkpoints",
    "reward_model.reward_manager=batch",
    f"custom_reward_function.path={REPO_ROOT.as_posix()}/my_reward/judge_qwen.py",
    "custom_reward_function.name=compute_score",
}


def launcher_text() -> str:
    assert LAUNCHER.is_file(), f"missing task4 launcher: {LAUNCHER}"
    return LAUNCHER.read_text(encoding="utf-8")


def extract_overrides() -> list[str]:
    text = launcher_text()
    match = OVERRIDE_BLOCK.search(text)
    assert match is not None, "launcher must declare a TRAINER_OVERRIDES=( ... ) array"
    overrides = []
    for raw in match.group(1).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        expanded = bash_word(line)
        assert "$" not in expanded, f"unsubstituted variable left in override: {expanded}"
        overrides.append(expanded)
    return overrides


# ---------------------------------------------------------------------------
# foreground, fail-fast behaviour
# ---------------------------------------------------------------------------


def test_launcher_is_foreground_and_fail_fast():
    text = launcher_text()
    assert "set -euo pipefail" in text
    assert "nohup" not in text


def test_launcher_requires_every_run_input():
    text = launcher_text()
    for name in ("MODEL_PATH", "TINY_DATASET", "TASK4_RUN_ID", "TASK4_RUN_DIR"):
        assert f"${{{name}:?" in text, f"launcher must fail fast when {name} is unset"


def test_launcher_preserves_the_trainer_exit_code():
    text = launcher_text()
    assert "set -o pipefail" in text
    assert "PIPESTATUS[0]" in text
    assert re.search(r"^\s*exit\s+\"?\$TRAINER_STATUS\"?\s*$", text, re.M), (
        "launcher must exit with the trainer's own status"
    )


def test_launcher_captures_train_log_and_keeps_running_in_the_foreground():
    text = launcher_text()
    assert "train.log" in text
    assert "tee" in text


def test_launcher_refuses_to_report_success_without_captured_train_log():
    """An unchecked `tee` failure would report a clean run with no evidence."""
    text = launcher_text()
    assert re.search(r'\[ ! -s "\$TASK4_RUN_DIR/train\.log" \]', text), (
        "launcher must verify train.log was actually written before reporting success"
    )
    assert text.count("train.log") >= 3


# ---------------------------------------------------------------------------
# precondition ordering and fail-closed guards
# ---------------------------------------------------------------------------


def test_run_dir_reuse_guard_is_fail_closed():
    """Being unable to list the run dir must never be read as 'empty'."""
    text = launcher_text()
    assert re.search(r'if ! entries="\$\(ls -A "\$TASK4_RUN_DIR"\)"', text), (
        "the emptiness probe must be checked, not silenced with `|| true`"
    )
    guard = text[text.index('if [ -e "$TASK4_RUN_DIR" ]') : text.index("RESOLVED_CONFIG_TMP=")]
    assert "2>/dev/null" not in guard, "the reuse guard must not swallow listing failures"
    assert guard.count("exit 2") >= 3, "refusal, non-directory and unlistable cases must all exit 2"


def test_relative_runtime_paths_are_absolutised_before_any_check():
    """A relative TASK4_RUN_DIR must not be validated against the caller's cwd.

    Regression: the `cd "$PROJECT_ROOT"` used to sit after the guards but before every
    write, so `TASK4_RUN_DIR=rel` was checked against the invocation directory and then
    used under PROJECT_ROOT, letting prior evidence be overwritten.
    """
    text = launcher_text()
    cd_index = text.index('cd "$PROJECT_ROOT"')
    guard_index = text.index('if [ -e "$TASK4_RUN_DIR" ]')
    assert cd_index < guard_index, "the launcher must cd before the run-dir guard"

    absolutise = re.search(
        r'case "\$TASK4_RUN_DIR" in\n\s*/\*\) ;;\n\s*\*\) TASK4_RUN_DIR="\$PROJECT_ROOT/\$TASK4_RUN_DIR" ;;\nesac',
        text,
    )
    assert absolutise is not None, "TASK4_RUN_DIR must be absolutised against PROJECT_ROOT"
    assert text.index(absolutise.group(0)) < guard_index, "TASK4_RUN_DIR must be absolutised before the guards"
    # REWARD_LOG is never taken from the environment, so it is absolute by construction, and it
    # must be derived before the preconditions that create and probe it
    assert 'REWARD_LOG="$PROJECT_ROOT/logs/judge_qwen3_8b.log"' in text
    assert text.index('REWARD_LOG="$PROJECT_ROOT/logs/judge_qwen3_8b.log"') < text.index(
        'mkdir -p "$(dirname "$REWARD_LOG")"'
    )


def test_tee_exit_status_is_checked():
    """A tee that writes partial output and fails must not be reported as success."""
    text = launcher_text()
    assert 'pipeline_status=("${PIPESTATUS[@]}")' in text, (
        "PIPESTATUS must be snapshotted in one step: reading ${PIPESTATUS[0]} first resets the "
        "array and makes ${PIPESTATUS[1]} unbound under `set -u`"
    )
    assert 'TEE_STATUS=${pipeline_status[1]:-0}' in text
    assert re.search(r'if \[ "\$TEE_STATUS" -ne 0 \]', text), "tee's own status must gate success"


def test_reward_log_is_not_overridable_and_follows_the_reward_module(tmp_path):
    """`judge_qwen.py:28-32` derives its log path from its own module and reads no env var."""
    text = launcher_text()
    assert "REWARD_LOG=\"${REWARD_LOG" not in text, (
        "a REWARD_LOG override would be a lie: the production reward never reads it"
    )
    assert 'REWARD_LOG="$PROJECT_ROOT/logs/judge_qwen3_8b.log"' in text
    assert re.search(r'if \[ ! -f "\$PROJECT_ROOT/my_reward/judge_qwen\.py" \]', text), (
        "PROJECT_ROOT must be proved to be the reward's own root"
    )


def test_reward_log_preconditions_cover_creation_readability_and_writability():
    """A clean checkout has no logs/ dir, so step 1's evidence would be silently lost."""
    text = launcher_text()
    assert 'mkdir -p "$(dirname "$REWARD_LOG")"' in text, (
        "the reward log's parent must exist before the trainer starts"
    )
    assert re.search(r'if ! touch "\$REWARD_LOG"; then', text), "the log must be created up front"
    assert re.search(r'if \[ ! -r "\$REWARD_LOG" \]', text)
    assert re.search(r'if \[ ! -w "\$REWARD_LOG" \]', text), (
        "the log must be writable, since safe_log swallows a failed append"
    )
    assert 'if ! REWARD_LOG_START_BYTES="$(wc -c < "$REWARD_LOG"' in text, (
        "the reward-log sizing probe must not abort the run through `set -e`"
    )


def test_run_dir_is_created_only_after_every_precondition():
    text = launcher_text()
    creation = text.index('mkdir -p "$TASK4_RUN_DIR"')
    for precondition in (
        'if [ -e "$TASK4_RUN_DIR" ]',
        'if [ ! -f "$PROJECT_ROOT/my_reward/judge_qwen.py" ]',
        'mkdir -p "$(dirname "$REWARD_LOG")"',
        'if ! REWARD_LOG_START_BYTES="$(wc -c < "$REWARD_LOG"',
        "RESOLVED_CONFIG_TMP=",
    ):
        assert text.index(precondition) < creation, f"{precondition} must run before the run dir exists"


def test_launcher_writes_the_resolved_hydra_config_into_the_run_dir():
    text = launcher_text()
    assert "resolved_config.yaml" in text
    assert "OmegaConf.to_yaml" in text
    assert text.index("resolved_config.yaml") < text.index(TRAINER_INVOCATION)


def test_provenance_capture_cannot_abort_the_run_before_training():
    """Every pre-trainer probe in the launcher must tolerate its own failure.

    Regression: under `set -e`, failing probes (`git rev-parse HEAD` returning
    non-zero via git's safe.directory refusal, a `wc` that cannot read the reward
    log) aborted the launcher before the trainer started and reported a status
    that was indistinguishable from a trainer failure.
    """
    text = launcher_text()
    blocks = {
        "environment.txt": re.search(r"\{\n(.*?)\n\} > \"\$TASK4_RUN_DIR/environment\.txt\"", text, re.S),
        "git_state.txt": re.search(r"\{\n(.*?)\n\} > \"\$TASK4_RUN_DIR/git_state\.txt\"", text, re.S),
    }
    checked = 0
    for name, match in blocks.items():
        assert match is not None, f"missing provenance capture block for {name}"
        for raw in match.group(1).splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            probes = line.startswith(("git ", "pip ")) or "$(python" in line
            if probes:
                checked += 1
                assert "||" in line, (
                    f"{name} probe can abort the run under `set -e` before the trainer starts: {line}"
                )
    assert checked >= 4, f"expected to inspect several probes, saw {checked}"

    # Probes outside those blocks are covered by the other guarded forms.
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"' in text
    assert 'if ! entries="$(ls -A "$TASK4_RUN_DIR")"' in text
    assert 'if ! REWARD_LOG_START_BYTES="$(wc -c < "$REWARD_LOG"' in text


def test_launcher_creates_the_run_dir_before_writing_evidence():
    text = launcher_text()
    assert text.index('mkdir -p "$TASK4_RUN_DIR"') < text.index("reward_log_window.json")


def test_launcher_records_the_reward_log_window_before_the_trainer_starts():
    text = launcher_text()
    assert text.index("reward_log_window.json") < text.index(TRAINER_INVOCATION), (
        "reward-log window must be recorded before the trainer invocation"
    )
    assert "judge_qwen3_8b.log" in text
    assert "start_offset_bytes" in text


def test_launcher_creates_the_run_dir_even_when_it_is_absent():
    assert 'mkdir -p "$TASK4_RUN_DIR"' in launcher_text()


# ---------------------------------------------------------------------------
# fixed smoke settings
# ---------------------------------------------------------------------------


def test_launcher_selects_explicit_identity_and_synchronous_vllm():
    text = launcher_text()
    assert "grouping.identity_mode=explicit" in text
    assert "actor_rollout_ref.rollout.name=vllm" in text
    assert "actor_rollout_ref.rollout.mode=sync" in text
    assert "identity_mode=legacy_index" not in text
    assert "rollout.mode=async" not in text


def test_launcher_pins_the_two_step_smoke_schedule():
    text = launcher_text()
    assert "trainer.total_training_steps=2" in text
    assert "trainer.save_freq=1" in text
    assert "data.train_batch_size=16" in text
    assert "actor_rollout_ref.rollout.n=2" in text


def test_launcher_isolates_tensorboard_per_run():
    text = launcher_text()
    assert 'TENSORBOARD_DIR="$TASK4_RUN_DIR/tensorboard"' in text
    assert "export TENSORBOARD_DIR" in text


def test_launcher_pins_every_override_value():
    """Every override value is part of the PASS contract, not just the headline ones."""
    overrides = extract_overrides()
    assert len(overrides) == 57, f"expected 57 overrides, found {len(overrides)}"
    assert len(set(overrides)) == 57, "override list contains duplicates"
    assert set(overrides) == EXPECTED_OVERRIDES


# ---------------------------------------------------------------------------
# real Hydra composition
# ---------------------------------------------------------------------------


@pytest.mark.skipif(importlib.util.find_spec("hydra") is None, reason="requires hydra-core")
def test_every_launcher_override_resolves_and_forces_explicit_identity():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    overrides = extract_overrides()
    assert overrides, "launcher must pass overrides to the trainer"

    key_overrides = [override.split("=", 1)[0] for override in overrides]

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config"), version_base=None):
            cfg = compose(config_name="ppo_trainer", overrides=overrides)
    finally:
        GlobalHydra.instance().clear()

    assert cfg.grouping.identity_mode == "explicit"
    for key in key_overrides:
        assert OmegaConf.select(cfg, key) is not None, f"launcher override does not resolve: {key}"


def test_reward_log_directory_failure_uses_the_documented_exit_code():
    """A non-directory `logs` used to die on `set -e` with exit 1, contradicting the contract."""
    text = launcher_text()
    assert re.search(r'if ! mkdir -p "\$\(dirname "\$REWARD_LOG"\)"; then', text), (
        "the log-directory creation must carry its own exit-2 guard"
    )
    assert re.search(r'if \[ ! -f "\$REWARD_LOG" \]; then', text), (
        "a non-regular file at the log path (e.g. a FIFO) must be refused before the trainer blocks on it"
    )