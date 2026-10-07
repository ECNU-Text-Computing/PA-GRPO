"""Resident integration test: the real launcher and the real verifier must agree.

Every other test in this suite either inspects the launcher's text or feeds the verifier
synthetic evidence, so none of them proves the two agree on the run-directory contract. Two
defects this session were reachable only by executing the launcher (a `PIPESTATUS` reset that
made every run exit 1, and a `cd` ordering that let a relative run directory clobber prior
evidence), while the text assertions stayed green.

This test runs the real launcher with the trainer emulated in the real evidence formats
(`tests_permstudy/task4_probe/trainer_emulator.py`) and then verifies the produced run directory
with the real verifier.

It needs a genuine POSIX `bash` and a Python 3 with the repository's runtime available to that
bash. On Windows `bash` resolves to WSL, whose filesystem view is separate from the interpreter
running pytest, so the test skips there; it runs on the Linux GPU host, which is where the
launcher is used.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_task4_real_gpu_smoke.sh"
EMULATOR = REPO_ROOT / "tests_permstudy" / "task4_probe" / "trainer_emulator.py"
SOURCE_PARQUET = REPO_ROOT / "dataset" / "train" / "chatbot_arena_raw_2perm_thinking.parquet"

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="needs a POSIX bash whose filesystem view matches the interpreter running pytest",
)


def _load(module_name: str, path: Path):
    assert path.is_file(), f"missing {path}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_shims(stage: Path) -> Path:
    shim_dir = stage / "bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    python_shim = shim_dir / "python"
    python_shim.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "${1:-}" = "-c" ]; then\n'
        '  case "${2:-}" in *OmegaConf*) echo "composed: config"; exit 0 ;; *) echo "3.12.3"; exit 0 ;; esac\n'
        "fi\n"
        'if [ "${1:-}" = "-m" ]; then exec python3 "$TASK4_EMULATOR"; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    python_shim.chmod(0o755)
    return shim_dir


def _fixture_and_adapter(stage: Path) -> tuple[Path, Path]:
    """Build the Task 1 fixture and a real bf16 adapter with the interpreter running pytest."""
    builder = _load("task4_build_tiny_pairwise_smoke", REPO_ROOT / "scripts_permstudy" / "task4" / "build_tiny_pairwise_smoke.py")
    manifest_path = stage / "dataset_manifest.json"
    fixture_path = stage / "task4_tiny_8pairs.parquet"
    builder.build_tiny_pairwise_smoke(
        source=SOURCE_PARQUET,
        model_path="unused",
        pair_count=8,
        max_prompt_length=4096,
        output=fixture_path,
        manifest=manifest_path,
        tokenizer=_WhitespaceTokenizer(),
    )
    adapter_path = stage / "adapter_model.safetensors"
    import torch
    from safetensors.torch import save_file

    save_file(
        {
            "base_model.model.q_proj.lora_A.weight": torch.zeros((4, 4), dtype=torch.bfloat16),
            "base_model.model.q_proj.lora_B.weight": torch.full((4, 4), 0.125, dtype=torch.bfloat16),
        },
        str(adapter_path),
    )
    return fixture_path, adapter_path


class _WhitespaceTokenizer:
    """Enough of the tokenizer protocol for the fixture builder; no model weights required."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        parts = [f"{message['role']}: {message['content']}" for message in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)

    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_real_launcher_output_satisfies_the_real_verifier(tmp_path):
    if shutil.which("bash") is None:
        pytest.skip("no bash on PATH")
    if not SOURCE_PARQUET.is_file():
        pytest.skip("source parquet not present")

    stage = tmp_path / "stage"
    stage.mkdir()
    fixture_path, adapter_path = _fixture_and_adapter(stage)
    shim_dir = _write_shims(stage)
    run_dir = stage / "run"

    env = dict(os.environ)
    env.update(
        PATH=f"{shim_dir}{os.pathsep}{env['PATH']}",
        PROJECT_ROOT=str(REPO_ROOT),
        MODEL_PATH="/tmp/Qwen3-8B",
        TINY_DATASET=str(fixture_path),
        TASK4_RUN_ID="integration_probe",
        TASK4_RUN_DIR=str(run_dir),
        TASK4_EMULATOR=str(EMULATOR),
        TASK4_MANIFEST=str(stage / "dataset_manifest.json"),
        TASK4_ADAPTER_SRC=str(adapter_path),
    )
    env.pop("REWARD_LOG", None)

    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, f"launcher failed: {completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"

    # the launcher must have created the reward log's directory before the trainer ran
    reward_log = REPO_ROOT / "logs" / "judge_qwen3_8b.log"
    assert reward_log.is_file(), "the launcher must create the reward log before starting the trainer"
    window = json.loads((run_dir / "reward_log_window.json").read_text(encoding="utf-8"))
    assert Path(window["reward_log_path"]).resolve() == reward_log.resolve()

    verifier = _load("task4_verify_real_gpu_smoke", REPO_ROOT / "scripts_permstudy" / "task4" / "verify_real_gpu_smoke.py")
    summary = verifier.verify_real_gpu_smoke(
        verifier.VerifyPaths(
            run_dir=run_dir,
            dataset_manifest=stage / "dataset_manifest.json",
            reward_log_window=run_dir / "reward_log_window.json",
            train_log=run_dir / "train.log",
            rollout_dir=run_dir / "rollouts",
            checkpoint_dir=run_dir / "checkpoints",
            merged_adapter_candidates=[verifier.merged_adapter_path(run_dir)],
            controlled_eval_dir=run_dir / "controlled_eval",
        )
    )

    assert summary["passed"] is True
    assert summary["optimization_steps"] == [1, 2]
    assert summary["identity"]["pair_check_count"] == 32
    assert summary["identity"]["reorder_observed"] is True
    assert summary["lora"]["adapter_source"] == "merger_output"
    assert summary["lora"]["nonzero_lora_b"] is True
    assert summary["adapter_config"]["r"] == 32
    assert summary["adapter_config"]["lora_alpha"] == 64
