"""Tests for the Task 4 real-GPU smoke evidence verifier.

The verifier must decide PASS/FAIL strictly from artifacts the run produced, and
must never substitute an expected value for missing evidence.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts_permstudy" / "task4" / "verify_real_gpu_smoke.py"

PAIR_COUNT = 8
ROWS_PER_STEP = 32


def load_verifier():
    assert SCRIPT.is_file(), f"missing task4 evidence verifier: {SCRIPT}"
    spec = importlib.util.spec_from_file_location("task4_verify_real_gpu_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# synthetic evidence
# ---------------------------------------------------------------------------


def write_dataset_manifest(run_dir: Path, pairs: int = PAIR_COUNT, reverse: bool = False) -> dict:
    entries = []
    for index in range(pairs):
        pair_id = f"q{index:04d}"
        entries.append(
            {
                "pair_id": pair_id,
                "original_question_id": pair_id,
                "winner": "model_a",
                "max_prompt_tokens": 100 + index,
                "rows": [
                    {"permutation": 0, "row_index": 2 * index, "ground_truth": "A",
                     "prompt_tokens": 90 + index, "prompt_sha256": "0" * 64},
                    {"permutation": 1, "row_index": 2 * index + 1, "ground_truth": "B",
                     "prompt_tokens": 100 + index, "prompt_sha256": "1" * 64},
                ],
            }
        )
    if reverse:
        entries = entries[::-1]
    manifest = {
        "schema_version": "task4_tiny_pairwise_smoke_v1",
        "pairs": entries,
        "pair_count": pairs,
        "row_count": pairs * 2,
        "fixture_sha256": "a" * 64,
    }
    (run_dir / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def canonical_sequence(manifest: dict) -> list[tuple[str, int]]:
    sequence: list[tuple[str, int]] = []
    for pair in manifest["pairs"]:
        for permutation in (0, 1):
            sequence.extend([(pair["pair_id"], permutation)] * 2)
    return sequence


def write_train_log(
    run_dir: Path,
    *,
    steps=(1, 2),
    fallback: bool = False,
    grad_norm: float = 0.75,
    pair_baseline: bool = True,
    nan_metric: bool = False,
    vllm: bool = True,
    identity_error: bool = False,
) -> None:
    lines = []
    if vllm:
        # a real vLLM runtime line (not the resolved-config echo, which contains
        # `hybrid_engine`/`free_cache_engine`/`engine_kwargs` and must not satisfy this gate)
        lines.append("INFO 01-01 00:00:00 engine.py:88] Initializing an LLM engine (v0.8.5) with config:")
        lines.append("INFO 01-01 00:00:00 model_runner.py:907] Loading model weights took 15.1234 GB")
    for step in steps:
        parts = [
            f"step:{step}",
            f"training/global_step:{step}",
            "actor/pg_loss:0.5000",
            f"actor/grad_norm:{grad_norm}",
            "actor/kl_loss:0.0100",
            "reward/consistency_unpaired_rate:0.0",
        ]
        if pair_baseline:
            parts += [
                "pair_baseline/mean_of_pair_means:0.100000",
                "pair_baseline/mean_of_pair_stds:0.200000",
                "pair_baseline/std_of_pair_stds:0.300000",
            ]
        if nan_metric:
            parts.append("actor/kl_loss:nan")
        lines.append(" - ".join(parts))
    if fallback:
        # the exact string the trainer prints at verl/trainer/ppo/ray_trainer.py:326
        lines.append("[Warning] pair-baseline GRPO failed with error RuntimeError('probe'), fallback to original GRPO.")
    if identity_error:
        lines.append("RolloutIdentityAlignmentError: generation output reordered rollout identity for pair_id")
    (run_dir / "train.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_line(body: str, index: object = 0) -> str:
    return f"2026-01-01 00:00:00.000000 | idx={index} | pid=1 | cuda=cuda:0 | {body}"


def write_reward_log(
    run_dir: Path,
    manifest: dict,
    *,
    reorder: bool = True,
    unpaired: bool = False,
    duplicate: bool = False,
    blocks: int = 2,
    batch_position_indices: bool = False,
    bad_join: bool = False,
) -> None:
    path = run_dir / "reward.log"
    # historical evidence that must never be admissible
    path.write_text(
        _log_line("PAIR_CHECK | pair_id=stale_pair | rollout_slot=0 | is_consistent=True") + "\n",
        encoding="utf-8",
    )
    start = path.stat().st_size

    canonical = canonical_sequence(manifest)
    pair_ids = [pair["pair_id"] for pair in manifest["pairs"]]
    row_index_by_key = {
        (pair["pair_id"], int(row["permutation"])): int(row["row_index"])
        for pair in manifest["pairs"]
        for row in pair["rows"]
    }
    lines: list[str] = []
    for block_index in range(blocks):
        order = list(canonical)
        if reorder and block_index == 0:
            order = order[::-1]
        for index, (pair_id, permutation) in enumerate(order):
            # ``my_reward/judge_qwen.py`` logs ``idx = extra.get("index", i)``: the fixture's
            # source row index when present, otherwise the post-reorder batch position.
            logged_index = index if batch_position_indices else row_index_by_key[(pair_id, permutation)]
            lines.append(
                _log_line(
                    f"BASE | i={index} | pair_id={pair_id} | perm={permutation} | ans=A | gt=A | "
                    f"confidence=1.00 | method=boxed | base_score=0.500 | corr=0.500 | fmt=0.000 | len=0.000",
                    index=logged_index,
                )
            )
        checks = []
        for pair in manifest["pairs"]:
            pair_id = pair["pair_id"]
            rows_by_permutation = {int(row["permutation"]): int(row["row_index"]) for row in pair["rows"]}
            join0, join1 = rows_by_permutation[0], rows_by_permutation[1]
            if bad_join and pair_id == manifest["pairs"][0]["pair_id"]:
                join1 = join0  # joins permutation 0 with itself
            for slot in (0, 1):
                checks.append(
                    _log_line(
                        f"PAIR_CHECK | pair_id={pair_id} | rollout_slot={slot} | "
                        f"i0={join0},idx0={join0},ans0=A,conf0=1.00 | "
                        f"i1={join1},idx1={join1},ans1=A,conf1=1.00 | "
                        f"mapped(ans0)=A | is_consistent=True | pair_bonus=1.000 | "
                        f"final_score0=1.000 | final_score1=1.000"
                    )
                )
        if duplicate:
            checks.append(checks[0])
        lines.extend(checks)
        if unpaired and block_index == 0:
            lines.append(_log_line(f"PAIR_UNPAIRED | pair_id={pair_ids[0]} | rollout_slot=1 | i=9"))

    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    end = path.stat().st_size
    (run_dir / "reward_log_window.json").write_text(
        json.dumps({"reward_log_path": str(path), "start_offset_bytes": start, "_end": end}),
        encoding="utf-8",
    )


def write_rollouts(run_dir: Path, *, steps=(1, 2), rows: int = ROWS_PER_STEP) -> None:
    directory = run_dir / "rollouts"
    directory.mkdir(parents=True, exist_ok=True)
    for step in steps:
        with (directory / f"{step}.jsonl").open("w", encoding="utf-8") as handle:
            for _ in range(rows):
                handle.write(json.dumps({"input": "p", "output": "o", "gts": "A", "score": 1.0, "step": step}) + "\n")


def write_checkpoints(
    run_dir: Path,
    *,
    steps=(1, 2),
    latest: int = 2,
    adapter: bool = True,
    lora_b_zero: bool = False,
    non_finite_tensor: bool = False,
    include_lora_b: bool = True,
    bfloat16: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    adapter_config: bool = True,
    normalization: bool = True,
) -> None:
    checkpoints = run_dir / "checkpoints"
    for step in steps:
        (checkpoints / f"global_step_{step}").mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    (checkpoints / "latest_checkpointed_iteration.txt").write_text(f"{latest}\n", encoding="utf-8")
    # `verl/model_merger merge --target_dir …/actor/hf` nests the adapter under lora_adapter/
    target = checkpoints / "global_step_2" / "actor" / "hf" / "lora_adapter"
    target.mkdir(parents=True, exist_ok=True)
    if adapter_config:
        config_path = target / "adapter_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "r": lora_rank,
                    "lora_alpha": lora_alpha,
                    "target_modules": ["q_proj"],
                    "task_type": "CAUSAL_LM",
                    "peft_type": "LORA",
                }
            ),
            encoding="utf-8",
        )
        if normalization:
            import hashlib

            (target / "adapter_normalization.json").write_text(
                json.dumps(
                    {
                        "schema_version": "task4_adapter_normalization_v1",
                        "observed_rank": lora_rank,
                        "observed_alpha_before": 0,
                        "observed_alpha_after": lora_alpha,
                        "corrected": lora_alpha != 0,
                        "after_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
    if not adapter:
        return
    if bfloat16:
        import torch
        from safetensors.torch import save_file as save_torch_file

        save_torch_file(
            {
                "base_model.model.q_proj.lora_A.weight": torch.zeros((2, 2), dtype=torch.bfloat16),
                "base_model.model.q_proj.lora_B.weight": torch.full((2, 2), 0.25, dtype=torch.bfloat16),
            },
            str(target / "adapter_model.safetensors"),
        )
        return
    if non_finite_tensor:
        b_value = np.full((2, 2), np.inf, dtype=np.float32)
    elif lora_b_zero:
        b_value = np.zeros((2, 2), dtype=np.float32)
    else:
        b_value = np.full((2, 2), 0.25, dtype=np.float32)
    tensors = {"base_model.model.q_proj.lora_A.weight": np.zeros((2, 2), dtype=np.float32)}
    if include_lora_b:
        tensors["base_model.model.q_proj.lora_B.weight"] = b_value
    save_file(tensors, str(target / "adapter_model.safetensors"))


def write_controlled_eval(
    run_dir: Path,
    *,
    contract: str = "controlled",
    num_options: int = 2,
    evaluated_samples: int = 4,
) -> None:
    """Reproduce the real evaluator layout: `<output_dir>/<subdir>/summary.json`.

    `evaluation/evaluate_models.py:1136-1141` writes one summary keyed by dataset name,
    each entry carrying `evaluated_samples`, `evaluation_contract` and `num_options`
    (`:311-318`). There is no `completed` key.
    """
    subdir = run_dir / "controlled_eval" / "task4_tiny_8pairs_qwen3-8b"
    subdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task4_tiny_8pairs": {
            "evaluated_samples": evaluated_samples,
            "correct": 1,
            "total_with_answer": evaluated_samples,
            "accuracy": 0.25,
            "evaluation_contract": contract,
            "num_options": num_options,
        }
    }
    (subdir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def build_run(tmp_path: Path, **overrides):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_dataset_manifest(run_dir, reverse=overrides.get("manifest_reverse", False))
    write_train_log(
        run_dir,
        steps=overrides.get("train_steps", overrides.get("steps", (1, 2))),
        fallback=overrides.get("fallback", False),
        grad_norm=overrides.get("grad_norm", 0.75),
        pair_baseline=overrides.get("pair_baseline", True),
        nan_metric=overrides.get("nan_metric", False),
        vllm=overrides.get("vllm", True),
        identity_error=overrides.get("identity_error", False),
    )
    if overrides.get("rollouts_present", True):
        write_rollouts(run_dir, steps=overrides.get("steps", (1, 2)), rows=overrides.get("rows", ROWS_PER_STEP))
    if overrides.get("reward_present", True):
        write_reward_log(
            run_dir,
            manifest,
            reorder=overrides.get("reorder", True),
            unpaired=overrides.get("unpaired", False),
            duplicate=overrides.get("duplicate", False),
            blocks=overrides.get("reward_blocks", 2),
            batch_position_indices=overrides.get("batch_position_indices", False),
            bad_join=overrides.get("bad_join", False),
        )
    else:
        (run_dir / "reward_log_window.json").write_text(
            json.dumps({"reward_log_path": str(run_dir / "missing.log"), "start_offset_bytes": 0}),
            encoding="utf-8",
        )
    # the launcher always records the reward log it windowed, and the verifier cross-checks it
    window = json.loads((run_dir / "reward_log_window.json").read_text(encoding="utf-8"))
    reward_log_recorded = overrides.get("environment_reward_log", window["reward_log_path"])
    offset_recorded = overrides.get("environment_start_offset", window["start_offset_bytes"])
    (run_dir / "environment.txt").write_text(
        f"task4_run_id=probe\nreward_log={reward_log_recorded}\n"
        f"reward_log_start_offset_bytes={offset_recorded}\n",
        encoding="utf-8",
    )
    write_checkpoints(
        run_dir,
        steps=overrides.get("steps", (1, 2)),
        latest=overrides.get("latest", 2),
        adapter=overrides.get("adapter", True),
        lora_b_zero=overrides.get("lora_b_zero", False),
        non_finite_tensor=overrides.get("non_finite_tensor", False),
        include_lora_b=overrides.get("include_lora_b", True),
        bfloat16=overrides.get("bfloat16", False),
        lora_rank=overrides.get("lora_rank", 32),
        lora_alpha=overrides.get("lora_alpha", 64),
        adapter_config=overrides.get("adapter_config", True),
        normalization=overrides.get("normalization", True),
    )
    write_controlled_eval(
        run_dir,
        contract=overrides.get("contract", "controlled"),
        num_options=overrides.get("num_options", 2),
        evaluated_samples=overrides.get("evaluated_samples", 4),
    )
    return run_dir


def paths_for(verifier, run_dir: Path):
    return verifier.VerifyPaths(
        run_dir=run_dir,
        dataset_manifest=run_dir / "dataset_manifest.json",
        reward_log_window=run_dir / "reward_log_window.json",
        train_log=run_dir / "train.log",
        rollout_dir=run_dir / "rollouts",
        checkpoint_dir=run_dir / "checkpoints",
        merged_adapter_candidates=[
            run_dir / "checkpoints" / "global_step_2" / "actor" / "hf" / "lora_adapter" / "adapter_model.safetensors",
        ],
        controlled_eval_dir=run_dir / "controlled_eval",
    )


def run_verifier(tmp_path: Path, **overrides):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, **overrides)
    summary = verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))
    return verifier, run_dir, summary


# ---------------------------------------------------------------------------
# success path
# ---------------------------------------------------------------------------


def test_valid_evidence_passes(tmp_path):
    verifier, run_dir, summary = run_verifier(tmp_path)

    assert summary["optimization_steps"] == [1, 2]
    assert summary["fallback_detected"] is False
    assert summary["identity"]["unpaired_count"] == 0
    assert summary["identity"]["reorder_observed"] is True
    assert summary["lora"]["nonzero_lora_b"] is True
    assert summary["passed"] is True
    assert summary["failures"] == []

    for name in ("metrics_summary.json", "identity_pairing_summary.json", "checkpoint_manifest.json", "lora_update.json"):
        assert (run_dir / name).is_file(), f"verifier must write {name}"
    written = json.loads((run_dir / "identity_pairing_summary.json").read_text(encoding="utf-8"))
    assert written["pair_check_count"] == PAIR_COUNT * 2 * 2


def test_historical_reward_lines_outside_the_window_are_ignored(tmp_path):
    _verifier, run_dir, summary = run_verifier(tmp_path)
    # the seeded historical PAIR_CHECK for stale_pair must not count
    assert summary["identity"]["pair_check_count"] == PAIR_COUNT * 2 * 2
    assert "stale_pair" not in json.dumps(summary["identity"])


def test_cli_returns_zero_on_success(tmp_path, capsys):
    verifier, run_dir, _summary = run_verifier(tmp_path)
    code = verifier.main(["--run-dir", str(run_dir), "--dataset-manifest", str(run_dir / "dataset_manifest.json")])
    assert code == 0
    assert "PASS" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# hard failures
# ---------------------------------------------------------------------------


def test_fallback_warning_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, fallback=True)
    with pytest.raises(verifier.VerificationFailed, match="fallback to original GRPO"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_missing_pair_baseline_metric_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, pair_baseline=False)
    with pytest.raises(verifier.VerificationFailed, match="pair-baseline metric"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_unpaired_rollout_key_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, unpaired=True)
    with pytest.raises(verifier.VerificationFailed, match="PAIR_UNPAIRED"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_duplicate_rollout_key_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, duplicate=True)
    with pytest.raises(verifier.VerificationFailed, match="duplicate PAIR_CHECK"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_pair_check_that_does_not_join_the_two_permutations_fails(tmp_path):
    """Plan §3.2 wants a PAIR_CHECK that actually joins permutation 0 with 1."""
    verifier = load_verifier()
    run_dir = build_run(tmp_path, bad_join=True)
    with pytest.raises(verifier.VerificationFailed, match="do not join permutations 0 and 1"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_non_finite_in_any_namespaced_metric_fails(tmp_path):
    """A NaN in any `group/name` metric must fail, not only the named ones."""
    verifier = load_verifier()
    run_dir = build_run(tmp_path)
    with (run_dir / "train.log").open("a", encoding="utf-8") as handle:
        handle.write("step:2 - critic/advantages:nan - reward/total:inf\n")
    with pytest.raises(verifier.VerificationFailed, match="non-finite metric"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_rotated_reward_log_offset_mismatch_is_rejected(tmp_path):
    """A replacement log with the same path but a different window is caught."""
    verifier = load_verifier()
    run_dir = build_run(tmp_path, environment_start_offset=0)
    with pytest.raises(verifier.EvidenceError, match="rotated or replaced"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_malformed_manifest_pair_entry_is_a_contract_error(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path)
    (run_dir / "dataset_manifest.json").write_text(json.dumps({"pairs": [1, 2]}), encoding="utf-8")
    with pytest.raises(verifier.EvidenceError, match="pair entries must be objects"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_batch_position_indices_are_rejected_as_vacuous_reorder_evidence(tmp_path):
    """A fixture without `extra_info.index` logs batch positions, so the gate must fail closed."""
    verifier = load_verifier()
    run_dir = build_run(tmp_path, batch_position_indices=True)
    with pytest.raises(verifier.VerificationFailed, match="vacuous"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_missing_optimization_step_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, train_steps=(1,))
    with pytest.raises(verifier.VerificationFailed, match="optimization steps observed"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_zero_grad_norm_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, grad_norm=0.0)
    with pytest.raises(verifier.VerificationFailed, match="grad_norm is not positive"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_non_finite_metric_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, nan_metric=True)
    with pytest.raises(verifier.VerificationFailed, match="non-finite metric"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_absent_vllm_initialization_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, vllm=False)
    with pytest.raises(verifier.VerificationFailed, match="vLLM engine initialization"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_identity_error_in_train_log_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, identity_error=True)
    with pytest.raises(verifier.VerificationFailed, match="rollout identity error"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_no_observed_reorder_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, reorder=False)
    with pytest.raises(verifier.VerificationFailed, match="no balance_batch reorder"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_absent_checkpoint_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, steps=(1, 2), latest=1)
    with pytest.raises(verifier.VerificationFailed, match="latest_checkpointed_iteration"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_zero_lora_b_tensor_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, lora_b_zero=True)
    with pytest.raises(verifier.VerificationFailed, match="lora_B tensor is zero"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_missing_lora_b_tensor_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, include_lora_b=False)
    with pytest.raises(verifier.VerificationFailed, match="no lora_B tensor"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_non_finite_adapter_tensor_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, non_finite_tensor=True)
    with pytest.raises(verifier.VerificationFailed, match="non-finite tensors"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_wrong_evaluation_contract_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, contract="official")
    with pytest.raises(verifier.VerificationFailed, match="evaluation contract"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_wrong_num_options_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, num_options=4)
    with pytest.raises(verifier.VerificationFailed, match="num_options"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_evaluated_sample_count_is_enforced_from_the_real_summary(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, evaluated_samples=2)
    with pytest.raises(verifier.VerificationFailed, match="evaluated 2 sample"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_missing_eval_key_is_a_contract_error_not_a_default(tmp_path):
    """A summary lacking a required key must fail as evidence, never be defaulted."""
    verifier = load_verifier()
    run_dir = build_run(tmp_path)
    summary_path = next((run_dir / "controlled_eval").rglob("summary.json"))
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    for entry in payload.values():
        entry.pop("evaluation_contract")
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(verifier.EvidenceError, match="evaluation_contract"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_malformed_evidence_is_a_contract_error_not_a_crash(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path)
    summary_path = next((run_dir / "controlled_eval").rglob("summary.json"))
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    for entry in payload.values():
        entry["num_options"] = None
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(verifier.EvidenceError, match="num_options"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_bfloat16_adapter_is_readable(tmp_path):
    """`safe_open(framework="numpy")` cannot see bf16, and the merger writes bf16."""
    verifier = load_verifier()
    run_dir = build_run(tmp_path, bfloat16=True)
    summary = verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))
    assert summary["lora"]["adapter_source"] == "merger_output"
    assert summary["lora"]["nonzero_lora_b"] is True


def test_trainer_saved_adapter_cannot_satisfy_the_merge_contract(tmp_path):
    """Task 4 requires the step-2 checkpoint to MERGE; a trainer-saved adapter is not that."""
    verifier = load_verifier()
    run_dir = build_run(tmp_path)
    merged = run_dir / "checkpoints" / "global_step_2" / "actor" / "hf" / "lora_adapter"
    trainer_saved = run_dir / "checkpoints" / "global_step_2" / "actor" / "lora_adapter"
    trainer_saved.mkdir(parents=True, exist_ok=True)
    for item in merged.iterdir():
        item.rename(trainer_saved / item.name)
    merged.rmdir()
    # the default candidate no longer considers the trainer-saved location at all
    with pytest.raises(verifier.EvidenceError, match="missing merged LoRA adapter"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_an_explicitly_supplied_adapter_cannot_satisfy_the_merge_contract(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path)
    elsewhere = run_dir / "somewhere_else"
    elsewhere.mkdir()
    source = run_dir / "checkpoints" / "global_step_2" / "actor" / "hf" / "lora_adapter"
    for item in source.iterdir():
        item.rename(elsewhere / item.name)
    source.rmdir()
    paths = paths_for(verifier, run_dir)
    paths.merged_adapter_candidates = [elsewhere / "adapter_model.safetensors"]
    with pytest.raises(verifier.VerificationFailed, match="not the merged step-2 adapter"):
        verifier.verify_real_gpu_smoke(paths)


def test_merger_zero_lora_alpha_fails(tmp_path):
    """`base_model_merger.py:266-269` writes lora_alpha 0, so alpha/r would be 0."""
    verifier = load_verifier()
    run_dir = build_run(tmp_path, lora_alpha=0)
    with pytest.raises(verifier.VerificationFailed, match="lora_alpha is 0"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_wrong_lora_rank_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, lora_rank=16)
    with pytest.raises(verifier.VerificationFailed, match="adapter rank r is 16"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_missing_adapter_config_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, adapter_config=False)
    with pytest.raises(verifier.EvidenceError, match="adapter_config.json"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_vllm_gate_is_not_satisfied_by_the_config_echo(tmp_path):
    """The resolved-config echo contains `hybrid_engine`/`free_cache_engine`/`engine_kwargs`."""
    verifier = load_verifier()
    run_dir = build_run(tmp_path, vllm=False)
    (run_dir / "train.log").write_text(
        "train.log\n"
        "actor_rollout_ref:\n"
        "  rollout:\n"
        "    name: 'hf'\n"
        "    free_cache_engine: true\n"
        "    engine_kwargs: {'vllm': {}}\n"
        "    hybrid_engine: true\n"
        "step:1 - actor/pg_loss:0.5 - actor/grad_norm:0.75 - actor/kl_loss:0.01\n"
        "step:2 - actor/pg_loss:0.5 - actor/grad_norm:0.75 - actor/kl_loss:0.01\n",
        encoding="utf-8",
    )
    with pytest.raises(verifier.VerificationFailed, match="vLLM engine initialization"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_echoed_non_finite_text_does_not_fail_the_run(tmp_path):
    """Only the required metric keys may fail the run through a non-finite value."""
    verifier = load_verifier()
    run_dir = build_run(tmp_path)
    with (run_dir / "train.log").open("a", encoding="utf-8") as handle:
        handle.write("reward_manager.py:126] response: 'the value is x: nan here'\n")
    summary = verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))
    assert summary["nan_or_inf_metrics"] == []


def test_reward_log_path_mismatch_is_rejected(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, environment_reward_log="/somewhere/else/reward.log")
    with pytest.raises(verifier.EvidenceError, match="does not match"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_missing_launcher_environment_record_is_rejected(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path)
    (run_dir / "environment.txt").unlink()
    with pytest.raises(verifier.EvidenceError, match="environment record"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


# ---------------------------------------------------------------------------
# missing evidence must never be reconstructed
# ---------------------------------------------------------------------------


def test_absent_rollouts_are_rejected_rather_than_assumed(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, rollouts_present=False)
    with pytest.raises(verifier.EvidenceError, match="rollout"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_short_rollout_file_is_observed_not_assumed(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, rows=31)
    with pytest.raises(verifier.VerificationFailed, match="observed 31 rollout rows"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_missing_reward_log_is_rejected(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, reward_present=False)
    with pytest.raises(verifier.EvidenceError, match="reward log"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_rotated_reward_log_window_is_rejected(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path)
    window_path = run_dir / "reward_log_window.json"
    window = json.loads(window_path.read_text(encoding="utf-8"))
    # simulate rotation/truncation: the recorded start is now past the file end
    window["start_offset_bytes"] = (run_dir / "reward.log").stat().st_size + 10
    window_path.write_text(json.dumps(window), encoding="utf-8")
    with pytest.raises(verifier.EvidenceError, match="rotated|truncated"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_missing_dataset_manifest_is_rejected(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path)
    (run_dir / "dataset_manifest.json").unlink()
    with pytest.raises(verifier.EvidenceError, match="dataset manifest"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_missing_merged_adapter_is_rejected(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, adapter=False)
    with pytest.raises(verifier.EvidenceError, match="LoRA adapter"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_extra_reward_block_fails(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, reward_blocks=3)
    with pytest.raises(verifier.VerificationFailed, match="invocation block"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_cli_exit_codes_distinguish_evidence_from_gate_failure(tmp_path, capsys):
    verifier = load_verifier()
    failing = build_run(tmp_path / "gate", fallback=True)
    assert verifier.main(["--run-dir", str(failing)]) == 3
    assert "verification failed" in capsys.readouterr().err

    broken = build_run(tmp_path / "evidence")
    (broken / "train.log").unlink()
    assert verifier.main(["--run-dir", str(broken)]) == 2
    assert "evidence error" in capsys.readouterr().err


def test_reordered_manifest_still_defines_the_canonical_order(tmp_path):
    verifier = load_verifier()
    run_dir = build_run(tmp_path, manifest_reverse=True)
    summary = verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))
    assert summary["identity"]["reorder_observed"] is True
    assert summary["passed"] is True


def test_missing_normalization_evidence_fails(tmp_path):
    """Content checks alone cannot prove a merge: the trainer's own adapter carries r/alpha too."""
    verifier = load_verifier()
    run_dir = build_run(tmp_path, normalization=False)
    with pytest.raises(verifier.EvidenceError, match="adapter_normalization.json"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))


def test_config_edited_after_normalization_fails(tmp_path):
    """The normalization digest binds PASS to the config bytes the correction step produced."""
    verifier = load_verifier()
    run_dir = build_run(tmp_path)
    config = (
        run_dir / "checkpoints" / "global_step_2" / "actor" / "hf" / "lora_adapter" / "adapter_config.json"
    )
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["lora_alpha"] = 32  # a plausible-looking hand edit after normalization
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(verifier.EvidenceError, match="modified after the correction step"):
        verifier.verify_real_gpu_smoke(paths_for(verifier, run_dir))