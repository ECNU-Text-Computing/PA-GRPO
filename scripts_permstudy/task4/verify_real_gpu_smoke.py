"""Verify the evidence produced by a Task 4 real vLLM / GPU / GRPO smoke run.

The verifier parses only artifacts that the run actually produced. It never
substitutes an expected value for missing evidence: absent or malformed inputs
raise ``EvidenceError``, and a present-but-failing gate raises
``VerificationFailed``. Both exit non-zero.

Evidence sources, all under ``--run-dir`` unless overridden:

``train.log``                trainer console output (metrics, vLLM init, warnings)
``reward_log_window.json``   {"reward_log_path", "start_offset_bytes"} written by the launcher
``dataset_manifest.json``    Task 1 fixture manifest (canonical pair/permutation order)
``rollouts/{step}.jsonl``    one file per optimization step, one JSON record per generated row
``checkpoints/``             ``global_step_N`` directories and ``latest_checkpointed_iteration.txt``
``checkpoints/global_step_2/actor/hf/lora_adapter/adapter_model.safetensors``
                             the merged LoRA adapter (``verl/model_merger/base_model_merger.py:276-280``
                             always nests it under ``lora_adapter/``). Only this path can satisfy the
                             Task 4 merge contract, and its ``adapter_config.json`` must be accompanied
                             by the ``adapter_normalization.json`` the Task 5 correction step writes.
``adapter_normalization.json``  beside the merged adapter; its ``after_sha256`` must match the
                             current ``adapter_config.json``, so hand-editing the config afterwards fails
``controlled_eval/``         the real evaluator output tree; every ``summary.json`` beneath it
                             (``evaluation/evaluate_models.py:1136-1141``) is aggregated
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

FALLBACK_MARKER = "fallback to original GRPO"

IDENTITY_ERROR_MARKERS = (
    "RolloutIdentityAlignmentError",
    "IdentityMetadataError",
    "generation output reordered rollout identity",
    "generation output changed rollout batch size",
    "generation output did not preserve",
    "does not match rollout identity order",
    "requires extra_info metadata",
)

PAIR_BASELINE_METRICS = (
    "pair_baseline/mean_of_pair_means",
    "pair_baseline/mean_of_pair_stds",
    "pair_baseline/std_of_pair_stds",
)

PER_STEP_METRICS = ("actor/pg_loss", "actor/grad_norm", "actor/kl_loss")

# Every namespaced (`group/name`) numeric value in the log is a trainer metric and must be
# finite. Restricting the check to a fixed list of keys let a NaN `critic/advantages` or
# `reward/total` through, while scanning for bare `ident:number` made echoed prose such as a
# response containing "x: nan" fail the run. The `/`-in-the-key rule separates the two.

# Real vLLM runtime markers. The resolved-config echo printed by
# `verl/trainer/main_ppo.py:253` contains `hybrid_engine`, `free_cache_engine` and
# `engine_kwargs: {'vllm': {}}`, so a bare "vllm"+"engine" substring test is satisfied by a
# run that never started vLLM. These markers do not occur in that echo.
VLLM_RUNTIME_MARKERS = (
    "Initializing an LLM engine",
    "EngineCore",
    "Loading model weights took",
    "vLLM API server",
    "Starting vLLM",
    # With VLLM_LOGGING_LEVEL=WARN the INFO-level startup markers above
    # are suppressed. This string is emitted only after a real vLLM
    # GPU worker object has been constructed, so it is valid runtime
    # evidence and cannot be satisfied by the resolved-config echo.
    "vllm.v1.worker.gpu_worker.Worker",
)

_METRIC_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9_./]*)\s*[:=]\s*(-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|nan|NaN|inf|-inf)\b"
)
_BASE_RE = re.compile(r"BASE \| i=(\d+) \| pair_id=(\S+?) \| perm=(\d+)")
_UNPAIRED_RE = re.compile(r"PAIR_UNPAIRED \| pair_id=(\S+?) \| rollout_slot=(\d+)")
_CHECK_RE = re.compile(
    r"PAIR_CHECK \| pair_id=(\S+?) \| rollout_slot=(\d+) \| "
    r"i0=(\d+),idx0=(\d+),[^|]*\| i1=(\d+),idx1=(\d+),"
)
# The log line prefix is "... | idx=<source index> | pid=... |"; note that PAIR_CHECK
# lines carry "idx0=" / "idx1=", which this pattern deliberately does not match.
_IDX_RE = re.compile(r"\bidx=(\d+)\b")


class EvidenceError(Exception):
    """Evidence is missing, unreadable, or internally inconsistent (contract error)."""


class VerificationFailed(Exception):
    """Evidence is readable but a hard PASS gate is not satisfied."""


# ---------------------------------------------------------------------------
# small readers
# ---------------------------------------------------------------------------


def _read_text(path: Path, label: str) -> str:
    if not path.is_file():
        raise EvidenceError(f"missing {label}: {path.name}")
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise EvidenceError(f"cannot read {label}: {path.name}") from exc


def _read_json(path: Path, label: str) -> Any:
    text = _read_text(path, label)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"{label} is not valid JSON: {path.name}") from exc


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _as_int(value: Any, *, what: str) -> int:
    """Convert an evidence value to int, turning malformed input into a contract error."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise EvidenceError(f"{what} is not integer-like: {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{what} is not integer-like: {value!r}") from exc


# ---------------------------------------------------------------------------
# train log
# ---------------------------------------------------------------------------


def parse_train_log(text: str) -> dict[str, Any]:
    """Extract per-step metrics, fallback warnings, and identity failures."""
    metrics_by_step: dict[int, dict[str, float]] = {}
    current_step: int | None = None
    nan_or_inf: list[str] = []
    identity_errors: list[str] = []

    for line in text.splitlines():
        for marker in IDENTITY_ERROR_MARKERS:
            if marker in line:
                identity_errors.append(line.strip()[:240])
                break

        pairs = _METRIC_RE.findall(line)
        if not pairs:
            continue
        values: dict[str, float] = {}
        for key, raw in pairs:
            lowered = raw.lower()
            try:
                number = float(raw)
            except ValueError:
                continue
            if "/" in key and (lowered in {"nan", "inf", "-inf"} or not math.isfinite(number)):
                nan_or_inf.append(f"{key}={raw}")
            values[key] = number

        if "training/global_step" in values:
            current_step = int(values["training/global_step"])
        elif "step" in values:
            current_step = int(values["step"])
        if current_step is None:
            continue
        metrics_by_step.setdefault(current_step, {}).update(values)

    optimization_steps = sorted(
        step for step, values in metrics_by_step.items() if any(key in values for key in PER_STEP_METRICS)
    )
    return {
        "metrics_by_step": {str(step): metrics_by_step[step] for step in sorted(metrics_by_step)},
        "optimization_steps": optimization_steps,
        "fallback_detected": FALLBACK_MARKER in text,
        "nan_or_inf_metrics": nan_or_inf,
        "identity_errors": identity_errors,
        "vllm_runtime_markers": [marker for marker in VLLM_RUNTIME_MARKERS if marker in text],
        "vllm_engine_initialized": any(marker in text for marker in VLLM_RUNTIME_MARKERS),
    }


# ---------------------------------------------------------------------------
# reward log window and identity
# ---------------------------------------------------------------------------


def read_reward_window(window_meta: Mapping[str, Any], run_dir: Path) -> tuple[str, int, int]:
    """Return (path, start, end) for the reward log, rejecting unusable windows."""
    path_value = window_meta.get("reward_log_path")
    start = window_meta.get("start_offset_bytes")
    if not isinstance(path_value, str) or not path_value:
        raise EvidenceError("reward_log_window.json has no reward_log_path")
    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        raise EvidenceError("reward_log_window.json has no valid start_offset_bytes")

    path = Path(path_value)
    if not path.is_absolute():
        path = run_dir / path
    if not path.is_file():
        raise EvidenceError(f"reward log recorded by the window does not exist: {path.name}")

    end = path.stat().st_size
    if end <= start:
        raise EvidenceError(
            f"reward log window is empty or was rotated/truncated: start={start} end={end}"
        )
    return str(path), start, end


def _reward_blocks(lines: Sequence[str]) -> list[dict[str, list[Any]]]:
    """Split the windowed reward log into one block per reward-function invocation."""
    blocks: list[dict[str, list[Any]]] = []
    current: dict[str, list[Any]] | None = None

    def ensure() -> dict[str, list[Any]]:
        nonlocal current
        if current is None:
            current = {"base": [], "checks": [], "unpaired": []}
        return current

    for line in lines:
        base = _BASE_RE.search(line)
        if base:
            index = int(base.group(1))
            if index == 0 and current is not None and current["base"]:
                blocks.append(current)
                current = None
            source_match = _IDX_RE.search(line)
            source_index = int(source_match.group(1)) if source_match else None
            ensure()["base"].append((index, base.group(2), int(base.group(3)), source_index))
            continue
        unpaired = _UNPAIRED_RE.search(line)
        if unpaired:
            ensure()["unpaired"].append((unpaired.group(1), int(unpaired.group(2))))
            continue
        check = _CHECK_RE.search(line)
        if check:
            ensure()["checks"].append(
                (check.group(1), int(check.group(2)), int(check.group(4)), int(check.group(6)))
            )

    if current is not None and any(current.values()):
        blocks.append(current)
    return blocks


def canonical_placement_sequence(dataset_manifest: Mapping[str, Any]) -> list[tuple[str, int]]:
    """The pre-balance batch order: each pair's rows, each repeated for ``rollout.n=2``."""
    pairs = dataset_manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise EvidenceError("dataset manifest has no pairs")
    sequence: list[tuple[str, int]] = []
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise EvidenceError("dataset manifest pair entries must be objects")
        pair_id = pair.get("pair_id")
        rows = pair.get("rows")
        if not isinstance(pair_id, str) or not isinstance(rows, list) or len(rows) != 2:
            raise EvidenceError("dataset manifest pair entries must carry pair_id and two rows")
        permutations = [row.get("permutation") for row in rows]
        if sorted(permutations) != [0, 1]:
            raise EvidenceError(f"dataset manifest pair {pair_id} does not carry permutations 0 and 1")
        for permutation in (0, 1):
            sequence.extend([(pair_id, permutation)] * 2)
    return sequence


def canonical_source_index_sequence(dataset_manifest: Mapping[str, Any]) -> list[int]:
    """The source row indices in pre-balance batch order, each repeated for ``rollout.n=2``."""
    pairs = dataset_manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise EvidenceError("dataset manifest has no pairs")
    sequence: list[int] = []
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise EvidenceError("dataset manifest pair entries must be objects")
        rows = pair.get("rows")
        if not isinstance(rows, list) or len(rows) != 2:
            raise EvidenceError("dataset manifest pair entries must carry pair_id and two rows")
        for row in sorted(rows, key=lambda entry: int(entry["permutation"])):
            row_index = row.get("row_index")
            if not isinstance(row_index, int) or isinstance(row_index, bool):
                raise EvidenceError(
                    f"dataset manifest row for pair {pair.get('pair_id')} has no integer row_index"
                )
            sequence.extend([row_index] * 2)
    return sequence


def analyse_identity(
    window_lines: Sequence[str],
    dataset_manifest: Mapping[str, Any],
    expected_blocks: int,
) -> dict[str, Any]:
    canonical = canonical_placement_sequence(dataset_manifest)
    canonical_sources = canonical_source_index_sequence(dataset_manifest)
    expected_keys = set(canonical)
    blocks = _reward_blocks(window_lines)

    # source row index -> (pair_id, permutation), so a PAIR_CHECK's idx0/idx1 can be proved to
    # actually join the two permutations of the pair it names.
    row_lookup: dict[int, tuple[str, int]] = {}
    for pair in dataset_manifest.get("pairs") or []:
        if not isinstance(pair, Mapping):
            raise EvidenceError("dataset manifest pair entries must be objects")
        pair_id = pair.get("pair_id")
        rows = pair.get("rows")
        if not isinstance(pair_id, str) or not isinstance(rows, list) or len(rows) != 2:
            raise EvidenceError("dataset manifest pair entries must carry pair_id and two rows")
        for row in rows:
            row_index = row.get("row_index")
            if not isinstance(row_index, int) or isinstance(row_index, bool):
                raise EvidenceError(f"dataset manifest row for pair {pair_id} has no integer row_index")
            row_lookup[row_index] = (pair_id, int(row["permutation"]))

    unpaired_count = 0
    duplicate_keys = 0
    bad_joins: list[str] = []
    placement_reordered = False
    source_index_reordered = False
    source_indices_usable = bool(blocks)
    per_block: list[dict[str, Any]] = []

    for block in blocks:
        ordered = sorted(block["base"], key=lambda row: row[0])
        observed = [(pair_id, permutation) for _, pair_id, permutation, _ in ordered]
        observed_sources = [source for *_, source in ordered]
        checks = block["checks"]
        check_keys = [(pair_id, slot) for pair_id, slot, _, _ in checks]
        duplicates = len(check_keys) - len(set(check_keys))
        duplicate_keys += duplicates
        unpaired_count += len(block["unpaired"])

        block_bad_joins: list[str] = []
        for pair_id, slot, idx0, idx1 in checks:
            first = row_lookup.get(idx0)
            second = row_lookup.get(idx1)
            joined = (
                first is not None
                and second is not None
                and first[0] == pair_id
                and second[0] == pair_id
                and {first[1], second[1]} == {0, 1}
            )
            if not joined:
                block_bad_joins.append(f"{pair_id}#{slot}")
        bad_joins.extend(block_bad_joins)

        # The source-index channel is only admissible when the logged indices are
        # exactly the fixture's own row indices. If the fixture dropped
        # ``extra_info.index``, ``my_reward/judge_qwen.py`` logs the post-reorder
        # batch position instead and this multiset cannot match.
        block_usable = (
            bool(observed_sources)
            and all(source is not None for source in observed_sources)
            and sorted(observed_sources) == sorted(canonical_sources)
        )
        source_indices_usable = source_indices_usable and block_usable
        block_source_reordered = block_usable and observed_sources != canonical_sources

        block_placement_reordered = bool(observed) and observed != canonical
        placement_reordered = placement_reordered or block_placement_reordered
        source_index_reordered = source_index_reordered or block_source_reordered

        per_block.append(
            {
                "base_rows": len(observed),
                "pair_checks": len(checks),
                "unpaired": len(block["unpaired"]),
                "duplicate_check_keys": duplicates,
                "missing_check_keys": sorted(str(key) for key in expected_keys - set(check_keys)),
                "unexpected_check_keys": sorted(str(key) for key in set(check_keys) - expected_keys),
                "placement_reordered": block_placement_reordered,
                "source_indices_usable": block_usable,
                "source_index_reordered": block_source_reordered,
                "bad_pair_joins": block_bad_joins,
            }
        )

    return {
        "reward_log_blocks": len(blocks),
        "expected_blocks": expected_blocks,
        "unpaired_count": unpaired_count,
        "duplicate_check_keys": duplicate_keys,
        "pair_check_count": sum(entry["pair_checks"] for entry in per_block),
        "expected_pair_check_count_per_block": len(expected_keys),
        # The plan's §3.2 hook: reward log source-index order must differ from the
        # canonical interleaved input order for at least one step.
        "reorder_observed": source_index_reordered,
        "placement_reordered": placement_reordered,
        "source_index_reordered": source_index_reordered,
        "source_indices_usable": source_indices_usable,
        "canonical_placement_count": len(canonical),
        "bad_pair_joins": bad_joins,
        "blocks": per_block,
    }


# ---------------------------------------------------------------------------
# rollouts, checkpoints, LoRA, controlled eval
# ---------------------------------------------------------------------------


def inspect_rollouts(rollout_dir: Path, expected_steps: Sequence[int]) -> dict[str, Any]:
    if not rollout_dir.is_dir():
        raise EvidenceError(f"missing rollout directory: {rollout_dir.name}")
    rows_per_step: dict[str, int] = {}
    records_seen = 0
    for step in expected_steps:
        path = rollout_dir / f"{step}.jsonl"
        if not path.is_file():
            raise EvidenceError(f"missing rollout file for step {step}: {path.name}")
        lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        if not lines:
            raise EvidenceError(f"rollout file for step {step} is empty")
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceError(f"rollout file {path.name} contains a malformed record") from exc
            if _as_int(record.get("step", -1), what=f"{path.name} record step") != int(step):
                raise EvidenceError(f"rollout file {path.name} contains a record from another step")
            records_seen += 1
        rows_per_step[str(step)] = len(lines)
    return {"files": {str(step): f"{step}.jsonl" for step in expected_steps}, "rows_per_step": rows_per_step,
            "total_records": records_seen}


def inspect_checkpoints(checkpoint_dir: Path, expected_steps: Sequence[int]) -> dict[str, Any]:
    if not checkpoint_dir.is_dir():
        raise EvidenceError(f"missing checkpoint directory: {checkpoint_dir.name}")
    present = [step for step in expected_steps if (checkpoint_dir / f"global_step_{step}").is_dir()]
    latest_path = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not latest_path.is_file():
        raise EvidenceError("missing latest_checkpointed_iteration.txt")
    latest_raw = latest_path.read_text(encoding="utf-8", errors="replace").strip()
    try:
        latest = int(latest_raw)
    except ValueError as exc:
        raise EvidenceError(f"latest_checkpointed_iteration.txt is not an integer: {latest_raw!r}") from exc
    return {"expected_steps": list(expected_steps), "present_steps": present, "latest_iteration": latest}


def _read_adapter_arrays(adapter_path: Path) -> dict[str, np.ndarray]:
    """Load every adapter tensor as float64 numpy, tolerating bf16/fp16 safetensors.

    ``safetensors.safe_open(framework="numpy")`` raises ``TypeError: data type 'bfloat16'
    not understood`` for a bf16 adapter, and ``verl/model_merger/base_model_merger.py:296``
    builds the merged model in bfloat16. The torch framework is tried first and the numpy
    framework kept as a fallback.
    """
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise EvidenceError("safetensors is required to inspect the merged adapter") from exc

    last_error: Exception | None = None
    for framework in ("pt", "numpy"):
        arrays: dict[str, np.ndarray] = {}
        try:
            if framework == "pt":
                import torch  # local import: the numpy framework is the fallback
            with safe_open(str(adapter_path), framework=framework) as handle:
                for name in handle.keys():
                    tensor = handle.get_tensor(name)
                    if framework == "pt":
                        tensor = tensor.detach().to("cpu", dtype=torch.float32).numpy()
                    arrays[name] = np.asarray(tensor, dtype=np.float64)
            return arrays
        except Exception as exc:  # noqa: BLE001 - try the next framework, then report
            last_error = exc
    raise EvidenceError(
        f"cannot read LoRA adapter {adapter_path.name}: "
        f"{type(last_error).__name__}: {str(last_error)[:120]}"
    )


def merged_adapter_path(run_dir: Path) -> Path:
    """Where `verl.model_merger merge --target_dir …/actor/hf` actually writes the adapter."""
    return (
        run_dir / "checkpoints" / "global_step_2" / "actor" / "hf" / "lora_adapter" / "adapter_model.safetensors"
    )


def inspect_lora(adapter_candidates: Sequence[Path], run_dir: Path) -> dict[str, Any]:
    existing = [path for path in adapter_candidates if path.is_file()]
    if not existing:
        raise EvidenceError(
            "missing merged LoRA adapter; looked for "
            + ", ".join(str(path) for path in adapter_candidates)
        )
    adapter_path = existing[0]
    arrays = _read_adapter_arrays(adapter_path)

    lora_b_names = sorted(name for name in arrays if "lora_B" in name)
    all_finite = all(bool(np.all(np.isfinite(array))) for array in arrays.values())
    nonzero_lora_b = any(
        array.size and float(np.abs(array).max()) > 0.0 for name, array in arrays.items() if "lora_B" in name
    )
    return {
        "adapter": adapter_path.name,
        "adapter_path": str(adapter_path),
        # Task 4 requires the step-2 checkpoint to merge successfully and the MERGED adapter to be
        # loadable, so only the merger's own output can satisfy the contract. A trainer-saved
        # adapter, or one supplied from elsewhere, is accepted as evidence but not as a PASS.
        "adapter_source": "merger_output" if adapter_path == merged_adapter_path(run_dir) else "explicit",
        "expected_merged_path": str(merged_adapter_path(run_dir)),
        "tensor_count": len(arrays),
        "lora_b_tensor_count": len(lora_b_names),
        "lora_b_tensors": lora_b_names,
        "all_finite": all_finite,
        "nonzero_lora_b": nonzero_lora_b,
    }


def inspect_adapter_config(adapter_path: Path, expected_rank: int, expected_alpha: int) -> dict[str, Any]:
    """Read `adapter_config.json` and report the LoRA rank and scaling.

    `verl/model_merger/base_model_merger.py:266-269` hardcodes ``"lora_alpha": 0`` (with a
    comment saying an error should be raised), while training uses ``lora_rank=32`` and
    ``lora_alpha=64``. PEFT scales a LoRA update by ``alpha / r``, so a merged adapter left at
    ``0/32 = 0`` would zero the trained delta during evaluation: a non-zero ``lora_B`` alone
    does not prove the controlled evaluation saw the trained adapter.
    """
    config_path = adapter_path.parent / "adapter_config.json"
    if not config_path.is_file():
        raise EvidenceError(f"missing adapter_config.json next to {adapter_path.name}")
    payload = _read_json(config_path, "adapter_config.json")
    if not isinstance(payload, Mapping):
        raise EvidenceError("adapter_config.json must be a JSON object")
    for key in ("r", "lora_alpha"):
        if key not in payload:
            raise EvidenceError(f"adapter_config.json is missing {key}")
    rank = _as_int(payload["r"], what="adapter_config.r")
    alpha = _as_int(payload["lora_alpha"], what="adapter_config.lora_alpha")

    # `--merged-adapter` can point anywhere, and the trainer's own saved adapter
    # (`verl/workers/fsdp_workers.py:1049-1065`) carries the real peft values too, so content
    # checks alone cannot prove a merge happened. `adapter_normalization.json` is produced only
    # by the Task 5 correction step, and its after_sha256 binds it to the config's current bytes.
    normalization_path = adapter_path.parent / "adapter_normalization.json"
    if not normalization_path.is_file():
        raise EvidenceError(
            "missing adapter_normalization.json beside the merged adapter; the Task 5 correction step "
            "must run between model_merger merge and the controlled evaluation"
        )
    normalization = _read_json(normalization_path, "adapter_normalization.json")
    if not isinstance(normalization, Mapping):
        raise EvidenceError("adapter_normalization.json must be a JSON object")
    for key in ("observed_rank", "observed_alpha_after", "after_sha256"):
        if key not in normalization:
            raise EvidenceError(f"adapter_normalization.json is missing {key}")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if normalization["after_sha256"] != config_sha:
        raise EvidenceError(
            "adapter_config.json does not match the adapter_normalization.json digest, so it was "
            "modified after the correction step"
        )
    return {
        "path": config_path.name,
        "r": rank,
        "lora_alpha": alpha,
        "expected_rank": expected_rank,
        "expected_alpha": expected_alpha,
        "scaling": (alpha / rank) if rank else None,
        "expected_scaling": (expected_alpha / expected_rank) if expected_rank else None,
        "normalization": {
            "path": normalization_path.name,
            "observed_rank": _as_int(normalization["observed_rank"], what="normalization.observed_rank"),
            "observed_alpha_after": _as_int(
                normalization["observed_alpha_after"], what="normalization.observed_alpha_after"
            ),
            "corrected": bool(normalization.get("corrected", False)),
            "after_sha256_matches": True,
        },
    }


def inspect_controlled_eval(
    eval_dir: Path, expected_contract: str, expected_num_options: int, expected_num_samples: int
) -> dict[str, Any]:
    """Aggregate the real evaluator output tree (``<output_dir>/**/summary.json``).

    ``evaluation/evaluate_models.py:1136-1141`` writes one ``summary.json`` per run whose
    keys are dataset names, each carrying ``evaluated_samples``, ``evaluation_contract`` and
    ``num_options`` (``:311-318``). Nothing here is defaulted: a missing key is a contract
    error, so the gate is bound to captured output rather than to a CLI flag.
    """
    if not eval_dir.is_dir():
        raise EvidenceError(f"missing controlled-eval output directory: {eval_dir}")
    summary_files = sorted(eval_dir.rglob("summary.json"))
    if not summary_files:
        raise EvidenceError(f"controlled-eval output contains no summary.json under {eval_dir.name}")

    datasets: dict[str, Any] = {}
    contracts: set[str] = set()
    option_counts: set[int] = set()
    total_samples = 0
    for path in summary_files:
        payload = _read_json(path, "controlled-eval summary")
        if not isinstance(payload, Mapping) or not payload:
            raise EvidenceError(f"{path.name} must be a non-empty JSON object")
        for name in sorted(payload):
            entry = payload[name]
            if not isinstance(entry, Mapping):
                raise EvidenceError(f"controlled-eval entry {name!r} is not an object")
            for key in ("evaluated_samples", "evaluation_contract", "num_options"):
                if key not in entry:
                    raise EvidenceError(f"controlled-eval entry {name!r} is missing {key}")
            samples = _as_int(entry["evaluated_samples"], what=f"{name}.evaluated_samples")
            options = _as_int(entry["num_options"], what=f"{name}.num_options")
            contract = str(entry["evaluation_contract"])
            datasets[name] = {
                "evaluated_samples": samples,
                "evaluation_contract": contract,
                "num_options": options,
            }
            contracts.add(contract)
            option_counts.add(options)
            total_samples += samples

    return {
        "summary_files": [str(path.relative_to(eval_dir)) for path in summary_files],
        "datasets": datasets,
        "evaluation_contracts": sorted(contracts),
        "num_options_values": sorted(option_counts),
        "total_evaluated_samples": total_samples,
        "expected_contract": expected_contract,
        "expected_num_options": expected_num_options,
        "expected_num_samples": expected_num_samples,
    }


# ---------------------------------------------------------------------------
# coordination
# ---------------------------------------------------------------------------


@dataclass
class VerifyPaths:
    run_dir: Path
    dataset_manifest: Path
    reward_log_window: Path
    train_log: Path
    rollout_dir: Path
    checkpoint_dir: Path
    merged_adapter_candidates: Sequence[Path]
    controlled_eval_dir: Path


def verify_real_gpu_smoke(
    paths: VerifyPaths,
    *,
    expected_steps: Sequence[int] = (1, 2),
    expected_rows_per_step: int = 32,
    expected_eval_contract: str = "controlled",
    expected_num_options: int = 2,
    expected_num_samples: int = 4,
    expected_lora_rank: int = 32,
    expected_lora_alpha: int = 64,
    write_summaries: bool = True,
) -> dict[str, Any]:
    run_dir = paths.run_dir
    if not run_dir.is_dir():
        raise EvidenceError(f"missing run directory: {run_dir.name}")

    dataset_manifest = _read_json(paths.dataset_manifest, "dataset manifest")
    if not isinstance(dataset_manifest, Mapping):
        raise EvidenceError("dataset manifest must be a JSON object")
    expected_pairs = len(dataset_manifest.get("pairs") or [])
    if expected_pairs <= 0:
        raise EvidenceError("dataset manifest contains no pairs")

    train_text = _read_text(paths.train_log, "train log")
    train = parse_train_log(train_text)

    window_meta = _read_json(paths.reward_log_window, "reward log window")
    if not isinstance(window_meta, Mapping):
        raise EvidenceError("reward_log_window.json must be a JSON object")
    reward_path, start, end = read_reward_window(window_meta, run_dir)
    with open(reward_path, "rb") as handle:
        handle.seek(start)
        window_bytes = handle.read(end - start)
    window_lines = window_bytes.decode("utf-8", errors="replace").splitlines()

    # Path mismatch detector: the launcher records the reward log it windowed in
    # environment.txt, so a window JSON pointing at another run's log is caught rather
    # than silently accepted.
    environment_text = _read_text(run_dir / "environment.txt", "launcher environment record")
    recorded_logs = [
        line.split("=", 1)[1]
        for line in environment_text.splitlines()
        if line.startswith("reward_log=")
    ]
    if not recorded_logs:
        raise EvidenceError("environment.txt does not record the reward log path")
    recorded_log = recorded_logs[0]
    if Path(recorded_log).resolve() != Path(reward_path).resolve():
        raise EvidenceError(
            "reward log window path does not match the path the launcher recorded in environment.txt"
        )
    # A rotated or replaced log keeps the same path but presents a different window, so the
    # offset the launcher recorded must still equal the offset the window claims.
    recorded_offsets = [
        line.split("=", 1)[1]
        for line in environment_text.splitlines()
        if line.startswith("reward_log_start_offset_bytes=")
    ]
    if not recorded_offsets:
        raise EvidenceError("environment.txt does not record the reward log start offset")
    if _as_int(recorded_offsets[0], what="recorded reward_log_start_offset_bytes") != _as_int(
        window_meta.get("start_offset_bytes"), what="window start_offset_bytes"
    ):
        raise EvidenceError(
            "reward log window start offset does not match the offset the launcher recorded, "
            "so the log was rotated or replaced"
        )

    rollouts = inspect_rollouts(paths.rollout_dir, expected_steps)
    checkpoints = inspect_checkpoints(paths.checkpoint_dir, expected_steps)
    lora = inspect_lora(paths.merged_adapter_candidates, run_dir)
    adapter_config = inspect_adapter_config(
        Path(lora["adapter_path"]), expected_lora_rank, expected_lora_alpha
    )
    controlled_eval = inspect_controlled_eval(
        paths.controlled_eval_dir, expected_eval_contract, expected_num_options, expected_num_samples
    )
    identity = analyse_identity(window_lines, dataset_manifest, expected_blocks=len(expected_steps))
    identity["window"] = {
        "start_offset_bytes": start,
        "end_offset_bytes": end,
        "window_bytes": end - start,
        "line_count": len(window_lines),
    }

    baseline = {
        name: next(
            (
                train["metrics_by_step"][step].get(name)
                for step in train["metrics_by_step"]
                if name in train["metrics_by_step"][step]
            ),
            None,
        )
        for name in PAIR_BASELINE_METRICS
    }

    failures: list[str] = []

    if train["optimization_steps"] != list(expected_steps):
        failures.append(
            f"optimization steps observed {train['optimization_steps']}, expected {list(expected_steps)}"
        )
    if train["fallback_detected"]:
        failures.append("trainer log reports fallback to original GRPO")
    if train["identity_errors"]:
        failures.append(f"{len(train['identity_errors'])} rollout identity error(s) in the trainer log")
    if train["nan_or_inf_metrics"]:
        failures.append(f"non-finite metric value(s): {sorted(set(train['nan_or_inf_metrics']))[:5]}")
    if not train["vllm_engine_initialized"]:
        failures.append("trainer log shows no vLLM engine initialization")
    for name, value in baseline.items():
        if value is None:
            failures.append(f"missing pair-baseline metric: {name}")

    for step in expected_steps:
        values = train["metrics_by_step"].get(str(step), {})
        for name in PER_STEP_METRICS:
            value = values.get(name)
            if value is None:
                failures.append(f"step {step}: missing metric {name}")
            elif not _finite(value):
                failures.append(f"step {step}: {name} is not finite ({value})")
        grad_norm = values.get("actor/grad_norm")
        if grad_norm is not None and _finite(grad_norm) and float(grad_norm) <= 0.0:
            failures.append(f"step {step}: actor/grad_norm is not positive ({grad_norm})")
        rows = rollouts["rows_per_step"].get(str(step))
        if rows != expected_rows_per_step:
            failures.append(f"step {step}: observed {rows} rollout rows, expected {expected_rows_per_step}")

    if identity["reward_log_blocks"] != identity["expected_blocks"]:
        failures.append(
            f"reward log window holds {identity['reward_log_blocks']} invocation block(s), "
            f"expected {identity['expected_blocks']}"
        )
    if identity["unpaired_count"]:
        failures.append(f"{identity['unpaired_count']} PAIR_UNPAIRED record(s) in the reward window")
    if identity["duplicate_check_keys"]:
        failures.append(f"{identity['duplicate_check_keys']} duplicate PAIR_CHECK key(s)")
    for index, block in enumerate(identity["blocks"]):
        if block["pair_checks"] != identity["expected_pair_check_count_per_block"]:
            failures.append(
                f"reward block {index}: {block['pair_checks']} PAIR_CHECK record(s), "
                f"expected {identity['expected_pair_check_count_per_block']}"
            )
        if block["missing_check_keys"]:
            failures.append(f"reward block {index}: unpaired rollout keys {block['missing_check_keys'][:4]}")
    if identity["bad_pair_joins"]:
        failures.append(
            f"{len(identity['bad_pair_joins'])} PAIR_CHECK record(s) do not join permutations 0 and 1 of "
            f"the pair they name: {identity['bad_pair_joins'][:4]}"
        )
    if not identity["source_indices_usable"]:
        failures.append(
            "reward log source indices do not match the dataset manifest row indices, so the reorder evidence "
            "is vacuous: the fixture must carry extra_info.index, otherwise the logged idx= is the "
            "post-reorder batch position"
        )
    elif not identity["source_index_reordered"]:
        failures.append("no balance_batch reorder was observed in the reward log window")

    consistency_rate = train["metrics_by_step"].get(str(expected_steps[-1]), {}).get(
        "reward/consistency_unpaired_rate"
    )
    if consistency_rate is not None and _finite(consistency_rate) and float(consistency_rate) != 0.0:
        failures.append(f"reward/consistency_unpaired_rate is {consistency_rate}, expected 0")

    if checkpoints["present_steps"] != list(expected_steps):
        failures.append(
            f"checkpoint directories present {checkpoints['present_steps']}, expected {list(expected_steps)}"
        )
    if checkpoints["latest_iteration"] != int(expected_steps[-1]):
        failures.append(
            f"latest_checkpointed_iteration is {checkpoints['latest_iteration']}, "
            f"expected {int(expected_steps[-1])}"
        )

    if lora["adapter_source"] != "merger_output":
        failures.append(
            "the verified LoRA adapter is not the merged step-2 adapter "
            f"({lora['expected_merged_path']}); the Task 4 contract requires the checkpoint to merge "
            "successfully, so a trainer-saved or substituted adapter cannot satisfy it"
        )
    if not lora["all_finite"]:
        failures.append("merged LoRA adapter contains non-finite tensors")
    if lora["lora_b_tensor_count"] == 0:
        failures.append("merged LoRA adapter contains no lora_B tensor")
    elif not lora["nonzero_lora_b"]:
        failures.append("every lora_B tensor is zero: no optimizer update was saved")
    if adapter_config["r"] != expected_lora_rank:
        failures.append(
            f"merged adapter rank r is {adapter_config['r']}, expected {expected_lora_rank}"
        )
    if adapter_config["lora_alpha"] != expected_lora_alpha:
        failures.append(
            f"merged adapter lora_alpha is {adapter_config['lora_alpha']}, expected "
            f"{expected_lora_alpha}: PEFT scales the LoRA delta by alpha/r, so the merged config "
            f"({adapter_config['scaling']}) would not match training "
            f"({adapter_config['expected_scaling']}) and the evaluation would not see the trained adapter"
        )

    if controlled_eval["evaluation_contracts"] != [expected_eval_contract]:
        failures.append(
            f"controlled evaluation contracts are {controlled_eval['evaluation_contracts']}, "
            f"expected [{expected_eval_contract!r}]"
        )
    if controlled_eval["num_options_values"] != [expected_num_options]:
        failures.append(
            f"controlled evaluation num_options values are {controlled_eval['num_options_values']}, "
            f"expected [{expected_num_options}]"
        )
    if controlled_eval["total_evaluated_samples"] != expected_num_samples:
        failures.append(
            f"controlled evaluation evaluated {controlled_eval['total_evaluated_samples']} sample(s), "
            f"expected {expected_num_samples}"
        )

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "dataset_manifest": {"path": paths.dataset_manifest.name, "pairs": expected_pairs},
        "optimization_steps": train["optimization_steps"],
        "expected_optimization_steps": list(expected_steps),
        "fallback_detected": train["fallback_detected"],
        "nan_or_inf_metrics": sorted(set(train["nan_or_inf_metrics"])),
        "identity_errors": train["identity_errors"],
        "vllm_engine_initialized": train["vllm_engine_initialized"],
        "pair_baseline_metrics": baseline,
        "metrics_by_step": train["metrics_by_step"],
        "rollouts": rollouts,
        "identity": identity,
        "checkpoints": checkpoints,
        "lora": lora,
        "adapter_config": adapter_config,
        "controlled_eval": controlled_eval,
        "failures": failures,
        "passed": not failures,
    }

    if write_summaries:
        _write_json(run_dir / "metrics_summary.json", summary)
        _write_json(run_dir / "identity_pairing_summary.json", identity)
        _write_json(run_dir / "checkpoint_manifest.json", checkpoints)
        _write_json(run_dir / "lora_update.json", lora)

    if failures:
        raise VerificationFailed("; ".join(failures))
    return summary


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Task 4 real-GPU smoke evidence.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-manifest", default=None)
    parser.add_argument("--reward-log-window", default=None)
    parser.add_argument("--train-log", default=None)
    parser.add_argument("--rollout-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--merged-adapter", default=None)
    parser.add_argument("--expected-lora-rank", type=int, default=32)
    parser.add_argument("--expected-lora-alpha", type=int, default=64)
    parser.add_argument("--controlled-eval-dir", default=None)
    parser.add_argument("--expected-eval-contract", default="controlled")
    parser.add_argument("--expected-num-options", type=int, default=2)
    parser.add_argument("--expected-num-samples", type=int, default=4)
    return parser


def resolve_paths(args: argparse.Namespace) -> VerifyPaths:
    run_dir = Path(args.run_dir)
    return VerifyPaths(
        run_dir=run_dir,
        dataset_manifest=Path(args.dataset_manifest) if args.dataset_manifest else run_dir / "dataset_manifest.json",
        reward_log_window=Path(args.reward_log_window)
        if args.reward_log_window
        else run_dir / "reward_log_window.json",
        train_log=Path(args.train_log) if args.train_log else run_dir / "train.log",
        rollout_dir=Path(args.rollout_dir) if args.rollout_dir else run_dir / "rollouts",
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else run_dir / "checkpoints",
        merged_adapter_candidates=(
            [Path(args.merged_adapter)]
            if args.merged_adapter
            else [merged_adapter_path(run_dir)]
        ),
        controlled_eval_dir=Path(args.controlled_eval_dir) if args.controlled_eval_dir else run_dir / "controlled_eval",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        summary = verify_real_gpu_smoke(
            resolve_paths(args),
            expected_eval_contract=args.expected_eval_contract,
            expected_num_options=args.expected_num_options,
            expected_num_samples=args.expected_num_samples,
            expected_lora_rank=args.expected_lora_rank,
            expected_lora_alpha=args.expected_lora_alpha,
        )
    except EvidenceError as error:
        print(f"evidence error: {error}", file=sys.stderr)
        return 2
    except VerificationFailed as error:
        print(f"verification failed: {error}", file=sys.stderr)
        return 3
    print(
        "PASS steps={steps} pair_checks={checks} reorder_observed={reorder} nonzero_lora_b={lora}".format(
            steps=summary["optimization_steps"],
            checks=summary["identity"]["pair_check_count"],
            reorder=summary["identity"]["reorder_observed"],
            lora=summary["lora"]["nonzero_lora_b"],
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
