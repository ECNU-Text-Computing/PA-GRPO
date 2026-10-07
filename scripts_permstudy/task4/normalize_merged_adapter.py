"""Normalize the merged LoRA adapter config before the Task 4 controlled evaluation.

Why this exists
---------------
``verl/model_merger/base_model_merger.py:266-269`` writes the PEFT config for a merged LoRA
adapter as::

    peft_dict = {
        "r": lora_rank,
        "lora_alpha": 0,  # lora_alpha is not set. An error should be raised to inform the user to set it manually.
        ...
    }

Training runs with ``lora_rank=32`` and ``lora_alpha=64``, so PEFT's LoRA scaling
``alpha / r`` should be ``2``. Left at the merger's ``0``, the merged config scales the trained
delta to ``0 / 32 = 0``: a non-zero ``lora_B`` and a loadable adapter would then still not prove
that the controlled evaluation saw the trained adapter.

This script is the controlled correction step between ``model_merger merge`` and the
controlled evaluation. It never guesses: a rank other than the expected one is a hard error,
and any alpha correction is written to ``adapter_normalization.json`` with the before/after
SHA256 of ``adapter_config.json`` so the change is auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = "task4_adapter_normalization_v1"
DEFAULT_RANK = 32
DEFAULT_ALPHA = 64


class AdapterConfigError(ValueError):
    """The merged adapter config is missing, malformed, or contradicts the training config."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AdapterConfigError(f"missing merged adapter config: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AdapterConfigError(f"{path.name} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise AdapterConfigError(f"{path.name} must be a JSON object")
    return payload


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), prefix=path.name + ".", suffix=".tmp", delete=False
    )
    try:
        with handle:
            json.dump(payload, handle, indent=4, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def normalize_merged_adapter(
    adapter_dir: str | Path,
    expected_rank: int = DEFAULT_RANK,
    expected_alpha: int = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Verify the merged adapter's rank and bring its alpha onto the training config."""
    directory = Path(adapter_dir)
    config_path = directory / "adapter_config.json"
    before_bytes = config_path.read_bytes() if config_path.is_file() else b""
    config = _read_config(config_path)
    before_sha = _sha256_bytes(before_bytes) if before_bytes else None

    for key in ("r", "lora_alpha"):
        if key not in config:
            raise AdapterConfigError(f"{config_path.name} is missing {key}")
    try:
        rank = int(config["r"])
        alpha = int(config["lora_alpha"])
    except (TypeError, ValueError) as exc:
        raise AdapterConfigError(f"{config_path.name} has a non-integer r or lora_alpha") from exc

    if rank != expected_rank:
        raise AdapterConfigError(
            f"merged adapter rank r is {rank}, but training used {expected_rank}; refusing to rewrite "
            f"a config whose rank does not match the training configuration"
        )

    corrected = alpha != expected_alpha
    if corrected:
        config["lora_alpha"] = expected_alpha
        _write_atomic(config_path, config)

    after_sha = _sha256_bytes(config_path.read_bytes())
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "adapter_dir": directory.name,
        "config_file": config_path.name,
        "expected_rank": expected_rank,
        "expected_alpha": expected_alpha,
        "observed_rank": rank,
        "observed_alpha_before": alpha,
        "observed_alpha_after": expected_alpha if corrected else alpha,
        "corrected": corrected,
        "scaling_before": (alpha / rank) if rank else None,
        "scaling_after": (expected_alpha / rank) if rank else None,
        "before_sha256": before_sha,
        "after_sha256": after_sha,
    }
    evidence_path = directory / "adapter_normalization.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize a merged LoRA adapter config for evaluation.")
    parser.add_argument("--adapter-dir", required=True, help="directory holding adapter_config.json")
    parser.add_argument("--expected-rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--expected-alpha", type=int, default=DEFAULT_ALPHA)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        evidence = normalize_merged_adapter(args.adapter_dir, args.expected_rank, args.expected_alpha)
    except AdapterConfigError as error:
        print(f"adapter config error: {error}", file=sys.stderr)
        return 2
    print(
        "rank={rank} alpha {before}->{after} corrected={corrected} scaling={scaling}".format(
            rank=evidence["observed_rank"],
            before=evidence["observed_alpha_before"],
            after=evidence["observed_alpha_after"],
            corrected=evidence["corrected"],
            scaling=evidence["scaling_after"],
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
