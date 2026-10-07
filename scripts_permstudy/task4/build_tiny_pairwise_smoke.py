"""Build the deterministic 8-pair / 16-row fixture for the Task 4 real-GPU smoke.

The fixture feeds ``verl.trainer.main_ppo`` with ``grouping.identity_mode=explicit``,
so every row must carry identity metadata that
``permstudy.rollout_identity.attach_permutation_identity`` can consume directly:
``extra_info.original_question_id`` (or ``pair_id``) plus ``extra_info.permutation``.

Prompt lengths are measured exactly the way the trainer measures them under this
smoke's configuration (``data.filter_overlong_prompts=false``,
``data.truncation=error``)::

    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    length = len(tokenizer.encode(rendered, add_special_tokens=False))

A row whose length exceeds ``max_prompt_length`` makes the trainer raise
``RuntimeError``, so such pairs are excluded here instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

SCHEMA_VERSION = "task4_tiny_pairwise_smoke_v1"
SELECTION_SEED = 42
DEFAULT_PAIR_COUNT = 8
DEFAULT_MAX_PROMPT_LENGTH = 4096
PROMPT_KEY = "prompt"
EXTRA_INFO_KEY = "extra_info"
REWARD_KEY = "reward_model"

SELECTION_RULE = "prompt_length_quantile_bucket_v1"
ORDER_RULE = "alternating_extremes_v1"


class FixtureError(ValueError):
    """Raised when the source dataset cannot yield a valid deterministic fixture."""


# ---------------------------------------------------------------------------
# prompt measurement
# ---------------------------------------------------------------------------


def render_prompt(tokenizer: Any, messages: Sequence[Mapping[str, Any]]) -> str:
    """Render messages to the exact prompt text the trainer would tokenize."""
    return tokenizer.apply_chat_template(list(messages), add_generation_prompt=True, tokenize=False)


def prompt_token_length(tokenizer: Any, messages: Sequence[Mapping[str, Any]]) -> int:
    """Measure a prompt exactly as the trainer does with ``truncation=error``."""
    return len(tokenizer.encode(render_prompt(tokenizer, messages), add_special_tokens=False))


def _measure(tokenizer: Any, messages: Sequence[Mapping[str, Any]]) -> tuple[str, int]:
    rendered = render_prompt(tokenizer, messages)
    return rendered, len(tokenizer.encode(rendered, add_special_tokens=False))


def _load_tokenizer(model_path: str | Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise FixtureError("transformers is required to measure prompt lengths") from exc
    return AutoTokenizer.from_pretrained(str(model_path))


# ---------------------------------------------------------------------------
# source validation
# ---------------------------------------------------------------------------


def _require_columns(frame: pd.DataFrame, source: Path) -> None:
    required = {PROMPT_KEY, REWARD_KEY, EXTRA_INFO_KEY}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FixtureError(f"source dataset {source.name} is missing columns: {missing}")


def _as_int(value: Any, *, what: str, qid: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise FixtureError(f"question {qid} has a non-integer {what}: {value!r}") from exc


def _ground_truth(row: Mapping[str, Any]) -> str:
    reward = row.get(REWARD_KEY)
    if isinstance(reward, Mapping):
        return str(reward.get("ground_truth"))
    raise FixtureError("source row is missing reward_model.ground_truth")


def _expected_mirror(winner: str, permutation: int) -> str:
    if permutation == 0:
        return "A" if winner == "model_a" else "B"
    return "B" if winner == "model_a" else "A"


def _validate_pair(qid: str, first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    winner_first = str(first[EXTRA_INFO_KEY].get("winner"))
    winner_second = str(second[EXTRA_INFO_KEY].get("winner"))
    if winner_first != winner_second:
        raise FixtureError(
            f"question {qid} winner must agree across permutations, got "
            f"{winner_first!r} and {winner_second!r}"
        )
    for label, row, permutation in (("permutation 0", first, 0), ("permutation 1", second, 1)):
        expected = _expected_mirror(winner_first, permutation)
        actual = _ground_truth(row)
        if actual != expected:
            raise FixtureError(
                f"question {qid} {label} ground_truth must mirror the winner: "
                f"winner {winner_first!r} requires {expected!r}, got {actual!r}"
            )


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def _collect_eligible_pairs(frame: pd.DataFrame, tokenizer: Any, max_prompt_length: int):
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for position, record in enumerate(frame.to_dict(orient="records")):
        extra = record.get(EXTRA_INFO_KEY)
        if not isinstance(extra, Mapping):
            raise FixtureError(f"source row {position} has no extra_info mapping")
        qid = extra.get("original_question_id")
        if qid is None:
            raise FixtureError(f"source row {position} has no extra_info.original_question_id")
        grouped.setdefault(str(qid), []).append((position, record))

    eligible: list[dict[str, Any]] = []
    excluded_overlong = 0
    for qid in sorted(grouped):
        by_permutation: dict[int, list[tuple[int, Mapping[str, Any]]]] = {}
        for position, record in grouped[qid]:
            permutation = _as_int(record[EXTRA_INFO_KEY].get("permutation"), what="permutation", qid=qid)
            by_permutation.setdefault(permutation, []).append((position, record))

        shape = {key: len(value) for key, value in sorted(by_permutation.items())}
        if set(by_permutation) != {0, 1} or shape.get(0) != 1 or shape.get(1) != 1:
            raise FixtureError(
                f"question {qid} must have exactly one permutation 0 and one permutation 1, got {shape}"
            )

        first_position, first = by_permutation[0][0]
        second_position, second = by_permutation[1][0]
        _validate_pair(qid, first, second)

        first_text, first_tokens = _measure(tokenizer, first[PROMPT_KEY])
        second_text, second_tokens = _measure(tokenizer, second[PROMPT_KEY])
        if max(first_tokens, second_tokens) > max_prompt_length:
            excluded_overlong += 1
            continue

        eligible.append(
            {
                "original_question_id": qid,
                "pair_id": qid,
                "winner": str(first[EXTRA_INFO_KEY].get("winner")),
                "max_tokens": max(first_tokens, second_tokens),
                "rows": (
                    {
                        "position": int(first_position),
                        "permutation": 0,
                        "ground_truth": _ground_truth(first),
                        "prompt": [dict(message) for message in first[PROMPT_KEY]],
                        "prompt_text": first_text,
                        "prompt_tokens": first_tokens,
                    },
                    {
                        "position": int(second_position),
                        "permutation": 1,
                        "ground_truth": _ground_truth(second),
                        "prompt": [dict(message) for message in second[PROMPT_KEY]],
                        "prompt_text": second_text,
                        "prompt_tokens": second_tokens,
                    },
                ),
            }
        )
    return eligible, excluded_overlong


def _selection_score(qid: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}\0{qid}".encode("utf-8")).hexdigest()


def _select_pairs(eligible: Sequence[dict[str, Any]], pair_count: int) -> list[dict[str, Any]]:
    """Take one pair from each prompt-length quantile bucket, deterministically."""
    by_length = sorted(eligible, key=lambda pair: (pair["max_tokens"], pair["original_question_id"]))
    total = len(by_length)
    chosen: dict[int, tuple[str, dict[str, Any]]] = {}
    for rank, pair in enumerate(by_length):
        bucket = rank * pair_count // total
        candidate = (_selection_score(pair["original_question_id"]), pair)
        best = chosen.get(bucket)
        if best is None or candidate[0] < best[0]:
            chosen[bucket] = candidate
    return [chosen[bucket][1] for bucket in sorted(chosen)]


def _alternating_order(selected: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Interleave longest/shortest so ``balance_batch`` must move rows around."""
    ascending = sorted(selected, key=lambda pair: (pair["max_tokens"], pair["original_question_id"]))
    low, high = 0, len(ascending) - 1
    ordered: list[dict[str, Any]] = []
    while low <= high:
        ordered.append(ascending[high])
        high -= 1
        if low <= high:
            ordered.append(ascending[low])
            low += 1
    return ordered


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_tiny_pairwise_smoke(
    source: str | Path,
    model_path: str | Path,
    pair_count: int,
    max_prompt_length: int,
    output: str | Path,
    manifest: str | Path,
    *,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Write a deterministic ``pair_count``-pair fixture and return its manifest."""
    source_path = Path(source)
    output_path = Path(output)
    manifest_path = Path(manifest)
    if not source_path.is_file():
        raise FixtureError(f"source dataset not found: {source_path.name}")
    if pair_count <= 0:
        raise FixtureError(f"pair_count must be positive, got {pair_count}")
    if max_prompt_length <= 0:
        raise FixtureError(f"max_prompt_length must be positive, got {max_prompt_length}")

    frame = pd.read_parquet(source_path)
    _require_columns(frame, source_path)
    active_tokenizer = tokenizer if tokenizer is not None else _load_tokenizer(model_path)

    eligible, excluded_overlong = _collect_eligible_pairs(frame, active_tokenizer, max_prompt_length)
    if len(eligible) < pair_count:
        raise FixtureError(
            f"eligible pairs {len(eligible)} are fewer than the requested pair_count {pair_count}"
        )

    ordered = _alternating_order(_select_pairs(eligible, pair_count))

    fixture_rows: list[dict[str, Any]] = []
    pairs_payload: list[dict[str, Any]] = []
    canonical_fixture: list[dict[str, Any]] = []
    for pair in ordered:
        row_payloads: list[dict[str, Any]] = []
        for row in pair["rows"]:
            extra_info = {
                "pair_id": pair["pair_id"],
                "original_question_id": pair["original_question_id"],
                "permutation": row["permutation"],
                "winner": pair["winner"],
                "prompt_tokens": row["prompt_tokens"],
                # ``index`` is the source dataset's own provenance key and the value
                # ``my_reward/judge_qwen.py`` logs as ``idx=``. Dropping it would make every
                # reward-log source index fall back to the post-reorder batch position, which
                # destroys the plan's "reward log source-index order" reorder evidence.
                "index": row["position"],
            }
            fixture_rows.append(
                {
                    "data_source": "task4_tiny_pairwise_smoke",
                    "prompt": row["prompt"],
                    "ability": "text",
                    "reward_model": {"ground_truth": row["ground_truth"], "style": "rule"},
                    "extra_info": extra_info,
                }
            )
            row_payloads.append(
                {
                    "permutation": int(row["permutation"]),
                    "row_index": int(row["position"]),
                    "ground_truth": row["ground_truth"],
                    "prompt_tokens": int(row["prompt_tokens"]),
                    "prompt_sha256": _sha256_bytes(row["prompt_text"].encode("utf-8")),
                }
            )
            canonical_fixture.append(
                {
                    "pair_id": pair["pair_id"],
                    "permutation": int(row["permutation"]),
                    "ground_truth": row["ground_truth"],
                    "winner": pair["winner"],
                    "prompt_tokens": int(row["prompt_tokens"]),
                    "prompt": [dict(message) for message in row["prompt"]],
                }
            )
        pairs_payload.append(
            {
                "pair_id": pair["pair_id"],
                "original_question_id": pair["original_question_id"],
                "winner": pair["winner"],
                "max_prompt_tokens": int(pair["max_tokens"]),
                "rows": row_payloads,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fixture_rows).to_parquet(output_path, index=False)

    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "path": source_path.name,
            "sha256": _sha256_bytes(source_path.read_bytes()),
            "rows": int(len(frame)),
            "validated_questions": int(len(eligible) + excluded_overlong),
        },
        "selection": {
            "pair_count": int(pair_count),
            "max_prompt_length": int(max_prompt_length),
            "seed": SELECTION_SEED,
            "eligible_pairs": int(len(eligible)),
            "excluded_overlong_pairs": int(excluded_overlong),
            "rule": SELECTION_RULE,
            "order_rule": ORDER_RULE,
        },
        "pairs": pairs_payload,
        "pair_count": int(len(pairs_payload)),
        "row_count": int(len(fixture_rows)),
        "fixture_sha256": _sha256_bytes(_canonical_bytes(canonical_fixture)),
        "fixture_file_sha256": _sha256_bytes(output_path.read_bytes()),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="source pairwise parquet")
    parser.add_argument("--model-path", required=True, help="tokenizer/model path used for length measurement")
    parser.add_argument("--pair-count", type=int, default=DEFAULT_PAIR_COUNT)
    parser.add_argument("--max-prompt-length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH)
    parser.add_argument("--output", required=True, help="destination fixture parquet")
    parser.add_argument("--manifest", required=True, help="destination JSON manifest")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        manifest = build_tiny_pairwise_smoke(
            source=args.source,
            model_path=args.model_path,
            pair_count=args.pair_count,
            max_prompt_length=args.max_prompt_length,
            output=args.output,
            manifest=args.manifest,
        )
    except FixtureError as error:
        print(f"fixture error: {error}", file=sys.stderr)
        return 2
    print(
        "pairs={pair_count} rows={row_count} eligible={eligible} excluded_overlong={excluded} "
        "fixture_sha256={digest}".format(
            pair_count=manifest["pair_count"],
            row_count=manifest["row_count"],
            eligible=manifest["selection"]["eligible_pairs"],
            excluded=manifest["selection"]["excluded_overlong_pairs"],
            digest=manifest["fixture_sha256"],
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
