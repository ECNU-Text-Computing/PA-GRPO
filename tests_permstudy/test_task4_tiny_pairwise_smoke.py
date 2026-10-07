"""Tests for the Task 4 deterministic tiny pairwise smoke-fixture builder.

The builder must produce a 16-row / 8-pair fixture whose identity metadata is
directly consumable by ``permstudy.rollout_identity.attach_permutation_identity``
and whose prompt lengths are measured exactly the way the trainer measures them
under ``filter_overlong_prompts=false`` + ``truncation=error``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts_permstudy" / "task4" / "build_tiny_pairwise_smoke.py"

PAIR_COUNT = 8
MAX_PROMPT_LENGTH = 4096


def load_builder():
    """Import the builder by path so a missing script is an ordinary failure."""
    assert SCRIPT.is_file(), f"missing task4 fixture builder: {SCRIPT}"
    spec = importlib.util.spec_from_file_location("task4_build_tiny_pairwise_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RecordingTokenizer:
    """Whitespace tokenizer that records how the builder measured a prompt."""

    GENERATION_PROMPT = "assistant:"

    def __init__(self) -> None:
        self.chat_template_calls: list[dict] = []
        self.encode_calls: list[dict] = []

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        self.chat_template_calls.append(
            {"messages": messages, "add_generation_prompt": add_generation_prompt, "tokenize": tokenize}
        )
        assert tokenize is False, "length measurement must render the template to text first"
        parts = [f"{message['role']}: {message['content']}" for message in messages]
        if add_generation_prompt:
            parts.append(self.GENERATION_PROMPT)
        return "\n".join(parts)

    def encode(self, text, add_special_tokens=False):
        self.encode_calls.append({"text": text, "add_special_tokens": add_special_tokens})
        return text.split()


def expected_tokens(tokenizer, messages) -> int:
    """Independently compute the trainer's prompt length for these messages."""
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return len(tokenizer.encode(text, add_special_tokens=False))


def make_pair(qid: str, *, winner: str = "model_a", perm0_words: int = 10, perm1_words: int = 12):
    """Build one mirrored permutation-0/1 pair exactly as the source dataset does."""
    rows = []
    for permutation, words in ((0, perm0_words), (1, perm1_words)):
        if permutation == 0:
            gold = "A" if winner == "model_a" else "B"
        else:
            gold = "B" if winner == "model_a" else "A"
        rows.append(
            {
                "data_source": "chatbot_arena",
                "prompt": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": " ".join(f"w{i}" for i in range(words))},
                ],
                "ability": "text",
                "reward_model": {"ground_truth": gold, "style": "rule"},
                "extra_info": {
                    "index": len(rows),
                    "model_a": "name_a",
                    "model_b": "name_b",
                    "original_question_id": qid,
                    "permutation": permutation,
                    "split": "train",
                    "winner": winner,
                },
            }
        )
    return rows


def write_source(path: Path, rows) -> Path:
    pd.DataFrame(list(rows)).to_parquet(path, index=False)
    return path


def write_pairs_source(path: Path, pairs) -> Path:
    rows = [row for pair in pairs for row in pair]
    return write_source(path, rows)


def make_pairs(count: int, *, base_perm0: int = 10, base_perm1: int = 30, qid_prefix: str = "q"):
    return [
        make_pair(f"{qid_prefix}{i:04d}", perm0_words=base_perm0 + i, perm1_words=base_perm1 + i)
        for i in range(count)
    ]


def build(tmp_path: Path, pairs, **overrides):
    builder = load_builder()
    source = write_pairs_source(tmp_path / "source.parquet", pairs)
    kwargs = {
        "source": source,
        "model_path": str(tmp_path / "qwen3-8b"),
        "pair_count": PAIR_COUNT,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "output": tmp_path / "task4_tiny_8pairs.parquet",
        "manifest": tmp_path / "dataset_manifest.json",
        "tokenizer": RecordingTokenizer(),
    }
    kwargs.update(overrides)
    manifest = builder.build_tiny_pairwise_smoke(**kwargs)
    return builder, kwargs, manifest


# --------------------------------------------------------------------------
# fixture shape
# --------------------------------------------------------------------------


def test_builder_writes_eight_pairs_and_sixteen_rows(tmp_path):
    _builder, kwargs, _manifest = build(tmp_path, make_pairs(24))

    fixture = pd.read_parquet(kwargs["output"])
    assert len(fixture) == 16
    assert fixture["extra_info"].map(lambda e: e["original_question_id"]).nunique() == PAIR_COUNT

    groups = fixture.groupby(fixture["extra_info"].map(lambda e: e["original_question_id"]))
    assert all(sorted(group["extra_info"].map(lambda e: e["permutation"])) == [0, 1] for _, group in groups)


def test_pair_rows_stay_adjacent_and_mirrored(tmp_path):
    _builder, kwargs, _manifest = build(tmp_path, make_pairs(24))

    fixture = pd.read_parquet(kwargs["output"]).reset_index(drop=True)
    assert len(fixture) % 2 == 0
    for offset in range(0, len(fixture), 2):
        first, second = fixture.iloc[offset], fixture.iloc[offset + 1]
        assert first["extra_info"]["original_question_id"] == second["extra_info"]["original_question_id"]
        assert first["extra_info"]["permutation"] == 0
        assert second["extra_info"]["permutation"] == 1
        assert first["extra_info"]["winner"] == second["extra_info"]["winner"]
        assert {first["reward_model"]["ground_truth"], second["reward_model"]["ground_truth"]} == {"A", "B"}


def test_fixture_extra_info_is_identity_ready(tmp_path):
    _builder, kwargs, _manifest = build(tmp_path, make_pairs(24))

    fixture = pd.read_parquet(kwargs["output"])
    pair_ids = [row["pair_id"] for row in fixture["extra_info"]]
    assert len(set(pair_ids)) == PAIR_COUNT
    for row in fixture["extra_info"]:
        assert row["original_question_id"]
        assert int(row["permutation"]) in (0, 1)
        assert row["pair_id"] == row["original_question_id"]


def test_extra_info_preserves_index_for_reward_log_reorder_evidence(tmp_path):
    """`my_reward/judge_qwen.py` logs `idx = extra.get("index", i)`.

    If the fixture omits `index`, every reward-log source index degrades to the
    post-reorder batch position, so the plan's "reward log source-index order
    differs from canonical interleaved input order" reorder evidence cannot be
    satisfied — and a verifier comparing those ascending positions would report a
    reorder vacuously.
    """
    _builder, kwargs, manifest = build(tmp_path, make_pairs(24))

    fixture = pd.read_parquet(kwargs["output"])
    observed = [int(row["index"]) for row in fixture["extra_info"]]
    expected = [int(row["row_index"]) for pair in manifest["pairs"] for row in pair["rows"]]

    assert observed == expected
    assert len(set(observed)) == 16
    assert observed != sorted(observed), (
        "the fixture's alternating order must not also sort the source indices, "
        "otherwise a post-reorder batch position could masquerade as a source index"
    )


# --------------------------------------------------------------------------
# validation and rejection
# --------------------------------------------------------------------------


def test_missing_or_duplicate_permutation_is_rejected(tmp_path):
    builder = load_builder()
    pairs = make_pairs(10)
    # pair q0002 loses permutation 1; pair q0003 duplicates permutation 0
    rows = [row for pair in pairs for row in pair]
    rows = [row for row in rows if not (row["extra_info"]["original_question_id"] == "q0002" and row["extra_info"]["permutation"] == 1)]
    rows.append(dict(make_pair("q0003", perm0_words=40, perm1_words=41)[0]))
    source = write_source(tmp_path / "source.parquet", rows)

    with pytest.raises(builder.FixtureError, match="exactly one permutation 0 and one permutation 1"):
        builder.build_tiny_pairwise_smoke(
            source=source,
            model_path="unused",
            pair_count=PAIR_COUNT,
            max_prompt_length=MAX_PROMPT_LENGTH,
            output=tmp_path / "out.parquet",
            manifest=tmp_path / "m.json",
            tokenizer=RecordingTokenizer(),
        )


def test_non_mirrored_ground_truth_is_rejected(tmp_path):
    builder = load_builder()
    pairs = make_pairs(10)
    # a model_a winner makes permutation 1 mirror to "B"; "A" breaks the mirror
    pairs[4][1]["reward_model"]["ground_truth"] = "A"
    source = write_pairs_source(tmp_path / "source.parquet", pairs)

    with pytest.raises(builder.FixtureError, match="mirror"):
        builder.build_tiny_pairwise_smoke(
            source=source,
            model_path="unused",
            pair_count=PAIR_COUNT,
            max_prompt_length=MAX_PROMPT_LENGTH,
            output=tmp_path / "out.parquet",
            manifest=tmp_path / "m.json",
            tokenizer=RecordingTokenizer(),
        )


def test_winner_disagreement_within_a_pair_is_rejected(tmp_path):
    builder = load_builder()
    pairs = make_pairs(10)
    pairs[3][1]["extra_info"]["winner"] = "model_b"
    source = write_pairs_source(tmp_path / "source.parquet", pairs)

    with pytest.raises(builder.FixtureError, match="winner"):
        builder.build_tiny_pairwise_smoke(
            source=source,
            model_path="unused",
            pair_count=PAIR_COUNT,
            max_prompt_length=MAX_PROMPT_LENGTH,
            output=tmp_path / "out.parquet",
            manifest=tmp_path / "m.json",
            tokenizer=RecordingTokenizer(),
        )


def test_too_few_eligible_pairs_fails(tmp_path):
    builder = load_builder()
    source = write_pairs_source(tmp_path / "source.parquet", make_pairs(5))

    with pytest.raises(builder.FixtureError, match="eligible"):
        builder.build_tiny_pairwise_smoke(
            source=source,
            model_path="unused",
            pair_count=PAIR_COUNT,
            max_prompt_length=MAX_PROMPT_LENGTH,
            output=tmp_path / "out.parquet",
            manifest=tmp_path / "m.json",
            tokenizer=RecordingTokenizer(),
        )


# --------------------------------------------------------------------------
# length measurement and overlong exclusion
# --------------------------------------------------------------------------


def test_length_measurement_matches_trainer_semantics(tmp_path):
    builder = load_builder()
    tokenizer = RecordingTokenizer()
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "w0 w1 w2"}]

    measured = builder.prompt_token_length(tokenizer, messages)

    assert measured == expected_tokens(RecordingTokenizer(), messages)
    assert tokenizer.chat_template_calls[-1]["add_generation_prompt"] is True
    assert tokenizer.chat_template_calls[-1]["tokenize"] is False
    assert tokenizer.encode_calls[-1]["add_special_tokens"] is False


def test_overlong_pair_is_excluded(tmp_path):
    builder = load_builder()
    # token length of the fake renderer is words + 5 for a two-message prompt
    overlong = make_pair("q_overlong", perm0_words=200, perm1_words=200)
    fine = make_pairs(16, qid_prefix="ok")
    source = write_pairs_source(tmp_path / "source.parquet", [overlong, *fine])
    tokenizer = RecordingTokenizer()
    limit = 100

    manifest = builder.build_tiny_pairwise_smoke(
        source=source,
        model_path="unused",
        pair_count=PAIR_COUNT,
        max_prompt_length=limit,
        output=tmp_path / "out.parquet",
        manifest=tmp_path / "m.json",
        tokenizer=tokenizer,
    )

    assert "q_overlong" not in {pair["original_question_id"] for pair in manifest["pairs"]}
    assert manifest["selection"]["excluded_overlong_pairs"] == 1
    for pair in manifest["pairs"]:
        assert pair["max_prompt_tokens"] <= limit
        for row in pair["rows"]:
            assert row["prompt_tokens"] <= limit


# --------------------------------------------------------------------------
# determinism and ordering
# --------------------------------------------------------------------------


def test_selection_is_order_independent_and_hash_stable(tmp_path):
    builder = load_builder()
    pairs = make_pairs(24)
    forward = write_pairs_source(tmp_path / "forward.parquet", pairs)
    reversed_rows = [row for pair in reversed(pairs) for row in pair]
    backward = write_source(tmp_path / "backward.parquet", reversed_rows)

    def run(source: Path, tag: str):
        return builder.build_tiny_pairwise_smoke(
            source=source,
            model_path="unused",
            pair_count=PAIR_COUNT,
            max_prompt_length=MAX_PROMPT_LENGTH,
            output=tmp_path / f"{tag}.parquet",
            manifest=tmp_path / f"{tag}.json",
            tokenizer=RecordingTokenizer(),
        )

    first, second = run(forward, "first"), run(backward, "second")

    assert first["fixture_sha256"] == second["fixture_sha256"]
    assert [pair["original_question_id"] for pair in first["pairs"]] == [
        pair["original_question_id"] for pair in second["pairs"]
    ]

    def row_indices(manifest):
        return [row["row_index"] for pair in manifest["pairs"] for row in pair["rows"]]

    # provenance row indices must follow the reordered source, while the fixture content does not
    assert row_indices(first) != row_indices(second)


def test_selected_pairs_span_every_prompt_length_quantile(tmp_path):
    _builder, kwargs, manifest = build(tmp_path, make_pairs(40))

    ordered = sorted(make_pairs(40), key=lambda pair: pair[0]["extra_info"]["original_question_id"])
    rank = {pair[0]["extra_info"]["original_question_id"]: index for index, pair in enumerate(ordered)}
    buckets = {rank[pair["original_question_id"]] * PAIR_COUNT // 40 for pair in manifest["pairs"]}

    assert buckets == set(range(PAIR_COUNT))
    assert Path(kwargs["output"]).is_file()


def test_output_order_alternates_long_and_short(tmp_path):
    _builder, kwargs, manifest = build(tmp_path, make_pairs(40))

    lengths = [pair["max_prompt_tokens"] for pair in manifest["pairs"]]
    assert len(lengths) == PAIR_COUNT
    assert len(set(lengths)) == PAIR_COUNT
    for index in range(len(lengths) - 1):
        if index % 2 == 0:
            assert lengths[index] > lengths[index + 1], f"expected long-then-short at {index}: {lengths}"
        else:
            assert lengths[index] < lengths[index + 1], f"expected short-then-long at {index}: {lengths}"


# --------------------------------------------------------------------------
# manifest contract
# --------------------------------------------------------------------------


def test_manifest_records_ids_indices_lengths_labels_and_hashes(tmp_path):
    _builder, kwargs, manifest = build(tmp_path, make_pairs(24))

    assert manifest["schema_version"]
    assert manifest["pair_count"] == PAIR_COUNT
    assert manifest["row_count"] == 16
    assert manifest["source"]["sha256"] == hashlib.sha256(Path(kwargs["source"]).read_bytes()).hexdigest()
    assert len(manifest["source"]["sha256"]) == 64
    assert len(manifest["fixture_sha256"]) == 64

    fixture_bytes = Path(kwargs["output"]).read_bytes()
    assert manifest["fixture_file_sha256"] == hashlib.sha256(fixture_bytes).hexdigest()

    seen_indices = set()
    seen_rows = 0
    for pair in manifest["pairs"]:
        assert pair["pair_id"] == pair["original_question_id"]
        assert pair["winner"] in {"model_a", "model_b"}
        assert pair["max_prompt_tokens"] > 0
        assert [row["permutation"] for row in pair["rows"]] == [0, 1]
        for row in pair["rows"]:
            assert row["ground_truth"] in {"A", "B"}
            assert isinstance(row["row_index"], int) and row["row_index"] not in seen_indices
            seen_indices.add(row["row_index"])
            assert row["prompt_tokens"] > 0
            assert len(row["prompt_sha256"]) == 64
            seen_rows += 1
    assert seen_rows == 16

    assert json.loads(Path(kwargs["manifest"]).read_text(encoding="utf-8")) == manifest


def test_manifest_pair_count_matches_selection_option(tmp_path):
    builder = load_builder()
    source = write_pairs_source(tmp_path / "source.parquet", make_pairs(20))

    manifest = builder.build_tiny_pairwise_smoke(
        source=source,
        model_path="unused",
        pair_count=4,
        max_prompt_length=MAX_PROMPT_LENGTH,
        output=tmp_path / "out.parquet",
        manifest=tmp_path / "m.json",
        tokenizer=RecordingTokenizer(),
    )

    assert manifest["pair_count"] == 4
    assert manifest["row_count"] == 8
    assert len(pd.read_parquet(tmp_path / "out.parquet")) == 8
