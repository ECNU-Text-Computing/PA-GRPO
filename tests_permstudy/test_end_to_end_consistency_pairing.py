import importlib

import pytest


class _NullWriter:
    def add_scalar(self, *_args, **_kwargs):
        pass


class _RecordingWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, name, value, _step):
        self.scalars[name] = value


@pytest.mark.parametrize("module_name", ["my_reward.judge_llama", "my_reward.judge_qwen"])
def test_explicit_reward_pairs_by_pair_id_and_slot_after_reorder(monkeypatch, module_name):
    """Catches reverting to t-th-by-appearance pairing after batch balance."""
    reward_module = importlib.import_module(module_name)
    monkeypatch.setattr(reward_module, "get_tb_writer", lambda: _NullWriter())
    monkeypatch.setattr(reward_module, "safe_log", lambda *_args, **_kwargs: None)
    if hasattr(reward_module, "safe_log_output"):
        monkeypatch.setattr(reward_module, "safe_log_output", lambda *_args, **_kwargs: None)

    # Reordered positions are: BA slot 1, AB slot 0, BA slot 0, AB slot 1.
    result = reward_module.compute_score(
        data_sources=["judge"] * 4,
        solution_strs=[
            "<answer>A</answer>",
            "<answer>A</answer>",
            "<answer>B</answer>",
            "<answer>A</answer>",
        ],
        ground_truths=["A", "A", "B", "A"],
        extra_infos=[
            {"pair_id": "pair-1", "permutation_id": 1, "rollout_slot": 1},
            {"pair_id": "pair-1", "permutation_id": 0, "rollout_slot": 0},
            {"pair_id": "pair-1", "permutation_id": 1, "rollout_slot": 0},
            {"pair_id": "pair-1", "permutation_id": 0, "rollout_slot": 1},
        ],
        identity_mode="explicit",
        return_dict=True,
    )

    assert result["reward_extra_info"]["consistency"] == [-1.0, 1.0, 1.0, -1.0]
    assert result["reward_extra_info"]["consistency_unpaired"] == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("module_name", ["my_reward.judge_llama", "my_reward.judge_qwen"])
def test_explicit_reward_records_missing_partner_without_cross_slot_pairing(monkeypatch, module_name):
    """Catches pairing a missing slot with a rollout from another slot."""
    reward_module = importlib.import_module(module_name)
    writer = _RecordingWriter()
    monkeypatch.setattr(reward_module, "get_tb_writer", lambda: writer)
    monkeypatch.setattr(reward_module, "safe_log", lambda *_args, **_kwargs: None)
    if hasattr(reward_module, "safe_log_output"):
        monkeypatch.setattr(reward_module, "safe_log_output", lambda *_args, **_kwargs: None)

    result = reward_module.compute_score(
        data_sources=["judge"] * 3,
        solution_strs=["<answer>A</answer>", "<answer>B</answer>", "<answer>A</answer>"],
        ground_truths=["A", "B", "A"],
        extra_infos=[
            {"pair_id": "pair-1", "permutation_id": 0, "rollout_slot": 0},
            {"pair_id": "pair-1", "permutation_id": 1, "rollout_slot": 0},
            {"pair_id": "pair-1", "permutation_id": 0, "rollout_slot": 1},
        ],
        identity_mode="explicit",
        return_dict=True,
    )

    assert result["reward_extra_info"]["consistency"] == [1.0, 1.0, 0.0]
    assert result["reward_extra_info"]["consistency_unpaired"] == [0.0, 0.0, 1.0]
    assert writer.scalars["reward/consistency_unpaired_rate"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("module_name", ["my_reward.judge_llama", "my_reward.judge_qwen"])
def test_explicit_reward_rejects_duplicate_identity(monkeypatch, module_name):
    """Catches duplicate rollout identities being silently overwritten."""
    reward_module = importlib.import_module(module_name)
    monkeypatch.setattr(reward_module, "get_tb_writer", lambda: _NullWriter())
    monkeypatch.setattr(reward_module, "safe_log", lambda *_args, **_kwargs: None)
    if hasattr(reward_module, "safe_log_output"):
        monkeypatch.setattr(reward_module, "safe_log_output", lambda *_args, **_kwargs: None)

    duplicate = {"pair_id": "pair-1", "permutation_id": 0, "rollout_slot": 0}
    with pytest.raises(ValueError, match="duplicate rollout identity"):
        reward_module.compute_score(
            data_sources=["judge"] * 3,
            solution_strs=["<answer>A</answer>"] * 3,
            ground_truths=["A"] * 3,
            extra_infos=[duplicate, duplicate.copy(), {**duplicate, "permutation_id": 1}],
            identity_mode="explicit",
        )


@pytest.mark.parametrize("module_name", ["my_reward.judge_llama", "my_reward.judge_qwen"])
def test_legacy_reward_keeps_index_aligned_official_behavior(monkeypatch, module_name):
    """Catches controlled pairing replacing the default official behavior."""
    reward_module = importlib.import_module(module_name)
    monkeypatch.setattr(reward_module, "get_tb_writer", lambda: _NullWriter())
    monkeypatch.setattr(reward_module, "safe_log", lambda *_args, **_kwargs: None)
    if hasattr(reward_module, "safe_log_output"):
        monkeypatch.setattr(reward_module, "safe_log_output", lambda *_args, **_kwargs: None)

    result = reward_module.compute_score(
        data_sources=["judge"] * 4,
        solution_strs=[
            "<answer>A</answer>",
            "<answer>A</answer>",
            "<answer>B</answer>",
            "<answer>A</answer>",
        ],
        ground_truths=["A", "A", "B", "A"],
        extra_infos=[
            {"index": 1, "permutation": 1},
            {"index": 0, "permutation": 0},
            {"index": 1, "permutation": 1},
            {"index": 0, "permutation": 0},
        ],
        identity_mode="legacy_index",
        return_dict=True,
    )

    assert result["reward_extra_info"]["consistency"] == [-1.0, -1.0, 1.0, 1.0]
    assert "consistency_unpaired" not in result["reward_extra_info"]
