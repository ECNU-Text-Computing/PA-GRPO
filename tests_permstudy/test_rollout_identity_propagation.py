import importlib.util
import numpy as np
import pytest
import torch


pytestmark = pytest.mark.skipif(importlib.util.find_spec("ray") is None, reason="requires verl DataProto dependencies")


def _input_batch():
    from verl import DataProto

    return DataProto.from_single_dict(
        {
            "input_ids": torch.tensor([[10, 11], [20, 21]]),
            "extra_info": np.array(
                [
                    {"original_question_id": "pair-1", "permutation": 0},
                    {"original_question_id": "pair-1", "permutation": 1},
                ],
                dtype=object,
            ),
        }
    )


def test_explicit_repeat_assigns_slot_zero_through_n_minus_one_per_permutation():
    """Catches creating one slot before repeat, which would repeat slot zero."""
    from permstudy.rollout_identity import attach_permutation_identity, repeat_for_rollout

    batch = _input_batch()
    attach_permutation_identity(batch, identity_mode="explicit")

    repeated = repeat_for_rollout(batch, repeat_times=4, identity_mode="explicit")

    assert repeated.non_tensor_batch["pair_id"].tolist() == ["pair-1"] * 8
    assert repeated.non_tensor_batch["permutation_id"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert repeated.non_tensor_batch["rollout_slot"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


def test_actual_generation_output_must_preserve_expanded_identity_order():
    """Catches a generation backend losing or reordering identity metadata."""
    from verl import DataProto
    from permstudy.rollout_identity import (
        RolloutIdentityAlignmentError,
        attach_permutation_identity,
        repeat_for_rollout,
        validate_generation_identity,
    )

    batch = _input_batch()
    attach_permutation_identity(batch, identity_mode="explicit")
    generation_input = repeat_for_rollout(batch, repeat_times=2, identity_mode="explicit")

    aligned_output = DataProto.from_single_dict(
        {
            "responses": torch.tensor([[100], [101], [102], [103]]),
            **generation_input.non_tensor_batch,
        }
    )
    validate_generation_identity(generation_input, aligned_output, identity_mode="explicit")

    wrong_order = np.array([0, 2, 1, 3])
    misaligned_output = DataProto.from_single_dict(
        {
            "responses": torch.tensor([[100], [102], [101], [103]]),
            **{
                key: value[wrong_order]
                for key, value in generation_input.non_tensor_batch.items()
            },
        }
    )
    with pytest.raises(RolloutIdentityAlignmentError, match="generation output reordered rollout identity"):
        validate_generation_identity(generation_input, misaligned_output, identity_mode="explicit")


def test_vllm_request_outputs_are_checked_against_actual_prompt_order():
    """Catches response objects arriving in a different prompt order."""
    from types import SimpleNamespace

    from permstudy.rollout_identity import RolloutIdentityAlignmentError, validate_backend_prompt_order

    inputs = [
        {"prompt_token_ids": [10, 11]},
        {"prompt_token_ids": [20, 21]},
    ]
    aligned = [
        SimpleNamespace(prompt_token_ids=[10, 11]),
        SimpleNamespace(prompt_token_ids=[20, 21]),
    ]
    validate_backend_prompt_order(inputs, aligned)

    response_objects_reordered = [aligned[1], aligned[0]]
    with pytest.raises(RolloutIdentityAlignmentError, match="output prompt order"):
        validate_backend_prompt_order(inputs, response_objects_reordered)


def test_balance_reorder_keeps_all_identity_fields_aligned():
    """Catches identity arrays that are not carried by DataProto.reorder."""
    from permstudy.rollout_identity import attach_permutation_identity, repeat_for_rollout, rollout_ids_from_batch

    batch = _input_batch()
    attach_permutation_identity(batch, identity_mode="explicit")
    repeated = repeat_for_rollout(batch, repeat_times=2, identity_mode="explicit")

    repeated.reorder(torch.tensor([3, 0, 2, 1]))

    assert [
        (identity.pair_id, identity.permutation_id, identity.rollout_slot)
        for identity in rollout_ids_from_batch(repeated)
    ] == [
        ("pair-1", 1, 1),
        ("pair-1", 0, 0),
        ("pair-1", 1, 0),
        ("pair-1", 0, 1),
    ]


def test_explicit_identity_never_falls_back_to_index_pairing():
    """Catches silently deriving controlled pair IDs from index // 2."""
    from permstudy.rollout_identity import IdentityMetadataError, attach_permutation_identity

    batch = _input_batch()
    batch.non_tensor_batch["extra_info"] = np.array(
        [{"index": 0, "permutation": 0}, {"index": 1, "permutation": 1}], dtype=object
    )

    with pytest.raises(IdentityMetadataError, match="pair_id.*original_question_id"):
        attach_permutation_identity(batch, identity_mode="explicit")


def test_legacy_mode_leaves_official_batch_shape_and_metadata_unchanged():
    """Catches explicit identity leaking into the official reproduction path."""
    from permstudy.rollout_identity import attach_permutation_identity, repeat_for_rollout

    batch = _input_batch()
    attach_permutation_identity(batch, identity_mode="legacy_index")
    repeated = repeat_for_rollout(batch, repeat_times=2, identity_mode="legacy_index")

    assert set(repeated.non_tensor_batch) == {"extra_info"}
    assert len(repeated) == 4
