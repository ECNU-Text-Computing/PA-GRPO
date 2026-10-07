import pytest


def _rollout(pair_id: str, permutation_id: int, rollout_slot: int):
    from permstudy.ids import RolloutId

    return RolloutId(pair_id, permutation_id, rollout_slot)


def test_pairing_uses_explicit_slot_in_normal_order():
    from permstudy.grouping import pair_consistency_rollouts

    ids = [
        _rollout("pair-1", 0, 0),
        _rollout("pair-1", 0, 1),
        _rollout("pair-1", 0, 2),
        _rollout("pair-1", 1, 0),
        _rollout("pair-1", 1, 1),
        _rollout("pair-1", 1, 2),
    ]

    result = pair_consistency_rollouts(ids)

    assert result.pairs == {
        ("pair-1", 0): (0, 3),
        ("pair-1", 1): (1, 4),
        ("pair-1", 2): (2, 5),
    }
    assert result.unpaired == {}


def test_pairing_is_invariant_to_batch_reorder():
    from permstudy.grouping import pair_consistency_rollouts

    ids = [
        _rollout("pair-1", 1, 2),
        _rollout("pair-1", 0, 0),
        _rollout("pair-1", 1, 0),
        _rollout("pair-1", 0, 2),
        _rollout("pair-1", 0, 1),
        _rollout("pair-1", 1, 1),
    ]

    result = pair_consistency_rollouts(ids)

    assert result.pairs == {
        ("pair-1", 0): (1, 2),
        ("pair-1", 1): (4, 5),
        ("pair-1", 2): (3, 0),
    }
    assert result.unpaired == {}


def test_missing_slot_is_reported_without_cross_slot_pairing():
    from permstudy.grouping import pair_consistency_rollouts

    ids = [
        _rollout("pair-1", 0, 0),
        _rollout("pair-1", 0, 1),
        _rollout("pair-1", 0, 2),
        _rollout("pair-1", 1, 0),
        _rollout("pair-1", 1, 2),
    ]

    result = pair_consistency_rollouts(ids)

    assert result.pairs == {
        ("pair-1", 0): (0, 3),
        ("pair-1", 2): (2, 4),
    }
    assert result.unpaired == {("pair-1", 1): (1,)}


def test_duplicate_rollout_identity_raises_instead_of_overwriting():
    from permstudy.grouping import DuplicateRolloutKeyError, pair_consistency_rollouts

    ids = [
        _rollout("pair-1", 0, 1),
        _rollout("pair-1", 0, 1),
        _rollout("pair-1", 1, 1),
    ]

    with pytest.raises(DuplicateRolloutKeyError, match="pair-1.*permutation_id=0.*rollout_slot=1"):
        pair_consistency_rollouts(ids)
