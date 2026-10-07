"""Order-independent grouping for cross-permutation consistency rewards."""

from dataclasses import dataclass
from typing import Sequence

from permstudy.ids import RolloutId


ConsistencyKey = tuple[str, int]


class DuplicateRolloutKeyError(ValueError):
    """Raised when one permutation/slot identity occurs more than once."""


@dataclass(frozen=True)
class PairingResult:
    pairs: dict[ConsistencyKey, tuple[int, int]]
    unpaired: dict[ConsistencyKey, tuple[int, ...]]


def pair_consistency_rollouts(rollout_ids: Sequence[RolloutId]) -> PairingResult:
    """Pair permutation 0/1 samples by explicit ``(pair_id, rollout_slot)``.

    Returned tuple values are positions in the supplied sequence, ordered as
    ``(permutation_0_position, permutation_1_position)``.
    """

    grouped: dict[ConsistencyKey, dict[int, int]] = {}
    for position, rollout_id in enumerate(rollout_ids):
        if rollout_id.permutation_id not in (0, 1):
            raise ValueError(
                f"permutation_id must be 0 or 1, got {rollout_id.permutation_id} at position {position}"
            )
        if rollout_id.rollout_slot < 0:
            raise ValueError(f"rollout_slot must be non-negative, got {rollout_id.rollout_slot}")

        key = (rollout_id.pair_id, rollout_id.rollout_slot)
        permutation_positions = grouped.setdefault(key, {})
        if rollout_id.permutation_id in permutation_positions:
            raise DuplicateRolloutKeyError(
                "duplicate rollout identity: "
                f"pair_id={rollout_id.pair_id!r}, "
                f"permutation_id={rollout_id.permutation_id}, "
                f"rollout_slot={rollout_id.rollout_slot}"
            )
        permutation_positions[rollout_id.permutation_id] = position

    pairs: dict[ConsistencyKey, tuple[int, int]] = {}
    unpaired: dict[ConsistencyKey, tuple[int, ...]] = {}
    for key, permutation_positions in grouped.items():
        if set(permutation_positions) == {0, 1}:
            pairs[key] = (permutation_positions[0], permutation_positions[1])
        else:
            unpaired[key] = tuple(permutation_positions[perm] for perm in sorted(permutation_positions))

    return PairingResult(pairs=pairs, unpaired=unpaired)
