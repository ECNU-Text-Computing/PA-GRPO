"""Mode-aware pairing adapter used by the Judge reward functions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from permstudy.grouping import ConsistencyKey, pair_consistency_rollouts
from permstudy.ids import RolloutId
from permstudy.rollout_identity import EXPLICIT_IDENTITY_MODE, LEGACY_IDENTITY_MODE, validate_identity_mode


@dataclass(frozen=True)
class RewardPairing:
    pairs: tuple[tuple[ConsistencyKey, int, int], ...]
    unpaired: dict[ConsistencyKey, tuple[int, ...]]


def _required(extra: dict[str, Any], key: str, position: int) -> Any:
    value = extra.get(key)
    if value is None:
        raise ValueError(f"explicit identity requires {key} at reward position {position}")
    return value


def build_reward_pairing(
    *,
    identity_mode: str,
    extra_infos: Sequence[dict[str, Any]],
    legacy_pair_ids: Sequence[Any],
    legacy_permutations: Sequence[Any],
) -> RewardPairing:
    """Build either the untouched official index pairing or controlled pairing."""

    validate_identity_mode(identity_mode)
    if identity_mode == EXPLICIT_IDENTITY_MODE:
        rollout_ids = [
            RolloutId(
                pair_id=str(_required(extra, "pair_id", position)),
                permutation_id=int(_required(extra, "permutation_id", position)),
                rollout_slot=int(_required(extra, "rollout_slot", position)),
            )
            for position, extra in enumerate(extra_infos)
        ]
        result = pair_consistency_rollouts(rollout_ids)
        return RewardPairing(
            pairs=tuple((key, positions[0], positions[1]) for key, positions in result.pairs.items()),
            unpaired=result.unpaired,
        )

    if identity_mode != LEGACY_IDENTITY_MODE:  # Defensive; validate_identity_mode already checks this.
        raise AssertionError(identity_mode)
    grouped = defaultdict(lambda: {0: [], 1: []})
    for position, (pair_id, permutation_id) in enumerate(zip(legacy_pair_ids, legacy_permutations, strict=True)):
        if pair_id is None or permutation_id not in (0, 1):
            continue
        grouped[pair_id][permutation_id].append(position)

    pairs: list[tuple[ConsistencyKey, int, int]] = []
    for pair_id, permutation_positions in grouped.items():
        for rollout_index, (position_0, position_1) in enumerate(
            zip(permutation_positions[0], permutation_positions[1])
        ):
            pairs.append(((str(pair_id), rollout_index), position_0, position_1))
    return RewardPairing(pairs=tuple(pairs), unpaired={})
