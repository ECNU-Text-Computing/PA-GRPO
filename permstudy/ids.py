"""Explicit identities for permutation inputs and generated rollouts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PermutationSampleId:
    pair_id: str
    permutation_id: int


@dataclass(frozen=True)
class RolloutId:
    pair_id: str
    permutation_id: int
    rollout_slot: int
