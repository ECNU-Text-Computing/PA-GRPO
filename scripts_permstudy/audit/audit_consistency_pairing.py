"""Compare official order pairing with explicit rollout-identity pairing."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from permstudy.grouping import pair_consistency_rollouts  # noqa: E402
from permstudy.ids import RolloutId  # noqa: E402


@dataclass(frozen=True)
class Sample:
    rollout_id: RolloutId
    surface_answer: str
    response_length: int

    @property
    def semantic_answer(self) -> str:
        if self.rollout_id.permutation_id == 0:
            return "y1" if self.surface_answer == "A" else "y2"
        return "y2" if self.surface_answer == "A" else "y1"


SAMPLES = [
    Sample(RolloutId("pair-1", 0, 0), "A", 4),
    Sample(RolloutId("pair-1", 0, 1), "A", 9),
    Sample(RolloutId("pair-1", 0, 2), "B", 2),
    Sample(RolloutId("pair-1", 1, 0), "B", 7),
    Sample(RolloutId("pair-1", 1, 1), "A", 1),
    Sample(RolloutId("pair-1", 1, 2), "A", 6),
]


def _official_order_pairs(samples: list[Sample]) -> list[tuple[int, int]]:
    by_permutation: dict[int, list[int]] = {0: [], 1: []}
    for position, sample in enumerate(samples):
        by_permutation[sample.rollout_id.permutation_id].append(position)
    return list(zip(by_permutation[0], by_permutation[1]))


def _explicit_pairs(samples: list[Sample]) -> list[tuple[int, int]]:
    result = pair_consistency_rollouts([sample.rollout_id for sample in samples])
    if result.unpaired:
        raise AssertionError(f"audit fixture unexpectedly contains unpaired samples: {result.unpaired}")
    return list(result.pairs.values())


def _identity_pair(samples: list[Sample], pair: tuple[int, int]) -> tuple[int, int]:
    return tuple(samples[position].rollout_id.rollout_slot for position in pair)


def _reward_vector(samples: list[Sample], pairs: list[tuple[int, int]]) -> list[float]:
    rewards = [0.0] * len(samples)
    for left, right in pairs:
        reward = 1.0 if samples[left].semantic_answer == samples[right].semantic_answer else -1.0
        rewards[left] += reward
        rewards[right] += reward
    return rewards


def _scenario(name: str, order: list[int]) -> dict[str, object]:
    samples = [SAMPLES[index] for index in order]
    official_pairs = _official_order_pairs(samples)
    explicit_pairs = _explicit_pairs(samples)
    official_identity_pairs = [_identity_pair(samples, pair) for pair in official_pairs]
    explicit_identity_pairs = [_identity_pair(samples, pair) for pair in explicit_pairs]
    exact_pair_matches = sum(left_slot == right_slot for left_slot, right_slot in official_identity_pairs)

    official_rewards = _reward_vector(samples, official_pairs)
    explicit_rewards = _reward_vector(samples, explicit_pairs)
    reward_matches = sum(a == b for a, b in zip(official_rewards, explicit_rewards))
    mean_difference = sum(a - b for a, b in zip(official_rewards, explicit_rewards)) / len(samples)
    mean_absolute_difference = sum(
        abs(a - b) for a, b in zip(official_rewards, explicit_rewards)
    ) / len(samples)

    return {
        "name": name,
        "ordered_rollout_ids": [
            {
                "pair_id": sample.rollout_id.pair_id,
                "permutation_id": sample.rollout_id.permutation_id,
                "rollout_slot": sample.rollout_id.rollout_slot,
                "response_length": sample.response_length,
                "surface_answer": sample.surface_answer,
                "semantic_answer": sample.semantic_answer,
            }
            for sample in samples
        ],
        "official_slot_pairs": official_identity_pairs,
        "explicit_slot_pairs": explicit_identity_pairs,
        "official_consistency_rewards": official_rewards,
        "explicit_consistency_rewards": explicit_rewards,
        "metrics": {
            "pairing_match_rate": exact_pair_matches / len(explicit_pairs),
            "consistency_reward_match_rate": reward_matches / len(samples),
            "mean_reward_difference": mean_difference,
            "mean_absolute_reward_difference": mean_absolute_difference,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "audit" / "task1_5" / "consistency_pairing_reorder.json",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    random_order = list(range(len(SAMPLES)))
    random.Random(20261004).shuffle(random_order)
    length_order = sorted(range(len(SAMPLES)), key=lambda index: SAMPLES[index].response_length)
    balance_order = [3, 0, 4, 2, 5, 1]
    scenarios = [
        _scenario("original", list(range(len(SAMPLES)))),
        _scenario("random_shuffle_seed_20261004", random_order),
        _scenario("response_length_sort", length_order),
        _scenario("simulated_balance_reorder", balance_order),
    ]
    payload = {
        "fixture_note": "Synthetic semantic winners are chosen to make cross-slot mispairing observable.",
        "official_behavior_modeled": "pair t-th observed permutation-0 item with t-th observed permutation-1 item",
        "explicit_behavior": "pair by (pair_id, rollout_slot)",
        "scenarios": scenarios,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote consistency pairing audit to {args.output}")


if __name__ == "__main__":
    main()
