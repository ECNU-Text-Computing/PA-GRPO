"""Build the two-row pairwise Judge fixture used by Task 1A."""

from pathlib import Path

import pandas as pd


def _prompt(option_a: str, option_b: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Reply with only A or B."},
        {
            "role": "user",
            "content": (
                "Which response is more correct?\n"
                f"A: {option_a}\n"
                f"B: {option_b}\n"
                "Answer with A or B only."
            ),
        },
    ]


def main() -> None:
    rows = [
        {
            "prompt": _prompt("2 + 2 = 4", "2 + 2 = 5"),
            "reward_model": {"ground_truth": "A"},
            "data_source": "synthetic_pairwise_smoke",
            "ability": "arithmetic",
            "extra_info": {"pair_id": "smoke_pair", "permutation": 0},
        },
        {
            "prompt": _prompt("2 + 2 = 5", "2 + 2 = 4"),
            "reward_model": {"ground_truth": "B"},
            "data_source": "synthetic_pairwise_smoke",
            "ability": "arithmetic",
            "extra_info": {"pair_id": "smoke_pair", "permutation": 1},
        },
    ]

    repo_root = Path(__file__).resolve().parents[2]
    output = repo_root / "artifacts" / "smoke" / "task1a" / "smoke_judge_direct.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
