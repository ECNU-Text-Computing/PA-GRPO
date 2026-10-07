"""Generate the CPU-only PA paper/code advantage fidelity artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from permstudy.advantages import (  # noqa: E402
    AdvantageResult,
    global_advantage_official_compatible,
    global_advantage_paper,
)


REWARDS = torch.tensor([1.0, 1.0, -1.0, -1.0])
GROUP_IDS = ["pair-1"] * 4


def _mask(lengths: list[int]) -> torch.Tensor:
    width = max(lengths)
    return torch.tensor(
        [[1.0] * length + [0.0] * (width - length) for length in lengths],
        dtype=torch.float32,
    )


def _tensor_list(value: torch.Tensor) -> list[float] | list[bool]:
    return value.detach().cpu().tolist()


def _result_dict(result: AdvantageResult) -> dict[str, list[float] | list[bool]]:
    return {
        "advantage": _tensor_list(result.advantage),
        "group_mean": _tensor_list(result.group_mean),
        "group_std": _tensor_list(result.group_std),
        "gate_mask": _tensor_list(result.gate_mask),
    }


def _comparison(lengths: list[int]) -> dict[str, object]:
    paper = global_advantage_paper(REWARDS, GROUP_IDS)
    official = global_advantage_official_compatible(REWARDS, GROUP_IDS, _mask(lengths))
    difference = official.advantage - paper.advantage
    sign_disagreement = torch.sign(official.advantage) != torch.sign(paper.advantage)
    return {
        "setup": {
            "scalar_rewards": _tensor_list(REWARDS),
            "group_ids": GROUP_IDS,
            "response_lengths": lengths,
        },
        "paper": _result_dict(paper),
        "official_compatible": _result_dict(official),
        "metrics": {
            "adv_l1_diff_official_vs_paper": float(difference.abs().mean().item()),
            "adv_linf_diff_official_vs_paper": float(difference.abs().max().item()),
            "adv_sign_disagreement_rate": float(sign_disagreement.float().mean().item()),
        },
        "scope_note": (
            "official_compatible reproduces the equations in the pinned ray_trainer.py path; "
            "direct import/execution of the Ray trainer remains a Linux-stack check"
        ),
    }


def _sigma_gate_results() -> dict[str, object]:
    cases = {
        "equal": torch.tensor([1.0, 1.0, 1.0, 1.0]),
        "nearly_equal": torch.tensor([1.0, 1.0, 1.0001, 1.0]),
        "normal_variance": REWARDS,
    }
    threshold = 1e-3
    output: dict[str, object] = {
        "threshold": threshold,
        "threshold_source": "synthetic audit value; not claimed as a paper default",
        "cases": {},
    }
    for name, rewards in cases.items():
        ungated = global_advantage_paper(rewards, GROUP_IDS)
        gated = global_advantage_paper(
            rewards,
            GROUP_IDS,
            sigma_gate_threshold=threshold,
        )
        output["cases"][name] = {
            "rewards": _tensor_list(rewards),
            "ungated": _result_dict(ungated),
            "gated": _result_dict(gated),
        }
    return output


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "audit" / "task1_5",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    equal = _comparison([8, 8, 8, 8])
    unequal = _comparison([2, 8, 4, 10])
    sigma = _sigma_gate_results()

    _write_json(args.output_dir / "advantage_equal_length.json", equal)
    _write_json(args.output_dir / "advantage_unequal_length.json", unequal)
    _write_json(args.output_dir / "sigma_gate_results.json", sigma)

    environment = (
        f"platform: {platform.platform()}\n"
        f"python: {platform.python_version()}\n"
        f"torch: {torch.__version__}\n"
        "device: CPU\n"
        "upstream_commit: 0ee9abd903cb4ac4945f1176e943d20436470096\n"
        "official_execution: source-equation characterization; direct Ray trainer execution pending Linux smoke\n"
    )
    (args.output_dir / "environment.txt").write_text(environment, encoding="utf-8")
    print(f"Wrote advantage audit artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
