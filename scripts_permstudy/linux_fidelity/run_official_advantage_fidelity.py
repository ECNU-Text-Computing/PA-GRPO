"""Generate Task 2A official-real Linux fidelity artifacts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import ray
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from permstudy.linux_fidelity import run_official_advantage_case  # noqa: E402


def _as_list(value: torch.Tensor) -> list[float]:
    return value.detach().cpu().tolist()


def _difference_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    difference = left - right
    return {
        "l1_difference": float(difference.abs().mean().item()),
        "linf_difference": float(difference.abs().max().item()),
        "sign_disagreement_rate": float((torch.sign(left) != torch.sign(right)).float().mean().item()),
    }


def _case(lengths: list[int]) -> dict[str, object]:
    rewards = [1.0, 1.0, -1.0, -1.0]
    group_ids = [0, 0, 0, 0]
    result = run_official_advantage_case(rewards, lengths, group_ids)
    return {
        "setup": {
            "scalar_rewards": rewards,
            "response_lengths": lengths,
            "group_ids": group_ids,
        },
        "official_real": _as_list(result.official_real),
        "official_compatible": _as_list(result.official_compatible),
        "paper": _as_list(result.paper),
        "original_grpo": _as_list(result.original_grpo),
        "path_proof": {
            "fallback_message_present": "fallback to original GRPO" in result.trainer_output,
            "pair_baseline_metrics_present": bool(result.pair_baseline_metrics),
            "pair_baseline_metrics": result.pair_baseline_metrics,
            "sentinel_test": "tests_permstudy/test_linux_official_advantage.py",
        },
        "metrics": {
            "official_real_vs_compatible": _difference_metrics(
                result.official_real, result.official_compatible
            ),
            "official_real_vs_paper": _difference_metrics(result.official_real, result.paper),
            "official_real_vs_original_grpo": _difference_metrics(
                result.official_real, result.original_grpo
            ),
        },
    }


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _command_output(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    output = completed.stdout.strip() or completed.stderr.strip()
    return output if output else f"unavailable (exit={completed.returncode})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--git-commit",
        default=None,
        help="Explicit source commit, useful when a Windows-created worktree is executed through WSL.",
    )
    args = parser.parse_args()
    output_dir = REPO_ROOT / "artifacts" / "linux_fidelity"
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        "advantage_equal_length.json": _case([8, 8, 8, 8]),
        "advantage_unequal_length.json": _case([2, 8, 4, 10]),
    }
    for filename, payload in cases.items():
        (output_dir / filename).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    environment = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "ray": ray.__version__,
        "verl_source": str((REPO_ROOT / "verl").resolve()),
        "vllm": _version("vllm"),
        "gpu": _command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "git_commit": args.git_commit or _command_output(["git", "rev-parse", "HEAD"]),
        "upstream_pa_commit": "0ee9abd903cb4ac4945f1176e943d20436470096",
        "execution_device": "CPU tensors; WSL GPU visibility recorded separately",
    }
    (output_dir / "environment.txt").write_text(
        "".join(f"{key}: {value}\n" for key, value in environment.items()),
        encoding="utf-8",
    )
    print(f"Wrote Linux fidelity artifacts to {output_dir}")


if __name__ == "__main__":
    main()
