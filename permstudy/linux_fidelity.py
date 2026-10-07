"""Linux-only harness for the real PA-GRPO advantage path."""

from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
import torch

from permstudy.advantages import global_advantage_official_compatible, global_advantage_paper


@dataclass(frozen=True)
class OfficialAdvantageFidelityResult:
    official_real: torch.Tensor
    official_compatible: torch.Tensor
    paper: torch.Tensor
    original_grpo: torch.Tensor
    pair_baseline_metrics: dict[str, float | None]
    trainer_output: str


def _masked_response_scalar(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum(dim=-1) / mask.sum(dim=-1)


def json_safe_metric(value: float) -> float | None:
    """Convert a numeric diagnostic to strict-JSON-compatible form."""

    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def run_official_advantage_case(
    rewards: torch.Tensor | Sequence[float],
    lengths: Sequence[int],
    group_ids: Sequence[Hashable],
) -> OfficialAdvantageFidelityResult:
    """Run the pinned repository's real DataProto GRPO dispatch on CPU."""

    from verl import DataProto
    from verl.trainer.ppo import core_algos, ray_trainer
    from verl.trainer.ppo.core_algos import AdvantageEstimator

    rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
    lengths_list = [int(length) for length in lengths]
    groups = list(group_ids)
    if rewards_tensor.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    if len(lengths_list) != rewards_tensor.numel() or len(groups) != rewards_tensor.numel():
        raise ValueError("rewards, lengths, and group_ids must have equal length")
    if not lengths_list or min(lengths_list) <= 0:
        raise ValueError("response lengths must be positive")

    width = max(lengths_list)
    response_mask = torch.zeros((len(lengths_list), width), dtype=torch.float32)
    token_level_rewards = torch.zeros_like(response_mask)
    for row, (reward, length) in enumerate(zip(rewards_tensor, lengths_list)):
        response_mask[row, :length] = 1.0
        token_level_rewards[row, length - 1] = reward

    original_advantages, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards.clone(),
        response_mask=response_mask,
        index=np.asarray(groups),
        norm_adv_by_std_in_grpo=False,
    )
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": token_level_rewards,
            "response_mask": response_mask,
        },
        non_tensors={"uid": np.asarray(groups)},
        meta_info={},
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result_data = ray_trainer.compute_advantage(data, AdvantageEstimator.GRPO, config={})
    trainer_output = stdout.getvalue() + stderr.getvalue()
    if "fallback to original GRPO" in trainer_output:
        raise RuntimeError(f"official pair-baseline path fell back:\n{trainer_output}")

    pair_metrics = result_data.meta_info.get("pair_baseline_metrics")
    if not pair_metrics:
        raise RuntimeError("pair_baseline_metrics missing; PA pair-baseline execution was not proven")

    official_real = _masked_response_scalar(result_data.batch["advantages"], response_mask)
    original_grpo = _masked_response_scalar(original_advantages, response_mask)
    paper = global_advantage_paper(rewards_tensor, groups)
    compatible = global_advantage_official_compatible(rewards_tensor, groups, response_mask)

    return OfficialAdvantageFidelityResult(
        official_real=official_real,
        official_compatible=compatible.advantage,
        paper=paper.advantage,
        original_grpo=original_grpo,
        pair_baseline_metrics={key: json_safe_metric(value) for key, value in pair_metrics.items()},
        trainer_output=trainer_output,
    )
