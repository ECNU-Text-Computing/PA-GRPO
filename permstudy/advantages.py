"""Pure response-level advantage functions used by fidelity audits."""

from dataclasses import dataclass
from typing import Hashable, Sequence

import torch


@dataclass(frozen=True)
class AdvantageResult:
    """Per-sample advantage and the group statistics used to produce it."""

    advantage: torch.Tensor
    group_mean: torch.Tensor
    group_std: torch.Tensor
    gate_mask: torch.Tensor


def _prepare_inputs(
    rewards: torch.Tensor | Sequence[float], group_ids: Sequence[Hashable]
) -> tuple[torch.Tensor, list[Hashable]]:
    rewards_tensor = torch.as_tensor(rewards)
    if rewards_tensor.ndim != 1:
        raise ValueError(f"rewards must be one-dimensional, got shape {tuple(rewards_tensor.shape)}")
    if not rewards_tensor.is_floating_point():
        rewards_tensor = rewards_tensor.float()
    groups = list(group_ids)
    if len(groups) != rewards_tensor.numel():
        raise ValueError(f"rewards/group_ids length mismatch: {rewards_tensor.numel()} != {len(groups)}")
    return rewards_tensor, groups


def _positions_by_group(group_ids: Sequence[Hashable]) -> dict[Hashable, list[int]]:
    positions: dict[Hashable, list[int]] = {}
    for position, group_id in enumerate(group_ids):
        try:
            positions.setdefault(group_id, []).append(position)
        except TypeError as exc:
            raise TypeError(f"group_id at position {position} is not hashable: {group_id!r}") from exc
    return positions


def _population_stats(
    values: torch.Tensor, group_ids: Sequence[Hashable]
) -> tuple[torch.Tensor, torch.Tensor]:
    means = torch.empty_like(values)
    stds = torch.empty_like(values)
    for positions in _positions_by_group(group_ids).values():
        index = torch.tensor(positions, device=values.device, dtype=torch.long)
        group_values = values.index_select(0, index)
        mean = group_values.mean()
        variance = ((group_values - mean) ** 2).mean()
        means.index_fill_(0, index, mean)
        stds.index_fill_(0, index, torch.sqrt(torch.clamp_min(variance, 0.0)))
    return means, stds


def global_advantage_paper(
    rewards: torch.Tensor | Sequence[float],
    group_ids: Sequence[Hashable],
    *,
    eps: float = 1e-6,
    sigma_gate_threshold: float | None = None,
) -> AdvantageResult:
    """Compute the paper-form response-level global advantage.

    Standard deviation is the population standard deviation (``correction=0``
    semantics). When configured, the sigma gate is applied per group using the
    paper's strict ``sigma < threshold`` condition.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    if sigma_gate_threshold is not None and sigma_gate_threshold < 0:
        raise ValueError("sigma_gate_threshold must be non-negative or None")

    rewards_tensor, groups = _prepare_inputs(rewards, group_ids)
    means, stds = _population_stats(rewards_tensor, groups)
    advantage = (rewards_tensor - means) / (stds + eps)

    if sigma_gate_threshold is None:
        gate_mask = torch.zeros_like(rewards_tensor, dtype=torch.bool)
    else:
        gate_mask = stds < sigma_gate_threshold
        advantage = torch.where(gate_mask, torch.zeros_like(advantage), advantage)

    return AdvantageResult(advantage=advantage, group_mean=means, group_std=stds, gate_mask=gate_mask)

def global_advantage_official_compatible(
    rewards: torch.Tensor | Sequence[float],
    group_ids: Sequence[Hashable],
    response_mask: torch.Tensor,
    *,
    variance_floor: float = 1e-6,
    clip_value: float = 5.0,
) -> AdvantageResult:
    """Reproduce the current PA repository's two-stage GRPO path.

    This is a characterization helper, not the controlled implementation. It
    models the current sequence: group-center scalar outcomes, broadcast through
    the response mask, mean over the padded token dimension, then apply the PA
    population-standardized group baseline and clipping.
    """

    if variance_floor < 0:
        raise ValueError("variance_floor must be non-negative")
    if clip_value <= 0:
        raise ValueError("clip_value must be positive")

    rewards_tensor, groups = _prepare_inputs(rewards, group_ids)
    mask = torch.as_tensor(response_mask, device=rewards_tensor.device, dtype=rewards_tensor.dtype)
    if mask.ndim != 2 or mask.shape[0] != rewards_tensor.numel():
        raise ValueError(
            "response_mask must have shape [num_rewards, response_length], "
            f"got {tuple(mask.shape)}"
        )

    first_stage_mean = torch.empty_like(rewards_tensor)
    for positions in _positions_by_group(groups).values():
        index = torch.tensor(positions, device=rewards_tensor.device, dtype=torch.long)
        group_rewards = rewards_tensor.index_select(0, index)
        mean = group_rewards.mean() if len(positions) > 1 else rewards_tensor.new_tensor(0.0)
        first_stage_mean.index_fill_(0, index, mean)

    centered_token_returns = (rewards_tensor - first_stage_mean).unsqueeze(-1) * mask
    collapsed_returns = centered_token_returns.reshape(rewards_tensor.shape[0], -1).mean(dim=-1)

    means, raw_stds = _population_stats(collapsed_returns, groups)
    stds = torch.sqrt(torch.clamp(raw_stds**2, min=variance_floor))
    advantage = torch.clamp((collapsed_returns - means) / stds.clamp_min(1e-6), -clip_value, clip_value)
    gate_mask = torch.zeros_like(rewards_tensor, dtype=torch.bool)
    return AdvantageResult(advantage=advantage, group_mean=means, group_std=stds, gate_mask=gate_mask)
