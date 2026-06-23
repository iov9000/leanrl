from __future__ import annotations

import torch

from .support_summary import QuantileSupportSummary


def quantile_quadratic_reward(
    projected_values: torch.Tensor,
    policy_summary: QuantileSupportSummary,
    expert_summary: QuantileSupportSummary,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = policy_summary.normalize_queries(projected_values)
    q = policy_summary.quantile_positions(projected_values)
    target = expert_summary.inverse_cdf(q)
    residual = z - target
    reward = -(residual * residual).mean(dim=1, keepdim=True)
    return reward, residual


def dual_potential_reward(
    projected_values: torch.Tensor,
    policy_summary: QuantileSupportSummary,
    potential_grid: torch.Tensor,
) -> torch.Tensor:
    phi = policy_summary.interp_on_support(projected_values, potential_grid)
    return -phi.mean(dim=1, keepdim=True)
