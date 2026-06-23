from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from .reward_computer import dual_potential_reward, quantile_quadratic_reward
from .support_summary import (
    QuantileSupportSummary,
    build_dual_potential_grid,
    build_quantile_summary,
)

try:
    from leanrl.il_utils import build_x
except ImportError:
    from il_utils import build_x


@dataclass
class RewardSnapshot:
    projection_family: str
    space: str
    phase_mode: str
    reward_mode: str
    reward_scale: float
    policy_summary: QuantileSupportSummary
    expert_summary: QuantileSupportSummary
    potential_grid: torch.Tensor | None
    critic: nn.Module | None = None
    linear_directions: torch.Tensor | None = None
    created_at_step: int = 0
    policy_sample_count: int = 0
    expert_sample_count: int = 0

    @torch.no_grad()
    def project(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        next_phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = build_x(
            observations,
            actions,
            next_observations,
            space=self.space,
            phase=phase,
            next_phase=next_phase,
            phase_mode=self.phase_mode,
        )
        if self.projection_family == "critic":
            if self.critic is None:
                raise RuntimeError("critic snapshot is missing")
            z = self.critic(x)
        elif self.projection_family == "linear":
            if self.linear_directions is None:
                raise RuntimeError("linear projection directions are missing")
            z = x @ self.linear_directions.transpose(0, 1)
        else:
            raise ValueError(f"Unknown projection_family={self.projection_family}")
        if z.ndim == 1:
            z = z.unsqueeze(-1)
        return z

    @torch.no_grad()
    def rewards(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        next_phase: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.project(
            observations,
            actions,
            next_observations,
            phase=phase,
            next_phase=next_phase,
        )
        if self.reward_mode == "quantile_quadratic":
            reward, residual = quantile_quadratic_reward(
                z, self.policy_summary, self.expert_summary
            )
            stats = {
                "residual_abs_mean": residual.abs().mean(),
                "projected_mean": self.policy_summary.normalize_queries(z).mean(),
                "projected_std": self.policy_summary.normalize_queries(z).std(
                    unbiased=False
                ),
            }
            return self.reward_scale * reward, stats
        if self.reward_mode == "dual_potential":
            if self.potential_grid is None:
                raise RuntimeError("dual potential grid is missing")
            reward = dual_potential_reward(z, self.policy_summary, self.potential_grid)
            stats = {
                "residual_abs_mean": torch.tensor(0.0, device=z.device),
                "projected_mean": self.policy_summary.normalize_queries(z).mean(),
                "projected_std": self.policy_summary.normalize_queries(z).std(
                    unbiased=False
                ),
            }
            return self.reward_scale * reward, stats
        raise ValueError(f"Unknown reward_mode={self.reward_mode}")

    def state_dict(self) -> dict[str, object]:
        critic_state = None
        if self.critic is not None:
            critic_state = copy.deepcopy(self.critic.state_dict())
        return {
            "projection_family": self.projection_family,
            "space": self.space,
            "phase_mode": self.phase_mode,
            "reward_mode": self.reward_mode,
            "reward_scale": float(self.reward_scale),
            "policy_summary": self.policy_summary.state_dict(),
            "expert_summary": self.expert_summary.state_dict(),
            "potential_grid": self.potential_grid,
            "critic_state_dict": critic_state,
            "linear_directions": self.linear_directions,
            "created_at_step": int(self.created_at_step),
            "policy_sample_count": int(self.policy_sample_count),
            "expert_sample_count": int(self.expert_sample_count),
        }


@torch.no_grad()
def build_reward_snapshot(
    projection_family: str,
    space: str,
    phase_mode: str,
    reward_mode: str,
    reward_scale: float,
    normalize_mode: str,
    num_quantiles: int,
    stat_eps: float,
    policy_batch: dict[str, torch.Tensor],
    expert_batch: dict[str, torch.Tensor],
    created_at_step: int,
    critic: nn.Module | None = None,
    linear_directions: torch.Tensor | None = None,
) -> RewardSnapshot:
    x_policy = build_x(
        policy_batch["observations"],
        policy_batch["actions"],
        policy_batch.get("next_observations", None),
        space=space,
        phase=policy_batch.get("phase", None),
        next_phase=policy_batch.get("next_phase", None),
        phase_mode=phase_mode,
    )
    x_expert = build_x(
        expert_batch["observations"],
        expert_batch["actions"],
        expert_batch.get("next_observations", None),
        space=space,
        phase=expert_batch.get("phase", None),
        next_phase=expert_batch.get("next_phase", None),
        phase_mode=phase_mode,
    )

    critic_snapshot = None
    if projection_family == "critic":
        if critic is None:
            raise ValueError("critic must be provided for critic snapshots")
        critic_snapshot = copy.deepcopy(critic).eval()
        critic_snapshot.requires_grad_(False)
        z_policy = critic_snapshot(x_policy)
        z_expert = critic_snapshot(x_expert)
    elif projection_family == "linear":
        if linear_directions is None:
            raise ValueError("linear_directions must be provided for linear snapshots")
        z_policy = x_policy @ linear_directions.transpose(0, 1)
        z_expert = x_expert @ linear_directions.transpose(0, 1)
    else:
        raise ValueError(f"Unknown projection_family={projection_family}")

    if z_policy.ndim == 1:
        z_policy = z_policy.unsqueeze(-1)
    if z_expert.ndim == 1:
        z_expert = z_expert.unsqueeze(-1)

    policy_summary = build_quantile_summary(
        z_policy,
        num_quantiles=num_quantiles,
        normalize_mode=normalize_mode,
        stat_eps=stat_eps,
    )

    if normalize_mode == "snapshot_zscore":
        z_expert = (z_expert - policy_summary.mean.unsqueeze(0)) / (
            policy_summary.std.unsqueeze(0) + stat_eps
        )
        expert_summary = build_quantile_summary(
            z_expert,
            num_quantiles=num_quantiles,
            normalize_mode="none",
            stat_eps=stat_eps,
        )
    else:
        expert_summary = build_quantile_summary(
            z_expert,
            num_quantiles=num_quantiles,
            normalize_mode="none",
            stat_eps=stat_eps,
        )

    potential_grid = None
    if reward_mode == "dual_potential":
        potential_grid = build_dual_potential_grid(policy_summary, expert_summary)

    return RewardSnapshot(
        projection_family=projection_family,
        space=space,
        phase_mode=phase_mode,
        reward_mode=reward_mode,
        reward_scale=reward_scale,
        policy_summary=policy_summary,
        expert_summary=expert_summary,
        potential_grid=potential_grid,
        critic=critic_snapshot,
        linear_directions=linear_directions,
        created_at_step=created_at_step,
        policy_sample_count=int(x_policy.shape[0]),
        expert_sample_count=int(x_expert.shape[0]),
    )


@torch.no_grad()
def compute_snapshot_queue_rewards(
    snapshots: Sequence[RewardSnapshot],
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_observations: torch.Tensor | None = None,
    phase: torch.Tensor | None = None,
    next_phase: torch.Tensor | None = None,
    weighting: str = "uniform",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if len(snapshots) == 0:
        raise RuntimeError("No reward snapshots are available")

    reward_parts = []
    residual_abs_means = []
    projected_means = []
    projected_stds = []
    ages = []

    if weighting == "uniform":
        weights = torch.ones(len(snapshots), device=observations.device)
    elif weighting == "exponential":
        weights = torch.tensor(
            [0.5 ** (len(snapshots) - 1 - i) for i in range(len(snapshots))],
            device=observations.device,
            dtype=observations.dtype,
        )
    else:
        raise ValueError(f"Unknown weighting={weighting}")
    weights = weights / weights.sum()

    for idx, snapshot in enumerate(snapshots):
        reward_k, stats_k = snapshot.rewards(
            observations,
            actions,
            next_observations,
            phase=phase,
            next_phase=next_phase,
        )
        reward_parts.append(reward_k)
        residual_abs_means.append(stats_k["residual_abs_mean"])
        projected_means.append(stats_k["projected_mean"])
        projected_stds.append(stats_k["projected_std"])
        ages.append(torch.tensor(float(snapshot.created_at_step), device=observations.device))

    rewards = torch.stack(reward_parts, dim=0)
    reward = (weights.view(-1, 1, 1) * rewards).sum(dim=0)
    stats = {
        "residual_abs_mean": torch.stack(residual_abs_means).mean(),
        "projected_mean": torch.stack(projected_means).mean(),
        "projected_std": torch.stack(projected_stds).mean(),
        "snapshot_created_at_mean": torch.stack(ages).mean(),
    }
    return reward, stats
