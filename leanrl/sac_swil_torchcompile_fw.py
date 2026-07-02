"""
SAC + regularized projected SWIL with an optional FW-adjacent outer loop.

This file is a sibling to the existing LeanRL SAC+SWIL scripts. The inner SAC
update remains standard and compile-friendly; the control-flow-heavy outer loop
for occupancy mixing, candidate snapshots, and coarse line search stays in
plain Python outside compiled regions.
"""

import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import copy
import math
import pickle
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import TensorDict, from_modules
from tensordict.nn import CudaGraphModule
from torchrl.data import LazyTensorStorage, ReplayBuffer

try:
    from leanrl.il_utils import (
        build_x,
        compute_demo_phases,
        compute_online_phases,
        phase_feature_dim,
        validate_phase_mode,
    )
    from leanrl.irl.swil.reward_computer import quantile_quadratic_reward
    from leanrl.irl.swil.support_summary import (
        QuantileSupportSummary,
        build_quantile_summary,
    )
except ImportError:
    from il_utils import (
        build_x,
        compute_demo_phases,
        compute_online_phases,
        phase_feature_dim,
        validate_phase_mode,
    )
    from irl.swil.reward_computer import quantile_quadratic_reward
    from irl.swil.support_summary import (
        QuantileSupportSummary,
        build_quantile_summary,
    )


def _first_present(data, keys: list[str]):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _as_feature_array(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D after reshape, got shape {arr.shape}")
    return arr.astype(np.float32)


def _as_done_array(x: Optional[np.ndarray], n: int) -> np.ndarray:
    if x is None:
        return np.zeros((n,), dtype=np.bool_)
    arr = np.asarray(x).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"dones length {arr.shape[0]} does not match obs length {n}")
    return (arr > 0.5).astype(np.bool_)


def _compute_next_obs_from_obs_done(obs: np.ndarray, dones: np.ndarray) -> np.ndarray:
    next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
    done_idx = np.where(dones)[0]
    next_obs[done_idx] = obs[done_idx]
    return next_obs.astype(np.float32)


def load_expert_dataset(
    expert_path: str, n_expert_trajs: int = 0
) -> dict[str, np.ndarray]:
    if not os.path.isfile(expert_path):
        raise FileNotFoundError(f"expert_path not found: {expert_path}")

    ext = os.path.splitext(expert_path)[1].lower()
    if ext == ".npz":
        data = np.load(expert_path, allow_pickle=True)
    elif ext == ".pkl":
        with open(expert_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "all" in data and isinstance(data["all"], dict):
            nested = data["all"]
            if _first_present(data, ["obs", "observations"]) is None:
                data = nested
    else:
        raise ValueError(f"Unsupported expert dataset extension: {ext}")

    obs_raw = _first_present(data, ["obs", "observations"])
    act_raw = _first_present(data, ["actions", "acs", "act"])
    next_obs_raw = _first_present(data, ["next_obs", "next_observations"])
    dones_raw = _first_present(data, ["dones", "done", "terminals"])
    traj_ids_raw = _first_present(
        data, ["traj_ids", "traj_id", "episode_ids", "ep_ids"]
    )

    if obs_raw is None or act_raw is None:
        raise ValueError("Expert dataset must contain obs/observations and actions/acs")

    obs = _as_feature_array(obs_raw, "obs")
    actions = _as_feature_array(act_raw, "actions")
    if actions.shape[0] != obs.shape[0]:
        raise ValueError("Expert obs/actions lengths do not match")

    if n_expert_trajs > 0 and traj_ids_raw is not None:
        traj_ids = np.asarray(traj_ids_raw).reshape(-1)
        keep = np.unique(traj_ids)[:n_expert_trajs]
        mask = np.isin(traj_ids, keep)
        obs = obs[mask]
        actions = actions[mask]
        traj_ids = traj_ids[mask]
        if next_obs_raw is not None:
            next_obs_raw = np.asarray(next_obs_raw)[mask]
        if dones_raw is not None:
            dones_raw = np.asarray(dones_raw)[mask]
    elif traj_ids_raw is not None:
        traj_ids = np.asarray(traj_ids_raw).reshape(-1)
    else:
        traj_ids = None

    dones = _as_done_array(dones_raw, obs.shape[0])
    if next_obs_raw is not None:
        next_obs = _as_feature_array(next_obs_raw, "next_obs")
    else:
        next_obs = _compute_next_obs_from_obs_done(obs, dones)
    phase, next_phase = compute_demo_phases(dones, traj_ids=traj_ids)

    return {
        "obs": obs,
        "actions": actions,
        "next_obs": next_obs,
        "dones": dones,
        "phase": phase,
        "next_phase": next_phase,
    }


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


class SoftQNetwork(nn.Module):
    def __init__(self, n_obs: int, n_act: int, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs + n_act, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc3 = nn.Linear(256, 1, device=device)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0


class Actor(nn.Module):
    def __init__(self, envs, n_obs: int, n_act: int, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc_mean = nn.Linear(256, n_act, device=device)
        self.fc_logstd = nn.Linear(256, n_act, device=device)
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
                device=device,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
                device=device,
            ),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def get_action(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


class RegularizedProjectionCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        output_normalize: bool,
        use_bounded_alpha: bool,
        alpha_min: float,
        alpha_max: float,
        projection_norm_mode: str,
        post_step_max_norm: float,
        eps: float = 1e-6,
        device=None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.output_normalize = output_normalize
        self.use_bounded_alpha = use_bounded_alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.projection_norm_mode = projection_norm_mode
        self.post_step_max_norm = post_step_max_norm
        self.eps = eps

        def make_linear(in_dim: int, out_dim: int):
            layer = nn.Linear(in_dim, out_dim, device=device)
            if self.projection_norm_mode == "spectral_norm":
                layer = nn.utils.parametrizations.spectral_norm(layer)
            return layer

        self.net = nn.Sequential(
            make_linear(input_dim, hidden_dim),
            nn.Tanh(),
            make_linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            make_linear(hidden_dim, output_dim),
        )
        self.alpha_raw = nn.Parameter(torch.zeros(output_dim, device=device))
        self.register_buffer("running_mean", torch.zeros(output_dim, device=device))
        self.register_buffer("running_std", torch.ones(output_dim, device=device))

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def bounded_alpha(self) -> torch.Tensor:
        if self.use_bounded_alpha:
            return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(
                self.alpha_raw
            )
        return F.softplus(self.alpha_raw) + self.alpha_min

    def _normalize(
        self, raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        if not self.output_normalize:
            return raw
        return (raw - mean.unsqueeze(0)) / (std.unsqueeze(0) + self.eps)

    @torch.no_grad()
    def update_running_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.running_mean.copy_(mean.detach())
        self.running_std.copy_(std.detach().clamp_min(self.eps))

    def project_with_running_stats(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.raw(x)
        norm = self._normalize(raw, self.running_mean, self.running_std)
        alpha = self.bounded_alpha()
        proj = norm * alpha.unsqueeze(0)
        return {
            "raw": raw,
            "normalized": norm,
            "projected": proj,
            "mean": self.running_mean,
            "std": self.running_std,
            "alpha": alpha,
        }

    def project_pair(
        self,
        x_policy: torch.Tensor,
        x_expert: torch.Tensor,
        update_running_stats: bool = False,
    ) -> dict[str, torch.Tensor]:
        raw_policy = self.raw(x_policy)
        raw_expert = self.raw(x_expert)
        union_raw = torch.cat([raw_expert, raw_policy], dim=0)
        mean = union_raw.mean(dim=0)
        std = union_raw.std(dim=0, unbiased=False).clamp_min(self.eps)
        if update_running_stats:
            self.update_running_stats(mean, std)

        norm_policy = self._normalize(raw_policy, mean, std)
        norm_expert = self._normalize(raw_expert, mean, std)
        alpha = self.bounded_alpha()
        proj_policy = norm_policy * alpha.unsqueeze(0)
        proj_expert = norm_expert * alpha.unsqueeze(0)
        union_projected = torch.cat([proj_expert, proj_policy], dim=0)
        return {
            "raw_policy": raw_policy,
            "raw_expert": raw_expert,
            "norm_policy": norm_policy,
            "norm_expert": norm_expert,
            "proj_policy": proj_policy,
            "proj_expert": proj_expert,
            "raw_mean": mean,
            "raw_std": std,
            "alpha": alpha,
            "effective_std": union_projected.std(dim=0, unbiased=False),
        }

    def weight_penalty(self) -> torch.Tensor:
        penalty = None
        for module in self.modules():
            if isinstance(module, nn.Linear):
                term = module.weight.pow(2).mean()
                penalty = term if penalty is None else penalty + term
        if penalty is None:
            return self.alpha_raw.new_tensor(0.0)
        return penalty

    @torch.no_grad()
    def post_step_clip_weights(self) -> None:
        if self.projection_norm_mode != "post_step_clip":
            return
        max_norm = max(self.post_step_max_norm, 1e-6)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight
                fro_norm = weight.norm()
                if fro_norm > max_norm:
                    weight.mul_(max_norm / (fro_norm + 1e-12))


def projective_w2_sq(
    policy_proj: torch.Tensor, expert_proj: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if policy_proj.ndim == 1:
        policy_proj = policy_proj.unsqueeze(-1)
    if expert_proj.ndim == 1:
        expert_proj = expert_proj.unsqueeze(-1)
    n = min(policy_proj.shape[0], expert_proj.shape[0])
    pi_sorted, pi_idx = torch.sort(policy_proj[:n], dim=0)
    ex_sorted, _ = torch.sort(expert_proj[:n], dim=0)
    residual = pi_sorted - ex_sorted
    reward_sorted = -(residual * residual)
    rewards_per_proj = torch.empty_like(reward_sorted)
    rewards_per_proj.scatter_(0, pi_idx, reward_sorted)
    rewards = rewards_per_proj.mean(dim=1, keepdim=True)
    w2_sq = (residual * residual).mean()
    return w2_sq, rewards


def normalize_rewards_for_replay(reward: torch.Tensor, args) -> torch.Tensor:
    if not args.reward_normalize_before_replay:
        return reward
    scale = max(args.reward_clip_scale, 1e-6)
    return torch.tanh(reward / scale) * scale


def clone_actor(envs, n_obs: int, n_act: int, device: torch.device) -> Actor:
    actor = Actor(envs, n_obs=n_obs, n_act=n_act, device=device)
    actor.eval()
    actor.requires_grad_(False)
    return actor


@torch.no_grad()
def copy_actor_(src: Actor, dst: Actor) -> None:
    dst.load_state_dict(copy.deepcopy(src.state_dict()))
    dst.eval()
    dst.requires_grad_(False)


@torch.no_grad()
def ema_actor_(dst: Actor, src: Actor, mix: float) -> None:
    mix = float(np.clip(mix, 0.0, 1.0))
    dst_state = dst.state_dict()
    src_state = src.state_dict()
    for key, value in dst_state.items():
        if torch.is_floating_point(value):
            value.lerp_(src_state[key], mix)
        else:
            value.copy_(src_state[key])
    dst.load_state_dict(dst_state)
    dst.eval()
    dst.requires_grad_(False)


@torch.no_grad()
def select_mixed_rollout_actions(
    obs: torch.Tensor,
    pi_accept: Actor,
    pi_cand: Actor,
    use_candidate: torch.Tensor,
) -> torch.Tensor:
    actions = torch.empty(
        (obs.shape[0], pi_accept.action_scale.shape[0]),
        device=obs.device,
        dtype=obs.dtype,
    )
    accept_idx = (~use_candidate).nonzero(as_tuple=False).flatten()
    cand_idx = use_candidate.nonzero(as_tuple=False).flatten()
    if accept_idx.numel() > 0:
        actions[accept_idx] = pi_accept.get_action(obs[accept_idx])[0]
    if cand_idx.numel() > 0:
        actions[cand_idx] = pi_cand.get_action(obs[cand_idx])[0]
    return actions


def assign_candidate_mask(
    num_envs: int,
    eta: float,
    device: torch.device,
) -> torch.Tensor:
    eta = float(np.clip(eta, 0.0, 1.0))
    return torch.rand(num_envs, device=device) < eta


@torch.no_grad()
def collect_policy_measurements(
    actor: Actor,
    env_id: str,
    seed: int,
    batch_size: int,
    num_batches: int,
    device: torch.device,
    space: str,
    phase_mode: str,
    time_horizon: int,
) -> torch.Tensor:
    env = gym.vector.SyncVectorEnv([make_env(env_id, seed, 0, False, "fw_eval")])
    obs_np, _ = env.reset(seed=seed)
    obs = torch.as_tensor(obs_np, device=device, dtype=torch.float32)
    episode_steps = torch.zeros(obs.shape[0], device=device, dtype=torch.float32)
    xs = []
    total_steps = batch_size * num_batches
    steps = 0
    while steps < total_steps:
        action = actor.get_action(obs)[0]
        next_obs_np, _, terminations, truncations, infos = env.step(
            action.cpu().numpy()
        )
        next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
        real_next_obs = next_obs.clone()
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if infos["_final_observation"][idx]:
                    real_next_obs[idx] = torch.as_tensor(
                        final_obs, device=device, dtype=torch.float32
                    )
        dones = torch.as_tensor(
            terminations | truncations, device=device, dtype=torch.bool
        )
        phase = None
        next_phase = None
        if phase_mode != "none":
            phase, next_phase = compute_online_phases(
                episode_steps,
                horizon=time_horizon,
                done=dones,
            )
        x = build_x(
            obs,
            action,
            real_next_obs if space == "sas" else None,
            space=space,
            phase=phase,
            next_phase=next_phase,
            phase_mode=phase_mode,
        )
        xs.append(x)
        obs = next_obs
        episode_steps = torch.where(
            dones,
            torch.zeros_like(episode_steps),
            episode_steps + 1.0,
        )
        steps += x.shape[0]
    env.close()
    return torch.cat(xs, dim=0)[:total_steps]


@torch.no_grad()
def sample_expert_measurements(
    expert_transitions: TensorDict,
    sample_size: int,
    space: str,
    phase_mode: str,
) -> torch.Tensor:
    n = expert_transitions["observations"].shape[0]
    idx = torch.randint(0, n, (sample_size,), device=expert_transitions.device)
    return build_x(
        expert_transitions["observations"][idx],
        expert_transitions["actions"][idx],
        expert_transitions["next_observations"][idx] if space == "sas" else None,
        space=space,
        phase=expert_transitions["phase"][idx],
        next_phase=expert_transitions["next_phase"][idx],
        phase_mode=phase_mode,
    )


@torch.no_grad()
def mixed_measurements(
    accept_x: torch.Tensor,
    cand_x: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    total = min(accept_x.shape[0], cand_x.shape[0])
    n_cand = int(round(float(np.clip(eta, 0.0, 1.0)) * total))
    n_accept = total - n_cand
    accept_idx = torch.randperm(accept_x.shape[0], device=accept_x.device)[:n_accept]
    cand_idx = torch.randperm(cand_x.shape[0], device=cand_x.device)[:n_cand]
    parts = []
    if n_accept > 0:
        parts.append(accept_x[accept_idx])
    if n_cand > 0:
        parts.append(cand_x[cand_idx])
    if not parts:
        return accept_x[:0]
    return torch.cat(parts, dim=0)


@torch.no_grad()
def estimate_sw_objective(
    critic: RegularizedProjectionCritic,
    expert_x: torch.Tensor,
    policy_x: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    projected = critic.project_pair(policy_x, expert_x, update_running_stats=False)
    w2_sq, _ = projective_w2_sq(projected["proj_policy"], projected["proj_expert"])
    if mode in {"sw2", "msw2", "current_projected_objective"}:
        return 0.5 * w2_sq
    raise ValueError(f"Unknown line_search_objective={mode}")


@torch.no_grad()
def coarse_line_search_eta(
    critic: RegularizedProjectionCritic,
    expert_x: torch.Tensor,
    accept_x: torch.Tensor,
    cand_x: torch.Tensor,
    eta_grid: list[float],
    mode: str,
) -> tuple[float, dict[str, float]]:
    best_eta = float(eta_grid[0])
    best_obj = None
    metrics: dict[str, float] = {}
    for eta in eta_grid:
        mix_x = mixed_measurements(accept_x, cand_x, eta)
        obj = float(estimate_sw_objective(critic, expert_x, mix_x, mode).cpu().item())
        metrics[f"eta_{eta:.2f}"] = obj
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_eta = float(eta)
    metrics["best_objective"] = float(best_obj if best_obj is not None else 0.0)
    return best_eta, metrics


@torch.no_grad()
def compute_queue_rewards(
    critic_queue: Deque[RegularizedProjectionCritic],
    critic: RegularizedProjectionCritic,
    policy_batch: TensorDict,
    expert_batch: TensorDict,
    space: str,
    phase_mode: str,
    reward_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    critics = list(critic_queue) if len(critic_queue) > 0 else [critic]
    x_pi = build_x(
        policy_batch["observations"],
        policy_batch["actions"],
        policy_batch["next_observations"] if space == "sas" else None,
        space=space,
        phase=policy_batch["phase"],
        next_phase=policy_batch["next_phase"],
        phase_mode=phase_mode,
    )
    x_exp = build_x(
        expert_batch["observations"],
        expert_batch["actions"],
        expert_batch["next_observations"] if space == "sas" else None,
        space=space,
        phase=expert_batch["phase"],
        next_phase=expert_batch["next_phase"],
        phase_mode=phase_mode,
    )

    rewards = []
    w2_sq_vals = []
    raw_means = []
    raw_stds = []
    proj_stds = []
    alphas = []
    for critic_k in critics:
        proj_pi = critic_k.project_with_running_stats(x_pi)
        proj_exp = critic_k.project_with_running_stats(x_exp)
        w2_sq, reward_k = projective_w2_sq(proj_pi["projected"], proj_exp["projected"])
        rewards.append(reward_k)
        w2_sq_vals.append(w2_sq)
        raw_union = torch.cat([proj_pi["raw"], proj_exp["raw"]], dim=0)
        proj_union = torch.cat([proj_pi["projected"], proj_exp["projected"]], dim=0)
        raw_means.append(raw_union.mean())
        raw_stds.append(raw_union.std(unbiased=False))
        proj_stds.append(proj_union.std(unbiased=False))
        alphas.append(proj_pi["alpha"].mean())

    reward = reward_scale * torch.stack(rewards, dim=0).mean(dim=0)
    stats = {
        "w2_sq": torch.stack(w2_sq_vals).mean(),
        "raw_mean": torch.stack(raw_means).mean(),
        "raw_std": torch.stack(raw_stds).mean(),
        "projected_std": torch.stack(proj_stds).mean(),
        "alpha": torch.stack(alphas).mean(),
    }
    return reward, stats


def _project_snapshot_batch(
    critic: RegularizedProjectionCritic,
    batch: TensorDict,
    space: str,
    phase_mode: str,
) -> torch.Tensor:
    x = build_x(
        batch["observations"],
        batch["actions"],
        batch["next_observations"] if space == "sas" else None,
        space=space,
        phase=batch["phase"],
        next_phase=batch["next_phase"],
        phase_mode=phase_mode,
    )
    z = critic.project_with_running_stats(x)["projected"]
    if z.ndim == 1:
        z = z.unsqueeze(-1)
    return z


@dataclass
class FWRewardSnapshot:
    critic: RegularizedProjectionCritic
    space: str
    phase_mode: str
    reward_scale: float
    policy_summary: QuantileSupportSummary
    expert_summary: QuantileSupportSummary
    created_at_step: int
    policy_sample_count: int
    expert_sample_count: int

    @torch.no_grad()
    def rewards(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        phase: torch.Tensor,
        next_phase: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "next_observations": next_observations,
                "phase": phase,
                "next_phase": next_phase,
            },
            batch_size=observations.shape[:1],
            device=observations.device,
        )
        z = _project_snapshot_batch(
            self.critic,
            batch=batch,
            space=self.space,
            phase_mode=self.phase_mode,
        )
        reward, residual = quantile_quadratic_reward(
            z,
            self.policy_summary,
            self.expert_summary,
        )
        stats = {
            "residual_abs_mean": residual.abs().mean(),
            "projected_mean": self.policy_summary.normalize_queries(z).mean(),
            "projected_std": self.policy_summary.normalize_queries(z).std(
                unbiased=False
            ),
        }
        return self.reward_scale * reward, stats


@torch.no_grad()
def build_fw_reward_snapshot(
    critic: RegularizedProjectionCritic,
    policy_batch: TensorDict,
    expert_batch: TensorDict,
    space: str,
    phase_mode: str,
    reward_scale: float,
    normalize_mode: str,
    num_quantiles: int,
    stat_eps: float,
    created_at_step: int,
) -> FWRewardSnapshot:
    critic_snapshot = copy.deepcopy(critic).eval()
    critic_snapshot.requires_grad_(False)

    z_policy = _project_snapshot_batch(
        critic_snapshot,
        batch=policy_batch,
        space=space,
        phase_mode=phase_mode,
    )
    z_expert = _project_snapshot_batch(
        critic_snapshot,
        batch=expert_batch,
        space=space,
        phase_mode=phase_mode,
    )

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

    return FWRewardSnapshot(
        critic=critic_snapshot,
        space=space,
        phase_mode=phase_mode,
        reward_scale=reward_scale,
        policy_summary=policy_summary,
        expert_summary=expert_summary,
        created_at_step=created_at_step,
        policy_sample_count=int(z_policy.shape[0]),
        expert_sample_count=int(z_expert.shape[0]),
    )


@torch.no_grad()
def compute_fw_snapshot_queue_rewards(
    snapshots: Sequence[FWRewardSnapshot],
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_observations: torch.Tensor,
    phase: torch.Tensor,
    next_phase: torch.Tensor,
    weighting: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if len(snapshots) == 0:
        raise RuntimeError("No FW reward snapshots are available")

    if weighting == "uniform":
        weights = torch.ones(len(snapshots), device=observations.device)
    elif weighting == "exponential":
        weights = torch.tensor(
            [0.5 ** (len(snapshots) - 1 - i) for i in range(len(snapshots))],
            device=observations.device,
            dtype=observations.dtype,
        )
    else:
        raise ValueError(f"Unknown reward_snapshot_queue_weighting={weighting}")
    weights = weights / weights.sum()

    reward_parts = []
    residual_abs_means = []
    projected_means = []
    projected_stds = []
    snapshot_steps = []
    for snapshot in snapshots:
        reward_k, stats_k = snapshot.rewards(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            phase=phase,
            next_phase=next_phase,
        )
        reward_parts.append(reward_k)
        residual_abs_means.append(stats_k["residual_abs_mean"])
        projected_means.append(stats_k["projected_mean"])
        projected_stds.append(stats_k["projected_std"])
        snapshot_steps.append(
            torch.tensor(float(snapshot.created_at_step), device=observations.device)
        )

    rewards = torch.stack(reward_parts, dim=0)
    reward = (weights.view(-1, 1, 1) * rewards).sum(dim=0)
    latest = snapshots[-1]
    latest_w2_sq = (
        (latest.policy_summary.quantiles - latest.expert_summary.quantiles)
        .pow(2)
        .mean()
    )
    stats = {
        "w2_sq": latest_w2_sq,
        "residual_abs_mean": torch.stack(residual_abs_means).mean(),
        "projected_mean": torch.stack(projected_means).mean(),
        "projected_std": torch.stack(projected_stds).mean(),
        "snapshot_created_at_mean": torch.stack(snapshot_steps).mean(),
        "snapshot_policy_samples": torch.tensor(
            float(latest.policy_sample_count), device=observations.device
        ),
        "snapshot_expert_samples": torch.tensor(
            float(latest.expert_sample_count), device=observations.device
        ),
    }
    return reward, stats


def enqueue_critic_snapshot(
    critic_queue: Deque[RegularizedProjectionCritic],
    critic: RegularizedProjectionCritic,
) -> None:
    snapshot = copy.deepcopy(critic).eval()
    snapshot.requires_grad_(False)
    critic_queue.append(snapshot)


def make_q_params(n_obs: int, n_act: int, device: torch.device):
    qf1 = SoftQNetwork(n_obs=n_obs, n_act=n_act, device=device)
    qf2 = SoftQNetwork(n_obs=n_obs, n_act=n_act, device=device)
    qnet_params = from_modules(qf1, qf2, as_module=True)
    qnet_target = qnet_params.data.clone()
    qnet = SoftQNetwork(n_obs=n_obs, n_act=n_act, device="meta")
    qnet_params.to_module(qnet)
    return qnet_params, qnet_target, qnet


def evaluate_policy(
    actor: Actor, env_id: str, seed: int, n_episodes: int, device: torch.device
) -> float:
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    returns = []
    obs, _ = env.reset(seed=seed)
    while len(returns) < n_episodes:
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = actor.get_action(obs_t)[2]
        next_obs, _, term, trunc, info = env.step(action.cpu().numpy()[0])
        if term or trunc:
            returns.append(float(info["episode"]["r"]))
            obs, _ = env.reset()
        else:
            obs = next_obs
    env.close()
    return float(np.mean(returns)) if returns else 0.0


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False

    env_id: str = "HalfCheetah-v5"
    num_envs: int = 1
    total_timesteps: int = 1_000_000
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 250
    learning_starts: int = 5_000
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True

    compile: bool = True
    cudagraphs: bool = True
    compile_fw_inner_updates: bool = True
    measure_burnin: int = 3

    expert_path: str = "demos/demo_hf_HalfCheetah-v5_10.pkl"
    n_expert_trajs: int = 0
    space: str = "sa"
    imitation_phase_mode: str = "none"
    imitation_time_horizon: int = 0

    critic_hidden: int = 128
    critic_output_dim: int = 1
    critic_lr: float = 3e-4
    critic_updates_per_sac_update: int = 1
    critic_queue_len: int = 10
    critic_queue_update_freq: int = 1
    reward_scale: float = 1.0
    summary_num_quantiles: int = 257
    summary_stat_eps: float = 1e-6
    summary_policy_sample_size: int = 8192
    summary_expert_sample_size: int = 0
    reward_snapshot_refresh_interval: int = 1000
    reward_snapshot_queue_len: int = 1
    reward_snapshot_queue_weighting: str = "uniform"
    reward_snapshot_normalize_mode: str = "snapshot_zscore"

    reward_normalize_before_replay: bool = False
    reward_clip_scale: float = 5.0

    mgsw_output_normalize: bool = True
    mgsw_use_bounded_alpha: bool = True
    mgsw_alpha_min: float = 0.1
    mgsw_alpha_max: float = 5.0
    mgsw_var_reg_coef: float = 1e-2
    mgsw_target_std: float = 1.0
    mgsw_weight_reg_coef: float = 1e-6
    mgsw_projection_norm_mode: str = "weight_penalty"
    mgsw_projection_post_step_max_norm: float = 10.0

    use_fw_adjacent_mixing: bool = False
    outer_update_interval: int = 25_000
    candidate_snapshot_interval: int = 5_000
    eta_mode: str = "fixed"
    fixed_eta: float = 0.25
    eta_grid: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    accept_update_rule: str = "keep_anchor"
    occupancy_eval_batch_size: int = 256
    occupancy_eval_num_batches: int = 4
    line_search_objective: str = "current_projected_objective"
    debug_log_fw_metrics: bool = True

    eval_interval: int = 10_000
    eval_episodes: int = 5
    print_eval: bool = True
    target_eval_return: float = 0.0
    target_eval_metric: str = "max"
    bc_pretrain_steps: int = 0
    bc_batch_size: int = 1024
    bc_eval_episodes: int = 5
    save_model: bool = False
    save_dir: str = "checkpoints"


if __name__ == "__main__":
    args = tyro.cli(Args)
    validate_phase_mode(args.imitation_phase_mode)
    if args.space not in {"s", "sa", "sas"}:
        raise ValueError("--space must be one of s|sa|sas")
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be >= 1")
    if args.eta_mode not in {"fixed", "grid_search"}:
        raise ValueError("--eta_mode must be fixed or grid_search")
    if args.accept_update_rule not in {
        "keep_anchor",
        "promote_candidate",
        "ema_weights",
    }:
        raise ValueError(
            "--accept_update_rule must be keep_anchor, promote_candidate, or ema_weights"
        )
    if args.line_search_objective not in {"sw2", "msw2", "current_projected_objective"}:
        raise ValueError("--line_search_objective is invalid")
    if args.critic_queue_len <= 0:
        raise ValueError("--critic_queue_len must be >= 1")
    if args.target_eval_metric not in {"train", "accept", "max"}:
        raise ValueError("--target_eval_metric must be train, accept, or max")
    if args.bc_pretrain_steps < 0:
        raise ValueError("--bc_pretrain_steps must be >= 0")
    if args.bc_batch_size <= 0:
        raise ValueError("--bc_batch_size must be >= 1")
    if args.bc_eval_episodes <= 0:
        raise ValueError("--bc_eval_episodes must be >= 1")
    if args.summary_num_quantiles <= 1:
        raise ValueError("--summary_num_quantiles must be >= 2")
    if args.summary_policy_sample_size <= 0:
        raise ValueError("--summary_policy_sample_size must be >= 1")
    if args.reward_snapshot_refresh_interval <= 0:
        raise ValueError("--reward_snapshot_refresh_interval must be >= 1")
    if args.reward_snapshot_queue_len <= 0:
        raise ValueError("--reward_snapshot_queue_len must be >= 1")
    if args.reward_snapshot_queue_weighting not in {"uniform", "exponential"}:
        raise ValueError(
            "--reward_snapshot_queue_weighting must be one of uniform|exponential"
        )
    if args.reward_snapshot_normalize_mode not in {"none", "snapshot_zscore"}:
        raise ValueError(
            "--reward_snapshot_normalize_mode must be one of none|snapshot_zscore"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{ts}__{args.exp_name}__{args.env_id}__seed{args.seed}"

    wandb.init(
        project="sac_swil_fw",
        name=run_name,
        config=vars(args),
        save_code=True,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    use_compile = bool(args.compile and args.compile_fw_inner_updates)
    use_cudagraphs = bool(args.cudagraphs and device.type == "cuda")

    effective_norm_mode = args.mgsw_projection_norm_mode
    if use_compile and effective_norm_mode == "spectral_norm":
        print(
            "Falling back from spectral_norm to weight_penalty for compile stability."
        )
        effective_norm_mode = "weight_penalty"

    expert_np = load_expert_dataset(
        args.expert_path, n_expert_trajs=args.n_expert_trajs
    )
    expert_obs = torch.as_tensor(expert_np["obs"], dtype=torch.float32, device=device)
    expert_actions = torch.as_tensor(
        expert_np["actions"], dtype=torch.float32, device=device
    )
    expert_next_obs = torch.as_tensor(
        expert_np["next_obs"], dtype=torch.float32, device=device
    )
    expert_dones = torch.as_tensor(expert_np["dones"], dtype=torch.bool, device=device)
    expert_phase = torch.as_tensor(
        expert_np["phase"], dtype=torch.float32, device=device
    )
    expert_next_phase = torch.as_tensor(
        expert_np["next_phase"], dtype=torch.float32, device=device
    )
    expert_n = expert_obs.shape[0]

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed, idx, args.capture_video, run_name)
            for idx in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    n_obs = int(np.prod(envs.single_observation_space.shape))
    n_act = int(np.prod(envs.single_action_space.shape))
    phase_dim = phase_feature_dim(args.imitation_phase_mode)
    phase_horizon = int(
        args.imitation_time_horizon
        or getattr(envs.envs[0].spec, "max_episode_steps", None)
        or 1000
    )

    actor = Actor(envs, n_obs=n_obs, n_act=n_act, device=device)
    pi_accept = clone_actor(envs, n_obs=n_obs, n_act=n_act, device=device)
    pi_cand = clone_actor(envs, n_obs=n_obs, n_act=n_act, device=device)
    copy_actor_(actor, pi_accept)
    copy_actor_(actor, pi_cand)

    qnet_params, qnet_target, qnet = make_q_params(
        n_obs=n_obs, n_act=n_act, device=device
    )

    x_dim = n_obs
    if args.space in {"sa", "sas"}:
        x_dim += n_act
    if args.space == "sas":
        x_dim += n_obs
    x_dim += phase_dim
    if args.space == "sas":
        x_dim += phase_dim

    critic = RegularizedProjectionCritic(
        input_dim=x_dim,
        hidden_dim=args.critic_hidden,
        output_dim=args.critic_output_dim,
        output_normalize=args.mgsw_output_normalize,
        use_bounded_alpha=args.mgsw_use_bounded_alpha,
        alpha_min=args.mgsw_alpha_min,
        alpha_max=args.mgsw_alpha_max,
        projection_norm_mode=effective_norm_mode,
        post_step_max_norm=args.mgsw_projection_post_step_max_norm,
        device=device,
    )

    actor_optimizer = optim.Adam(
        actor.parameters(),
        lr=args.policy_lr,
        # CudaGraphModule wraps these updates when cudagraphs is enabled, so the
        # underlying optimizers must remain capturable even if torch.compile is on.
        capturable=use_cudagraphs,
    )
    q_optimizer = optim.Adam(qnet.parameters(), lr=args.q_lr, capturable=use_cudagraphs)
    critic_optimizer = optim.Adam(
        critic.parameters(),
        lr=args.critic_lr,
        capturable=use_cudagraphs,
    )

    critic_queue: Deque[RegularizedProjectionCritic] = deque(
        maxlen=args.critic_queue_len
    )
    enqueue_critic_snapshot(critic_queue, critic)
    queue_pushes = 1
    reward_snapshots: Deque[FWRewardSnapshot] = deque(
        maxlen=args.reward_snapshot_queue_len
    )

    if args.autotune:
        target_entropy = -torch.prod(
            torch.tensor(envs.single_action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.detach().exp()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, capturable=use_cudagraphs)
    else:
        alpha = torch.as_tensor(args.alpha, device=device)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))
    expert_rb = ReplayBuffer(storage=LazyTensorStorage(expert_n, device=device))
    expert_transitions = TensorDict(
        observations=expert_obs,
        next_observations=expert_next_obs,
        actions=expert_actions,
        phase=expert_phase,
        next_phase=expert_next_phase,
        rewards=torch.zeros((expert_n, 1), device=device),
        dones=expert_dones,
        terminations=expert_dones,
        truncations=torch.zeros((expert_n,), dtype=torch.bool, device=device),
        batch_size=expert_n,
        device=device,
    )
    expert_rb.extend(expert_transitions)

    def _sample_summary_batch(
        replay: ReplayBuffer,
        sample_size: int,
        fallback: TensorDict | None = None,
    ) -> TensorDict:
        if fallback is not None and (
            sample_size <= 0 or sample_size >= int(fallback["observations"].shape[0])
        ):
            return fallback
        return replay.sample(sample_size)

    def refresh_reward_snapshot(global_step: int) -> FWRewardSnapshot | None:
        if len(rb) < max(args.batch_size, args.summary_policy_sample_size):
            return None
        policy_batch = _sample_summary_batch(rb, args.summary_policy_sample_size)
        expert_batch = _sample_summary_batch(
            expert_rb,
            args.summary_expert_sample_size,
            fallback=expert_transitions
            if args.summary_expert_sample_size <= 0
            else None,
        )
        return build_fw_reward_snapshot(
            critic=critic,
            policy_batch=policy_batch,
            expert_batch=expert_batch,
            space=args.space,
            phase_mode=args.imitation_phase_mode,
            reward_scale=args.reward_scale,
            normalize_mode=args.reward_snapshot_normalize_mode,
            num_quantiles=args.summary_num_quantiles,
            stat_eps=args.summary_stat_eps,
            created_at_step=global_step,
        )

    if args.bc_pretrain_steps > 0:
        actor.train()
        for bc_step in range(args.bc_pretrain_steps):
            idx = torch.randint(0, expert_n, (args.bc_batch_size,), device=device)
            mean, _ = actor(expert_obs[idx])
            pred_actions = torch.tanh(mean) * actor.action_scale + actor.action_bias
            bc_loss = F.mse_loss(pred_actions, expert_actions[idx])
            actor_optimizer.zero_grad(set_to_none=True)
            bc_loss.backward()
            actor_optimizer.step()
        actor.eval()
        copy_actor_(actor, pi_accept)
        copy_actor_(actor, pi_cand)
        bc_eval = evaluate_policy(
            actor, args.env_id, args.seed + 500, args.bc_eval_episodes, device
        )
        print(
            f"bc_pretrain complete steps={args.bc_pretrain_steps} loss={bc_loss.detach().cpu().item():.6f} eval_return={bc_eval:.2f}",
            flush=True,
        )
        wandb.log(
            {
                "bc/loss": float(bc_loss.detach().cpu().item()),
                "bc/eval_return": bc_eval,
            },
            step=0,
        )

    def batched_qf(params, obs, act, next_q_value=None):
        with params.to_module(qnet):
            vals = qnet(obs, act)
            if next_q_value is not None:
                return F.mse_loss(vals.view(-1), next_q_value)
            return vals

    def update_q(data: TensorDict):
        q_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            next_action, next_log_pi, _ = actor.get_action(data["next_observations"])
            qf_next_target = torch.vmap(batched_qf, (0, None, None))(
                qnet_target, data["next_observations"], next_action
            )
            min_qf_next_target = qf_next_target.min(dim=0).values - alpha * next_log_pi
            next_q_value = data["rewards"].flatten() + (
                ~data["dones"].flatten()
            ).float() * args.gamma * min_qf_next_target.view(-1)

        q_losses = torch.vmap(batched_qf, (0, None, None, None))(
            qnet_params, data["observations"], data["actions"], next_q_value
        )
        qf_loss = q_losses.sum(0)
        qf_loss.backward()
        q_optimizer.step()
        return TensorDict({"qf_loss": qf_loss.detach()}, batch_size=[], device=device)

    def update_actor(data: TensorDict):
        actor_optimizer.zero_grad(set_to_none=True)
        pi, log_pi, _ = actor.get_action(data["observations"])
        qf_pi = torch.vmap(batched_qf, (0, None, None))(
            qnet_params.data, data["observations"], pi
        )
        min_qf_pi = qf_pi.min(0).values
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
        actor_loss.backward()
        actor_optimizer.step()

        alpha_loss = torch.tensor(0.0, device=device)
        if args.autotune:
            a_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                _, log_pi2, _ = actor.get_action(data["observations"])
            alpha_loss = (-log_alpha.exp() * (log_pi2 + target_entropy)).mean()
            alpha_loss.backward()
            a_optimizer.step()

        return TensorDict(
            {
                "actor_loss": actor_loss.detach(),
                "alpha_loss": alpha_loss.detach(),
                "alpha": alpha.detach(),
            },
            batch_size=[],
            device=device,
        )

    def update_critic(td: TensorDict):
        critic_optimizer.zero_grad(set_to_none=True)
        x_pi = build_x(
            td["pi_observations"],
            td["pi_actions"],
            td["pi_next_observations"] if args.space == "sas" else None,
            space=args.space,
            phase=td["pi_phase"],
            next_phase=td["pi_next_phase"],
            phase_mode=args.imitation_phase_mode,
        )
        x_exp = build_x(
            td["exp_observations"],
            td["exp_actions"],
            td["exp_next_observations"] if args.space == "sas" else None,
            space=args.space,
            phase=td["exp_phase"],
            next_phase=td["exp_next_phase"],
            phase_mode=args.imitation_phase_mode,
        )
        projected = critic.project_pair(x_pi, x_exp, update_running_stats=True)
        w2_sq, _ = projective_w2_sq(projected["proj_policy"], projected["proj_expert"])

        target_std = torch.full_like(projected["effective_std"], args.mgsw_target_std)
        var_reg = args.mgsw_var_reg_coef * (
            (
                torch.log(projected["effective_std"] + 1e-6)
                - torch.log(target_std + 1e-6)
            )
            .pow(2)
            .mean()
        )
        weight_reg = projected["alpha"].new_tensor(0.0)
        if effective_norm_mode == "weight_penalty":
            weight_reg = args.mgsw_weight_reg_coef * critic.weight_penalty()

        critic_loss = -w2_sq + var_reg + weight_reg
        critic_loss.backward()
        critic_optimizer.step()
        critic.post_step_clip_weights()

        return TensorDict(
            {
                "critic_loss": critic_loss.detach(),
                "w2_sq": w2_sq.detach(),
                "var_reg": var_reg.detach(),
                "weight_reg": weight_reg.detach(),
                "raw_mean": torch.cat(
                    [projected["raw_policy"], projected["raw_expert"]], dim=0
                )
                .mean()
                .detach(),
                "raw_std": torch.cat(
                    [projected["raw_policy"], projected["raw_expert"]], dim=0
                )
                .std(unbiased=False)
                .detach(),
                "projected_std": projected["effective_std"].mean().detach(),
                "alpha_value": projected["alpha"].mean().detach(),
            },
            batch_size=[],
            device=device,
        )

    if use_compile:
        update_q = torch.compile(update_q, mode=None)
        update_actor = torch.compile(update_actor, mode=None)
        update_critic = torch.compile(update_critic, mode=None)

    if use_cudagraphs:
        # The FW training updates perform optimizer/backward work that currently is
        # not capture-safe with TensorDict's CudaGraphModule in this script.
        # Keep the flag and capturable optimizers for future work, but avoid wrapping
        # the mutating update steps so the default compile+cudagraphs setup runs.
        pass

    obs_np, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(obs_np, device=device, dtype=torch.float32)
    episode_steps = torch.zeros(args.num_envs, device=device, dtype=torch.float32)
    episode_use_candidate = assign_candidate_mask(args.num_envs, 0.0, device)
    episode_assignment_total = int(args.num_envs)
    episode_assignment_candidate = int(episode_use_candidate.sum().item())

    eta_current = 0.0
    fw_metrics: dict[str, float] = {}
    last_fw_selected_eta = 0.0
    last_fw_objective = 0.0
    last_critic_out = TensorDict({}, batch_size=[], device=device)
    last_reward_stats = {
        "w2_sq": torch.tensor(0.0, device=device),
        "residual_abs_mean": torch.tensor(0.0, device=device),
        "projected_mean": torch.tensor(0.0, device=device),
        "projected_std": torch.tensor(1.0, device=device),
        "snapshot_created_at_mean": torch.tensor(-1.0, device=device),
        "snapshot_policy_samples": torch.tensor(0.0, device=device),
        "snapshot_expert_samples": torch.tensor(0.0, device=device),
    }

    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    measure_burnin = None
    avg_returns = deque(maxlen=20)
    max_ep_ret = -float("inf")
    desc = ""
    critic_updates = 0

    if args.save_model:
        os.makedirs(args.save_dir, exist_ok=True)

    for global_step in pbar:
        if global_step == args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            with torch.no_grad():
                if args.use_fw_adjacent_mixing:
                    action_t = select_mixed_rollout_actions(
                        obs, pi_accept, pi_cand, episode_use_candidate
                    )
                else:
                    action_t = actor.get_action(obs)[0]
            actions = action_t.cpu().numpy()

        next_obs_np, _, terminations, truncations, infos = envs.step(actions)
        dones_np = terminations | truncations

        if "episode" in infos:
            for r in infos["episode"]["r"][infos["episode"]["_r"]]:
                r = float(r)
                max_ep_ret = max(max_ep_ret, r)
                avg_returns.append(r)
            desc = f"step={global_step}, ep_ret={np.mean(avg_returns):.2f} (max={max_ep_ret:.2f})"

        next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
        real_next_obs = next_obs.clone()
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if infos["_final_observation"][idx]:
                    real_next_obs[idx] = torch.as_tensor(
                        final_obs, device=device, dtype=torch.float32
                    )

        action_t = torch.as_tensor(actions, device=device, dtype=torch.float32)
        done_t = torch.as_tensor(dones_np, device=device, dtype=torch.bool)
        phase, next_phase = compute_online_phases(
            episode_steps,
            horizon=phase_horizon,
            done=done_t,
        )

        if global_step >= args.learning_starts and (
            len(reward_snapshots) == 0
            or global_step % args.reward_snapshot_refresh_interval == 0
        ):
            snapshot = refresh_reward_snapshot(global_step)
            if snapshot is not None:
                reward_snapshots.append(snapshot)

        raw_reward = torch.zeros((args.num_envs, 1), device=device)
        reward_insert = raw_reward
        if global_step >= args.learning_starts and len(reward_snapshots) > 0:
            raw_reward, reward_stats = compute_fw_snapshot_queue_rewards(
                snapshots=list(reward_snapshots),
                observations=obs,
                actions=action_t,
                next_observations=real_next_obs,
                phase=phase,
                next_phase=next_phase,
                weighting=args.reward_snapshot_queue_weighting,
            )
            last_reward_stats = reward_stats
            reward_insert = normalize_rewards_for_replay(raw_reward, args)

        transition = TensorDict(
            {
                "observations": obs,
                "next_observations": real_next_obs,
                "actions": action_t,
                "phase": phase,
                "next_phase": next_phase,
                "rewards": reward_insert,
                "raw_rewards": raw_reward,
                "dones": done_t,
                "terminations": torch.as_tensor(
                    terminations, device=device, dtype=torch.bool
                ),
                "truncations": torch.as_tensor(
                    truncations, device=device, dtype=torch.bool
                ),
            },
            batch_size=obs.shape[0],
            device=device,
        )
        rb.extend(transition)
        obs = next_obs
        episode_steps = torch.where(
            done_t,
            torch.zeros_like(episode_steps),
            episode_steps + 1.0,
        )

        if args.use_fw_adjacent_mixing and done_t.any():
            new_mask = assign_candidate_mask(
                int(done_t.sum().item()), eta_current, device
            )
            episode_use_candidate[done_t] = new_mask
            episode_assignment_total += int(done_t.sum().item())
            episode_assignment_candidate += int(new_mask.sum().item())

        if global_step <= args.learning_starts:
            continue

        data = rb.sample(args.batch_size)

        for _ in range(args.critic_updates_per_sac_update):
            batch_pi = rb.sample(args.batch_size)
            batch_exp = expert_rb.sample(args.batch_size)
            critic_td = TensorDict(
                {
                    "pi_observations": batch_pi["observations"],
                    "pi_actions": batch_pi["actions"],
                    "pi_next_observations": batch_pi["next_observations"],
                    "pi_phase": batch_pi["phase"],
                    "pi_next_phase": batch_pi["next_phase"],
                    "exp_observations": batch_exp["observations"],
                    "exp_actions": batch_exp["actions"],
                    "exp_next_observations": batch_exp["next_observations"],
                    "exp_phase": batch_exp["phase"],
                    "exp_next_phase": batch_exp["next_phase"],
                },
                batch_size=batch_pi["observations"].shape[:1],
                device=device,
            )
            last_critic_out = update_critic(critic_td)
            critic_updates += 1
            if critic_updates % args.critic_queue_update_freq == 0:
                enqueue_critic_snapshot(critic_queue, critic)
                queue_pushes += 1

        out_q = update_q(data)
        out_actor = TensorDict(
            {
                "actor_loss": torch.tensor(0.0, device=device),
                "alpha_loss": torch.tensor(0.0, device=device),
                "alpha": alpha.detach(),
            },
            batch_size=[],
            device=device,
        )
        if global_step % args.policy_frequency == 0:
            for _ in range(args.policy_frequency):
                out_actor = update_actor(data)
                if args.autotune:
                    alpha.copy_(log_alpha.detach().exp())

        if global_step % args.target_network_frequency == 0:
            qnet_target.lerp_(qnet_params.data, args.tau)

        if global_step % args.candidate_snapshot_interval == 0:
            copy_actor_(actor, pi_cand)

        if (
            args.use_fw_adjacent_mixing
            and global_step % args.outer_update_interval == 0
        ):
            copy_actor_(actor, pi_cand)
            eval_steps = (
                args.occupancy_eval_batch_size * args.occupancy_eval_num_batches
            )
            expert_x = sample_expert_measurements(
                expert_transitions,
                eval_steps,
                args.space,
                args.imitation_phase_mode,
            )
            accept_x = collect_policy_measurements(
                actor=pi_accept,
                env_id=args.env_id,
                seed=args.seed + global_step + 11,
                batch_size=args.occupancy_eval_batch_size,
                num_batches=args.occupancy_eval_num_batches,
                device=device,
                space=args.space,
                phase_mode=args.imitation_phase_mode,
                time_horizon=phase_horizon,
            )
            cand_x = collect_policy_measurements(
                actor=pi_cand,
                env_id=args.env_id,
                seed=args.seed + global_step + 29,
                batch_size=args.occupancy_eval_batch_size,
                num_batches=args.occupancy_eval_num_batches,
                device=device,
                space=args.space,
                phase_mode=args.imitation_phase_mode,
                time_horizon=phase_horizon,
            )

            if args.eta_mode == "fixed":
                eta_current = float(np.clip(args.fixed_eta, 0.0, 1.0))
                mix_x = mixed_measurements(accept_x, cand_x, eta_current)
                fw_obj = estimate_sw_objective(
                    critic, expert_x, mix_x, args.line_search_objective
                )
                fw_metrics = {
                    "eta_fixed": eta_current,
                    "best_objective": float(fw_obj.cpu().item()),
                }
            else:
                eta_current, fw_metrics = coarse_line_search_eta(
                    critic=critic,
                    expert_x=expert_x,
                    accept_x=accept_x,
                    cand_x=cand_x,
                    eta_grid=args.eta_grid,
                    mode=args.line_search_objective,
                )

            last_fw_selected_eta = eta_current
            last_fw_objective = fw_metrics.get("best_objective", 0.0)

            # Episode-level policy assignment is the practical occupancy-mixing surrogate.
            episode_use_candidate = assign_candidate_mask(
                args.num_envs, eta_current, device
            )
            episode_assignment_total += args.num_envs
            episode_assignment_candidate += int(episode_use_candidate.sum().item())

            # The line search targets the projected SW objective, not the shaped reward.
            if args.accept_update_rule == "promote_candidate":
                copy_actor_(pi_cand, pi_accept)
            elif args.accept_update_rule == "ema_weights":
                ema_actor_(pi_accept, pi_cand, eta_current)

        if args.eval_interval > 0 and global_step % args.eval_interval == 0:
            eval_train = evaluate_policy(
                actor, args.env_id, args.seed + 1000, args.eval_episodes, device
            )
            eval_accept = evaluate_policy(
                pi_accept, args.env_id, args.seed + 2000, args.eval_episodes, device
            )
            target_eval_value = {
                "train": eval_train,
                "accept": eval_accept,
                "max": max(eval_train, eval_accept),
            }[args.target_eval_metric]
            if args.print_eval:
                print(
                    (
                        f"eval step={global_step} train_return={eval_train:.2f} "
                        f"accept_return={eval_accept:.2f} target_metric={args.target_eval_metric} "
                        f"target_value={target_eval_value:.2f}"
                    ),
                    flush=True,
                )
            wandb.log(
                {
                    "eval/episodic_return_train": eval_train,
                    "eval/episodic_return_accept": eval_accept,
                },
                step=global_step,
            )
            if (
                args.target_eval_return > 0
                and target_eval_value >= args.target_eval_return
            ):
                print(
                    (
                        f"target reached at step={global_step}: "
                        f"{args.target_eval_metric}_return={target_eval_value:.2f}"
                    ),
                    flush=True,
                )
                break

        if start_time is not None and global_step % 100 == 0:
            speed = (global_step - measure_burnin) / (time.time() - start_time)
            pbar.set_description(f"{speed:5.1f} sps, {desc}")
            log_data = {
                "speed_sps": speed,
                "train/episode_return": float(np.mean(avg_returns))
                if len(avg_returns) > 0
                else 0.0,
                "train/qf_loss": float(out_q["qf_loss"].cpu().item())
                if "qf_loss" in out_q.keys()
                else 0.0,
                "train/actor_loss": float(out_actor["actor_loss"].cpu().item()),
                "train/alpha_loss": float(out_actor["alpha_loss"].cpu().item()),
                "train/alpha": float(alpha.detach().cpu().item()),
                "swil/reward_raw_mean": float(raw_reward.mean().cpu().item()),
                "swil/reward_raw_std": float(
                    raw_reward.std(unbiased=False).cpu().item()
                ),
                "swil/reward_raw_min": float(raw_reward.min().cpu().item()),
                "swil/reward_raw_max": float(raw_reward.max().cpu().item()),
                "swil/reward_insert_mean": float(reward_insert.mean().cpu().item()),
                "swil/reward_insert_std": float(
                    reward_insert.std(unbiased=False).cpu().item()
                ),
                "swil/reward_queue_w2_sq": float(
                    last_reward_stats["w2_sq"].cpu().item()
                ),
                "swil/reward_queue_residual_abs_mean": float(
                    last_reward_stats["residual_abs_mean"].cpu().item()
                ),
                "swil/reward_queue_projected_mean": float(
                    last_reward_stats["projected_mean"].cpu().item()
                ),
                "swil/reward_queue_projected_std": float(
                    last_reward_stats["projected_std"].cpu().item()
                ),
                "swil/reward_snapshot_age": float(
                    -1.0
                    if last_reward_stats["snapshot_created_at_mean"].cpu().item() < 0
                    else global_step
                    - last_reward_stats["snapshot_created_at_mean"].cpu().item()
                ),
                "swil/reward_snapshot_policy_samples": float(
                    last_reward_stats["snapshot_policy_samples"].cpu().item()
                ),
                "swil/reward_snapshot_expert_samples": float(
                    last_reward_stats["snapshot_expert_samples"].cpu().item()
                ),
                "swil/critic_loss": float(
                    last_critic_out.get("critic_loss", torch.tensor(0.0, device=device))
                    .cpu()
                    .item()
                ),
                "swil/critic_w2_sq": float(
                    last_critic_out.get("w2_sq", torch.tensor(0.0, device=device))
                    .cpu()
                    .item()
                ),
                "swil/critic_var_reg": float(
                    last_critic_out.get("var_reg", torch.tensor(0.0, device=device))
                    .cpu()
                    .item()
                ),
                "swil/critic_weight_reg": float(
                    last_critic_out.get("weight_reg", torch.tensor(0.0, device=device))
                    .cpu()
                    .item()
                ),
                "swil/critic_raw_mean": float(
                    last_critic_out.get("raw_mean", torch.tensor(0.0, device=device))
                    .cpu()
                    .item()
                ),
                "swil/critic_raw_std": float(
                    last_critic_out.get("raw_std", torch.tensor(0.0, device=device))
                    .cpu()
                    .item()
                ),
                "swil/critic_projected_std": float(
                    last_critic_out.get(
                        "projected_std", torch.tensor(0.0, device=device)
                    )
                    .cpu()
                    .item()
                ),
                "swil/critic_alpha_value": float(
                    last_critic_out.get("alpha_value", torch.tensor(0.0, device=device))
                    .cpu()
                    .item()
                ),
                "swil/critic_queue_len": len(critic_queue),
                "swil/critic_queue_pushes": queue_pushes,
                "fw/eta_selected": float(last_fw_selected_eta),
                "fw/objective_selected": float(last_fw_objective),
                "fw/candidate_episode_fraction": float(
                    episode_assignment_candidate / max(episode_assignment_total, 1)
                ),
                "fw/current_episode_candidate_fraction": float(
                    episode_use_candidate.float().mean().cpu().item()
                ),
                "fw/feature_enabled": 1.0 if args.use_fw_adjacent_mixing else 0.0,
            }
            if args.debug_log_fw_metrics:
                for key, value in fw_metrics.items():
                    log_data[f"fw/{key}"] = value
            wandb.log(log_data, step=global_step)

    envs.close()

    if args.save_model:
        os.makedirs(args.save_dir, exist_ok=True)
        actor_path = os.path.join(args.save_dir, f"{run_name}_actor_final.pt")
        accept_path = os.path.join(args.save_dir, f"{run_name}_accept_final.pt")
        cand_path = os.path.join(args.save_dir, f"{run_name}_cand_final.pt")
        critic_path = os.path.join(args.save_dir, f"{run_name}_critic_final.pt")
        torch.save(actor.state_dict(), actor_path)
        torch.save(pi_accept.state_dict(), accept_path)
        torch.save(pi_cand.state_dict(), cand_path)
        torch.save(critic.state_dict(), critic_path)
        wandb.save(actor_path, policy="now")
        wandb.save(accept_path, policy="now")
        wandb.save(cand_path, policy="now")
        wandb.save(critic_path, policy="now")
