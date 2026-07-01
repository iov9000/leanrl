# docs and experiment results for the SAC substrate:
# https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import csv
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
import wandb
from stable_baselines3.common.buffers import ReplayBuffer
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule

try:
    from leanrl.il_utils import compute_demo_phases, compute_online_phases
except ImportError:
    from il_utils import compute_demo_phases, compute_online_phases

BASE_OCCUPANCY_GEOMETRIES = {"s", "sa", "sas", "sasde"}
OCCUPANCY_GEOMETRIES = BASE_OCCUPANCY_GEOMETRIES | {f"{geometry}t" for geometry in BASE_OCCUPANCY_GEOMETRIES}

try:
    from leanrl.swil_exact_dual import (
        LearnedSliceProjector,
        PotentialBank,
        SUPPORTED_REWARD_MODES,
        build_potential_bank,
        fit_state_action_normalizer,
        load_expert_dataset,
        maximize_projected_w2,
        normalized_swil_rewards,
        sample_slice_projector,
        state_action_samples,
    )
except ImportError:
    from swil_exact_dual import (
        LearnedSliceProjector,
        PotentialBank,
        SUPPORTED_REWARD_MODES,
        build_potential_bank,
        fit_state_action_normalizer,
        load_expert_dataset,
        maximize_projected_w2,
        normalized_swil_rewards,
        sample_slice_projector,
        state_action_samples,
    )


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False
    track: bool = True
    wandb_project_name: str = "sac_swil_exact_dual"
    wandb_entity: Optional[str] = None
    csv_log_path: str = ""

    # SAC arguments
    env_id: str = "HalfCheetah-v5"
    total_timesteps: int = 1_000_000
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5_000
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True
    measure_burnin: int = 3
    compile: bool = False
    cudagraphs: bool = False

    # Exact-dual SWIL arguments
    expert_path: str = ""
    normalize_sa: bool = True
    norm_source: str = "expert_only"  # expert_only | expert_and_initial_policy
    occupancy_geometry: str = "sa"
    """Occupancy features: s | sa | sas | sasde; append t for sin/cos phase, e.g. sat or sasdet."""
    occupancy_time_horizon: int = 0
    """If >0, horizon used for occupancy phase t; otherwise use env max_episode_steps."""
    absorbing_state: bool = False
    """Append an absorbing-state indicator to occupancy states and mark terminal next states as absorbing."""
    num_projections: int = 32
    projection_seed: int = 0
    projection_type: str = "linear_random"
    """linear_random | poly_random | circular_random | nn_random | mixed_linear_nn"""
    projection_degree: int = 2
    projection_radius: float = 2.0
    nn_slice_features: int = 256
    nn_slice_activation: str = "silu"  # relu | silu | tanh | erf
    nn_slice_linear_count: int = 0
    nn_slice_calibrate: bool = True
    learned_projections: bool = False
    """Use a trainable critic that maximizes projected W₂² (Danskin formulation)."""
    critic_hidden_dim: int = 256
    critic_steps: int = 50
    """Gradient ascent steps for the inner W₂² maximization per bank rebuild."""
    critic_lr: float = 1e-3
    critic_spectral_norm: bool = True
    swil_reward_mode: str = "dual"
    """dual uses the legacy center/std flags; dual_raw, dual_centered, and rpl disable per-batch normalization."""
    swil_reward_scale: float = 1.0
    swil_reward_center: bool = True
    swil_reward_std_norm: bool = True
    potential_interp: str = "linear"
    swil_update_freq: int = 1000
    swil_num_learner_samples: int = 8192
    swil_num_expert_samples: int = 8192
    swil_warmup_steps: int = 5000

    # Checkpointing / evaluation
    save_dir: str = "checkpoints"
    save_interval: int = 100000
    eval: bool = False
    load_path: Optional[str] = None
    eval_episodes: int = 10
    eval_interval: int = 0


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


def validate_occupancy_geometry(value: str) -> str:
    if value not in OCCUPANCY_GEOMETRIES:
        raise ValueError(f"Unsupported occupancy_geometry={value}; expected one of {sorted(OCCUPANCY_GEOMETRIES)}")
    return value


def validate_swil_reward_mode(value: str) -> str:
    if value not in SUPPORTED_REWARD_MODES:
        raise ValueError(f"Unsupported swil_reward_mode={value}; expected one of {sorted(SUPPORTED_REWARD_MODES)}")
    return value


def swil_reward_kwargs(args: Args) -> dict[str, object]:
    mode = validate_swil_reward_mode(args.swil_reward_mode)
    if mode == "dual":
        return {
            "mode": "dual",
            "center": args.swil_reward_center,
            "center_mode": None,
            "std_norm": args.swil_reward_std_norm,
        }
    if mode == "dual_centered":
        return {"mode": "dual", "center": False, "center_mode": "per_bank", "std_norm": False}
    if mode == "dual_raw":
        return {"mode": "dual", "center": False, "center_mode": "none", "std_norm": False}
    return {"mode": "rpl", "center": False, "center_mode": "none", "std_norm": False}


def occupancy_base_geometry(value: str) -> str:
    value = validate_occupancy_geometry(value)
    if value.endswith("t"):
        return value[:-1]
    return value


def occupancy_uses_time(value: str) -> bool:
    return validate_occupancy_geometry(value).endswith("t")


def encode_occupancy_phase(phase: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    phase = phase.to(device=device, dtype=dtype).reshape(-1, 1)
    angle = phase * (2.0 * torch.pi)
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=1)


def augment_absorbing_state(
    observations: torch.Tensor,
    *,
    dones: torch.Tensor | None = None,
    next_state: bool = False,
) -> torch.Tensor:
    if observations.ndim != 2:
        raise ValueError(f"observations must be 2D, got {tuple(observations.shape)}")
    indicator = torch.zeros(observations.shape[0], 1, device=observations.device, dtype=observations.dtype)
    values = observations
    if next_state and dones is not None:
        done_mask = dones.to(device=observations.device, dtype=torch.bool).reshape(-1, 1)
        if done_mask.shape[0] != observations.shape[0]:
            raise ValueError(f"dones must have length {observations.shape[0]}, got {done_mask.shape[0]}")
        values = torch.where(done_mask, torch.zeros_like(observations), observations)
        indicator = done_mask.to(dtype=observations.dtype)
    return torch.cat([values, indicator], dim=1)


def build_occupancy_features(
    observations: torch.Tensor,
    actions: torch.Tensor,
    next_observations: torch.Tensor | None,
    geometry: str,
    phase: torch.Tensor | None = None,
    dones: torch.Tensor | None = None,
    absorbing_state: bool = False,
) -> torch.Tensor:
    geometry = validate_occupancy_geometry(geometry)
    base_geometry = occupancy_base_geometry(geometry)
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError(
            f"observations and actions must be 2D, got {tuple(observations.shape)} and {tuple(actions.shape)}"
        )
    if observations.shape[0] != actions.shape[0]:
        raise ValueError("observations and actions must have matching batch sizes")
    state_obs = augment_absorbing_state(observations) if absorbing_state else observations
    if base_geometry == "s":
        features = state_obs
    elif base_geometry == "sa":
        features = state_action_samples(state_obs, actions)
    else:
        if next_observations is None:
            raise ValueError(f"next_observations is required for occupancy_geometry={geometry}")
        if next_observations.ndim != 2 or next_observations.shape != observations.shape:
            raise ValueError(
                "next_observations must be 2D and match observations for "
                f"occupancy_geometry={geometry}, got {tuple(next_observations.shape)} vs {tuple(observations.shape)}"
            )
        next_state_obs = (
            augment_absorbing_state(next_observations, dones=dones, next_state=True)
            if absorbing_state
            else next_observations
        )
        transition_obs = next_state_obs if base_geometry == "sas" else next_state_obs - state_obs
        features = torch.cat([state_obs, actions, transition_obs], dim=1)
    if not occupancy_uses_time(geometry):
        return features
    if phase is None:
        raise ValueError(f"phase is required for occupancy_geometry={geometry}")
    phase_features = encode_occupancy_phase(phase, dtype=features.dtype, device=features.device)
    if phase_features.shape[0] != features.shape[0]:
        raise ValueError(f"phase must have length {features.shape[0]}, got {phase_features.shape[0]}")
    return torch.cat([features, phase_features], dim=1)


def shifted_next_observations(observations: torch.Tensor, terminals: torch.Tensor | None = None) -> torch.Tensor:
    if observations.ndim != 2:
        raise ValueError(f"observations must be 2D, got {tuple(observations.shape)}")
    next_observations = torch.cat([observations[1:], observations[-1:].clone()], dim=0)
    if terminals is not None:
        terminals = terminals.to(device=observations.device, dtype=torch.bool).reshape(-1)
        if terminals.shape[0] != observations.shape[0]:
            raise ValueError("terminals must have the same length as observations")
        next_observations = torch.where(terminals[:, None], observations, next_observations)
    return next_observations


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.array(env.single_observation_space.shape).prod())
        act_dim = int(np.prod(env.single_action_space.shape))
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.array(env.single_observation_space.shape).prod())
        act_dim = int(np.prod(env.single_action_space.shape))
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x, noise: torch.Tensor | None = None):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        if noise is None:
            x_t = normal.rsample()
        else:
            x_t = mean + std * noise
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def replay_size(rb: ReplayBuffer) -> int:
    return int(rb.buffer_size if rb.full else rb.pos)


def sample_replay_sa(
    rb: ReplayBuffer,
    n_samples: int,
    device: torch.device,
    *,
    recent: bool,
    occupancy_geometry: str = "sa",
    phase_buffer: np.ndarray | None = None,
    absorbing_state: bool = False,
) -> torch.Tensor:
    size = replay_size(rb)
    if size < 2:
        raise RuntimeError("Need at least two replay samples")
    count = min(size, n_samples)
    if recent:
        if rb.full:
            indices = (np.arange(rb.pos - count, rb.pos) % rb.buffer_size).astype(np.int64)
        else:
            indices = np.arange(size - count, size, dtype=np.int64)
    else:
        replace = size < n_samples
        indices = np.random.choice(size, size=count, replace=replace)

    obs = np.asarray(rb.observations[indices, 0], dtype=np.float32)
    actions = np.asarray(rb.actions[indices, 0], dtype=np.float32)
    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
    act_t = torch.as_tensor(actions, device=device, dtype=torch.float32)
    dones_t = None
    if absorbing_state:
        dones = np.asarray(rb.dones[indices, 0], dtype=np.float32)
        dones_t = torch.as_tensor(dones, device=device, dtype=torch.float32)
    next_obs_t = None
    if occupancy_base_geometry(occupancy_geometry) in {"sas", "sasde"}:
        next_obs = np.asarray(rb.next_observations[indices, 0], dtype=np.float32)
        next_obs_t = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
    phase_t = None
    if occupancy_uses_time(occupancy_geometry):
        if phase_buffer is None:
            raise ValueError(f"phase_buffer is required for occupancy_geometry={occupancy_geometry}")
        phase = np.asarray(phase_buffer[indices, 0], dtype=np.float32)
        phase_t = torch.as_tensor(phase, device=device, dtype=torch.float32)
    return build_occupancy_features(
        obs_t,
        act_t,
        next_obs_t,
        occupancy_geometry,
        phase_t,
        dones_t,
        absorbing_state,
    )


def sample_replay_batch_with_indices(rb: ReplayBuffer, batch_size: int) -> tuple[object, np.ndarray]:
    upper = rb.buffer_size if rb.full else rb.pos
    if upper <= 0:
        raise RuntimeError("Cannot sample from an empty replay buffer")
    batch_inds = np.random.randint(0, upper, size=batch_size)
    return rb._get_samples(batch_inds), batch_inds


def sample_rows(x: torch.Tensor, n_samples: int) -> torch.Tensor:
    if x.shape[0] <= n_samples:
        return x
    idx = torch.randperm(x.shape[0], device=x.device)[:n_samples]
    return x[idx]


@torch.no_grad()
def soft_update_params(source: nn.Module, target: nn.Module, tau: float) -> None:
    source_params = list(source.parameters())
    target_params = list(target.parameters())
    torch._foreach_mul_(target_params, 1.0 - tau)
    torch._foreach_add_(target_params, source_params, alpha=tau)


def open_csv_logger(path: str, run_name: str):
    if not path:
        path = os.path.join("runs", f"{run_name}.csv")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "global_step",
            "speed",
            "episode_return",
            "eval_return_env_reward",
            "swil_reward_mean",
            "swil_reward_std",
            "projected_w2",
            "fw_duality_gap",
            "proxy_fw_gap",
            "critic_w2",
            "q_loss",
            "policy_loss",
            "alpha",
        ],
    )
    if file.tell() == 0:
        writer.writeheader()
    return file, writer


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError("stable_baselines3>=2.0 is required")

    args = tyro.cli(Args)
    if not args.expert_path:
        raise ValueError("--expert_path is required for exact-dual SWIL")
    if args.potential_interp != "linear":
        raise ValueError("--potential_interp currently only supports linear")
    validate_occupancy_geometry(args.occupancy_geometry)
    reward_kwargs = swil_reward_kwargs(args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    use_cudagraphs = bool(args.cudagraphs and device.type == "cuda")
    if args.cudagraphs and not use_cudagraphs:
        print("CUDA graphs requested but CUDA is not active; running without CUDA graphs.")

    expert_data = load_expert_dataset(args.expert_path)
    expert_obs = torch.as_tensor(expert_data["observations"], device=device, dtype=torch.float32)
    expert_actions = torch.as_tensor(expert_data["actions"], device=device, dtype=torch.float32)
    expert_terminals = None
    if "terminals" in expert_data:
        expert_terminals = torch.as_tensor(expert_data["terminals"], device=device, dtype=torch.bool)
    expert_next_obs = shifted_next_observations(expert_obs, expert_terminals)
    expert_phase = None
    if occupancy_uses_time(args.occupancy_geometry):
        terminal_np = (
            expert_data["terminals"]
            if "terminals" in expert_data
            else np.zeros(expert_obs.shape[0], dtype=np.bool_)
        )
        expert_phase_np, _ = compute_demo_phases(terminal_np)
        expert_phase = torch.as_tensor(expert_phase_np, device=device, dtype=torch.float32)
    expert_sa_raw = build_occupancy_features(
        expert_obs,
        expert_actions,
        expert_next_obs,
        args.occupancy_geometry,
        expert_phase,
        expert_terminals,
        args.absorbing_state,
    )

    print(expert_sa_raw.shape)

    print(args.normalize_sa)

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=args.q_lr,
        capturable=use_cudagraphs and not args.compile,
    )
    actor_optimizer = optim.Adam(
        list(actor.parameters()),
        lr=args.policy_lr,
        capturable=use_cudagraphs and not args.compile,
    )

    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.detach().exp()
        a_optimizer = optim.Adam(
            [log_alpha],
            lr=args.q_lr,
            capturable=use_cudagraphs and not args.compile,
        )
    else:
        alpha = torch.as_tensor(args.alpha, device=device)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    replay_phase = np.zeros((args.buffer_size, envs.num_envs), dtype=np.float32)

    sa_dim = expert_sa_raw.shape[1]
    slice_projector = sample_slice_projector(
        sa_dim,
        args.num_projections,
        seed=args.projection_seed,
        projection_type=args.projection_type,
        projection_degree=args.projection_degree,
        projection_radius=args.projection_radius,
        nn_feature_dim=args.nn_slice_features,
        nn_activation=args.nn_slice_activation,
        nn_linear_count=args.nn_slice_linear_count,
        device=device,
    )
    learned_projector: LearnedSliceProjector | None = None
    if args.learned_projections:
        learned_projector = LearnedSliceProjector(
            sa_dim,
            args.num_projections,
            hidden_dim=args.critic_hidden_dim,
            activation=args.nn_slice_activation,
            use_spectral_norm=args.critic_spectral_norm,
        ).to(device)

    normalizer = None
    if args.norm_source == "expert_only":
        normalizer = fit_state_action_normalizer(
            expert_sa_raw,
            normalize_sa=args.normalize_sa,
            norm_source=args.norm_source,
        )
        if not args.learned_projections and args.nn_slice_calibrate:
            slice_projector = slice_projector.with_calibration(normalizer.transform(expert_sa_raw))
    elif args.norm_source != "expert_and_initial_policy":
        raise ValueError(f"Unsupported --norm_source {args.norm_source}")

    potential_bank: PotentialBank | None = None
    last_swil_update = -args.swil_update_freq
    critic_w2 = 0.0
    critic_optimizer: torch.optim.Optimizer | None = None

    def current_alpha() -> torch.Tensor:
        if args.autotune:
            return log_alpha.detach().exp()
        return alpha

    def update_critic(data: TensorDict) -> TensorDict:
        q_optimizer.zero_grad()
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(
                data["next_observations"],
                data["next_action_noise"],
            )
            qf1_next_target = qf1_target(data["next_observations"], next_state_actions)
            qf2_next_target = qf2_target(data["next_observations"], next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - current_alpha() * next_state_log_pi
            next_q_value = data["swil_rewards"].flatten() + (
                1.0 - data["dones"].flatten()
            ) * args.gamma * min_qf_next_target.view(-1)

        qf1_a_values = qf1(data["observations"], data["actions"]).view(-1)
        qf2_a_values = qf2(data["observations"], data["actions"]).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        qf_loss.backward()
        q_optimizer.step()
        return TensorDict(q_loss=qf_loss.detach())

    def update_actor_and_alpha(data: TensorDict) -> TensorDict:
        actor_optimizer.zero_grad()
        pi, log_pi, _ = actor.get_action(data["observations"], data["policy_action_noise"])
        qf1_pi = qf1(data["observations"], pi)
        qf2_pi = qf2(data["observations"], pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        alpha_t = current_alpha()
        actor_loss = ((alpha_t * log_pi) - min_qf_pi).mean()

        actor_loss.backward()
        actor_optimizer.step()

        alpha_loss = torch.zeros((), device=device)
        if args.autotune:
            a_optimizer.zero_grad()
            with torch.no_grad():
                _, log_pi_alpha, _ = actor.get_action(data["observations"], data["alpha_action_noise"])
            alpha_loss = (-log_alpha.exp() * (log_pi_alpha + target_entropy)).mean()
            alpha_loss.backward()
            a_optimizer.step()

        return TensorDict(
            actor_loss=actor_loss.detach(),
            alpha_loss=alpha_loss.detach(),
            alpha=current_alpha().detach(),
        )

    if args.compile:
        update_critic = torch.compile(update_critic, mode=None)
        update_actor_and_alpha = torch.compile(update_actor_and_alpha, mode=None)

    if use_cudagraphs:
        update_critic = CudaGraphModule(update_critic, in_keys=[], out_keys=[])
        update_actor_and_alpha = CudaGraphModule(update_actor_and_alpha, in_keys=[], out_keys=[])

    def load_actor(weights_path: str):
        state = torch.load(weights_path, map_location=device)
        actor.load_state_dict(state)
        actor.eval()

    def evaluate_policy(n_episodes: int) -> float:
        eval_env = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + 10_000, 0, False, run_name)])
        ep_returns = []
        obs_eval, _ = eval_env.reset(seed=args.seed + 10_000)
        with torch.no_grad():
            while len(ep_returns) < n_episodes:
                mean_action = actor.get_action(torch.as_tensor(obs_eval, device=device, dtype=torch.float32))[2]
                next_obs_eval, _, _, _, infos_eval = eval_env.step(mean_action.cpu().numpy())
                if "episode" in infos_eval:
                    for r in infos_eval["episode"]["r"][infos_eval["episode"]["_r"]]:
                        ep_returns.append(float(r))
                obs_eval = next_obs_eval
        eval_env.close()
        return float(np.mean(ep_returns)) if ep_returns else 0.0

    if args.eval:
        assert args.load_path is not None and os.path.isfile(args.load_path), "Provide a valid --load_path for eval"
        load_actor(args.load_path)
        avg_ret = evaluate_policy(args.eval_episodes)
        print(f"Eval average environment return over {args.eval_episodes} episodes: {avg_ret:.2f}")
        raise SystemExit(0)

    csv_file, csv_writer = open_csv_logger(args.csv_log_path, run_name)
    os.makedirs(args.save_dir, exist_ok=True)

    obs, _ = envs.reset(seed=args.seed)
    episode_steps = torch.zeros(envs.num_envs, device=device, dtype=torch.float32)
    phase_horizon = int(
        args.occupancy_time_horizon
        or getattr(envs.envs[0].spec, "max_episode_steps", None)
        or 1000
    )
    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=20)
    desc = ""
    measure_burnin = 0
    latest_train_logs: dict[str, float] = {}

    for global_step in pbar:
        if global_step == args.measure_burnin + max(args.learning_starts, args.swil_warmup_steps):
            start_time = time.time()
            measure_burnin = global_step

        exploring = global_step < max(args.learning_starts, args.swil_warmup_steps) or potential_bank is None
        if exploring:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions, _, _ = actor.get_action(torch.as_tensor(obs, device=device, dtype=torch.float32))
            actions = actions.cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        done_mask = torch.as_tensor(terminations | truncations, device=device, dtype=torch.bool)
        phase, _ = compute_online_phases(episode_steps, horizon=phase_horizon, done=done_mask)

        if "episode" in infos:
            for r in infos["episode"]["r"][infos["episode"]["_r"]]:
                max_ep_ret = max(max_ep_ret, r)
                avg_returns.append(r)
            desc = (
                f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} "
                f"(max={max_ep_ret: 4.2f})"
            )

        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if infos["_final_observation"][idx]:
                    real_next_obs[idx] = torch.as_tensor(final_obs, device=device, dtype=torch.float32)
        replay_phase[rb.pos, : envs.num_envs] = phase.detach().cpu().numpy()
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs
        episode_steps = torch.where(done_mask, torch.zeros_like(episode_steps), episode_steps + 1.0)

        can_update_swil = (
            global_step >= args.swil_warmup_steps
            and replay_size(rb) >= 2
            and (potential_bank is None or global_step - last_swil_update >= args.swil_update_freq)
        )
        if can_update_swil:
            learner_sa_raw = sample_replay_sa(
                rb,
                args.swil_num_learner_samples,
                device,
                recent=True,
                occupancy_geometry=args.occupancy_geometry,
                phase_buffer=replay_phase,
                absorbing_state=args.absorbing_state,
            )
            if normalizer is None:
                normalizer = fit_state_action_normalizer(
                    expert_sa_raw,
                    learner_sa_raw,
                    normalize_sa=args.normalize_sa,
                    norm_source=args.norm_source,
                )
                if not args.learned_projections and args.nn_slice_calibrate:
                    slice_projector = slice_projector.with_calibration(normalizer.transform(expert_sa_raw))
            expert_batch = sample_rows(expert_sa_raw, args.swil_num_expert_samples)
            learner_sa = normalizer.transform(learner_sa_raw)
            expert_sa = normalizer.transform(expert_batch)

            if args.learned_projections:
                critic_w2, critic_optimizer = maximize_projected_w2(
                    learned_projector,
                    learner_sa,
                    expert_sa,
                    n_steps=args.critic_steps,
                    lr=args.critic_lr,
                    optimizer=critic_optimizer,
                )
                potential_bank = build_potential_bank(learner_sa, expert_sa, learned_projector)
                latest_train_logs["critic_w2"] = critic_w2
            else:
                potential_bank = build_potential_bank(learner_sa, expert_sa, slice_projector)

            last_swil_update = global_step

        if global_step > args.learning_starts and potential_bank is not None and normalizer is not None:
            data, batch_indices = sample_replay_batch_with_indices(rb, args.batch_size)
            batch_phase = None
            if occupancy_uses_time(args.occupancy_geometry):
                batch_phase = torch.as_tensor(
                    replay_phase[batch_indices, 0],
                    device=device,
                    dtype=torch.float32,
                )
            batch_sa_raw = build_occupancy_features(
                data.observations,
                data.actions,
                data.next_observations,
                args.occupancy_geometry,
                batch_phase,
                data.dones,
                args.absorbing_state,
            )
            batch_sa = normalizer.transform(batch_sa_raw)
            with torch.no_grad():
                batch_reward_kwargs = reward_kwargs
                if args.swil_reward_mode == "rpl":
                    batch_reward_kwargs = {
                        **reward_kwargs,
                        "rpl_remove_indices": torch.as_tensor(
                            batch_indices % potential_bank.z_grid.shape[1],
                            device=device,
                            dtype=torch.long,
                        ),
                    }
                swil_rewards = normalized_swil_rewards(
                    potential_bank,
                    batch_sa,
                    reward_scale=args.swil_reward_scale,
                    **batch_reward_kwargs,
                )

            update_batch = TensorDict(
                observations=data.observations,
                actions=data.actions,
                next_observations=data.next_observations,
                dones=data.dones.float(),
                swil_rewards=swil_rewards,
                next_action_noise=torch.randn_like(data.actions),
                policy_action_noise=torch.randn_like(data.actions),
                alpha_action_noise=torch.randn_like(data.actions),
                batch_size=[args.batch_size],
                device=device,
            )
            out_main = update_critic(update_batch)
            actor_loss_value = 0.0
            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    out_main.update(update_actor_and_alpha(update_batch))
                    actor_loss_value = float(out_main["actor_loss"].item())

            if global_step % args.target_network_frequency == 0:
                soft_update_params(qf1, qf1_target, args.tau)
                soft_update_params(qf2, qf2_target, args.tau)

            with torch.no_grad():
                probe_sa = normalizer.transform(
                    sample_replay_sa(
                        rb,
                        min(args.batch_size, replay_size(rb)),
                        device,
                        recent=True,
                        occupancy_geometry=args.occupancy_geometry,
                        phase_buffer=replay_phase,
                        absorbing_state=args.absorbing_state,
                    )
                )
                probe_reward_kwargs = reward_kwargs
                if args.swil_reward_mode == "rpl":
                    probe_reward_kwargs = {
                        **reward_kwargs,
                        "rpl_remove_indices": torch.arange(
                            probe_sa.shape[0],
                            device=device,
                            dtype=torch.long,
                        )
                        % potential_bank.z_grid.shape[1],
                    }
                probe_reward = normalized_swil_rewards(
                    potential_bank,
                    probe_sa,
                    reward_scale=args.swil_reward_scale,
                    **probe_reward_kwargs,
                )
                proxy_fw_gap = potential_bank.raw_rewards(probe_sa).mean() - potential_bank.policy_reward_mean
                fw_duality_gap = potential_bank.fw_duality_gap(probe_sa)
                latest_train_logs = {
                    "train/swil_reward_mean": float(probe_reward.mean().item()),
                    "train/swil_reward_std": float(probe_reward.std(unbiased=False).item()),
                    "train/projected_w2": float(potential_bank.projected_w2.item()),
                    "train/fw_duality_gap": float(fw_duality_gap.item()),
                    "train/proxy_fw_gap": float(proxy_fw_gap.item()),
                    "train/q_loss": float(out_main["q_loss"].item()),
                    "train/policy_loss": actor_loss_value,
                    "train/alpha": float(current_alpha().item()),
                    **({"train/critic_w2": critic_w2} if args.learned_projections else {}),
                }

        if args.save_interval and global_step > 0 and global_step % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f"{run_name}_actor_step{global_step}.pt")
            torch.save(actor.state_dict(), ckpt_path)
            if args.track:
                wandb.save(ckpt_path, policy="now")

        if global_step % 100 == 0:
            speed = 0.0
            if start_time is not None:
                speed = (global_step - measure_burnin) / max(time.time() - start_time, 1e-6)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)

            eval_return = None
            if args.eval_interval > 0 and global_step > 0 and global_step % args.eval_interval == 0:
                eval_return = evaluate_policy(args.eval_episodes)

            logs = {
                "speed": speed,
                "charts/episodic_return": float(torch.tensor(avg_returns).mean().item()) if avg_returns else 0.0,
                **latest_train_logs,
            }
            if eval_return is not None:
                logs["eval/return_env_reward"] = eval_return
            if args.track:
                wandb.log(logs, step=global_step)

            csv_writer.writerow(
                {
                    "global_step": global_step,
                    "speed": speed,
                    "episode_return": logs["charts/episodic_return"],
                    "eval_return_env_reward": "" if eval_return is None else eval_return,
                    "swil_reward_mean": latest_train_logs.get("train/swil_reward_mean", ""),
                    "swil_reward_std": latest_train_logs.get("train/swil_reward_std", ""),
                    "projected_w2": latest_train_logs.get("train/projected_w2", ""),
                    "fw_duality_gap": latest_train_logs.get("train/fw_duality_gap", ""),
                    "proxy_fw_gap": latest_train_logs.get("train/proxy_fw_gap", ""),
                    "critic_w2": latest_train_logs.get("train/critic_w2", ""),
                    "q_loss": latest_train_logs.get("train/q_loss", ""),
                    "policy_loss": latest_train_logs.get("train/policy_loss", ""),
                    "alpha": latest_train_logs.get("train/alpha", float(current_alpha().item())),
                }
            )
            csv_file.flush()

    envs.close()
    csv_file.close()
    final_path = os.path.join(args.save_dir, f"{run_name}_actor_final.pt")
    torch.save(actor.state_dict(), final_path)
    if args.track:
        wandb.save(final_path, policy="now")
