import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
import wandb
from torch.distributions.normal import Normal

try:
    from leanrl.envs.mjlab_torch_vec_env import MjlabTorchVecEnv, ObsBatch
except ImportError:
    from envs.mjlab_torch_vec_env import MjlabTorchVecEnv, ObsBatch


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    device: str = "auto"
    capture_video: bool = False

    env_id: str = "Mjlab-Velocity-Flat-Unitree-G1"
    num_envs: int = 4096
    num_steps: int = 24
    total_timesteps: int = 100_000_000
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 5
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    actor_hidden_dims: tuple[int, ...] = (256, 128, 128)
    critic_hidden_dims: tuple[int, ...] = (256, 128, 128)
    activation: str = "elu"
    policy_observation_group: str = "actor"
    critic_observation_group: str = "critic"
    clip_actions: float | None = 1.0

    normalize_observations: bool = False
    normalize_rewards: bool = False
    reward_clip: float | None = None
    observation_clip: float | None = 10.0

    compile: bool = False
    cudagraphs: bool = False
    measure_burnin: int = 3

    track: bool = False
    wandb_project_name: str = "ppo_mjlab"
    wandb_entity: str | None = None
    save_model: bool = False
    save_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    resume_from: str | None = None
    play_checkpoint: str | None = None
    play_num_envs: int = 1
    play_steps: int = 1000
    video_dir: str = "videos/mjlab_ppo"
    video_length: int = 1000
    deterministic_play: bool = True
    mock_env: bool = False

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
    policy_obs_dim: int = 0
    critic_obs_dim: int = 0
    action_dim: int = 0


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def activation_module(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation={name!r}; expected elu, tanh, or relu")


def mlp(in_dim: int, hidden_dims: tuple[int, ...], out_dim: int, activation: str, out_std: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for hidden in hidden_dims:
        layers += [layer_init(nn.Linear(last, hidden)), activation_module(activation)]
        last = hidden
    layers.append(layer_init(nn.Linear(last, out_dim), std=out_std))
    return nn.Sequential(*layers)


class Agent(nn.Module):
    def __init__(
        self,
        policy_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        actor_hidden_dims: tuple[int, ...],
        critic_hidden_dims: tuple[int, ...],
        activation: str,
        device: torch.device,
    ):
        super().__init__()
        self.actor_mean = mlp(policy_obs_dim, actor_hidden_dims, action_dim, activation, out_std=0.01).to(device)
        self.critic = mlp(critic_obs_dim, critic_hidden_dims, 1, activation, out_std=1.0).to(device)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim, device=device))

    def get_value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs)

    def get_action_and_value(
        self, policy_obs: torch.Tensor, critic_obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = self.actor_mean(policy_obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean + action_std * torch.randn_like(action_mean)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(critic_obs)


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...], device: torch.device, eps: float = 1e-4):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = torch.tensor(eps, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, *self.mean.shape)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], device=x.device, dtype=self.count.dtype)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta.square() * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, x: torch.Tensor, clip: float | None) -> torch.Tensor:
        y = (x - self.mean) / torch.sqrt(self.var + 1e-8)
        return y if clip is None else torch.clamp(y, -clip, clip)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.mean = state["mean"].to(self.mean.device)
        self.var = state["var"].to(self.var.device)
        self.count = state["count"].to(self.count.device)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(rewards.shape[1], device=rewards.device)
    z = terminated | truncated
    for t in range(rewards.shape[0] - 1, -1, -1):
        terminal_nonbootstrap = (~terminated[t]).float()
        boundary_nonterminal = (~z[t]).float()
        delta = rewards[t] + gamma * terminal_nonbootstrap * bootstrap_values[t] - values[t]
        lastgaelam = delta + gamma * gae_lambda * boundary_nonterminal * lastgaelam
        advantages[t] = lastgaelam
    return advantages, advantages + values


def ppo_update(
    agent: Agent,
    optimizer: optim.Optimizer,
    policy_obs: torch.Tensor,
    critic_obs: torch.Tensor,
    actions: torch.Tensor,
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    args: Args,
) -> dict[str, torch.Tensor]:
    optimizer.zero_grad()
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(policy_obs, critic_obs, actions)
    logratio = newlogprob - logprobs
    ratio = logratio.exp()
    with torch.no_grad():
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1.0) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    pg_loss = torch.max(
        -advantages * ratio,
        -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
    ).mean()
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = values + torch.clamp(newvalue - values, -args.clip_coef, args.clip_coef)
        v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - returns) ** 2).mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()
    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()
    return {
        "policy_loss": pg_loss.detach(),
        "value_loss": v_loss.detach(),
        "entropy": entropy_loss.detach(),
        "approx_kl": approx_kl.detach(),
        "old_approx_kl": old_approx_kl.detach(),
        "clip_fraction": clipfrac.detach(),
        "grad_norm": torch.as_tensor(grad_norm, device=policy_obs.device),
    }


class MockMjlabLikeEnv:
    def __init__(self, num_envs: int, device: torch.device, policy_dim: int = 5, critic_dim: int = 7, action_dim: int = 3, horizon: int = 4):
        self.num_envs = num_envs
        self.device = device
        self.policy_dim = policy_dim
        self.critic_dim = critic_dim
        self.num_actions = action_dim
        self.horizon = horizon
        self.step_count = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.state = torch.zeros(num_envs, policy_dim, device=device)
        self.action_space = gym.spaces.Box(
            low=np.full((action_dim,), -1.0, dtype=np.float32),
            high=np.full((action_dim,), 1.0, dtype=np.float32),
            dtype=np.float32,
        )

    def _obs(self) -> dict[str, torch.Tensor]:
        critic_pad = torch.zeros(self.num_envs, self.critic_dim - self.policy_dim, device=self.device)
        policy = self.state.clone()
        return {"actor": policy, "policy": policy, "critic": torch.cat([self.state, critic_pad], dim=-1)}

    def reset(self, seed: int | None = None, env_ids: torch.Tensor | None = None):
        if seed is not None:
            torch.manual_seed(seed)
        if env_ids is None:
            self.step_count.zero_()
            self.state = torch.randn_like(self.state) * 0.01
        else:
            self.step_count[env_ids] = 0
            self.state[env_ids] = torch.randn((env_ids.numel(), self.policy_dim), device=self.device) * 0.01
        return self._obs(), {}

    def step(self, action: torch.Tensor):
        self.step_count += 1
        padded_action = torch.zeros_like(self.state)
        padded_action[:, : action.shape[-1]] = action
        self.state = self.state + 0.05 * padded_action
        reward = -self.state.square().sum(dim=-1)
        terminated = self.state.norm(dim=-1) > 100.0
        truncated = self.step_count >= self.horizon
        return self._obs(), reward, terminated, truncated, {}

    def close(self) -> None:
        pass


def select_device(args: Args) -> torch.device:
    if args.device != "auto":
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


def make_env(args: Args, device: torch.device) -> MjlabTorchVecEnv:
    if args.mock_env:
        return MjlabTorchVecEnv(
            args.env_id,
            args.num_envs,
            device,
            seed=args.seed,
            policy_observation_group=args.policy_observation_group,
            critic_observation_group=args.critic_observation_group,
            clip_actions=args.clip_actions,
            env=MockMjlabLikeEnv(args.num_envs, device),
        )
    return MjlabTorchVecEnv(
        args.env_id,
        args.num_envs,
        device,
        seed=args.seed,
        policy_observation_group=args.policy_observation_group,
        critic_observation_group=args.critic_observation_group,
        clip_actions=args.clip_actions,
    )


def make_render_env(args: Args, device: torch.device, video: bool) -> MjlabTorchVecEnv:
    if args.mock_env:
        raise RuntimeError("Video playback is only supported for real mjlab environments, not --mock-env.")
    raw_env = MjlabTorchVecEnv(
        args.env_id,
        args.play_num_envs,
        device,
        seed=args.seed,
        policy_observation_group=args.policy_observation_group,
        critic_observation_group=args.critic_observation_group,
        clip_actions=args.clip_actions,
        render_mode="rgb_array" if video else None,
    )
    if not video:
        return raw_env
    from mjlab.utils.wrappers import VideoRecorder

    raw_env.env = VideoRecorder(
        raw_env.env,
        video_folder=Path(args.video_dir),
        step_trigger=lambda step: step == 0,
        video_length=args.video_length,
        name_prefix=f"{args.env_id}__leanrl-ppo",
    )
    return raw_env


def save_checkpoint(
    path: Path,
    agent: Agent,
    optimizer: optim.Optimizer,
    args: Args,
    global_step: int,
    iteration: int,
    normalizers: dict[str, RunningMeanStd],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "agent": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": asdict(args),
            "global_step": global_step,
            "iteration": iteration,
            "normalizers": {name: rms.state_dict() for name, rms in normalizers.items()},
            "env_id": args.env_id,
            "versions": {
                "torch": torch.__version__,
                "mjlab": _module_version("mjlab"),
                "mujoco_warp": _module_version("mujoco_warp"),
            },
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        path,
    )


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return getattr(module, "__version__", None)


def load_policy_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[Agent, Args, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = Args(**checkpoint["args"])
    agent = Agent(
        checkpoint["args"]["policy_obs_dim"],
        checkpoint["args"]["critic_obs_dim"],
        checkpoint["args"]["action_dim"],
        tuple(args.actor_hidden_dims),
        tuple(args.critic_hidden_dims),
        args.activation,
        torch.device(device),
    )
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    return agent, args, checkpoint


def _load_eval_normalizers(checkpoint: dict[str, Any], device: torch.device) -> dict[str, RunningMeanStd]:
    normalizers = {}
    for name, state in checkpoint.get("normalizers", {}).items():
        shape = tuple(state["mean"].shape)
        rms = RunningMeanStd(shape, device)
        rms.load_state_dict(state)
        normalizers[name] = rms
    return normalizers


def play(args: Args) -> None:
    if args.play_checkpoint is None:
        raise ValueError("--play-checkpoint is required for play mode")
    device = select_device(args)
    agent, checkpoint_args, checkpoint = load_policy_checkpoint(args.play_checkpoint, device=device)
    checkpoint_args.env_id = checkpoint.get("env_id", checkpoint_args.env_id)
    checkpoint_args.seed = args.seed
    checkpoint_args.cuda = args.cuda
    checkpoint_args.device = args.device
    checkpoint_args.play_num_envs = args.play_num_envs
    checkpoint_args.play_steps = args.play_steps
    checkpoint_args.video_dir = args.video_dir
    checkpoint_args.video_length = args.video_length
    checkpoint_args.capture_video = args.capture_video
    checkpoint_args.deterministic_play = args.deterministic_play
    if args.env_id != Args.env_id:
        checkpoint_args.env_id = args.env_id

    envs = make_render_env(checkpoint_args, device, video=args.capture_video)
    normalizers = _load_eval_normalizers(checkpoint, device)

    def norm_policy(x: torch.Tensor) -> torch.Tensor:
        rms = normalizers.get("policy_obs")
        return x if rms is None else rms.normalize(x, checkpoint_args.observation_clip)

    def norm_critic(x: torch.Tensor) -> torch.Tensor:
        rms = normalizers.get("critic_obs")
        return x if rms is None else rms.normalize(x, checkpoint_args.observation_clip)

    obs, _ = envs.reset(seed=checkpoint_args.seed)
    returns = torch.zeros(checkpoint_args.play_num_envs, device=device)
    lengths = torch.zeros(checkpoint_args.play_num_envs, device=device)
    completed_returns: list[float] = []
    with torch.no_grad():
        for _ in tqdm.tqdm(range(checkpoint_args.play_steps)):
            policy_obs = norm_policy(obs.policy)
            critic_obs = norm_critic(obs.critic)
            if checkpoint_args.deterministic_play:
                action = agent.actor_mean(policy_obs)
            else:
                action, _, _, _ = agent.get_action_and_value(policy_obs, critic_obs)
            step = envs.step(action)
            returns += step.reward
            lengths += 1
            done = step.terminated | step.truncated
            if bool(done.any().item()):
                completed_returns.extend(float(x) for x in returns[done].detach().cpu().tolist())
                returns[done] = 0
                lengths[done] = 0
            obs = step.obs
    envs.close()
    if completed_returns:
        print(f"mean completed return: {sum(completed_returns) / len(completed_returns):.3f}")
    if args.capture_video:
        print(f"video directory: {Path(args.video_dir).resolve()}")


def train(args: Args) -> dict[str, float]:
    if args.cudagraphs:
        raise RuntimeError("CUDA graphs are not enabled for this trainer because mjlab step/reset and done-row resets are not graph-safe yet.")
    if args.capture_video:
        raise RuntimeError("capture_video is not implemented for the Torch-native mjlab path.")

    args.batch_size = args.num_envs * args.num_steps
    if args.batch_size % args.num_minibatches != 0:
        raise ValueError(f"num_envs*num_steps={args.batch_size} must be divisible by num_minibatches={args.num_minibatches}")
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = max(args.total_timesteps // args.batch_size, 1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = select_device(args)
    envs = make_env(args, device)
    action_dim = math.prod(envs.single_action_space.shape)
    args.policy_obs_dim = envs.policy_obs_dim
    args.critic_obs_dim = envs.critic_obs_dim
    args.action_dim = action_dim
    args_dict = asdict(args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}"
    if args.track:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, name=run_name, config=args_dict, save_code=True)

    agent = Agent(
        envs.policy_obs_dim,
        envs.critic_obs_dim,
        action_dim,
        tuple(args.actor_hidden_dims),
        tuple(args.critic_hidden_dims),
        args.activation,
        device,
    )
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    start_iteration = 1
    global_step = 0
    resume_checkpoint = None
    if args.resume_from is not None:
        resume_checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        checkpoint = resume_checkpoint
        agent.load_state_dict(checkpoint["agent"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint["global_step"])
        start_iteration = int(checkpoint["iteration"]) + 1

    policy_obs = torch.zeros((args.num_steps, args.num_envs, envs.policy_obs_dim), device=device)
    critic_obs = torch.zeros((args.num_steps, args.num_envs, envs.critic_obs_dim), device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, action_dim), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    terminated = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=torch.bool)
    truncated = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=torch.bool)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    bootstrap_values = torch.zeros((args.num_steps, args.num_envs), device=device)

    normalizers: dict[str, RunningMeanStd] = {}
    if args.normalize_observations:
        normalizers["policy_obs"] = RunningMeanStd((envs.policy_obs_dim,), device)
        normalizers["critic_obs"] = RunningMeanStd((envs.critic_obs_dim,), device)
    if args.normalize_rewards:
        normalizers["rewards"] = RunningMeanStd((), device)
    if resume_checkpoint is not None:
        for name, state in resume_checkpoint.get("normalizers", {}).items():
            if name in normalizers:
                normalizers[name].load_state_dict(state)

    def norm_policy(x: torch.Tensor) -> torch.Tensor:
        if not args.normalize_observations:
            return x
        normalizers["policy_obs"].update(x)
        return normalizers["policy_obs"].normalize(x, args.observation_clip)

    def norm_critic(x: torch.Tensor) -> torch.Tensor:
        if not args.normalize_observations:
            return x
        normalizers["critic_obs"].update(x)
        return normalizers["critic_obs"].normalize(x, args.observation_clip)

    get_action_and_value = agent.get_action_and_value
    get_value = agent.get_value
    compiled_update = ppo_update
    compiled_gae = compute_gae
    if args.compile:
        get_action_and_value = torch.compile(get_action_and_value)
        get_value = torch.compile(get_value)
        compiled_update = torch.compile(ppo_update)
        compiled_gae = torch.compile(compute_gae, fullgraph=True)

    next_obs, _ = envs.reset(seed=args.seed)
    episodic_returns = torch.zeros(args.num_envs, device=device)
    episodic_lengths = torch.zeros(args.num_envs, device=device)
    completed_returns = deque(maxlen=100)
    completed_lengths = deque(maxlen=100)
    pbar = tqdm.tqdm(range(start_iteration, args.num_iterations + 1))
    train_start = time.time()
    measure_start_step = 0
    rollout_sps = 0.0
    update_sps = 0.0
    last_update: dict[str, torch.Tensor] = {}

    for iteration in pbar:
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate
        if device.type == "cuda":
            torch.cuda.synchronize()
        rollout_start = time.time()

        for step in range(args.num_steps):
            pi_obs_t = norm_policy(next_obs.policy)
            v_obs_t = norm_critic(next_obs.critic)
            policy_obs[step].copy_(pi_obs_t)
            critic_obs[step].copy_(v_obs_t)
            with torch.no_grad():
                action, logprob, _, value = get_action_and_value(pi_obs_t, v_obs_t)
            actions[step].copy_(action)
            logprobs[step].copy_(logprob)
            values[step].copy_(value.flatten())

            step_batch = envs.step(action)
            reward = step_batch.reward
            if args.normalize_rewards:
                normalizers["rewards"].update(reward)
                reward = normalizers["rewards"].normalize(reward, args.reward_clip)
            elif args.reward_clip is not None:
                reward = torch.clamp(reward, -args.reward_clip, args.reward_clip)
            rewards[step].copy_(reward)
            terminated[step].copy_(step_batch.terminated)
            truncated[step].copy_(step_batch.truncated)
            with torch.no_grad():
                bootstrap_values[step].copy_(get_value(norm_critic(step_batch.bootstrap_obs.critic)).flatten())

            episodic_returns += step_batch.reward
            episodic_lengths += 1
            done = step_batch.terminated | step_batch.truncated
            if bool(done.any().item()):
                for value_item in episodic_returns[done].detach().cpu().tolist():
                    completed_returns.append(float(value_item))
                for length_item in episodic_lengths[done].detach().cpu().tolist():
                    completed_lengths.append(float(length_item))
                episodic_returns[done] = 0
                episodic_lengths[done] = 0
            next_obs = step_batch.obs

        if device.type == "cuda":
            torch.cuda.synchronize()
        rollout_time = time.time() - rollout_start
        rollout_sps = args.batch_size / max(rollout_time, 1e-9)
        global_step += args.batch_size

        advantages, returns = compiled_gae(rewards, values, bootstrap_values, terminated, truncated, args.gamma, args.gae_lambda)

        b_policy_obs = policy_obs.reshape((-1, envs.policy_obs_dim))
        b_critic_obs = critic_obs.reshape((-1, envs.critic_obs_dim))
        b_actions = actions.reshape((-1, action_dim))
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        if device.type == "cuda":
            torch.cuda.synchronize()
        update_start = time.time()
        for _epoch in range(args.update_epochs):
            b_inds = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start : start + args.minibatch_size]
                last_update = compiled_update(
                    agent,
                    optimizer,
                    b_policy_obs[mb_inds],
                    b_critic_obs[mb_inds],
                    b_actions[mb_inds],
                    b_logprobs[mb_inds],
                    b_advantages[mb_inds],
                    b_returns[mb_inds],
                    b_values[mb_inds],
                    args,
                )
                if args.target_kl is not None and last_update["approx_kl"] > args.target_kl:
                    break
            if args.target_kl is not None and last_update["approx_kl"] > args.target_kl:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        update_time = time.time() - update_start
        update_sps = args.batch_size / max(update_time, 1e-9)

        if iteration == args.measure_burnin:
            train_start = time.time()
            measure_start_step = global_step
        end_to_end_sps = (global_step - measure_start_step) / max(time.time() - train_start, 1e-9)
        lr = optimizer.param_groups[0]["lr"]
        log_data = {
            "charts/mean_step_reward": rewards.mean().detach(),
            "charts/sps": end_to_end_sps,
            "charts/simulation_sps": rollout_sps,
            "charts/update_sps": update_sps,
            "charts/learning_rate": lr,
            "diagnostics/action_mean": actions.mean().detach(),
            "diagnostics/action_std": actions.std().detach(),
            "diagnostics/value_mean": values.mean().detach(),
            "diagnostics/advantage_mean": advantages.mean().detach(),
            "diagnostics/advantage_std": advantages.std().detach(),
            "diagnostics/terminated_fraction": terminated.float().mean().detach(),
            "diagnostics/truncated_fraction": truncated.float().mean().detach(),
        }
        if last_update:
            log_data.update(
                {
                    "losses/policy_loss": last_update["policy_loss"],
                    "losses/value_loss": last_update["value_loss"],
                    "losses/entropy": last_update["entropy"],
                    "losses/approx_kl": last_update["approx_kl"],
                    "losses/old_approx_kl": last_update["old_approx_kl"],
                    "losses/clip_fraction": last_update["clip_fraction"],
                    "losses/grad_norm": last_update["grad_norm"],
                }
            )
        if completed_returns:
            log_data["charts/episodic_return"] = sum(completed_returns) / len(completed_returns)
            log_data["charts/episodic_length"] = sum(completed_lengths) / len(completed_lengths)
        if args.track:
            wandb.log({key: (value.item() if torch.is_tensor(value) else value) for key, value in log_data.items()}, step=global_step)
        pbar.set_description(
            f"sps={end_to_end_sps:,.0f} sim_sps={rollout_sps:,.0f} reward={float(rewards.mean().detach().item()):.3f}"
        )
        if args.save_model and iteration % args.save_interval == 0:
            save_checkpoint(Path(args.checkpoint_dir) / f"{run_name}__step_{global_step}.pt", agent, optimizer, args, global_step, iteration, normalizers)

    if args.save_model:
        save_checkpoint(Path(args.checkpoint_dir) / f"{run_name}__final.pt", agent, optimizer, args, global_step, args.num_iterations, normalizers)
    envs.close()
    return {"global_step": float(global_step), "simulation_sps": float(rollout_sps), "update_sps": float(update_sps)}


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.play_checkpoint is not None:
        play(args)
    else:
        train(args)
