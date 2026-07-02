
import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule, TensorDictModule
from torchrl.data import LazyTensorStorage, ReplayBuffer

# Same demo loader used by your SAC-GAIL and SAC-SF-Sinkhorn scripts.
# Expected keys (at least): demos["all"]["obs"], demos["all"]["acs"], demos["all"]["done"] (and optionally "rew").
from irl.utils import load_hf_demos


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array" if capture_video and idx == 0 else None)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        return env

    return thunk


# ----------------------------
# Networks (SAC)
# ----------------------------
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


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, envs, n_obs: int, n_act: int, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc_mean = nn.Linear(256, n_act, device=device)
        self.fc_logstd = nn.Linear(256, n_act, device=device)
        self.register_buffer(
            "action_scale",
            torch.tensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0,
                         dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0,
                         dtype=torch.float32, device=device),
        )

    def forward(self, obs: torch.Tensor):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs: torch.Tensor):
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


# ----------------------------
# Args
# ----------------------------
@dataclass
class Args:
    exp_name: str = "sac_sqil_torchcompile"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False
    track: bool = False
    wandb_project_name: str = "sac_sqil"
    wandb_entity: str = None

    # Env
    env_id: str = "Ant-v5"
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

    # Perf
    compile: bool = False
    cudagraphs: bool = False
    measure_burnin: int = 3

    # Expert demos
    demo_dir: str = "./demos"
    n_demos: int = 10
    subsample: int = 1

    # SQIL specifics
    expert_fraction: float = 0.5     # fraction of batch sampled from expert buffer
    expert_reward: float = 1.0       # r=1 for expert transitions
    agent_reward: float = 0.0        # r=0 for agent transitions (pure SQIL)
    env_reward_scale: float = 0.0    # if >0, adds scaled env reward to agent transitions

    # Optional: only use agent states for actor update (sometimes slightly more stable)
    actor_update_agent_only: bool = False


def _build_next_obs_from_obs_done(obs: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
    """
    Demos sometimes provide only (obs, act, done). Construct next_obs by shifting obs.
    For terminal transitions, next_obs is set to obs (and done=1 ensures no bootstrap).
    """
    next_obs = torch.empty_like(obs)
    next_obs[:-1] = obs[1:]
    next_obs[-1] = obs[-1]
    if done is not None:
        # done is float32/0-1; treat >0.5 as terminal
        terminal = done.view(-1) > 0.5
        next_obs[terminal] = obs[terminal]
    return next_obs


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}"

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=f"{args.exp_name}-{run_name}",
            config=vars(args),
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Load expert demos
    demos = load_hf_demos(args, n_demos=args.n_demos)
    demos_all = demos["all"]
    for k in ["obs", "acs", "done", "rew"]:
        if k in demos_all:
            demos_all[k] = torch.as_tensor(demos_all[k], device=device, dtype=torch.float32)

    assert "obs" in demos_all and "acs" in demos_all, "Demos must provide obs and acs for SQIL."
    expert_obs = demos_all["obs"]
    expert_acs = demos_all["acs"]
    expert_done = demos_all.get("done", torch.zeros((expert_obs.shape[0],), device=device))
    expert_next_obs = _build_next_obs_from_obs_done(expert_obs, expert_done)

    # (Optional) If demos include rew, ignore it by default: SQIL uses sparse reward.
    expert_N = expert_obs.shape[0]

    # Env
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "Continuous action space only."

    n_act = int(np.prod(envs.single_action_space.shape))
    n_obs = int(np.prod(envs.single_observation_space.shape))

    # Actor + policy module (detach copy for TD sampling)
    actor = Actor(envs, n_obs=n_obs, n_act=n_act, device=device)
    actor_detach = Actor(envs, n_obs=n_obs, n_act=n_act, device=device)
    from_module(actor).data.to_module(actor_detach)

    policy = TensorDictModule(
        actor_detach.get_action,
        in_keys=["observation"],
        out_keys=["action", "log_prob", "mean"],
    )

    # Q nets (two critics)
    def get_q_params():
        qf1 = SoftQNetwork(n_obs=n_obs, n_act=n_act, device=device)
        qf2 = SoftQNetwork(n_obs=n_obs, n_act=n_act, device=device)
        qnet_params = from_modules(qf1, qf2, as_module=True)
        qnet_target = qnet_params.data.clone()
        qnet = SoftQNetwork(n_obs=n_obs, n_act=n_act, device="meta")
        qnet_params.to_module(qnet)
        return qnet_params, qnet_target, qnet

    qnet_params, qnet_target, qnet = get_q_params()

    q_optimizer = optim.Adam(qnet.parameters(), lr=args.q_lr, capturable=args.cudagraphs and not args.compile)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, capturable=args.cudagraphs and not args.compile)

    # Entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.detach().exp()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, capturable=args.cudagraphs and not args.compile)
    else:
        alpha = torch.as_tensor(args.alpha, device=device)

    # Replay buffers
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    # Expert replay (static)
    expert_rb = ReplayBuffer(storage=LazyTensorStorage(expert_N, device=device))
    expert_td = TensorDict(
        observations=expert_obs,
        next_observations=expert_next_obs,
        actions=expert_acs,
        rewards=torch.full((expert_N, 1), float(args.expert_reward), device=device),
        dones=(expert_done.view(-1, 1) > 0.5),
        terminations=(expert_done.view(-1, 1) > 0.5),
        batch_size=[expert_N],
        device=device,
    )
    expert_rb.extend(expert_td)

    # Utility: batched Q
    def batched_qf(params, obs, action, next_q_value=None):
        with params.to_module(qnet):
            vals = qnet(obs, action)
            if next_q_value is not None:
                return F.mse_loss(vals.view(-1), next_q_value)
            return vals

    # Sample a SQIL batch: concat expert+agent
    @torch.no_grad()
    def sample_sqil_batch(batch_size: int) -> TensorDict:
        n_exp = int(round(batch_size * args.expert_fraction))
        n_exp = max(0, min(batch_size, n_exp))
        n_ag = batch_size - n_exp
        exp = expert_rb.sample(n_exp) if n_exp > 0 else None
        ag = rb.sample(n_ag) if n_ag > 0 else None

        if exp is None:
            out = ag
        elif ag is None:
            out = exp
        else:
            out = TensorDict(
                {
                    "observations": torch.cat([exp["observations"], ag["observations"]], dim=0),
                    "next_observations": torch.cat([exp["next_observations"], ag["next_observations"]], dim=0),
                    "actions": torch.cat([exp["actions"], ag["actions"]], dim=0),
                    "rewards": torch.cat([exp["rewards"], ag["rewards"]], dim=0),
                    "dones": torch.cat([exp["dones"], ag["dones"]], dim=0),
                    "terminations": torch.cat([exp["terminations"], ag["terminations"]], dim=0),
                },
                batch_size=[batch_size],
                device=device,
            )
        return out

    # SAC updates
    def update_main(data: TensorDict):
        q_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(data["next_observations"])
            qf_next_target = torch.vmap(batched_qf, (0, None, None))(
                qnet_target, data["next_observations"], next_state_actions
            )
            min_qf_next_target = qf_next_target.min(dim=0).values - alpha * next_state_log_pi

            next_q_value = data["rewards"].flatten() + (~data["dones"].flatten()).float() * args.gamma * min_qf_next_target.view(-1)

        qf_losses = torch.vmap(batched_qf, (0, None, None, None))(
            qnet_params, data["observations"], data["actions"], next_q_value
        )
        qf_loss = qf_losses.sum(0)
        qf_loss.backward()
        q_optimizer.step()
        return TensorDict(qf_loss=qf_loss.detach())

    def update_pol(data_for_actor: TensorDict):
        actor_optimizer.zero_grad(set_to_none=True)
        pi, log_pi, _ = actor.get_action(data_for_actor["observations"])
        qf_pi = torch.vmap(batched_qf, (0, None, None))(
            qnet_params.data, data_for_actor["observations"], pi
        )
        min_qf_pi = qf_pi.min(0).values
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
        actor_loss.backward()
        actor_optimizer.step()

        alpha_loss = torch.tensor(0.0, device=device)
        if args.autotune:
            a_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                _, log_pi2, _ = actor.get_action(data_for_actor["observations"])
            alpha_loss = (-log_alpha.exp() * (log_pi2 + target_entropy)).mean()
            alpha_loss.backward()
            a_optimizer.step()

        return TensorDict(
            alpha=alpha.detach(),
            actor_loss=actor_loss.detach(),
            alpha_loss=alpha_loss.detach(),
        )

    if args.compile:
        update_main = torch.compile(update_main)
        update_pol = torch.compile(update_pol)

    if args.cudagraphs:
        update_main = CudaGraphModule(update_main, in_keys=[], out_keys=[])
        update_pol = CudaGraphModule(update_pol, in_keys=[], out_keys=[])

    # Main loop
    obs, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(obs, device=device, dtype=torch.float32)

    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    measure_burnin_step = None
    avg_returns = deque(maxlen=20)
    max_ep_ret = -float("inf")
    desc = ""

    for global_step in pbar:
        if global_step == args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin_step = global_step

        # Act
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            td_in = TensorDict({"observation": obs}, batch_size=obs.shape[0], device=device)
            td_out = policy(td_in)
            actions = td_out["action"].detach().cpu().numpy()

        next_obs, env_rewards, terminations, truncations, infos = envs.step(actions)

        # Episode logging
        if "episode" in infos:
            for r in infos["episode"]["r"][infos["episode"]["_r"]]:
                max_ep_ret = max(max_ep_ret, float(r))
                avg_returns.append(float(r))
            desc = f"step={global_step}, ep_ret={np.mean(avg_returns):.1f} (max={max_ep_ret:.1f})"

        next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
        real_next_obs = next_obs.clone()
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if infos["_final_observation"][idx]:
                    real_next_obs[idx] = torch.as_tensor(final_obs, device=device, dtype=torch.float32)

        # SQIL: agent transitions get sparse reward (usually 0), optionally plus scaled env reward
        r_agent = float(args.agent_reward) + float(args.env_reward_scale) * torch.as_tensor(
            env_rewards, device=device, dtype=torch.float32
        )
        r_agent = r_agent.view(-1, 1)

        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=torch.as_tensor(actions, device=device, dtype=torch.float32),
            rewards=r_agent,
            dones=torch.as_tensor(terminations, device=device).view(-1, 1),
            terminations=torch.as_tensor(terminations, device=device).view(-1, 1),
            batch_size=obs.shape[0],
            device=device,
        )
        obs = next_obs
        rb.extend(transition)

        # Train
        if global_step > args.learning_starts:
            data = sample_sqil_batch(args.batch_size)
            out_main = update_main(data)

            # Actor update data choice
            data_for_actor = data
            if args.actor_update_agent_only:
                # resample agent-only for actor; keep size fixed
                data_for_actor = rb.sample(args.batch_size)

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    out_main.update(update_pol(data_for_actor))
                    if args.autotune:
                        alpha.copy_(log_alpha.detach().exp())

            if global_step % args.target_network_frequency == 0:
                qnet_target.lerp_(qnet_params.data, args.tau)

            # Logging
            if start_time is not None and (global_step % 100 == 0):
                speed = (global_step - measure_burnin_step) / (time.time() - start_time)
                pbar.set_description(f"{speed:5.1f} sps, {desc}")

                if args.track:
                    wandb.log(
                        {
                            "speed_sps": speed,
                            "train/qf_loss": out_main["qf_loss"].mean().item(),
                            "train/actor_loss": out_main.get("actor_loss", torch.tensor(0.0)).mean().item(),
                            "train/alpha": float(alpha.detach().cpu().item()),
                            "sqil/expert_fraction": float(args.expert_fraction),
                            "sqil/expert_reward": float(args.expert_reward),
                            "sqil/agent_reward": float(args.agent_reward),
                        },
                        step=global_step,
                    )

    envs.close()
