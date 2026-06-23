import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import math
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
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule, TensorDictModule
from torchrl.data import LazyTensorStorage, ReplayBuffer

try:
    from leanrl.irl.utils import load_hf_demos
except ImportError:
    from irl.utils import load_hf_demos


"""
SAC + IQ-Learn baseline (continuous action), torch.compile ready.

Example:
python leanrl/sac_iqlearn_torchcompile.py \
  --env_id HalfCheetah-v5 --n_demos 10 --track --compile --cudagraphs
"""


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array" if capture_video and idx == 0 else None)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
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


def _build_next_obs_from_obs_done(
    obs: torch.Tensor, terminal: torch.Tensor | None
) -> torch.Tensor:
    next_obs = torch.empty_like(obs)
    next_obs[:-1] = obs[1:]
    next_obs[-1] = obs[-1]
    if terminal is not None:
        terminal = terminal.view(-1).bool()
        next_obs[terminal] = obs[terminal]
    return next_obs


def _extract_init_obs(obs: torch.Tensor, done: torch.Tensor | None) -> torch.Tensor:
    if done is None:
        return obs[: max(1, min(4096, obs.shape[0]))]
    done_mask = (done.view(-1) > 0.5).detach().cpu().numpy()
    starts = [0]
    starts.extend((np.where(done_mask[:-1])[0] + 1).tolist())
    starts = [idx for idx in starts if idx < obs.shape[0]]
    if len(starts) == 0:
        starts = [0]
    idx = torch.as_tensor(starts, device=obs.device, dtype=torch.long)
    return obs[idx]


def _infer_demo_terminations(done: torch.Tensor, time_limit: int | None) -> torch.Tensor:
    done_bool = done.view(-1).bool()
    terminations = done_bool.clone()
    if time_limit is None or time_limit <= 0:
        return terminations

    done_indices = torch.nonzero(done_bool, as_tuple=False).view(-1)
    episode_start = 0
    for done_idx in done_indices.tolist():
        episode_len = done_idx - episode_start + 1
        if episode_len >= time_limit:
            terminations[done_idx] = False
        episode_start = done_idx + 1
    return terminations


def _as_critic_batch(values: torch.Tensor) -> torch.Tensor:
    values = values.squeeze(-1)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    return values


def iq_learn_loss_terms(
    q_expert: torch.Tensor,
    q_policy: torch.Tensor,
    v_expert: torch.Tensor,
    v_policy: torch.Tensor,
    v_init: torch.Tensor,
    next_v_expert: torch.Tensor,
    next_v_policy: torch.Tensor,
    expert_done: torch.Tensor,
    policy_done: torch.Tensor,
    gamma: float,
    chi2_coeff: float,
    value_coeff: float,
    initial_state_coeff: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    q_expert = _as_critic_batch(q_expert)
    q_policy = _as_critic_batch(q_policy)
    v_expert = v_expert.view(-1)
    v_policy = v_policy.view(-1)
    v_init = v_init.view(-1)
    next_v_expert = next_v_expert.view(-1)
    next_v_policy = next_v_policy.view(-1)

    not_done_expert = (~expert_done.view(-1).bool()).float()
    not_done_policy = (~policy_done.view(-1).bool()).float()
    y_expert = gamma * not_done_expert * next_v_expert
    y_policy = gamma * not_done_policy * next_v_policy

    expert_residual = q_expert - y_expert.unsqueeze(0)
    policy_residual = q_policy - y_policy.unsqueeze(0)

    expert_term = -expert_residual.mean()
    value_term = value_coeff * torch.cat(
        [v_expert - y_expert, v_policy - y_policy], dim=0
    ).mean()

    chi2_term = q_expert.new_tensor(0.0)
    if chi2_coeff > 0:
        mixed_residual = torch.cat([expert_residual, policy_residual], dim=1)
        chi2_term = 0.25 * chi2_coeff * mixed_residual.pow(2).mean()

    init_term = q_expert.new_tensor(0.0)
    if initial_state_coeff > 0:
        init_term = initial_state_coeff * (1.0 - gamma) * v_init.mean()

    q_loss = expert_term + value_term + chi2_term + init_term
    return q_loss, {
        "expert_term": expert_term,
        "value_term": value_term,
        "chi2_term": chi2_term,
        "init_term": init_term,
    }


@dataclass
class Args:
    exp_name: str = "sac_iqlearn_torchcompile"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False
    track: bool = False
    wandb_project_name: str = "sac_iqlearn"
    wandb_entity: Optional[str] = None

    env_id: str = "HalfCheetah-v5"
    total_timesteps: int = 1_000_000
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5_000
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True

    compile: bool = False
    cudagraphs: bool = False
    measure_burnin: int = 3

    demo_dir: str = "./demos"
    n_demos: int = 10
    subsample: int = 1

    # IQ-Learn objective coefficients
    iq_chi2_coeff: float = 1.0
    iq_value_coeff: float = 1.0
    iq_initial_state_coeff: float = 0.0
    iq_use_target_next_v: bool = True
    q_grad_clip: float = 10.0

    # Optional actor BC regularization from expert demonstrations.
    actor_bc_coef: float = 0.0
    actor_bc_warmup_steps: int = 0


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
    use_compile = bool(args.compile)
    use_cudagraphs = bool(args.cudagraphs and device.type == "cuda")
    if args.cudagraphs and not use_cudagraphs:
        print("Disabling --cudagraphs because CUDA is unavailable.")
    if use_compile and use_cudagraphs:
        print("Disabling --compile for IQ-Learn update steps because --cudagraphs is active.")
        use_compile = False

    env_time_limit = None
    try:
        env_time_limit = gym.spec(args.env_id).max_episode_steps
    except Exception:
        pass

    demos = load_hf_demos(args, n_demos=args.n_demos)
    demos_all = demos["all"]
    for k in ["obs", "acs", "done"]:
        if k in demos_all:
            demos_all[k] = torch.as_tensor(demos_all[k], device=device, dtype=torch.float32)

    assert "obs" in demos_all and "acs" in demos_all, "Demos must provide obs and acs."
    expert_obs = demos_all["obs"]
    expert_acs = demos_all["acs"]
    expert_done = demos_all.get(
        "done", torch.zeros((expert_obs.shape[0],), device=device)
    )
    expert_terminations = _infer_demo_terminations(expert_done, env_time_limit)
    expert_next_obs = _build_next_obs_from_obs_done(expert_obs, expert_terminations)
    expert_init_obs = _extract_init_obs(expert_obs, expert_done)

    expert_n = expert_obs.shape[0]
    init_n = expert_init_obs.shape[0]

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "Continuous action only"

    n_act = int(np.prod(envs.single_action_space.shape))
    n_obs = int(np.prod(envs.single_observation_space.shape))

    actor = Actor(envs, n_obs=n_obs, n_act=n_act, device=device)
    actor_detach = Actor(envs, n_obs=n_obs, n_act=n_act, device=device)
    from_module(actor).data.to_module(actor_detach)
    policy = TensorDictModule(
        actor_detach.get_action,
        in_keys=["observation"],
        out_keys=["action", "log_prob", "mean"],
    )

    def get_q_params():
        qf1 = SoftQNetwork(n_obs=n_obs, n_act=n_act, device=device)
        qf2 = SoftQNetwork(n_obs=n_obs, n_act=n_act, device=device)
        qnet_params = from_modules(qf1, qf2, as_module=True)
        qnet_target = qnet_params.data.clone()
        qnet = SoftQNetwork(n_obs=n_obs, n_act=n_act, device="meta")
        qnet_params.to_module(qnet)
        return qnet_params, qnet_target, qnet

    qnet_params, qnet_target, qnet = get_q_params()

    q_optimizer = optim.Adam(
        qnet.parameters(),
        lr=args.q_lr,
        capturable=use_cudagraphs,
    )
    actor_optimizer = optim.Adam(
        actor.parameters(),
        lr=args.policy_lr,
        capturable=use_cudagraphs,
    )

    if args.autotune:
        target_entropy = -torch.prod(torch.tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.detach().exp()
        a_optimizer = optim.Adam(
            [log_alpha],
            lr=args.q_lr,
            capturable=use_cudagraphs,
        )
    else:
        alpha = torch.as_tensor(args.alpha, device=device)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    expert_rb = ReplayBuffer(storage=LazyTensorStorage(expert_n, device=device))
    expert_td = TensorDict(
        observations=expert_obs,
        next_observations=expert_next_obs,
        actions=expert_acs,
        rewards=torch.zeros((expert_n, 1), dtype=torch.float32, device=device),
        dones=expert_terminations.view(-1, 1),
        terminations=expert_terminations.view(-1, 1),
        batch_size=[expert_n],
        device=device,
    )
    expert_rb.extend(expert_td)

    init_rb = ReplayBuffer(storage=LazyTensorStorage(init_n, device=device))
    init_td = TensorDict(
        observations=expert_init_obs,
        batch_size=[init_n],
        device=device,
    )
    init_rb.extend(init_td)

    def batched_qf(params, obs, action):
        with params.to_module(qnet):
            return qnet(obs, action)

    def v_from_params(params, obs, detach_actor: bool = True):
        if detach_actor:
            with torch.no_grad():
                act, log_pi, _ = actor.get_action(obs)
        else:
            act, log_pi, _ = actor.get_action(obs)
        q_vals = torch.vmap(batched_qf, (0, None, None))(params, obs, act)
        min_q = q_vals.min(dim=0).values
        return (min_q - alpha * log_pi).view(-1)

    def update_iq_q(td: TensorDict):
        q_optimizer.zero_grad(set_to_none=True)

        p_obs = td["policy_observations"]
        p_act = td["policy_actions"]
        p_nobs = td["policy_next_observations"]
        p_done = td["policy_dones"].view(-1)

        e_obs = td["expert_observations"]
        e_act = td["expert_actions"]
        e_nobs = td["expert_next_observations"]
        e_done = td["expert_dones"].view(-1)

        init_obs = td["init_observations"]

        q_e = torch.vmap(batched_qf, (0, None, None))(qnet_params, e_obs, e_act).squeeze(-1)
        q_p = torch.vmap(batched_qf, (0, None, None))(qnet_params, p_obs, p_act).squeeze(-1)

        with torch.no_grad():
            if args.iq_use_target_next_v:
                v_e_next = v_from_params(qnet_target, e_nobs, detach_actor=True)
                v_p_next = v_from_params(qnet_target, p_nobs, detach_actor=True)
            else:
                v_e_next = v_from_params(qnet_params.data, e_nobs, detach_actor=True)
                v_p_next = v_from_params(qnet_params.data, p_nobs, detach_actor=True)

        v_p = v_from_params(qnet_params, p_obs, detach_actor=True)
        v_e = v_from_params(qnet_params, e_obs, detach_actor=True)
        v_init = v_from_params(qnet_params, init_obs, detach_actor=True)

        q_loss, loss_terms = iq_learn_loss_terms(
            q_expert=q_e,
            q_policy=q_p,
            v_expert=v_e,
            v_policy=v_p,
            v_init=v_init,
            next_v_expert=v_e_next,
            next_v_policy=v_p_next,
            expert_done=e_done,
            policy_done=p_done,
            gamma=args.gamma,
            chi2_coeff=args.iq_chi2_coeff,
            value_coeff=args.iq_value_coeff,
            initial_state_coeff=args.iq_initial_state_coeff,
        )
        q_loss.backward()

        if args.q_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(qnet.parameters(), args.q_grad_clip)

        q_optimizer.step()

        return {
            "q_loss": q_loss.detach(),
            "iq_expert_term": loss_terms["expert_term"].detach(),
            "iq_value_term": loss_terms["value_term"].detach(),
            "iq_chi2_term": loss_terms["chi2_term"].detach(),
            "iq_init_term": loss_terms["init_term"].detach(),
            "q_expert_mean": q_e.mean().detach(),
            "q_policy_mean": q_p.mean().detach(),
            "v_policy_mean": v_p.mean().detach(),
        }

    def update_actor(td: TensorDict):
        actor_optimizer.zero_grad(set_to_none=True)

        pi, log_pi, _ = actor.get_action(td["observations"])
        qf_pi = torch.vmap(batched_qf, (0, None, None))(qnet_params.data, td["observations"], pi)
        min_qf_pi = qf_pi.min(0).values
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

        bc_loss = torch.tensor(0.0, device=device)
        if args.actor_bc_coef > 0:
            _, _, exp_mean_action = actor.get_action(td["expert_observations"])
            bc_loss = F.mse_loss(exp_mean_action, td["expert_actions"])
            actor_loss = actor_loss + args.actor_bc_coef * td["bc_active"].mean() * bc_loss

        actor_loss.backward()
        actor_optimizer.step()

        alpha_loss = torch.tensor(0.0, device=device)
        if args.autotune:
            a_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                _, log_pi2, _ = actor.get_action(td["observations"])
            alpha_loss = (-log_alpha.exp() * (log_pi2 + target_entropy)).mean()
            alpha_loss.backward()
            a_optimizer.step()

        return {
            "actor_loss": actor_loss.detach(),
            "alpha_loss": alpha_loss.detach(),
            "alpha": alpha.detach(),
            "bc_loss": bc_loss.detach(),
        }

    if use_compile:
        update_iq_q = torch.compile(update_iq_q)
        update_actor = torch.compile(update_actor)
    elif use_cudagraphs:
        # Do not pass in_keys=[] here. That selects TensorDict-module mode and
        # locks TensorDict outputs during capture. These update functions are
        # generic callables that take a TensorDict input and return plain dicts.
        update_iq_q = CudaGraphModule(update_iq_q)
        update_actor = CudaGraphModule(update_actor)

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

        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                td_in = TensorDict({"observation": obs}, batch_size=obs.shape[0], device=device)
                td_out = policy(td_in)
            actions = td_out["action"].detach().cpu().numpy()

        next_obs, env_rewards, terminations, truncations, infos = envs.step(actions)

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

        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=torch.as_tensor(actions, device=device, dtype=torch.float32),
            rewards=torch.as_tensor(env_rewards, device=device, dtype=torch.float32).view(-1, 1),
            dones=torch.as_tensor(terminations, device=device, dtype=torch.bool).view(-1, 1),
            terminations=torch.as_tensor(terminations, device=device, dtype=torch.bool).view(-1, 1),
            truncations=torch.as_tensor(truncations, device=device, dtype=torch.bool).view(-1, 1),
            batch_size=obs.shape[0],
            device=device,
        )
        obs = next_obs
        rb.extend(transition)

        if global_step <= args.learning_starts:
            continue

        policy_batch = rb.sample(args.batch_size)
        expert_batch = expert_rb.sample(args.batch_size)
        init_batch = init_rb.sample(args.batch_size)

        iq_batch = TensorDict(
            {
                "policy_observations": policy_batch["observations"],
                "policy_actions": policy_batch["actions"],
                "policy_next_observations": policy_batch["next_observations"],
                "policy_dones": policy_batch["dones"],
                "expert_observations": expert_batch["observations"],
                "expert_actions": expert_batch["actions"],
                "expert_next_observations": expert_batch["next_observations"],
                "expert_dones": expert_batch["dones"],
                "init_observations": init_batch["observations"],
            },
            batch_size=[args.batch_size],
            device=device,
        )

        out_q = update_iq_q(iq_batch)

        out_actor = {
            "actor_loss": torch.tensor(0.0, device=device),
            "alpha_loss": torch.tensor(0.0, device=device),
            "bc_loss": torch.tensor(0.0, device=device),
        }
        if global_step % args.policy_frequency == 0:
            bc_active = float(global_step < args.actor_bc_warmup_steps)
            actor_batch = TensorDict(
                {
                    "observations": torch.cat(
                        [policy_batch["observations"], expert_batch["observations"]], dim=0
                    ),
                    "expert_observations": expert_batch["observations"],
                    "expert_actions": expert_batch["actions"],
                    "bc_active": torch.full((args.batch_size, 1), bc_active, device=device),
                },
                batch_size=[],
                device=device,
            )
            for _ in range(args.policy_frequency):
                out_actor = update_actor(actor_batch)
                if args.autotune:
                    alpha.copy_(log_alpha.detach().exp())
            from_module(actor).data.to_module(actor_detach)

        if global_step % args.target_network_frequency == 0:
            qnet_target.lerp_(qnet_params.data, args.tau)

        if start_time is not None and (global_step % 100 == 0):
            speed = (global_step - measure_burnin_step) / (time.time() - start_time)
            pbar.set_description(f"{speed:5.1f} sps, {desc}")

            if args.track:
                wandb.log(
                    {
                        "speed_sps": speed,
                        "train/episodic_return": np.mean(avg_returns) if len(avg_returns) > 0 else 0.0,
                        "train/q_loss": out_q["q_loss"].mean().item(),
                        "train/actor_loss": out_actor["actor_loss"].mean().item(),
                        "train/alpha_loss": out_actor["alpha_loss"].mean().item(),
                        "train/alpha": float(alpha.detach().cpu().item()),
                        "iq/expert_term": out_q["iq_expert_term"].mean().item(),
                        "iq/value_term": out_q["iq_value_term"].mean().item(),
                        "iq/chi2_term": out_q["iq_chi2_term"].mean().item(),
                        "iq/init_term": out_q["iq_init_term"].mean().item(),
                        "iq/q_expert_mean": out_q["q_expert_mean"].mean().item(),
                        "iq/q_policy_mean": out_q["q_policy_mean"].mean().item(),
                        "iq/v_policy_mean": out_q["v_policy_mean"].mean().item(),
                        "iq/bc_loss": out_actor["bc_loss"].mean().item(),
                    },
                    step=global_step,
                )

    envs.close()
