
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

# Reuse your existing demo loader if you have it (as in the provided script).
# If not, replace this with your own demo loading.
from irl.utils import load_hf_demos


# ----------------------------
# Sinkhorn OT (log-domain)
# ----------------------------
@torch.no_grad()
def sinkhorn_plan(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    epsilon: float = 0.05,
    n_iters: int = 50,
) -> torch.Tensor:
    """
    Compute entropic OT plan P between discrete measures a (n,) and b (m,)
    with cost matrix C (n,m), using stabilized log-domain Sinkhorn.

    Returns:
        P: (n,m) transport plan with row sums ~a and col sums ~b.
    """
    # K = exp(-C/eps) but in log form
    logK = -C / epsilon  # (n,m)

    # Duals in log space
    loga = torch.log(a + 1e-12)  # (n,)
    logb = torch.log(b + 1e-12)  # (m,)

    u = torch.zeros_like(a)  # actually stores log u
    v = torch.zeros_like(b)  # log v

    # Alternate normalization
    for _ in range(n_iters):
        u = loga - torch.logsumexp(logK + v.unsqueeze(0), dim=1)
        v = logb - torch.logsumexp(logK.transpose(0, 1) + u.unsqueeze(0), dim=1)

    # Plan in linear domain
    P = torch.exp(logK + u.unsqueeze(1) + v.unsqueeze(0))
    return P


@torch.no_grad()
def ot_reward_from_plan(
    P: torch.Tensor,
    C: torch.Tensor,
    a: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample reward proxy for learner points (rows):
        r_i = - E_{j ~ P(j|i)}[C_ij]
            = - (sum_j P_ij * C_ij) / a_i
    where a_i is the row marginal mass.
    """
    row_cost = (P * C).sum(dim=1)
    r = -row_cost / (a + 1e-12)
    return r


# ----------------------------
# Env helper
# ----------------------------
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
# Networks (SAC + SF)
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


class PhiNet(nn.Module):
    """Instantaneous feature map phi(s,a)."""
    def __init__(self, n_obs: int, n_act: int, d: int, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs + n_act, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc3 = nn.Linear(256, d, device=device)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PsiNet(nn.Module):
    """Successor feature embedding psi(s,a)."""
    def __init__(self, n_obs: int, n_act: int, d: int, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs + n_act, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc3 = nn.Linear(256, d, device=device)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ----------------------------
# Args
# ----------------------------
@dataclass
class Args:
    exp_name: str = "sac_sinkhorn_sf_torchcompile"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False
    track: bool = False
    wandb_project_name: str = "sac_sinkhorn_sf"
    wandb_entity: str = None

    env_id: str = "HalfCheetah-v4"
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

    compile: bool = False
    cudagraphs: bool = False
    measure_burnin: int = 3

    # Expert demos
    demo_dir: str = "./demos"
    n_demos: int = 10
    subsample: int = 1

    # OT / Sinkhorn reward
    irl_start: int = 5_000          # start using OT reward after this many steps
    irl_period: int = 1             # compute OT reward every update (keep 1 for simplest)
    ot_epsilon: float = 0.05
    ot_iters: int = 50
    ot_cost: str = "cosine"         # cosine|l2
    ot_reward_scale: float = 1.0
    ot_reward_clip: float = 10.0    # 0 disables clipping

    # Successor features
    sf_dim: int = 64
    sf_lr: float = 3e-4
    sf_tau: float = 0.005
    sf_update_frequency: int = 1
    sf_warmup_updates: int = 1000   # updates before we trust OT reward (stabilizes early training)
    sf_grad_clip: float = 10.0     # 0 disables grad clipping
    sf_l2_reg: float = 0.0         # L2 reg on psi/phi weights (e.g., 1e-6)
    sf_use_mean_next_action: bool = True  # reduce SF target noise

    sf_gamma: float = 0.97          # SF discount (often smaller than SAC gamma for Humanoid/Walker2d)
    sf_loss: str = "huber"        # mse|huber
    sf_huber_delta: float = 10.0    # beta for smooth_l1_loss
    sf_done_on_trunc: bool = True   # treat truncations as done for SF target
    sf_recent_buffer: bool = True   # use a small recent buffer for SF updates (reduces off-policy drift)
    sf_recent_buffer_size: int = 200_000

    ot_use_psi_target: bool = True # compute OT embedding with psi_target (stabilizes geometry)

    # Mix env reward (debug only; for pure imitation set env_reward_scale=0)
    env_reward_scale: float = 0.0


# ----------------------------
# Main
# ----------------------------
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

    # Load expert demos (expects dict with keys obs, acs, done, ...)
    demos = load_hf_demos(args, n_demos=args.n_demos)
    demos_all = demos["all"]
    for k in ["obs", "acs", "done"]:
        if k in demos_all:
            demos_all[k] = torch.as_tensor(demos_all[k], device=device, dtype=torch.float32)
    expert_obs = demos_all["obs"]
    expert_acs = demos_all["acs"]
    expert_N = expert_obs.shape[0]

    # Envs
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "Continuous action space only."

    n_act = int(np.prod(envs.single_action_space.shape))
    n_obs = int(np.prod(envs.single_observation_space.shape))

    # Actor + policy module
    actor = Actor(envs, n_obs=n_obs, n_act=n_act, device=device)
    actor_detach = Actor(envs, n_obs=n_obs, n_act=n_act, device=device)
    from_module(actor).data.to_module(actor_detach)

    policy = TensorDictModule(
        actor_detach.get_action,
        in_keys=["observation"],
        out_keys=["action", "log_prob", "mean"],
    )

    # Q nets
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

    # Successor feature models
    phi = PhiNet(n_obs, n_act, d=args.sf_dim, device=device)
    psi = PsiNet(n_obs, n_act, d=args.sf_dim, device=device)
    psi_target = PsiNet(n_obs, n_act, d=args.sf_dim, device=device)
    psi_target.load_state_dict(psi.state_dict())
    sf_optimizer = optim.Adam(list(phi.parameters()) + list(psi.parameters()), lr=args.sf_lr, capturable=args.cudagraphs and not args.compile)

    # Replay
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    # Optional small recent buffer for SF to reduce off-policy drift on hard envs
    sf_rb = None
    if args.sf_recent_buffer:
        sf_rb = ReplayBuffer(storage=LazyTensorStorage(args.sf_recent_buffer_size, device=device))

    # Utility: batched Q
    def batched_qf(params, obs, action, next_q_value=None):
        with params.to_module(qnet):
            vals = qnet(obs, action)
            if next_q_value is not None:
                return F.mse_loss(vals.view(-1), next_q_value)
            return vals

    
    # Successor feature TD update (residual formulation so phi can learn too)
    def update_sf(data: TensorDict):
        sf_optimizer.zero_grad(set_to_none=True)
        obs = data["observations"]
        act = data["actions"]
        next_obs = data["next_observations"]
        dones = data["dones"].float().view(-1, 1)

        # Next action for the target term (mean action reduces variance)
        with torch.no_grad():
            if args.sf_use_mean_next_action:
                _, _, next_act = actor.get_action(next_obs)  # mean_action
            else:
                next_act, _, _ = actor.get_action(next_obs)  # sampled action
            psi_next = psi_target(next_obs, next_act)

        psi_sa = psi(obs, act)
        phi_sa = phi(obs, act)

        # Bellman residual: psi(s,a) - [phi(s,a) + gamma(1-d) psi_tgt(s',a')]
                # Optionally treat time-limit truncations as terminal for SF (helps Humanoid/Walker2d)
        if args.sf_done_on_trunc and "truncations" in data.keys():
            dones_sf = (data["dones"].view(-1, 1) | data["truncations"].view(-1, 1)).float()
        else:
            dones_sf = dones

        residual = psi_sa - (phi_sa + (1.0 - dones_sf) * args.sf_gamma * psi_next)
        if args.sf_loss == "mse":
            loss = (residual * residual).mean()
        elif args.sf_loss == "huber":
            loss = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=args.sf_huber_delta)
        else:
            raise ValueError(f"Unknown --sf_loss {args.sf_loss}")

        # Optional L2 regularization to prevent norm blow-up
        if args.sf_l2_reg and args.sf_l2_reg > 0:
            l2 = torch.tensor(0.0, device=device)
            for p in list(phi.parameters()) + list(psi.parameters()):
                l2 = l2 + (p * p).sum()
            loss = loss + args.sf_l2_reg * l2

        loss.backward()
        if args.sf_grad_clip and args.sf_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(phi.parameters()) + list(psi.parameters()), args.sf_grad_clip)
        sf_optimizer.step()

        # soft-update target
        with torch.no_grad():
            for p_t, p in zip(psi_target.parameters(), psi.parameters()):
                p_t.lerp_(p, args.sf_tau)

        return TensorDict(sf_loss=loss.detach())

    @torch.no_grad()
    def compute_ot_rewards(data: TensorDict):
        obs = data["observations"]
        act = data["actions"]
        n = obs.shape[0]

        # Sample expert batch of same size
        idx = torch.randint(0, expert_N, (n,), device=device)
        e_obs = expert_obs[idx]
        e_act = expert_acs[idx]

        embed = psi_target if args.ot_use_psi_target else psi
        z_pi = F.normalize(embed(obs, act), dim=-1)
        z_e = F.normalize(embed(e_obs, e_act), dim=-1)

        if args.ot_cost == "cosine":
            C = 1.0 - (z_pi @ z_e.t())
        elif args.ot_cost == "l2":
            C = torch.cdist(z_pi, z_e, p=2)
        else:
            raise ValueError(f"Unknown --ot_cost {args.ot_cost}")

        a = torch.full((n,), 1.0 / n, device=device)
        b = torch.full((n,), 1.0 / n, device=device)

        P = sinkhorn_plan(a, b, C, epsilon=args.ot_epsilon, n_iters=args.ot_iters)
        r_ot = ot_reward_from_plan(P, C, a)  # (n,)
        sinkhorn_cost = (P * C).sum()

        # scale + clip
        r_ot = r_ot * args.ot_reward_scale
        if args.ot_reward_clip and args.ot_reward_clip > 0:
            r_ot = torch.clamp(r_ot, -args.ot_reward_clip, args.ot_reward_clip)

        return r_ot.view(-1, 1), sinkhorn_cost.detach()

    # SAC updates (use OT reward when enabled)
    def update_main(data: TensorDict, rewards_override: torch.Tensor | None = None):
        q_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(data["next_observations"])
            qf_next_target = torch.vmap(batched_qf, (0, None, None))(
                qnet_target, data["next_observations"], next_state_actions
            )
            min_qf_next_target = qf_next_target.min(dim=0).values - alpha * next_state_log_pi

            rewards = data["rewards"]
            if rewards_override is not None:
                rewards = rewards_override

            next_q_value = rewards.flatten() + (~data["dones"].flatten()).float() * args.gamma * min_qf_next_target.view(-1)

        qf_losses = torch.vmap(batched_qf, (0, None, None, None))(
            qnet_params, data["observations"], data["actions"], next_q_value
        )
        qf_loss = qf_losses.sum(0)
        qf_loss.backward()
        q_optimizer.step()
        return TensorDict(qf_loss=qf_loss.detach())

    def update_pol(data: TensorDict):
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
            alpha=alpha.detach(),
            actor_loss=actor_loss.detach(),
            alpha_loss=alpha_loss.detach(),
        )

    if args.compile:
        mode = None
        update_main = torch.compile(update_main, mode=mode)
        update_pol = torch.compile(update_pol, mode=mode)
        update_sf = torch.compile(update_sf, mode=mode)

    if args.cudagraphs:
        update_main = CudaGraphModule(update_main, in_keys=[], out_keys=[])
        update_pol = CudaGraphModule(update_pol, in_keys=[], out_keys=[])
        update_sf = CudaGraphModule(update_sf, in_keys=[], out_keys=[])

    # Main loop
    obs, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    measure_burnin = None
    avg_returns = deque(maxlen=20)
    max_ep_ret = -float("inf")
    sf_updates = 0
    desc = ""

    for global_step in pbar:
        if global_step == args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

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

        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=torch.as_tensor(actions, device=device, dtype=torch.float32),
            rewards=torch.as_tensor(env_rewards, device=device, dtype=torch.float32).view(-1, 1),
                        dones=(terminations | truncations),
            terminations=terminations,
            truncations=truncations,
            batch_size=obs.shape[0],
            device=device,
        )
        obs = next_obs
        rb.extend(transition)
        if sf_rb is not None:
            sf_rb.extend(transition)

        # Train
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            # Update successor features
            out_sf = None
            if (global_step % args.sf_update_frequency) == 0:
                out_sf = update_sf(data)
                sf_updates += 1

            # Compute OT reward (optionally gated)
            use_ot = (global_step >= args.irl_start) and (sf_updates >= args.sf_warmup_updates)
            r_ot = None
            sinkhorn_cost = None
            if use_ot and (global_step % args.irl_period == 0):
                r_ot, sinkhorn_cost = compute_ot_rewards(data)

            # Mix env reward if desired
            rewards_override = None
            if use_ot and r_ot is not None:
                rewards_override = r_ot + args.env_reward_scale * data["rewards"]

            out_main = update_main(data, rewards_override=rewards_override)

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    out_main.update(update_pol(data))
                    if args.autotune:
                        alpha.copy_(log_alpha.detach().exp())

            if global_step % args.target_network_frequency == 0:
                qnet_target.lerp_(qnet_params.data, args.tau)

            # Logging
            if start_time is not None and (global_step % 100 == 0):
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed:5.1f} sps, {desc}")

                if args.track:
                    logs = {
                        "speed_sps": speed,
                        "train/qf_loss": out_main["qf_loss"].mean().item(),
                        "train/actor_loss": out_main.get("actor_loss", torch.tensor(0.0)).mean().item(),
                        "train/alpha": float(alpha.detach().cpu().item()),
                        "train/episodic_return": np.mean(avg_returns) if len(avg_returns) > 0 else 0.0,
                    }
                    if out_sf is not None:
                        logs["sf/loss"] = out_sf["sf_loss"].mean().item()
                        with torch.no_grad():
                            zpsi = psi(data["observations"], data["actions"]).norm(dim=-1).mean()
                            zphi = phi(data["observations"], data["actions"]).norm(dim=-1).mean()
                        logs["sf/psi_norm"] = float(zpsi.cpu().item())
                        logs["sf/phi_norm"] = float(zphi.cpu().item())
                    if sinkhorn_cost is not None:
                        logs["ot/sinkhorn_cost"] = float(sinkhorn_cost.cpu().item())
                        logs["ot/reward_mean"] = float(r_ot.mean().cpu().item())
                    wandb.log(logs, step=global_step)

    envs.close()
