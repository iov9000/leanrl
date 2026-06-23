import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional
import pickle

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

from irl.firl import fIRLDiscriminator, firlReward
from irl.utils import load_hf_demos, prepare_batch_update_irl


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False
    track: bool = False
    wandb_project_name: str = "sac_firl"
    wandb_entity: str = None

    save_video: bool = False
    save_video_length: int = 1000

    # Algorithm specific arguments
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
    disc_period: int = 100  # -1 to update on episode completion
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True

    compile: bool = False
    cudagraphs: bool = False
    measure_burnin: int = 3

    # SWIL / IRL specific
    demo_dir: str = "./demos"
    n_demos: int = 10
    subsample: int = 1
    normalize_irl_rewards: bool = False
    
    # Discriminator architecture/behavior
    use_actions: bool = True
    use_dones: bool = False
    use_next_obs: bool = False
    d_layer_dims: List[int] = field(default_factory=lambda: [128, 128])
    disc_lr: float = 3e-4
    scheduler_gamma: float = 1.0
    
    use_cnn_base: bool = False
    linear_proj: bool = False
    proj_layer: bool = False
    use_disc_bias: bool = False
    use_weight_norm: bool = False
    use_spectral_norm: bool = False
    use_ll_weight_norm: bool = False
    disc_nonlin: str = "tanh"  # relu/leakyrelu/prelu/tanh/id
    irm_coeff: float = 0.0
    lip_coeff: float = 0.0
    lip_p: float = 1.0
    l2_coeff: float = 0.0
    # compatibility flags used by IRL utils
    on_policy: bool = False

    # Checkpoint / evaluation
    save_dir: str = "checkpoints"
    save_interval: int = 100000
    eval: bool = False
    load_path: str = ""
    disc_load_path: str = ""
    eval_episodes: int = 10
    # Demo saving (eval)
    save_demo: bool = False
    demo_out: str = ""


def make_env(args, env_id, seed, idx, capture_video, run_name, disc=None):
    def thunk():
        env = gym.make(
            env_id, render_mode="rgb_array" if capture_video and idx == 0 else None
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        
        if disc is not None:
            env = firlReward(env, disc)
            if args.normalize_irl_rewards:
                env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
                env = gym.wrappers.TransformReward(
                    env, lambda r: np.clip(r, -10, 10)
                )
        
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, n_act, n_obs, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_act + n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc3 = nn.Linear(256, 1, device=device)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env, n_obs, n_act, device=None):
        super().__init__()
        self.fc1 = nn.Linear(n_obs, 256, device=device)
        self.fc2 = nn.Linear(256, 256, device=device)
        self.fc_mean = nn.Linear(256, n_act, device=device)
        self.fc_logstd = nn.Linear(256, n_act, device=device)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
                device=device,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
                device=device,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}"

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
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

    # Create a single env for discriminator shape init
    shape_env = gym.make(args.env_id)
    disc = fIRLDiscriminator(shape_env, args)
    # Ensure discriminator optimizer supports CUDA graph capture when requested
    disc.d_optimizer = optim.Adam(
        disc.parameters(),
        lr=args.disc_lr,
        weight_decay=args.l2_coeff,
        capturable=args.cudagraphs and not args.compile,
    )

    # env setup (vectorized with IRL reward wrapper)
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args,
                args.env_id,
                args.seed,
                0,
                args.capture_video,
                run_name,
                disc,
            )
        ]
    )
    n_act = math.prod(envs.single_action_space.shape)
    n_obs = math.prod(envs.single_observation_space.shape)
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    actor = Actor(envs, device=device, n_act=n_act, n_obs=n_obs)
    actor_detach = Actor(envs, device=device, n_act=n_act, n_obs=n_obs)
    from_module(actor).data.to_module(actor_detach)
    policy = TensorDictModule(
        actor_detach.get_action,
        in_keys=["observation"],
        out_keys=["action", "log_prob", "mean"],
    )

    def get_q_params():
        qf1 = SoftQNetwork(envs, device=device, n_act=n_act, n_obs=n_obs)
        qf2 = SoftQNetwork(envs, device=device, n_act=n_act, n_obs=n_obs)
        qnet_params = from_modules(qf1, qf2, as_module=True)
        qnet_target = qnet_params.data.clone()
        qnet = SoftQNetwork(envs, device="meta", n_act=n_act, n_obs=n_obs)
        qnet_params.to_module(qnet)
        return qnet_params, qnet_target, qnet

    qnet_params, qnet_target, qnet = get_q_params()

    q_optimizer = optim.Adam(
        qnet.parameters(), lr=args.q_lr, capturable=args.cudagraphs and not args.compile
    )
    actor_optimizer = optim.Adam(
        list(actor.parameters()),
        lr=args.policy_lr,
        capturable=args.cudagraphs and not args.compile,
    )

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.single_action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.detach().exp()
        a_optimizer = optim.Adam(
            [log_alpha], lr=args.q_lr, capturable=args.cudagraphs and not args.compile
        )
    else:
        alpha = torch.as_tensor(args.alpha, device=device)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    def batched_qf(params, obs, action, next_q_value=None):
        with params.to_module(qnet):
            vals = qnet(obs, action)
            if next_q_value is not None:
                loss_val = F.mse_loss(vals.view(-1), next_q_value)
                return loss_val
            return vals

    def update_main(data):
        q_optimizer.zero_grad()
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(
                data["next_observations"]
            )
            qf_next_target = torch.vmap(batched_qf, (0, None, None))(
                qnet_target, data["next_observations"], next_state_actions
            )
            min_qf_next_target = (
                qf_next_target.min(dim=0).values - alpha * next_state_log_pi
            )
            next_q_value = data["rewards"].flatten() + (
                ~data["dones"].flatten()
            ).float() * args.gamma * min_qf_next_target.view(-1)

        qf_a_values = torch.vmap(batched_qf, (0, None, None, None))(
            qnet_params, data["observations"], data["actions"], next_q_value
        )
        qf_loss = qf_a_values.sum(0)

        qf_loss.backward()
        q_optimizer.step()
        return TensorDict(qf_loss=qf_loss.detach())

    def update_pol(data):
        actor_optimizer.zero_grad()
        pi, log_pi, _ = actor.get_action(data["observations"])
        qf_pi = torch.vmap(batched_qf, (0, None, None))(
            qnet_params.data, data["observations"], pi
        )
        min_qf_pi = qf_pi.min(0).values
        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

        actor_loss.backward()
        actor_optimizer.step()

        if args.autotune:
            a_optimizer.zero_grad()
            with torch.no_grad():
                _, log_pi, _ = actor.get_action(data["observations"])
            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
            alpha_loss.backward()
            a_optimizer.step()
        return TensorDict(
            alpha=alpha.detach(),
            actor_loss=actor_loss.detach(),
            alpha_loss=alpha_loss.detach(),
        )

    # Discriminator update (compilable + cudagraph-eligible)
    def update_disc(ud):
        # Dual optimization loop for FIRL
        # 1. Update discriminator (reward network + discriminator layers)
        loss_dict = disc.update_disc(ud)
        
        # 2. Update reward network using f-divergence loss
        disc.update(ud)
        
        return TensorDict(
            d_loss=loss_dict["d_loss"].detach(),
            grad_penalty=torch.tensor(0.0, device=device), # FIRL doesn't use GP in the same way
        )

    if args.compile:
        mode = None
        update_main = torch.compile(update_main, mode=mode)
        update_pol = torch.compile(update_pol, mode=mode)
        policy = torch.compile(policy, mode=mode)
        update_disc = torch.compile(update_disc, mode=mode)

    if args.cudagraphs:
        update_main = CudaGraphModule(update_main, in_keys=[], out_keys=[])
        update_pol = CudaGraphModule(update_pol, in_keys=[], out_keys=[])
        # update_disc will be captured manually below once we see a first batch.

    # Eval helpers: load and evaluate actor
    def load_actor(weights_path: str):
        state = torch.load(weights_path, map_location=device)
        actor.load_state_dict(state)
        actor.eval()

    def load_disc(weights_path: str):
        state = torch.load(weights_path, map_location=device)
        disc.load_state_dict(state)
        disc.eval()

    def evaluate_policy(n_episodes: int, use_swil_wrapper: bool = False) -> float:
        base_env = gym.vector.SyncVectorEnv(
            [
                make_env(
                    args,
                    args.env_id,
                    args.seed,
                    0,
                    False,
                    run_name,
                    disc if use_swil_wrapper else None,
                )
            ]
        )
        ep_returns = []
        obs_buf, acs_buf, rew_buf, done_buf = [], [], [], []
        obs, _ = base_env.reset(seed=args.seed)
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float)
        with torch.no_grad():
            while len(ep_returns) < n_episodes:
                mean_action = actor.get_action(obs_t)[2]
                next_obs, rewards, terminations, truncations, infos = base_env.step(
                    mean_action.cpu().numpy()
                )
                obs_buf.append(np.asarray(obs)[0])
                acs_buf.append(np.asarray(mean_action.cpu().numpy())[0])
                rew_buf.append(float(rewards[0]))
                done_buf.append(bool(terminations[0] or truncations[0]))
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        ep_returns.append(float(info["episode"]["r"]))
                obs = next_obs
                obs_t = torch.as_tensor(obs, device=device, dtype=torch.float)
        base_env.close()
        avg_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        if args.save_demo:
            import os, numpy as np, pickle

            os.makedirs(args.demo_dir, exist_ok=True)
            out_path = (
                args.demo_out
                if args.demo_out
                else os.path.join(args.demo_dir, f"demo_hf_{args.env_id}_policy.pkl")
            )
            demo = {
                "obs": np.asarray(obs_buf, dtype=np.float32),
                "acs": np.asarray(acs_buf, dtype=np.float32),
                "rew": np.asarray(rew_buf, dtype=np.float32),
                "done": np.asarray(done_buf, dtype=np.float32),
            }
            demo["support_obs"] = demo["obs"]
            demo["support_acs"] = demo["acs"]
            demo["support_rew"] = demo["rew"]
            demo["support_done"] = demo["done"]
            with open(out_path, "wb") as f:
                pickle.dump(demo, f)
            print(
                f"Saved demo to {out_path} (obs:{demo['obs'].shape}, acs:{demo['acs'].shape})"
            )
        return avg_ret

    if args.eval:
        assert args.load_path and os.path.isfile(args.load_path), (
            "Provide a valid --load_path for eval"
        )
        load_actor(args.load_path)
        use_irl = bool(args.disc_load_path) and os.path.isfile(args.disc_load_path)
        if use_irl:
            load_disc(args.disc_load_path)
        avg_ret = evaluate_policy(args.eval_episodes, use_swil_wrapper=use_irl)
        print(f"Eval average return over {args.eval_episodes} episodes: {avg_ret:.2f}")
        raise SystemExit(0)

    def record_checkpoint_video(step_id: int):
        if not args.save_video:
            return
        video_dir = os.path.join("videos", run_name)
        os.makedirs(video_dir, exist_ok=True)
        env = gym.make(args.env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda e: True)
        obs, _ = env.reset(seed=args.seed)
        t = 0
        with torch.no_grad():
            while t < args.save_video_length:
                obs_t = torch.as_tensor(obs, device=device, dtype=torch.float)
                mean_action = actor.get_action(obs_t)[2]
                obs, _, term, trunc, _ = env.step(mean_action.cpu().numpy())
                t += 1
                if term or trunc:
                    break
        env.close()

    # Main loop
    obs, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(obs, dtype=torch.float)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=20)
    desc = ""
    disc_graph = None
    disc_buffers = None

    os.makedirs(args.save_dir, exist_ok=True)
    for global_step in pbar:
        irl_trigger = False
        if global_step == args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            td_in = TensorDict(
                {"observation": obs}, batch_size=obs.shape[0], device=device
            )
            td_out = policy(td_in)
            actions = td_out["action"].detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "episode" in infos:
            irl_trigger = True
            for r in infos["episode"]["r"][infos["episode"]["_r"]]:
                max_ep_ret = max(max_ep_ret, r)
                avg_returns.append(r)
            desc = f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})"

        next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float)
        real_next_obs = next_obs.clone()
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if infos["_final_observation"][idx]:
                    real_next_obs[idx] = torch.as_tensor(
                        final_obs, device=device, dtype=torch.float
                    )

        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=torch.as_as_tensor(actions, device=device, dtype=torch.float),
            rewards=torch.as_as_tensor(rewards, device=device, dtype=torch.float),
            terminations=terminations,
            dones=terminations,
            batch_size=obs.shape[0],
            device=device,
        )

        obs = next_obs
        rb.extend(transition)

        # ALGO LOGIC: training SAC
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            out_disc = None
            out_main = update_main(data)
            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    out_main.update(update_pol(data))
                    if args.autotune:
                        alpha.copy_(log_alpha.detach().exp())

            if global_step % args.target_network_frequency == 0:
                qnet_target.lerp_(qnet_params.data, args.tau)

            irl_condition = global_step % args.disc_period == 0
            if args.disc_period == -1:
                irl_condition = irl_trigger
            if irl_condition:
                ud = prepare_batch_update_irl(
                    envs,
                    args,
                    demos_all,
                    data["observations"],
                    data["next_observations"],
                    data["actions"],
                    data["dones"],
                    actor,
                )
                if args.cudagraphs:
                    if disc_graph is None or any(
                        disc_buffers[k].shape != ud[k].shape for k in ud
                    ):
                        disc_buffers = {k: v.clone() for k, v in ud.items()}
                        disc_graph = torch.cuda.make_graphed_callables(
                            update_disc, (disc_buffers,)
                        )
                        out_disc = disc_graph(disc_buffers)
                    else:
                        for k in disc_buffers:
                            disc_buffers[k].copy_(ud[k])
                        out_disc = disc_graph(disc_buffers)
                else:
                    out_disc = update_disc(ud)

            if args.track and global_step % 100 == 0 and out_disc is not None:
                wandb.log(
                    {
                        "irl/d_loss": out_disc["d_loss"].mean().item(),
                    },
                    step=global_step,
                )

            if args.save_interval and global_step % args.save_interval == 0:
                ckpt_actor = os.path.join(
                    args.save_dir, f"{run_name}_actor_step{global_step}.pt"
                )
                torch.save(actor.state_dict(), ckpt_actor)
                if args.track:
                    wandb.save(ckpt_actor, policy="now")
                ckpt_disc = os.path.join(
                    args.save_dir, f"{run_name}_disc_step{global_step}.pt"
                )
                torch.save(disc.state_dict(), ckpt_disc)
                if args.track:
                    wandb.save(ckpt_disc, policy="now")
                record_checkpoint_video(global_step)
            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                if args.track:
                    with torch.no_grad():
                        logs = {
                            "episode_return": torch.tensor(avg_returns).mean(),
                            "actor_loss": out_main.get(
                                "actor_loss", torch.tensor(0.0)
                            ).mean(),
                            "alpha_loss": out_main.get("alpha_loss", torch.tensor(0.0)),
                            "qf_loss": out_main["qf_loss"].mean(),
                        }
                    wandb.log(
                        {
                            "speed": speed,
                            **{
                                k: (v.item() if isinstance(v, torch.Tensor) else v)
                                for k, v in logs.items()
                            },
                        },
                        step=global_step,
                    )

    envs.close()
    final_path = os.path.join(args.save_dir, f"{run_name}_actor_final.pt")
    torch.save(actor.state_dict(), final_path)
    if args.track:
        wandb.save(final_path, policy="now")
    final_disc = os.path.join(args.save_dir, f"{run_name}_disc_final.pt")
    torch.save(disc.state_dict(), final_disc)
    if args.track:
        wandb.save(final_disc, policy="now")
