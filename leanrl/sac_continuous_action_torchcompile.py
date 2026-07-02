# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional
import pickle
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

# from stable_baselines3.common.buffers import ReplayBuffer
from torchrl.data import LazyTensorStorage, ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    normalized_action_entropy: bool = False
    """Use tanh-normalized action coordinates for SAC entropy, omitting the env action-scale log-Jacobian constant."""

    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile."""

    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""
    # Checkpointing / evaluation
    save_dir: str = "checkpoints"
    save_interval: int = 100000
    eval: bool = False
    load_path: Optional[str] = None
    eval_episodes: int = 10
    # Demo saving (eval)
    save_demo: bool = False
    demo_dir: str = "demos"
    demo_out: Optional[str] = None
    # Checkpoint video capture
    save_video: bool = False
    save_video_length: int = 1000
    # Demo saving (eval)
    save_demo: bool = False
    demo_dir: str = "demos"
    demo_out: Optional[str] = None


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
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
    def __init__(self, env, n_obs, n_act, device=None, include_action_scale_in_log_prob: bool = True):
        super().__init__()
        self.include_action_scale_in_log_prob = include_action_scale_in_log_prob
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
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        tanh_jacobian = 1 - y_t.pow(2)
        if self.include_action_scale_in_log_prob:
            tanh_jacobian = self.action_scale * tanh_jacobian
        log_prob -= torch.log(tanh_jacobian + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.compile}__{args.cudagraphs}"

    wandb.init(
        project="sac_continuous_action",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    n_act = math.prod(envs.single_action_space.shape)
    n_obs = math.prod(envs.single_observation_space.shape)
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    max_action = float(envs.single_action_space.high[0])

    include_action_scale_in_log_prob = not args.normalized_action_entropy
    actor = Actor(
        envs,
        device=device,
        n_act=n_act,
        n_obs=n_obs,
        include_action_scale_in_log_prob=include_action_scale_in_log_prob,
    )
    actor_detach = Actor(
        envs,
        device=device,
        n_act=n_act,
        n_obs=n_obs,
        include_action_scale_in_log_prob=include_action_scale_in_log_prob,
    )
    # Copy params to actor_detach without grad
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

        # discard params of net
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
        if args.alpha <= 0:
            raise ValueError(f"--alpha must be positive when --autotune is enabled, got {args.alpha}")
        log_alpha = torch.log(torch.as_tensor([args.alpha], device=device))
        log_alpha.requires_grad_()
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
        # optimize the model
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

    def extend_and_sample(transition):
        rb.extend(transition)
        return rb.sample(args.batch_size)

    is_extend_compiled = False
    if args.compile:
        mode = None  # "reduce-overhead" if not args.cudagraphs else None
        update_main = torch.compile(update_main, mode=mode)
        update_pol = torch.compile(update_pol, mode=mode)
        policy = torch.compile(policy, mode=mode)

    if args.cudagraphs:
        update_main = CudaGraphModule(update_main, in_keys=[], out_keys=[])
        update_pol = CudaGraphModule(update_pol, in_keys=[], out_keys=[])
        # policy = CudaGraphModule(policy)

    # Eval-only helpers
    def load_actor(weights_path: str):
        state = torch.load(weights_path, map_location=device)
        actor.load_state_dict(state)
        actor.eval()

    def evaluate_policy(n_episodes: int) -> float:
        # Use single env instead of SyncVectorEnv to avoid potential hangs
        eval_env = make_env(args.env_id, args.seed, 0, False, run_name)()
        
        ep_returns = []
        obs_buf, acs_buf, rew_buf, done_buf = [], [], [], []
        obs, _ = eval_env.reset(seed=args.seed)
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            while len(ep_returns) < n_episodes:
                mean_action = actor.get_action(obs_t)[2]
                next_obs, rewards, terminations, truncations, infos = eval_env.step(
                    mean_action.cpu().numpy()[0]
                )
                obs_buf.append(np.asarray(obs))
                acs_buf.append(np.asarray(mean_action.cpu().numpy())[0])
                rew_buf.append(float(rewards))
                done_buf.append(bool(terminations or truncations))
                
                if terminations or truncations:
                    if "episode" in infos:
                        ep_returns.append(float(infos["episode"]["r"]))
                    else:
                        # Fallback if RecordEpisodeStatistics doesn't trigger or is different
                        # But RecordEpisodeStatistics is used in make_env
                        # It usually puts 'episode' in info on done
                        pass
                    
                    # Reset if done
                    next_obs, _ = eval_env.reset()
                
                obs = next_obs
                obs_t = torch.as_tensor(obs, device=device, dtype=torch.float).unsqueeze(0)
        eval_env.close()
        avg_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        if args.save_demo:
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
            print(f"Saved demo to {out_path} (obs:{demo['obs'].shape}, acs:{demo['acs'].shape})")
        return avg_ret

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

    if args.eval:
        assert args.load_path is not None and os.path.isfile(args.load_path), (
            "Provide a valid --load_path for eval"
        )
        load_actor(args.load_path)
        avg_ret = evaluate_policy(args.eval_episodes)
        print(f"Eval average return over {args.eval_episodes} episodes: {avg_ret:.2f}")
        raise SystemExit(0)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(obs, device=device, dtype=torch.float)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    start_time = None
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=20)
    desc = ""

    os.makedirs(args.save_dir, exist_ok=True)
    for global_step in pbar:
        if global_step == args.measure_burnin + args.learning_starts:
            start_time = time.time()
            measure_burnin = global_step

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions, _, _ = policy(obs)
            actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "episode" in infos:
            for r in infos["episode"]["r"][infos["episode"]["_r"]]:
                max_ep_ret = max(max_ep_ret, r)
                avg_returns.append(r)
            desc = f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})"

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float)
        real_next_obs = next_obs.clone()
        if "final_observation" in infos:
            for idx, final_obs in enumerate(infos["final_observation"]):
                if infos["_final_observation"][idx]:
                    real_next_obs[idx] = torch.as_tensor(
                        final_obs, device=device, dtype=torch.float
                    )
        # obs = torch.as_tensor(obs, device=device, dtype=torch.float)
        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=torch.as_tensor(actions, device=device, dtype=torch.float),
            rewards=torch.as_tensor(rewards, device=device, dtype=torch.float),
            terminations=terminations,
            dones=terminations,
            batch_size=obs.shape[0],
            device=device,
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        data = extend_and_sample(transition)

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            out_main = update_main(data)
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    out_main.update(update_pol(data))

                    alpha.copy_(log_alpha.detach().exp())

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                qnet_target.lerp_(qnet_params.data, args.tau)

            if args.save_interval and global_step % args.save_interval == 0:
                ckpt_path = os.path.join(
                    args.save_dir, f"{run_name}_actor_step{global_step}.pt"
                )
                torch.save(actor.state_dict(), ckpt_path)
                wandb.save(ckpt_path, policy="now")
            if args.save_interval and global_step % args.save_interval == 0:
                ckpt_path = os.path.join(args.save_dir, f"{run_name}_actor_step{global_step}.pt")
                torch.save(actor.state_dict(), ckpt_path)
                wandb.save(ckpt_path, policy="now")
                record_checkpoint_video(global_step)
            if global_step % 100 == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        "episode_return": torch.tensor(avg_returns).mean(),
                        "actor_loss": out_main["actor_loss"].mean(),
                        "alpha_loss": out_main.get("alpha_loss", 0),
                        "qf_loss": out_main["qf_loss"].mean(),
                    }
                wandb.log(
                    {
                        "speed": speed,
                        **logs,
                    },
                    step=global_step,
                )

    envs.close()

    if args.save_demo:
        evaluate_policy(args.eval_episodes)

    final_path = os.path.join(args.save_dir, f"{run_name}_actor_final.pt")
    torch.save(actor.state_dict(), final_path)
    wandb.save(final_path, policy="now")
