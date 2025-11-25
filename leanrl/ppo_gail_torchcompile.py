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

from irl.gail import GAILDiscriminator, GailReward
from irl.utils import load_hf_demos, prepare_batch_update_irl


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False
    track: bool = False
    wandb_project_name: str = "ppo_gail"
    wandb_entity: str = None

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    num_envs: int = 1
    num_steps: int = 2048
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None

    compile: bool = False
    cudagraphs: bool = False
    measure_burnin: int = 3

    # GAIL / IRL specific
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
    div: str = "rkl" # fkl, rkl, js
    # compatibility flags used by IRL utils
    on_policy: bool = True
    use_sb_ppo: bool = False
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
            env = GailReward(env, disc)
            if args.normalize_irl_rewards:
                env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
                env = gym.wrappers.TransformReward(
                    env, lambda r: np.clip(r, -10, 10)
                )
        
        env.action_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
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
    disc = GAILDiscriminator(shape_env, args).to(device)
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
                args.seed + i,
                i,
                args.capture_video,
                run_name,
                disc,
            )
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(
        agent.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # Discriminator update (compilable + cudagraph-eligible)
    def update_discriminator(ud):
        loss_dict = disc.compute_loss(ud)
        total_loss = (
            loss_dict["d_loss"]
            + args.irm_coeff * loss_dict["grad_penalty"]
            + args.lip_coeff * loss_dict["lip_penalty"]
        )
        disc.d_optimizer.zero_grad()
        total_loss.backward()
        disc.d_optimizer.step()
        return TensorDict(
            d_loss=loss_dict["d_loss"].detach(),
            grad_penalty=torch.as_tensor(loss_dict["grad_penalty"]).detach()
            if isinstance(loss_dict["grad_penalty"], torch.Tensor)
            else torch.tensor(0.0, device=next(iter(disc.parameters())).device),
        )

    # PPO Update logic
    def get_action_and_value(next_obs, next_done):
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
        return action, logprob, value

    def update_ppo(
        b_obs,
        b_logprobs,
        b_actions,
        b_advantages,
        b_returns,
        b_values,
        clip_coef,
        norm_adv,
        ent_coef,
        vf_coef,
    ):
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
            b_obs, b_actions
        )
        logratio = newlogprob - b_logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs = ((ratio - 1.0).abs() > clip_coef).float().mean()

        if norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (
                b_advantages.std() + 1e-8
            )

        # Policy loss
        pg_loss1 = -b_advantages * ratio
        pg_loss2 = -b_advantages * torch.clamp(
            ratio, 1 - clip_coef, 1 + clip_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if args.clip_vloss:
            v_loss_unclipped = (newvalue - b_returns) ** 2
            v_clipped = b_values + torch.clamp(
                newvalue - b_values,
                -clip_coef,
                clip_coef,
            )
            v_loss_clipped = (v_clipped - b_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()
        return (
            old_approx_kl,
            approx_kl,
            clipfracs,
            pg_loss,
            v_loss,
            entropy_loss,
        )

    if args.compile:
        update_ppo = torch.compile(update_ppo)
        get_action_and_value = torch.compile(get_action_and_value)

    if args.cudagraphs:
        update_ppo = CudaGraphModule(update_ppo, in_keys=[], out_keys=[])
        get_action_and_value = CudaGraphModule(
            get_action_and_value, in_keys=[], out_keys=[]
        )

    for update in tqdm.tqdm(range(1, num_updates + 1)):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            action, logprob, value = get_action_and_value(next_obs, next_done)
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        if args.track:
                            wandb.log(
                                {
                                    "charts/episodic_return": info["episode"]["r"],
                                    "charts/episodic_length": info["episode"]["l"],
                                },
                                step=global_step,
                            )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t]
                    + args.gamma * nextvalues * nextnonterminal
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_dones = dones.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                (
                    old_approx_kl,
                    approx_kl,
                    mb_clipfracs,
                    pg_loss,
                    v_loss,
                    entropy_loss,
                ) = update_ppo(
                    b_obs[mb_inds],
                    b_logprobs[mb_inds],
                    b_actions[mb_inds],
                    b_advantages[mb_inds],
                    b_returns[mb_inds],
                    b_values[mb_inds],
                    args.clip_coef,
                    args.norm_adv,
                    args.ent_coef,
                    args.vf_coef,
                )
                clipfracs.append(mb_clipfracs.item())

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Discriminator Update
        # Prepare batch for IRL update
        # We use the collected rollout data
        # Note: prepare_batch_update_irl expects specific format
        # PPO collects (T, N, ...) tensors. We flattened them to (T*N, ...).
        # prepare_batch_update_irl handles this if we pass the flattened tensors.
        
        ud = prepare_batch_update_irl(
            envs,
            args,
            demos_all,
            b_obs,
            b_actions,
            b_dones,
            agent, # PPO agent is used for importance sampling if needed, but GAIL usually doesn't need it for disc update unless using specific losses
        )
        
        out_disc = update_discriminator(ud)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        )

        if args.track:
            wandb.log(
                {
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                    "irl/d_loss": out_disc["d_loss"].mean().item(),
                    "irl/grad_penalty": out_disc.get(
                        "grad_penalty", torch.tensor(0.0)
                    )
                    .mean()
                    .item(),
                },
                step=global_step,
            )

    envs.close()
    
    os.makedirs(args.save_dir, exist_ok=True)
    final_path = os.path.join(args.save_dir, f"{run_name}_agent_final.pt")
    torch.save(agent.state_dict(), final_path)
    if args.track:
        wandb.save(final_path, policy="now")
    final_disc = os.path.join(args.save_dir, f"{run_name}_disc_final.pt")
    torch.save(disc.state_dict(), final_disc)
    if args.track:
        wandb.save(final_disc, policy="now")
