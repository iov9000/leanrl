from __future__ import annotations

from argparse import Namespace
from typing import Dict, Mapping, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm
from torch.optim import Adam

try:
    from leanrl.il_utils import (
        compute_online_phases,
        encode_phase_features,
        phase_feature_dim,
        validate_phase_mode,
    )
except ImportError:
    from il_utils import (
        compute_online_phases,
        encode_phase_features,
        phase_feature_dim,
        validate_phase_mode,
    )

TensorDict = Mapping[str, torch.Tensor]


def _resolve_phase_horizon(opt: Namespace, env: gym.Env | None = None) -> int:
    explicit = int(getattr(opt, "imitation_time_horizon", 0))
    if explicit > 0:
        return explicit
    if env is not None:
        spec = getattr(env, "spec", None)
        max_steps = getattr(spec, "max_episode_steps", None)
        if max_steps is not None:
            return int(max_steps)
    return 1000

def layer_init(
    layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def linlayer(
    in_dim: int,
    out_dim: int,
    bias: bool = True,
    wnorm: bool = False,
    snorm: bool = False,
) -> nn.Linear:
    if wnorm:
        return weight_norm(nn.Linear(in_dim, out_dim, bias=bias), "weight")
    elif snorm:
        return spectral_norm(nn.Linear(in_dim, out_dim, bias=bias), "weight")
    else:
        return nn.Linear(in_dim, out_dim, bias=bias)

class GAILDiscriminator(nn.Module):
    def __init__(self, env: gym.Env, args: Namespace) -> None:
        super(GAILDiscriminator, self).__init__()

        self.env = env
        self.args = args
        self.layer_dims = args.d_layer_dims
        self.lr = args.disc_lr
        self.use_actions = getattr(args, "use_actions", True)
        self.use_dones = getattr(args, "use_dones", False)
        self.use_next_obs = getattr(args, "use_next_obs", False)
        self.phase_mode = validate_phase_mode(getattr(args, "imitation_phase_mode", "none"))
        self.phase_dim = phase_feature_dim(self.phase_mode)
        self.irm_coeff = args.irm_coeff
        self.l2_coeff = args.l2_coeff
        self.lip_coeff = args.lip_coeff
        self.bias = args.use_disc_bias
        self.snorm = args.use_spectral_norm
        self.wnorm = args.use_weight_norm

        if args.disc_nonlin == 'relu':
            nonlin = nn.ReLU()
        elif args.disc_nonlin == 'tanh':
            nonlin = nn.Tanh()
        elif args.disc_nonlin == 'leakyrelu':
            nonlin = nn.LeakyReLU()
        elif args.disc_nonlin == 'silu':
            nonlin = nn.SiLU()
        else:
            nonlin = nn.PReLU()

        if hasattr(env, "single_observation_space"):
            if isinstance(env.single_observation_space, gym.spaces.Dict):
                ob_shapes = list(env.single_observation_space['observation'].shape)
            else:
                ob_shapes = list(env.single_observation_space.shape)
            ac_shapes = list(env.single_action_space.shape)
        else:
            if isinstance(env.observation_space, gym.spaces.Dict):
                ob_shapes = list(env.observation_space['observation'].shape)
            else:
                ob_shapes = list(env.observation_space.shape)
            ac_shapes = list(env.action_space.shape)

        if not ac_shapes:
            ac_shapes = [1]

        dim0 = ob_shapes[-1]
        if self.use_actions:
            dim0 = dim0 + ac_shapes[-1]
        if self.use_dones:
            dim0 = dim0 + 1
        if self.use_next_obs:
            dim0 = dim0 + ob_shapes[-1]
        dim0 = dim0 + self.phase_dim
        if self.use_next_obs:
            dim0 = dim0 + self.phase_dim

        self.layer_dims = [dim0] + self.layer_dims
        
        # Ensure layer_dims are ints
        self.layer_dims = [int(x) for x in self.layer_dims]
        
        layer_dims = self.layer_dims

        self.base = nn.Sequential(
            linlayer(
                self.layer_dims[0],
                self.layer_dims[1],
                self.bias,
                self.wnorm,
                self.snorm,
            ),
            nonlin,
        )

        self.discriminator_layers = []
        for i in range(2, len(layer_dims)):
            self.discriminator_layers += [
                linlayer(
                    in_dim=layer_dims[i - 1],
                    out_dim=layer_dims[i],
                    bias=self.bias,
                    wnorm=self.wnorm,
                    snorm=self.snorm,
                ),
                nonlin,
            ]

        self.discriminator_layers += [
            linlayer(
                in_dim=layer_dims[-1],
                out_dim=1,
                bias=self.bias,
                wnorm=self.wnorm,
                snorm=self.snorm,
            )
        ]

        self.discriminator = nn.Sequential(*self.discriminator_layers)

        self.d_optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coeff)

    def base_fwd(
        self,
        ob: torch.Tensor,
        ac: torch.Tensor | None,
        nob: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        next_phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        #  match tensor sizes
        if ac is not None:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
        if d is not None and len(ob.shape) != len(d.shape):
            d = torch.unsqueeze(d, -1)

        input_ = [ob]
        phase_features = encode_phase_features(phase, self.phase_mode)
        if self.use_actions:
            input_.append(ac)
        if self.use_next_obs:
            input_.append(nob)
            if phase_features is not None:
                input_.append(encode_phase_features(next_phase, self.phase_mode))
        if self.use_dones:
            input_.append(d)
        if phase_features is not None:
            input_.append(phase_features)

        base_out = self.base(torch.cat(input_, axis=-1))

        return base_out

    def forward(
        self,
        ob: torch.Tensor,
        ac: torch.Tensor | None = None,
        nob: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        next_phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_out = self.base_fwd(ob, ac, nob, d, phase, next_phase)

        d_out = self.discriminator(base_out)

        div = getattr(self.args, "div", "rkl")
        if div == 'fkl':
            d_out_ = torch.exp(d_out) # (N*T,) p/q TODO: clip
        elif div == 'rkl':
            d_out_ = d_out # (N*T,) log (p/q)
        elif div == 'js':  # https://pytorch.org/docs/master/generated/torch.nn.Softplus.html
            d_out_ = F.softplus(d_out) # (N*T,) log (1 + p/q)
        else:
             d_out_ = d_out

        return d_out_

    def get_reward(
        self,
        ob: torch.Tensor,
        ac: torch.Tensor | None,
        nob: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
        phase: torch.Tensor | None = None,
        next_phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_out = self.base_fwd(ob, ac, nob, d, phase, next_phase)

        d_out = self.discriminator(base_out)

        # Allow choosing reward shaping: default is -log(1 - D) (standard GAIL).
        # Some setups used -log(D); keep a toggle for backwards compatibility.
        use_neg_log_d = getattr(self.args, "gail_reward_neg_log_d", False)
        if use_neg_log_d:
            # -log D = softplus(-logits)
            reward = F.softplus(-d_out)
        else:
            # -log(1 - D) = softplus(logits)
            reward = F.softplus(d_out)

        self.reward = reward.squeeze(-1)
        return self.reward

    def irm_penalty(
        self, logits: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = torch.tensor(1.0, device=logits.device).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2), grad

    def lip_penalty(
        self, update_dict: TensorDict, p: float = 1
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        policy_obs = update_dict['policy_obs']
        policy_acs = update_dict['policy_acs']
        policy_obs_next = update_dict['policy_obs_next']
        policy_dones = update_dict['policy_dones']
        policy_phase = update_dict["policy_phase"]
        policy_next_phase = update_dict["policy_next_phase"]
        exp_obs = update_dict['expert_obs']
        exp_acs = update_dict['expert_acs']
        exp_obs_next = update_dict['expert_obs_next']
        exp_dones = update_dict['expert_dones']
        exp_phase = update_dict["expert_phase"]
        exp_next_phase = update_dict["expert_next_phase"]

        # Handle mismatched batch sizes by subsampling the larger batch
        min_bs = min(policy_obs.shape[0], exp_obs.shape[0])
        if policy_obs.shape[0] != min_bs:
            idx = torch.randperm(policy_obs.shape[0], device=policy_obs.device)[:min_bs]
            policy_obs = policy_obs[idx]
            policy_acs = policy_acs[idx]
            policy_obs_next = policy_obs_next[idx]
            policy_dones = policy_dones[idx]
            policy_phase = policy_phase[idx]
            policy_next_phase = policy_next_phase[idx]
        if exp_obs.shape[0] != min_bs:
            idx = torch.randperm(exp_obs.shape[0], device=exp_obs.device)[:min_bs]
            exp_obs = exp_obs[idx]
            exp_acs = exp_acs[idx]
            exp_obs_next = exp_obs_next[idx]
            exp_dones = exp_dones[idx]
            exp_phase = exp_phase[idx]
            exp_next_phase = exp_next_phase[idx]

        if len(exp_obs.shape) != len(exp_dones.shape):
            exp_dones = torch.unsqueeze(exp_dones, -1)

        if len(policy_obs.shape) != len(policy_dones.shape):
            policy_dones = torch.unsqueeze(policy_dones, -1)

        obs_epsilon = torch.rand(policy_obs.shape, device=policy_obs.device)
        interp_obs = obs_epsilon * policy_obs + (1 - obs_epsilon) * exp_obs
        interp_obs.requires_grad = True  # For gradient calculation
        
        input_ = [interp_obs]
        if self.use_actions:
            acs_epsilon = torch.rand(policy_acs.shape, device=policy_acs.device)
            interp_acs = acs_epsilon * policy_acs + (1 - acs_epsilon) * exp_acs
            interp_acs.requires_grad = True  # For gradient calculation
            input_.append(interp_acs)

        if self.use_next_obs:
            nobs_epsilon = torch.rand(policy_obs_next.shape, device=policy_obs_next.device)
            interp_next_obs = nobs_epsilon * policy_obs_next + (1 - nobs_epsilon) * exp_obs_next
            interp_next_obs.requires_grad = True  # For gradient calculation
            input_.append(interp_next_obs)
            interp_next_phase = 0.5 * (policy_next_phase + exp_next_phase)
        else:
            interp_next_phase = None
        if self.use_dones:
            d_epsilon = torch.rand(policy_dones.shape, device=policy_dones.device)
            interp_d = d_epsilon * policy_dones + (1 - d_epsilon) * exp_dones
            interp_d.requires_grad = True  # For gradient calculation
            input_.append(interp_d)
        interp_phase = 0.5 * (policy_phase + exp_phase)

        estimate = self.forward(*input_, phase=interp_phase, next_phase=interp_next_phase)

        grads = torch.autograd.grad(estimate.sum(), input_, create_graph=True)
        # Combine gradients from all inputs before computing the penalty
        grads_flat = [g.reshape(g.size(0), -1) for g in grads if g is not None]
        if len(grads_flat) == 0:
            return torch.tensor(0.0, device=estimate.device), None

        # Norm's gradient could be NaN at 0. Use our own safe_norm
        safe_norm = (sum((g ** 2).sum(dim=1) for g in grads_flat) + 1e-8).sqrt()
        # L1 penalty toward target Lipschitz constant p
        gradient_mag = torch.mean((safe_norm - p) ** 2)

        # Return concatenated grad for logging
        gradient_mix = torch.cat(grads_flat, dim=1)
        return gradient_mag, gradient_mix

    def compute_loss(self, update_dict: TensorDict) -> Dict[str, torch.Tensor]:
        self.policy_obs = update_dict['policy_obs']
        self.policy_acs = update_dict['policy_acs']
        policy_obs_next = update_dict['policy_obs_next']
        policy_dones = update_dict['policy_dones']
        policy_phase = update_dict["policy_phase"]
        policy_next_phase = update_dict["policy_next_phase"]
        exp_obs = update_dict['expert_obs']
        exp_acs = update_dict['expert_acs']
        exp_obs_next = update_dict['expert_obs_next']
        exp_dones = update_dict['expert_dones']
        exp_phase = update_dict["expert_phase"]
        exp_next_phase = update_dict["expert_next_phase"]

        policy_out = self.forward(
            self.policy_obs,
            self.policy_acs,
            policy_obs_next,
            policy_dones,
            policy_phase,
            policy_next_phase,
        )
        expert_out = self.forward(
            exp_obs,
            exp_acs,
            exp_obs_next,
            exp_dones,
            exp_phase,
            exp_next_phase,
        )

        d_out = torch.cat([expert_out, policy_out])

        expert_loss = F.binary_cross_entropy_with_logits(
            expert_out,
            torch.ones(expert_out.size(), device=expert_out.device))
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_out,
            torch.zeros(policy_out.size(), device=policy_out.device))

        # labels aligned with concatenation order: expert first (1), policy second (0)
        labels = torch.cat([torch.ones(expert_out.size(), device=expert_out.device),
                            torch.zeros(policy_out.size(), device=policy_out.device)])

        self.bce_loss = F.binary_cross_entropy_with_logits(d_out, labels)
        
        lip_penalty = 0
        grad_mix_norm = 0
        if self.lip_coeff > 0:
            lip_penalty, grad_mix = self.lip_penalty(update_dict, self.args.lip_p)
            grad_mix_norm = torch.norm(grad_mix) if grad_mix is not None else torch.tensor(0.0, device=self.policy_obs.device)

        self.grad_penalty = 0
        grad_irm = 0
        if self.irm_coeff > 0:
            # Default to simple IRM penalty on combined output
            self.grad_penalty, grad_irm = self.irm_penalty(d_out, labels)

        self.loss = self.bce_loss + self.irm_coeff * self.grad_penalty + self.lip_coeff * lip_penalty

        output_dict = {}
        output_dict['d_loss'] = self.bce_loss
        output_dict['lip_penalty'] = lip_penalty
        output_dict['lip_grad_mix'] = grad_mix_norm
        output_dict['expert_bce_loss'] = expert_loss
        output_dict['policy_bce_loss'] = policy_loss
        output_dict['grad_penalty'] = self.grad_penalty
        output_dict['irm_grad'] = grad_irm

        return output_dict
        
    def update(self, loss: torch.Tensor) -> None:
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()

class GailReward(gym.Wrapper):
    def __init__(self, env, disc):
        super().__init__(env=env)
        self.discriminator = disc
        self.args = getattr(disc, "args", Namespace())
        self.phase_mode = validate_phase_mode(
            getattr(self.args, "imitation_phase_mode", "none")
        )
        self.phase_horizon = _resolve_phase_horizon(self.args, env)
        self.obs = None
        self.step_index = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs = obs
        self.step_index = 0
        return obs, info

    def step(self, action):
        next_obs, gt_reward, term, trunc, info = self.env.step(action)
        done = term or trunc
        
        if self.obs is None:
             self.obs = next_obs # Should be set by reset, but fallback

        obs_t = torch.tensor(self.obs, dtype=torch.float32, device=next(self.discriminator.parameters()).device).unsqueeze(0)
        acs_t = torch.tensor(action, dtype=torch.float32, device=next(self.discriminator.parameters()).device).unsqueeze(0)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=next(self.discriminator.parameters()).device).unsqueeze(0)
        done_t = torch.tensor([done], dtype=torch.float32, device=next(self.discriminator.parameters()).device).unsqueeze(0)
        phase_t = None
        next_phase_t = None
        if self.phase_mode != "none":
            phase_t, next_phase_t = compute_online_phases(
                torch.tensor(
                    [self.step_index],
                    dtype=torch.float32,
                    device=obs_t.device,
                ),
                horizon=self.phase_horizon,
                done=done_t.view(-1),
            )

        with torch.no_grad():
            irl_reward = self.discriminator.get_reward(
                obs_t,
                acs_t,
                next_obs_t,
                done_t,
                phase=phase_t,
                next_phase=next_phase_t,
            ).cpu().numpy()[0]

        self.obs = next_obs
        self.step_index = 0 if done else self.step_index + 1
        return next_obs, irl_reward, term, trunc, info
