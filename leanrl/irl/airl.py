from __future__ import annotations

from argparse import Namespace
from typing import Dict, Mapping, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

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

class AIRLDiscriminator(nn.Module):
    def __init__(self, env: gym.Env, args: Namespace) -> None:
        super().__init__()
        self.env = env
        self.args = args
        
        if hasattr(env, "single_observation_space"):
            n_obs = np.prod(env.single_observation_space.shape)
            n_act = np.prod(env.single_action_space.shape)
        else:
            n_obs = np.prod(env.observation_space.shape)
            n_act = np.prod(env.action_space.shape)
        
        self.use_actions = getattr(args, "use_actions", True)
        self.phase_mode = validate_phase_mode(
            getattr(args, "imitation_phase_mode", "none")
        )
        self.phase_dim = phase_feature_dim(self.phase_mode)
        self.gamma = getattr(args, "gamma", 0.99)
        self.irm_coeff = getattr(args, "irm_coeff", 0.0)
        self.lip_coeff = getattr(args, "lip_coeff", 0.0)
        self.lip_p = getattr(args, "lip_p", 1.0)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and getattr(args, "cuda", True) else "cpu"
        )
        
        in_dim = n_obs
        if self.use_actions:
            in_dim += n_act
        in_dim += self.phase_dim
        value_in_dim = n_obs + self.phase_dim
            
        # Reward function g(s,a)
        self.reward_net = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
        
        # Shaping function h(s)
        self.value_net = nn.Sequential(
            layer_init(nn.Linear(value_in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        # Move parameters to GPU when available/allowed so downstream tensors match
        self.to(self.device)
        
    def forward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        acs: torch.Tensor | None,
        lprobs: torch.Tensor | None,
        phase: torch.Tensor | None = None,
        next_phase: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # g(s,a)
        phase_features = encode_phase_features(phase, self.phase_mode)
        if self.use_actions:
            x = torch.cat([obs, acs], dim=-1)
        else:
            x = obs
        if phase_features is not None:
            x = torch.cat([x, phase_features], dim=-1)
        reward = self.reward_net(x)
        
        # h(s)
        obs_value = obs
        next_obs_value = next_obs
        if phase_features is not None:
            next_phase_features = encode_phase_features(next_phase, self.phase_mode)
            obs_value = torch.cat([obs_value, phase_features], dim=-1)
            next_obs_value = torch.cat([next_obs_value, next_phase_features], dim=-1)
        value = self.value_net(obs_value)
        # h(s')
        next_value = self.value_net(next_obs_value)
        
        # f(s,a,s') = g(s,a) + gamma * h(s') - h(s)
        f_val = reward + self.gamma * next_value - value

        # Discriminator output D(s,a,s') = sigmoid(f(s,a,s') - log_pi(a|s))
        # But we return logits for BCEWithLogitsLoss: f(s,a,s') - log_pi(a|s)
        if lprobs is None:
            log_prob_term = torch.zeros_like(reward)
        else:
            log_prob_term = lprobs
            # Ensure shape is broadcastable to reward/logit shape
            if log_prob_term.dim() < reward.dim():
                log_prob_term = log_prob_term.unsqueeze(-1)
            if log_prob_term.shape[-1] != 1:
                log_prob_term = log_prob_term.sum(dim=-1, keepdim=True)
            log_prob_term = log_prob_term.expand_as(reward)

        logits = f_val - log_prob_term

        return logits, reward, value, next_value

    def get_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        acs: torch.Tensor | None,
        lprobs: torch.Tensor | None,
        phase: torch.Tensor | None = None,
        next_phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Return the AIRL logit f(s,a,s') - log_pi(a|s), which is the shaped reward.
        logits, _, _, _ = self.forward(
            obs,
            next_obs,
            acs,
            lprobs,
            phase=phase,
            next_phase=next_phase,
        )
        return logits.squeeze(-1)

    def irm_penalty(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
        loss = F.binary_cross_entropy_with_logits(logits * scale, labels)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2), grad

    def lip_penalty(
        self, update_dict: TensorDict, p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        policy_obs = update_dict["policy_obs"]
        policy_obs_next = update_dict["policy_obs_next"]
        policy_acs = update_dict["policy_acs"]
        policy_lprobs = update_dict["policy_lprobs"]
        policy_phase = update_dict["policy_phase"]
        policy_next_phase = update_dict["policy_next_phase"]

        expert_obs = update_dict["expert_obs"]
        expert_obs_next = update_dict["expert_obs_next"]
        expert_acs = update_dict["expert_acs"]
        expert_lprobs = update_dict["expert_lprobs"]
        expert_phase = update_dict["expert_phase"]
        expert_next_phase = update_dict["expert_next_phase"]

        min_bs = min(policy_obs.shape[0], expert_obs.shape[0])

        def subsample_group(*tensors):
            if tensors[0].shape[0] == min_bs:
                return tensors
            idx = torch.randperm(tensors[0].shape[0], device=tensors[0].device)[:min_bs]
            return tuple(t.index_select(0, idx) for t in tensors)

        (
            policy_obs,
            policy_obs_next,
            policy_acs,
            policy_lprobs,
            policy_phase,
            policy_next_phase,
        ) = subsample_group(
            policy_obs,
            policy_obs_next,
            policy_acs,
            policy_lprobs,
            policy_phase,
            policy_next_phase,
        )
        (
            expert_obs,
            expert_obs_next,
            expert_acs,
            expert_lprobs,
            expert_phase,
            expert_next_phase,
        ) = subsample_group(
            expert_obs,
            expert_obs_next,
            expert_acs,
            expert_lprobs,
            expert_phase,
            expert_next_phase,
        )

        obs_eps = torch.rand_like(policy_obs)
        ac_eps = torch.rand_like(policy_acs)
        next_obs_eps = torch.rand_like(policy_obs_next)
        lp_eps = torch.rand_like(policy_lprobs)

        interp_obs = (obs_eps * policy_obs + (1 - obs_eps) * expert_obs).requires_grad_(True)
        interp_acs = (ac_eps * policy_acs + (1 - ac_eps) * expert_acs).requires_grad_(True)
        interp_next_obs = (
            next_obs_eps * policy_obs_next + (1 - next_obs_eps) * expert_obs_next
        ).requires_grad_(True)
        interp_lprobs = lp_eps * policy_lprobs + (1 - lp_eps) * expert_lprobs
        interp_phase = 0.5 * (policy_phase + expert_phase)
        interp_next_phase = 0.5 * (policy_next_phase + expert_next_phase)

        logits, _, _, _ = self.forward(
            interp_obs,
            interp_next_obs,
            interp_acs,
            interp_lprobs,
            phase=interp_phase,
            next_phase=interp_next_phase,
        )
        grads = torch.autograd.grad(
            logits.sum(),
            [interp_obs, interp_acs, interp_next_obs],
            create_graph=True,
        )

        grads_flat = [g.reshape(g.size(0), -1) for g in grads if g is not None]
        if not grads_flat:
            return torch.tensor(0.0, device=logits.device), None

        safe_norm = (sum((g ** 2).sum(dim=1) for g in grads_flat) + 1e-8).sqrt()
        penalty = torch.mean((safe_norm - p) ** 2)
        grad_mix = torch.cat(grads_flat, dim=1)
        return penalty, grad_mix

    def compute_loss(self, update_dict: TensorDict) -> Dict[str, torch.Tensor | None]:
        # Expert/policy batches
        expert_obs = update_dict["expert_obs"].to(self.device, non_blocking=True)
        expert_obs_next = update_dict["expert_obs_next"].to(self.device, non_blocking=True)
        expert_acs = update_dict["expert_acs"].to(self.device, non_blocking=True)
        expert_lprobs = update_dict["expert_lprobs"].to(self.device, non_blocking=True)
        expert_phase = update_dict["expert_phase"].to(self.device, non_blocking=True)
        expert_next_phase = update_dict["expert_next_phase"].to(self.device, non_blocking=True)

        policy_obs = update_dict["policy_obs"].to(self.device, non_blocking=True)
        policy_obs_next = update_dict["policy_obs_next"].to(self.device, non_blocking=True)
        policy_acs = update_dict["policy_acs"].to(self.device, non_blocking=True)
        policy_lprobs = update_dict["policy_lprobs"].to(self.device, non_blocking=True)
        policy_phase = update_dict["policy_phase"].to(self.device, non_blocking=True)
        policy_next_phase = update_dict["policy_next_phase"].to(self.device, non_blocking=True)

        # Flatten actions/log probs to match obs dims when necessary
        def ensure_2d(tensor):
            if tensor is None:
                return None
            return tensor.unsqueeze(-1) if tensor.dim() == 1 else tensor

        expert_acs = ensure_2d(expert_acs)
        policy_acs = ensure_2d(policy_acs)
        expert_lprobs = ensure_2d(expert_lprobs)
        policy_lprobs = ensure_2d(policy_lprobs)

        expert_logits, _, _, _ = self.forward(
            expert_obs,
            expert_obs_next,
            expert_acs,
            expert_lprobs,
            phase=expert_phase,
            next_phase=expert_next_phase,
        )
        policy_logits, _, _, _ = self.forward(
            policy_obs,
            policy_obs_next,
            policy_acs,
            policy_lprobs,
            phase=policy_phase,
            next_phase=policy_next_phase,
        )

        expert_labels = torch.ones_like(expert_logits)
        policy_labels = torch.zeros_like(policy_logits)

        expert_loss = F.binary_cross_entropy_with_logits(
            expert_logits, expert_labels
        )
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_logits, policy_labels
        )
        d_loss = expert_loss + policy_loss

        grad_penalty = torch.tensor(0.0, device=d_loss.device)
        lip_penalty = torch.tensor(0.0, device=d_loss.device)
        lip_grad = None
        if getattr(self, "lip_coeff", 0.0) > 0:
            lip_penalty, lip_grad = self.lip_penalty(update_dict, self.lip_p)
        if getattr(self, "irm_coeff", 0.0) > 0:
            logits = torch.cat([expert_logits, policy_logits], dim=0)
            labels = torch.cat([expert_labels, policy_labels], dim=0)
            grad_penalty, _ = self.irm_penalty(logits, labels)

        return {
            "d_loss": d_loss,
            "grad_penalty": grad_penalty,
            "expert_bce_loss": expert_loss,
            "policy_bce_loss": policy_loss,
            "lip_penalty": lip_penalty,
            "lip_grad": lip_grad,
        }

class AirlReward(gym.Wrapper):
    def __init__(self, env, discriminator):
        super().__init__(env)
        self.discriminator = discriminator
        self.args = getattr(discriminator, "args", Namespace())
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
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # Compute IRL reward
        with torch.no_grad():
            # Use stored current observation
            if self.obs is None:
                 # Should not happen if reset is called, but handle just in case or for first step if reset not wrapped properly? 
                 # Actually reset is wrapped.
                 self.obs = next_obs # Fallback? Or error?
            
            obs_t = torch.tensor(self.obs, dtype=torch.float32, device=next(self.discriminator.parameters()).device).unsqueeze(0)
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=next(self.discriminator.parameters()).device).unsqueeze(0)
            acs_t = torch.tensor(action, dtype=torch.float32, device=next(self.discriminator.parameters()).device).unsqueeze(0)
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
                    done=torch.tensor([terminated or truncated], dtype=torch.bool, device=obs_t.device),
                )

            lprobs_dummy = torch.zeros(1, device=next(self.discriminator.parameters()).device)
            irl_reward = self.discriminator.get_reward(
                obs_t,
                next_obs_t,
                acs_t,
                lprobs_dummy,
                phase=phase_t,
                next_phase=next_phase_t,
            )
            
        self.obs = next_obs
        self.step_index = 0 if (terminated or truncated) else self.step_index + 1
        return next_obs, irl_reward.item(), terminated, truncated, info
