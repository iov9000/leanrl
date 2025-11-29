import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class AIRLDiscriminator(nn.Module):
    def __init__(self, env, args):
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
        self.gamma = getattr(args, "gamma", 0.99)
        self.irm_coeff = getattr(args, "irm_coeff", 0.0)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and getattr(args, "cuda", True) else "cpu"
        )
        
        in_dim = n_obs
        if self.use_actions:
            in_dim += n_act
            
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
            layer_init(nn.Linear(n_obs, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        # Move parameters to GPU when available/allowed so downstream tensors match
        self.to(self.device)
        
    def forward(self, obs, next_obs, acs, lprobs):
        # g(s,a)
        if self.use_actions:
            x = torch.cat([obs, acs], dim=-1)
        else:
            x = obs
        reward = self.reward_net(x)
        
        # h(s)
        value = self.value_net(obs)
        # h(s')
        next_value = self.value_net(next_obs)
        
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

    def get_reward(self, obs, next_obs, acs, lprobs):
        # Return the AIRL logit f(s,a,s') - log_pi(a|s), which is the shaped reward.
        logits, _, _, _ = self.forward(obs, next_obs, acs, lprobs)
        return logits.squeeze(-1)

    def irm_penalty(self, logits, labels):
        scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
        loss = F.binary_cross_entropy_with_logits(logits * scale, labels)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2), grad

    def compute_loss(self, update_dict):
        # Expert/policy batches
        expert_obs = update_dict["expert_obs"].to(self.device, non_blocking=True)
        expert_obs_next = update_dict["expert_obs_next"].to(self.device, non_blocking=True)
        expert_acs = update_dict["expert_acs"].to(self.device, non_blocking=True)
        expert_lprobs = update_dict["expert_lprobs"].to(self.device, non_blocking=True)

        policy_obs = update_dict["policy_obs"].to(self.device, non_blocking=True)
        policy_obs_next = update_dict["policy_obs_next"].to(self.device, non_blocking=True)
        policy_acs = update_dict["policy_acs"].to(self.device, non_blocking=True)
        policy_lprobs = update_dict["policy_lprobs"].to(self.device, non_blocking=True)

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
            expert_obs, expert_obs_next, expert_acs, expert_lprobs
        )
        policy_logits, _, _, _ = self.forward(
            policy_obs, policy_obs_next, policy_acs, policy_lprobs
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
        if getattr(self, "irm_coeff", 0.0) > 0:
            logits = torch.cat([expert_logits, policy_logits], dim=0)
            labels = torch.cat([expert_labels, policy_labels], dim=0)
            grad_penalty, _ = self.irm_penalty(logits, labels)

        return {
            "d_loss": d_loss,
            "grad_penalty": grad_penalty,
            "expert_bce_loss": expert_loss,
            "policy_bce_loss": policy_loss,
        }

class AirlReward(gym.Wrapper):
    def __init__(self, env, discriminator):
        super().__init__(env)
        self.discriminator = discriminator
        self.obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs = obs
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
            
            lprobs_dummy = torch.zeros(1, device=next(self.discriminator.parameters()).device)
            irl_reward = self.discriminator.get_reward(obs_t, next_obs_t, acs_t, lprobs_dummy)
            
        self.obs = next_obs
        return next_obs, irl_reward.item(), terminated, truncated, info
