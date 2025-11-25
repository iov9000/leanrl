import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        logits = f_val - lprobs.unsqueeze(-1)
        
        return logits, reward, value, next_value

    def get_reward(self, obs, next_obs, acs, lprobs):
        # with torch.no_grad(): # Caller should handle no_grad if needed
        logits, reward, value, next_value = self.forward(obs, next_obs, acs, lprobs)
        
        # Default to 'reward_term' (g(s,a)) as it is the recovered reward
        return reward.squeeze(-1)

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

