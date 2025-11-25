import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import Adam
from torch.nn.utils import spectral_norm, weight_norm
from leanrl.irl.utils import MiniGridCNN

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    # torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def linlayer(in_dim, out_dim, bias=True, wnorm=False, snorm=False):
    if wnorm:
        return weight_norm(nn.Linear(in_dim, out_dim, bias=bias), 'weight')
    elif snorm:
        return spectral_norm(nn.Linear(in_dim, out_dim, bias=bias), 'weight')
    else:
        return nn.Linear(in_dim, out_dim, bias=bias)

class fIRLDiscriminator(nn.Module):
    def __init__(self, env, args):
        super().__init__()

        self.env = env
        self.args = args
        self.layer_dims = args.d_layer_dims
        self.lr = args.disc_lr
        self.use_actions = args.use_actions
        self.use_dones = False # args.use_dones
        self.use_next_obs = False # args.use_next_obs        
        self.irm_coeff = args.irm_coeff
        self.l2_coeff = args.l2_coeff
        self.lip_coeff = args.lip_coeff
        self.use_cnn_base = args.use_cnn_base
        self.bias = args.use_disc_bias
        self.is_atari = False # 'atari' in self.args.exp_name
        self.snorm = False # args.use_spectral_norm
        self.wnorm = False # args.use_weight_norm
        self.clamp_magnitude = 10.0

        # Handle vector env spaces
        if hasattr(env, "single_observation_space"):
            n_obs = np.prod(env.single_observation_space.shape)
            n_act = np.prod(env.single_action_space.shape)
            observation_space = env.single_observation_space
            action_space = env.single_action_space
        else:
            n_obs = np.prod(env.observation_space.shape)
            n_act = np.prod(env.action_space.shape)
            observation_space = env.observation_space
            action_space = env.action_space

        if isinstance(observation_space, gym.spaces.Dict):
            ob_shapes = list(observation_space['observation'].shape)
        else:
            ob_shapes = list(observation_space.shape)
        
        ac_shapes = list(action_space.shape)

        dim0 = ob_shapes[-1]
        if self.use_actions:
            dim0 = dim0 + ac_shapes[-1]
        
        # if args.use_dones:
        #     dim0 = dim0 + 1
        # if args.use_next_obs:
        #     dim0 = dim0 + ob_shapes[-1]

        self.layer_dims = [dim0] + list(self.layer_dims)
        
        # reward function g_\psi
        if self.use_cnn_base:
            self.base = MiniGridCNN(self.layer_dims, self.use_actions)
        else:
            self.base = nn.Sequential(
                linlayer(self.layer_dims[0], self.layer_dims[1], self.bias),
                nn.ReLU()
            )

        self.reward_layers = []
        for i in range(2, len(self.layer_dims)):
            self.reward_layers += [
                linlayer(self.layer_dims[i - 1], self.layer_dims[i], self.bias),
                nn.ReLU()
            ]

        self.reward_layers += [
            linlayer(self.layer_dims[-1], 1, self.bias)
        ]
        self.reward = nn.Sequential(*self.reward_layers)

        # shaping function h_\phi
        if self.use_cnn_base:
            self.base_v = MiniGridCNN(self.layer_dims, use_actions=False)
        else:
            self.base_v = nn.Sequential(
                linlayer(self.layer_dims[0], self.layer_dims[1], self.bias),
                nn.Tanh()
            )

        self.disc_layers = []
        for i in range(2, len(self.layer_dims)):
            self.disc_layers += [
                linlayer(self.layer_dims[i - 1], self.layer_dims[i], self.bias),
                nn.Tanh()
            ]

        self.disc_layers += [
            linlayer(self.layer_dims[-1], 1, self.bias)
        ]
        self.disc = nn.Sequential(*self.disc_layers)

        self.reward_optimizer = Adam(list(self.base.parameters()) + list(self.reward.parameters()), lr=self.lr, 
                                weight_decay=self.l2_coeff)
        self.d_optimizer = Adam(list(self.base_v.parameters()) +  list(self.disc.parameters()), lr=self.lr, 
                                weight_decay=self.l2_coeff)

    def forward(self, ob, next_ob, ac, lprobs):
        # forward the nn models
        reward = self.get_reward(ob, ac)
        reward = torch.clamp(reward, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return reward

    def disc_forward(self, ob, ac):
        if self.use_actions and self.use_cnn_base:
            base_out = self.base_v(ob, ac)
        elif self.use_actions and not self.use_cnn_base:
            base_out = self.base_v(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base_v(ob)

        d_out = self.disc(base_out)
        return d_out

    def get_reward(self, ob, ac):
        if self.use_actions and self.use_cnn_base:
            base_out = self.base(ob, ac)
        elif self.use_actions and not self.use_cnn_base:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        reward = self.reward(base_out)
        reward = torch.clamp(reward, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return reward

    def irm_penalty(self, logits, y):
        scale = torch.tensor(1.).to(logits.device).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

    def compute_loss(self, update_dict):
        reward_loss, norm_logits = self.f_div_disc_loss('rkl', False, update_dict)

        grad_penalty = 0
        # if self.irm_coeff > 0:
        #     grad_penalty = self.irm_penalty(...) 

        loss = reward_loss + self.irm_coeff * grad_penalty
        if self.irm_coeff > 1.0:
            loss /= self.irm_coeff

        output_dict = {}
        output_dict['total_loss'] = loss
        output_dict['d_loss'] = reward_loss
        output_dict['grad_penalty'] = grad_penalty
        return output_dict

    def update_disc(self, update_dict):
        d_out = self.disc_forward(update_dict['all_obs'], update_dict['all_acs'])
        expert_out, policy_out = torch.chunk(d_out, chunks=2, dim=0)

        # labels: 1 for expert, 0 for policy
        labels = torch.cat([torch.ones_like(expert_out),
                            torch.zeros_like(policy_out)])

        bce_loss = F.binary_cross_entropy_with_logits(d_out, labels)
        
        grad_penalty = 0
        if self.irm_coeff > 0:
            grad_penalty = self.irm_penalty(d_out, labels)

        loss = bce_loss + self.irm_coeff * grad_penalty
        if self.irm_coeff > 1.0:
            loss /= self.irm_coeff

        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step() 

        return loss

    def f_div_disc_loss(self, div: str, IS: bool, update_dict):
        assert div in ['fkl', 'rkl', 'js']
        
        T = len(update_dict['all_obs'])
        logits = self.disc_forward(update_dict['all_obs'], update_dict['all_acs']) # torch vector

        if div == 'fkl':
            t1 = torch.exp(logits) # (N*T,) p/q
        elif div == 'rkl':
            t1 = logits # (N*T,) log (p/q)
        elif div == 'js':
            t1 = F.softplus(logits) # (N*T,) log (1 + p/q)

        rew = self.get_reward(update_dict['all_obs'], update_dict['all_acs'])
        t2 = rew

        if IS:
            # Importance Sampling logic (simplified for now, assuming on-policy or close)
            # For PPO we might not need full IS if we trust the policy ratio
            pass
            
        surrogate_objective = (t1 * t2).mean() - t1.mean() * t2.mean() # sample covariance
        
        # surrogate_objective /= T # already mean?

        return surrogate_objective, t1.mean() 

    def update(self, loss):
        self.reward_optimizer.zero_grad()
        loss.backward()
        self.reward_optimizer.step()

class firlReward(gym.Wrapper):
    def __init__(self, env, disc):
        super().__init__(env=env)
        self.discriminator = disc
        self.obs = None

    def step(self, action):
        next_obs, gt_reward, term, trunc, info = self.env.step(action)
        
        if self.obs is not None:
            obs_t = torch.tensor(self.obs, dtype=torch.float32).unsqueeze(0) # Add batch dim
        else:
            obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

        acs_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            irl_reward = self.discriminator.get_reward(obs_t, acs_t).cpu().numpy()[0]
        
        self.obs = next_obs
        return next_obs, irl_reward[0], term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs = obs
        return obs, info
