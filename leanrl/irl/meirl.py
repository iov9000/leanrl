import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import Adam
from torch.nn.utils import spectral_norm, weight_norm
from leanrl.irl.utils import MiniGridCNN, AtariCNNBase

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

class MEIRLDiscriminator(nn.Module):
    def __init__(self, env, args, clamp_magnitude=10.0):
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
        self.identity_features = False # args.identity_features
        self.clamp_magnitude = clamp_magnitude

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

        self.layer_dims = [dim0] + list(self.layer_dims)
        
        if self.identity_features:
            self.base = nn.Identity()
            self.phi = nn.Identity()
            self.discriminator = linlayer(in_dim=dim0,
                                            out_dim=1,
                                            bias=self.bias,
                                            wnorm=self.wnorm,
                                            snorm=self.snorm)
        else:
            if self.is_atari:
                self.base = AtariCNNBase(args, env, self.use_actions)
                self.discriminator = nn.Linear(512, 1, bias=self.bias)
                self.phi = nn.Identity()
            else:
                if self.use_cnn_base:
                    self.base = MiniGridCNN(self.layer_dims, self.use_actions)
                else:
                    self.base = nn.Sequential(
                        linlayer(self.layer_dims[0], self.layer_dims[1], self.bias, self.wnorm, self.snorm),
                        nn.ReLU() # Assuming ReLU as default nonlin
                    )

                self.discriminator_layers = []
                for i in range(2, len(self.layer_dims)):
                    self.discriminator_layers += [
                        linlayer(self.layer_dims[i - 1], self.layer_dims[i], self.bias, self.wnorm, self.snorm),
                        nn.ReLU()
                    ]
            self.phi = nn.Sequential(*self.discriminator_layers)

            self.discriminator_layers += [
                linlayer(self.layer_dims[-1], 1, self.bias, self.wnorm, self.snorm)
            ]

            self.discriminator = nn.Sequential(*self.discriminator_layers)

        self.d_optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coeff)

    def base_fwd(self, ob, ac, nob=None, d=None):
        input_ = [ob]
        if self.use_actions:
            input_.append(ac)
        
        if (self.use_cnn_base or self.is_atari):
            base_out = self.base(*input_)
        else:
            base_out = self.base(torch.cat(input_, axis=-1))

        return base_out

    def forward(self, ob, ac=None, nob=None, d=None):
        base_out = self.base_fwd(ob, ac, nob, d)

        phi = self.phi(base_out)
        d_out = self.discriminator(phi)
        # output = torch.clamp(d_out, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return d_out, phi

    def get_reward(self, ob, ac):
        if self.use_actions and (self.use_cnn_base or self.is_atari):
            base_out = self.base(ob, ac)
        elif self.use_actions and not (self.use_cnn_base or self.is_atari):
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = self.base(torch.cat([ob, ac], axis=-1))
        else:
            base_out = self.base(ob)

        phi = self.phi(base_out)
        d_out = self.discriminator(phi)
        reward = torch.clamp(d_out, min=-1.0*self.clamp_magnitude, max=self.clamp_magnitude)
        return reward

    def irm_penalty(self, logits, y):
        scale = torch.tensor(1.).to(logits.device).requires_grad_()
        loss = F.mse_loss(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

    def compute_loss(self, update_dict):    
        r_policy, self.phi_policy = self.forward(update_dict['policy_obs'], update_dict['policy_acs'])
        r_expert, self.phi_expert = self.forward(update_dict['expert_obs'], update_dict['expert_acs'])

        # MaxEnt IRL objective: maximize expert reward and minimize log-partition (approximated by policy reward)
        # Equivalent to minimizing (policy mean - expert mean)
        self.diff_loss = r_policy.mean() - r_expert.mean()
        
        self.grad_penalty = 0
        if self.irm_coeff > 0:
            self.grad_penalty = self.irm_penalty(r_policy, r_expert)

        lip_penalty = 0
        # if self.args.lip_coeff > 0:
        #     lip_penalty, grad_mix = self.lip_penalty(update_dict, self.args.lip_p)

        self.loss = self.diff_loss + self.irm_coeff * self.grad_penalty + self.lip_coeff*lip_penalty

        output_dict = {}
        output_dict['total_loss'] = self.loss
        output_dict['d_loss'] = self.diff_loss
        output_dict['grad_penalty'] = self.grad_penalty
        output_dict['lip_penalty'] = lip_penalty

        return output_dict

    def update(self, loss):
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()

class MeirlReward(gym.Wrapper):
    def __init__(self, env, disc):
        super().__init__(env=env)
        self.discriminator = disc
        self.obs = None

    def step(self, action):
        next_obs, gt_reward, term, trunc, info = self.env.step(action)
        
        if self.obs is not None:
            obs_t = torch.tensor(self.obs, dtype=torch.float32).unsqueeze(0)
        else:
            obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

        acs_t = torch.tensor(action, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            irl_reward = self.discriminator.get_reward(obs_t, acs_t).cpu().numpy()[0]
        
        self.obs = next_obs
        return next_obs, float(irl_reward), term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs = obs
        return obs, info
