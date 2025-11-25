import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import Adam
from torch.nn.utils import spectral_norm, weight_norm
from irl.utils import MiniGridCNN, AtariCNNBase

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def linlayer(in_dim, out_dim, bias=True, wnorm=False, snorm=False):
    if wnorm:
        return weight_norm(nn.Linear(in_dim, out_dim, bias=bias), 'weight')
    elif snorm:
        return spectral_norm(nn.Linear(in_dim, out_dim, bias=bias), 'weight')
    else:
        return nn.Linear(in_dim, out_dim, bias=bias)

class GAILDiscriminator(nn.Module):
    def __init__(self, env, args):
        super(GAILDiscriminator, self).__init__()

        self.env = env
        self.args = args
        self.layer_dims = args.d_layer_dims
        self.lr = args.disc_lr
        self.use_actions = getattr(args, "use_actions", True)
        self.use_dones = getattr(args, "use_dones", False)
        self.use_next_obs = getattr(args, "use_next_obs", False)
        self.irm_coeff = args.irm_coeff
        self.l2_coeff = args.l2_coeff
        self.lip_coeff = args.lip_coeff
        self.use_cnn_base = args.use_cnn_base
        self.bias = args.use_disc_bias
        self.is_atari = 'atari' in args.exp_name.lower() if hasattr(args, 'exp_name') else False
        self.snorm = args.use_spectral_norm
        self.wnorm = args.use_weight_norm

        if args.disc_nonlin == 'relu':
            nonlin = nn.ReLU()
        elif args.disc_nonlin == 'tanh':
            nonlin = nn.Tanh()
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

        self.layer_dims = [dim0] + self.layer_dims
        
        # Ensure layer_dims are ints
        self.layer_dims = [int(x) for x in self.layer_dims]
        
        layer_dims = self.layer_dims

        if self.is_atari:
            self.base = AtariCNNBase(args, env, self.use_actions)
            self.discriminator = nn.Linear(512, 1, bias=self.bias)
        else:
            if self.use_cnn_base:
                self.base = MiniGridCNN(layer_dims, self.use_actions)
            else:
                self.base = nn.Sequential(linlayer(self.layer_dims[0], self.layer_dims[1], 
                                                    self.bias, self.wnorm, self.snorm),
                                        nonlin)

            self.discriminator_layers = []
            for i in range(2, len(layer_dims)):
                self.discriminator_layers += [linlayer(in_dim=layer_dims[i - 1],
                                                            out_dim=layer_dims[i],
                                                            bias=self.bias,
                                                            wnorm=self.wnorm,
                                                            snorm=self.snorm),
                                            nonlin]

            self.discriminator_layers += [linlayer(in_dim=layer_dims[-1],
                                                   out_dim=1,
                                                        bias=self.bias,
                                                        wnorm=self.wnorm,
                                                        snorm=self.snorm)]

            self.discriminator = nn.Sequential(*self.discriminator_layers)

        self.d_optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_coeff)

    def base_fwd(self, ob, ac, nob=None, d=None):
        #  match tensor sizes
        if ac is not None:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
        if d is not None and len(ob.shape) != len(d.shape):
            d = torch.unsqueeze(d, -1)

        input_ = [ob]
        if self.use_actions:
            input_.append(ac)
        if self.use_next_obs:
            input_.append(nob)
        if self.use_dones:
            input_.append(d)

        if (self.use_cnn_base or self.is_atari):
            base_out = self.base(*input_)

        else:
            base_out = self.base(torch.cat(input_, axis=-1))

        return base_out

    def forward(self, ob, ac=None, nob=None, d=None):
        base_out = self.base_fwd(ob, ac, nob, d)

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

    def get_reward(self, ob, ac, nob=None, d=None):
        base_out = self.base_fwd(ob, ac, nob, d)

        d_out = self.discriminator(base_out)

        # XXX: log D vs log(1-D)!!!!
        # GAIL reward is typically -log(sigmoid(D)) or similar depending on formulation
        # Here we use -log(sigmoid(D) + epsilon)
        self.reward = - torch.log(torch.sigmoid(d_out) + 1e-8).squeeze(-1)
        return self.reward

    def irm_penalty(self, logits, y):
        scale = torch.tensor(1., device=logits.device).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2), grad

    def lip_penalty(self, update_dict, p=1):
        policy_obs = update_dict['policy_obs']
        policy_acs = update_dict['policy_acs']
        policy_obs_next = update_dict['policy_obs_next']
        policy_dones = update_dict['policy_dones']
        exp_obs = update_dict['expert_obs']
        exp_acs = update_dict['expert_acs']
        exp_obs_next = update_dict['expert_obs_next']
        exp_dones = update_dict['expert_dones']

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
        if self.use_dones:
            d_epsilon = torch.rand(policy_dones.shape, device=policy_dones.device)
            interp_d = d_epsilon * policy_dones + (1 - d_epsilon) * exp_dones
            interp_d.requires_grad = True  # For gradient calculation
            input_.append(interp_d)

        estimate = self.forward(*input_)

        gradient_mix = torch.autograd.grad(estimate.sum(), input_, create_graph=True)[0]

        # Norm's gradient could be NaN at 0. Use our own safe_norm
        safe_norm = (torch.sum(gradient_mix ** 2, dim=1) + 1e-8).sqrt()
        # L1
        gradient_mag = torch.mean((safe_norm - p) ** 2)

        return gradient_mag, gradient_mix

    def compute_loss(self, update_dict):
        self.policy_obs = update_dict['policy_obs']
        self.policy_acs = update_dict['policy_acs']
        policy_obs_next = update_dict['policy_obs_next']
        policy_dones = update_dict['policy_dones']
        exp_obs = update_dict['expert_obs']
        exp_acs = update_dict['expert_acs']
        exp_obs_next = update_dict['expert_obs_next']
        exp_dones = update_dict['expert_dones']

        policy_out = self.forward(self.policy_obs, self.policy_acs, policy_obs_next, policy_dones)
        expert_out = self.forward(exp_obs, exp_acs, exp_obs_next, exp_dones)

        d_out = torch.cat([expert_out, policy_out])

        expert_loss = F.binary_cross_entropy_with_logits(
            expert_out,
            torch.ones(expert_out.size(), device=expert_out.device))
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_out,
            torch.zeros(policy_out.size(), device=policy_out.device))

        labels = torch.cat([torch.zeros(expert_out.size(), device=expert_out.device),
                            torch.ones(policy_out.size(), device=policy_out.device)])

        self.bce_loss = F.binary_cross_entropy_with_logits(d_out, labels)
        
        lip_penalty = 0
        grad_mix_norm = 0
        if self.lip_coeff > 0:
            lip_penalty, grad_mix = self.lip_penalty(update_dict, self.args.lip_p)
            grad_mix_norm = torch.norm(grad_mix)

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
        
    def update(self, loss):
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()

class GailReward(gym.Wrapper):
    def __init__(self, env, disc):
        super().__init__(env=env)
        self.discriminator = disc
        self.obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs = obs
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

        with torch.no_grad():
            irl_reward = self.discriminator.get_reward(
                obs_t, acs_t, next_obs_t, done_t).cpu().numpy()[0]

        self.obs = next_obs
        return next_obs, irl_reward, term, trunc, info
