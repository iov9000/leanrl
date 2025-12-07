from __future__ import annotations

import copy
from argparse import Namespace
from collections import deque
from typing import Dict, Mapping, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from ..utils import gaussian_kld
from .gsw_utils import GSW

TensorDict = Mapping[str, torch.Tensor]


def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> None:
    if isinstance(layer, nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, std)
    # torch.nn.init.constant_(layer.bias, bias_const)
    # return layer


def linlayer(
    in_dim: int,
    out_dim: int,
    bias: bool = True,
    wnorm: bool = False,
    snorm: bool = False,
) -> nn.Linear:
    # return layer_init(nn.Linear(in_dim, out_dim, bias=bias))
    if wnorm:
        return weight_norm(nn.Linear(in_dim, out_dim, bias=bias), "weight")
    elif snorm:
        return spectral_norm(nn.Linear(in_dim, out_dim, bias=bias), "weight")
    else:
        return nn.Linear(in_dim, out_dim, bias=bias)


import torch
import torch.nn as nn
import torch.nn.functional as F


def get_obs_act_dim(env):
    """Try to infer obs / act dimensions from env; fall back to attributes."""
    # Observation dim
    if hasattr(env, "observation_space"):
        obs_dim = env.observation_space.shape[0]
    elif hasattr(env, "num_obs"):
        obs_dim = env.num_obs
    else:
        raise ValueError("Could not infer obs_dim from env")

    # Action dim
    if hasattr(env, "action_space"):
        act_shape = env.action_space.shape
        act_dim = act_shape[0] if act_shape is not None else env.action_space.n
    elif hasattr(env, "num_actions"):
        act_dim = env.num_actions
    else:
        raise ValueError("Could not infer act_dim from env")

    return obs_dim, act_dim


class ProjectionNet(nn.Module):
    """proj: R^(obs+act) -> R (scalar projection)."""

    def __init__(self, input_dim, hidden_dim=128, n_layers=2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, input_dim]
        return self.net(x).squeeze(-1)  # [B]


class Potential1D(nn.Module):
    """phi: R -> R (1D potential)."""

    def __init__(self, hidden_dim=64, n_layers=2):
        super().__init__()
        layers = []
        in_dim = 1
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        # z: [B] or [B,1]
        is_scalar = False
        if z.dim() == 0:
            z = z.unsqueeze(0)
            is_scalar = True
        if z.dim() == 1:
            z = z.unsqueeze(-1)
        out = self.net(z).squeeze(-1)  # [B]
        if is_scalar:
            return out.squeeze(0)
        return out


class SWILPotDiscriminator(nn.Module):
    """
    SWILPotDiscriminator implementing r(x) = phi(proj([obs, act])) with φ
    trained to behave like a 1D OT potential on the projected scalars.

    Usage:
        disc = SWILPotDiscriminator(env, opt)
        reward = disc(obs, act)  # [B]

        loss_critic = disc.compute_critic_loss(
            expert_obs, expert_act,
            policy_obs, policy_act,
            lambda_ot=1.0,
            lambda_monotone=1e-2
        )
        loss_critic.backward()
        optim_disc.step()
    """

    def __init__(
        self,
        env,
        opt=None,
        proj_hidden_dim=128,
        proj_layers=2,
        phi_hidden_dim=64,
        phi_layers=2,
    ):
        super().__init__()
        self.opt = opt
        obs_dim, act_dim = get_obs_act_dim(env)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.input_dim = obs_dim + act_dim

        # Projection network: (obs, act) -> scalar
        self.proj = ProjectionNet(
            input_dim=self.input_dim,
            hidden_dim=proj_hidden_dim,
            n_layers=proj_layers,
        )

        # 1D potential network: scalar -> scalar
        self.phi = Potential1D(
            hidden_dim=phi_hidden_dim,
            n_layers=phi_layers,
        )

    def _concat(self, obs, act, nob=None, d=None):
        """Concatenate obs and act along the last dimension."""
        input = [obs]
        if self.opt.use_actions:
            input.append(act)
        if self.opt.use_next_obs:
            input.append(nob)
        if self.opt.use_dones:
            input.append(d)

        return torch.cat(input, dim=-1)

    def forward(self, obs, act, nob=None, d=None):
        """
        Compute reward r(s,a) = phi(proj([s,a])).

        Args:
            obs: [B, obs_dim]
            act: [B, act_dim]
        Returns:
            reward: [B]
        """
        x = self._concat(obs, act, nob=nob, d=d)  # [B, obs+act]
        z = self.proj(x)  # [B]
        r = self.phi(z)  # [B]
        return r

    def get_reward(self, obs, act, nob=None, d=None, step=None):
        """Compatibility shim so wrappers can query rewards uniformly."""
        return self.forward(obs, act, nob=nob, d=d)

    # --------- Loss components ---------

    @staticmethod
    def _ot_potential_loss(
        phi,
        z_expert,
        z_policy,
        lambda_monotone: float = 0.0,
        detach_z_for_phi: bool = True,
    ):
        """
        Enforce that phi behaves like a 1D OT potential between z_expert, z_policy.

        Steps:
          - Sort z_expert, z_policy to get optimal 1D OT pairing.
          - Target gradient: g_target(zP_k) = zP_k - zE_k (for W2 with sq. cost).
          - Penalize (phi'(zP) - g_target)^2.
          - Optional monotonicity penalty on phi' (negative slopes).

        Args:
            phi: Potential1D
            z_expert: [B_e]
            z_policy: [B_p]
            lambda_monotone: weight of monotonicity regularizer
            detach_z_for_phi: if True, no gradients into z (and thus proj).
        Returns:
            scalar loss
        """
        # Flatten
        zE = z_expert.view(-1)
        zP = z_policy.view(-1)

        # Sort (1D OT coupling).
        zE_sorted, _ = torch.sort(zE)
        zP_sorted, _ = torch.sort(zP)

        n = min(zE_sorted.shape[0], zP_sorted.shape[0])
        zE_sorted = zE_sorted[:n]
        zP_sorted = zP_sorted[:n]

        if detach_z_for_phi:
            zP_for_phi = zP_sorted.detach()
        else:
            zP_for_phi = zP_sorted

        zP_for_phi = zP_for_phi.requires_grad_(True)

        # Evaluate phi and its gradient w.r.t. zP_for_phi
        phi_vals = phi(zP_for_phi)  # [n]
        (grad_phi,) = torch.autograd.grad(
            phi_vals.sum(), zP_for_phi, create_graph=True
        )  # [n]

        # Target gradient for squared W2 cost
        g_target = zP_sorted - zE_sorted  # [n]

        loss_ot = F.mse_loss(grad_phi, g_target)

        if lambda_monotone > 0.0:
            # Penalize negative slopes
            monotone_penalty = torch.relu(-grad_phi).mean()
            loss_ot = loss_ot + lambda_monotone * monotone_penalty

        return loss_ot

    def compute_ipm_loss(self, expert_obs, expert_act, policy_obs, policy_act):
        """
        IPM / IL objective:
            L_IL = - ( E_E[phi(proj(x))] - E_pi[phi(proj(x))] )
        so that maximizing w.r.t. critic increases the expert-policy gap.
        """
        rE = self.forward(expert_obs, expert_act)  # [B_e]
        rP = self.forward(policy_obs, policy_act)  # [B_p]
        return -(rE.mean() - rP.mean())

    def compute_ot_potential_loss(
        self,
        expert_obs,
        expert_act,
        policy_obs,
        policy_act,
        lambda_monotone: float = 0.0,
        detach_proj_for_phi: bool = True,
    ):
        """
        Convenience wrapper: compute OT potential loss in (obs,act) space.
        """
        with torch.no_grad() if detach_proj_for_phi else torch.enable_grad():
            zE = self.proj(self._concat(expert_obs, expert_act)).view(-1)
            zP = self.proj(self._concat(policy_obs, policy_act)).view(-1)

        return self._ot_potential_loss(
            self.phi,
            zE,
            zP,
            lambda_monotone=lambda_monotone,
            detach_z_for_phi=detach_proj_for_phi,
        )

    def compute_loss(
        self,
        update_dict: TensorDict,
        lambda_ot: float = 1.0,
        lambda_monotone: float = 0.0,
        detach_proj_for_phi: bool = True,
    ):
        """
        Full critic loss:
            L_critic = L_IL + lambda_ot * L_OT
        where:
            L_IL  = IPM loss between expert / policy
            L_OT  = OT potential gradient-matching + monotonicity penalty

        Args:
            update_dict: TensorDict containing expert/policy obs/acs
            lambda_ot: weight of OT potential term
            lambda_monotone: monotonicity regularizer for phi'
            detach_proj_for_phi: if True, do not backprop OT loss into proj.
        """
        expert_obs = update_dict["expert_obs"]
        expert_act = update_dict["expert_acs"]
        policy_obs = update_dict["policy_obs"]
        policy_act = update_dict["policy_acs"]

        loss_il = self.compute_ipm_loss(expert_obs, expert_act, policy_obs, policy_act)
        loss_ot = self.compute_ot_potential_loss(
            expert_obs,
            expert_act,
            policy_obs,
            policy_act,
            lambda_monotone=lambda_monotone,
            detach_proj_for_phi=detach_proj_for_phi,
        )
        loss = loss_il + lambda_ot * loss_ot
        return {
            "d_loss": loss,
            "grad_penalty": 0.0,
            "ib_loss": 0.0,
            "loss_il": loss_il.detach(),
            "loss_ot": loss_ot.detach(),
        }


class SWILDiscriminator(nn.Module):
    def __init__(self, env: gym.Env, opt: Namespace) -> None:
        super(SWILDiscriminator, self).__init__()

        self.env = env
        self.opt = opt
        self.gsw_df = opt.gsw_df
        self.layer_dims = opt.d_layer_dims
        self.lr = opt.disc_lr
        self.use_actions = opt.use_actions
        self.use_dones = opt.use_dones
        self.use_next_obs = opt.use_next_obs
        self.bias = opt.use_disc_bias
        bias = self.bias
        self.n_proj = opt.n_proj
        self.reward_type = opt.swil_reward_type
        self.swil_vb = opt.swil_vb
        self.proj_norm_coeff = opt.proj_norm_coeff
        self.i_c = opt.i_c
        self.beta = torch.tensor(opt.min_beta, dtype=torch.float)
        self.alpha_beta = opt.vb_coeff

        self.buffer_empty_cnt = 0

        if opt.disc_nonlin == "relu":
            nonlin = nn.ReLU()
        elif opt.disc_nonlin == "leakyrelu":
            nonlin = nn.LeakyReLU()
        elif opt.disc_nonlin == "prelu":
            nonlin = nn.PReLU()
        elif opt.disc_nonlin == "silu":
            nonlin = nn.SiLU()
        elif opt.disc_nonlin == "tanh":
            nonlin = nn.Tanh()
        elif opt.disc_nonlin == "id":
            nonlin = nn.Identity()
        else:
            nonlin = nn.PReLU()

        if isinstance(env.observation_space, gym.spaces.Dict):
            ob_shapes = list(env.observation_space["observation"].shape)
        else:
            ob_shapes = list(env.observation_space.shape)
        # ob_shapes = list(env.observation_space.shape)
        ac_shapes = list(env.action_space.shape)
        if not ac_shapes:
            ac_shapes = [1]

        dim0 = ob_shapes[-1]
        if self.use_actions:
            dim0 = dim0 + ac_shapes[-1]
        if opt.use_dones:
            dim0 = dim0 + 1
        if opt.use_next_obs:
            dim0 = dim0 + ob_shapes[-1]

        self.layer_dims = [dim0] + self.layer_dims

        ac_len = ac_shapes[0]

        if opt.gsw_df == "nn":
            if opt.linear_proj:
                self.base = linlayer(
                    self.layer_dims[0],
                    self.opt.n_proj,
                    bias,
                    opt.use_weight_norm,
                    opt.use_spectral_norm,
                )
                self.reward = nn.Identity()
            else:
                self.base = nn.Sequential(
                    linlayer(
                        self.layer_dims[0],
                        self.layer_dims[1],
                        bias,
                        opt.use_weight_norm,
                        opt.use_spectral_norm,
                    ),
                    nonlin,
                )

                if self.reward_type == "linear":
                    self.reward_layers = self._construct_network(opt, nonlin, bias)
                    self.reward = nn.Sequential(*self.reward_layers)
                else:
                    self.reward = nn.MultiheadAttention(
                        embed_dim=self.layer_dims[1], num_heads=1
                    )

            if self.opt.swil_rew == "replacement_nn":
                self.base_rew_replnn = copy.deepcopy(self.base)
                self.rew_replnn = copy.deepcopy(self.reward)

            if torch.cuda.is_available():
                self.base.cuda()
                self.reward.cuda()
                if self.opt.proj_layer:
                    self.proj_layer.cuda()

            self.d_optimizer = Adam(
                self.parameters(), lr=self.lr, weight_decay=self.opt.l2_coeff
            )  # , eps=1e-5)
            self.d_scheduler = ExponentialLR(
                self.d_optimizer, gamma=opt.scheduler_gamma
            )
        else:
            self.gsw_module = GSW(
                "poly", nofprojections=opt.n_proj, degree=opt.poly_degree
            )

        # init sorted atom queues
        self.pi_atoms_sorted = deque(maxlen=opt.max_q_len)
        self.pi_atoms_sorted_bkp = copy.deepcopy(self.pi_atoms_sorted)

        self.exp_atoms_sorted = deque(maxlen=opt.max_q_len)
        self.exp_atoms_sorted_bkp = copy.deepcopy(self.exp_atoms_sorted)

        self.pi_atoms_sorted_idx = deque(maxlen=opt.max_q_len)
        self.exp_atoms_sorted_idx = deque(maxlen=opt.max_q_len)

        self.policy_obs = torch.randn(
            [opt.batch_size, *ob_shapes]
        )  # , requires_grad=True)
        self.policy_acs = torch.randn(
            [opt.batch_size, *ac_shapes]
        )  # , requires_grad=True)
        self.expert_obs = torch.randn([opt.batch_size, *ob_shapes])
        self.expert_nobs = torch.randn([opt.batch_size, *ob_shapes])
        self.expert_acs = torch.randn([opt.batch_size, *ac_shapes])
        self.expert_nobs = torch.randn([opt.batch_size, *ob_shapes])

        # self.module_list = nn.ModuleList([self.base, self.base_v, self.reward, self.value])
        # self.d_optimizer = Adam(list(self.base.parameters()) + list(self.reward.parameters()),
        #                lr=self.lr, weight_decay=self.l2_coeff)

        # self.d_optimizer = Adam([self.policy_obs, self.policy_acs], lr=self.lr)

    def _construct_network(self, opt, nonlin: nn.Module, bias: bool) -> list:
        layers = []
        for i in range(2, len(self.layer_dims)):
            layers += [
                linlayer(
                    self.layer_dims[i - 1],
                    self.layer_dims[i],
                    bias,
                    opt.use_weight_norm,
                    opt.use_spectral_norm,
                ),
                nonlin,
            ]

        layers += [
            linlayer(
                self.layer_dims[-1],
                self.opt.n_proj,
                bias,
                opt.use_ll_weight_norm,
                opt.use_spectral_norm,
            ),
        ]

        return layers

    def forward(
        self,
        ob: torch.Tensor,
        ac: torch.Tensor,
        nob: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
        lprobs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # forward the nn models
        gsw_dist = self.gsw_dist(
            ob, ac, nob, d, self.expert_obs, self.expert_acs, random=False
        )

        # get random projections based on reward
        return gsw_dist

    def ot_potential_loss(
        self,
        phi: nn.Module,
        z_expert: torch.Tensor,
        z_policy: torch.Tensor,
        lambda_monotone: float = 0.0,
        detach_z_for_phi: bool = True,
    ) -> torch.Tensor:
        """
        Enforce that phi behaves like a 1D OT potential between z_expert and z_policy.

        Args:
            phi: Potential1D module
            z_expert: [B_e] tensor
            z_policy: [B_p] tensor
            lambda_monotone: weight for monotonicity regularizer
            detach_z_for_phi: if True, don't backprop into z (and hence proj)
                            when fitting phi; i.e. bilevel-ish.
        Returns:
            loss (scalar)
        """
        # Flatten to [B]
        zE = z_expert.view(-1)
        zP = z_policy.view(-1)

        # Sort: this is the optimal 1D OT coupling
        zE_sorted, _ = torch.sort(zE)
        zP_sorted, _ = torch.sort(zP)

        n = min(zE_sorted.shape[0], zP_sorted.shape[0])
        zE_sorted = zE_sorted[:n]
        zP_sorted = zP_sorted[:n]

        if detach_z_for_phi:
            zP_sorted = zP_sorted.detach()
        zP_sorted.requires_grad_(True)

        # Evaluate phi on policy scalars
        phi_vals = phi(zP_sorted)  # [n]

        # Gradient of phi w.r.t. z (this is phi'(z))
        (grad_phi,) = torch.autograd.grad(
            phi_vals.sum(), zP_sorted, create_graph=True
        )  # [n]

        # Target gradient for squared 2-W cost:
        # g_target(zP_k) = zP_k - zE_k (up to constant factors/signs)
        g_target = zP_sorted - zE_sorted

        # MSE between phi' and target gradient
        loss_ot = F.mse_loss(grad_phi, g_target)

        # Optional monotonicity penalty: penalize negative derivatives
        if lambda_monotone > 0:
            monotone_penalty = F.relu(-grad_phi).mean()
            loss_ot = loss_ot + lambda_monotone * monotone_penalty

        return loss_ot

    def proj(
        self,
        ob: torch.Tensor,
        ac: torch.Tensor,
        nob: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
        noise: bool = False,
    ) -> Tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        if self.opt.n_proj > 1 and not self.opt.linear_proj:
            with torch.no_grad():
                self.base.apply(layer_init)
                self.reward.apply(layer_init)

        base_out = self.base_fwd(self.base, ob, ac, nob, d)

        if self.swil_vb > 0:
            vb_out, z, mu, std = self.vb(base_out, noise=noise)
        else:
            vb_out = base_out
            z = mu = std = None
        # rew, v, v_n, d_out = self.forward(ob, next_ob, ac, lprobs) TODO??
        # potentially use more sophisticated mechanism with multiple projections
        if self.opt.proj_layer:
            return self.proj_layer(self.reward(vb_out)), z, mu, std
        else:
            # XXX: additional noise?
            rew = self.reward(vb_out)
            # rew = rew / (rew.norm(dim=-1, keepdim=True) + 1e-6)

            # perturb with noise -> TODO: sample from stochastic (e.g. Gaussian) process?
            if self.opt.n_proj > 1:
                rew += torch.randn_like(rew) * 0.01
            if self.opt.add_proj_noise:
                rew += torch.randn_like(rew) * 0.01

            # out = rew/rew.norm(-1)
            return rew, z, mu, std  # + torch.randn_like(self.reward(base_out))

    def base_fwd(
        self,
        base_fn: nn.Module,
        ob: torch.Tensor,
        ac: torch.Tensor,
        nob: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        #  match tensor sizes
        if len(ob.shape) != len(ac.shape):
            ac = torch.unsqueeze(ac, -1)
        if d is not None:
            if len(ob.shape) != len(d.shape):
                d = torch.unsqueeze(d, -1)

        input_ = [ob]
        if self.use_actions:
            input_.append(ac)
        if self.use_next_obs:
            input_.append(nob)
        if self.use_dones:
            input_.append(d)

        tensors = [t for t in input_ if t is not None]
        base_out = base_fn(torch.cat(tensors, axis=-1))
        return base_out

    def weight_reset(self) -> None:
        reset_parameters = getattr(self.reward, "reset_parameters", None)
        if callable(reset_parameters):
            self.reward.reset_parameters()

    def reset_all_weights(self) -> None:
        """
        refs:
            - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
            - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
            - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        """

        @torch.no_grad()
        def weight_reset(m: nn.Module):
            # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        self.base.apply(fn=weight_reset)
        self.reward.apply(fn=weight_reset)

    def vb(
        self, base_out: torch.Tensor, noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mustd = self.encoder_z(base_out)
        mu, logvar = torch.chunk(mustd, 2, -1)
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        if noise:
            z = mu + std * eps
        else:
            z = mu

        vb_out = self.decoder_z(z)

        return vb_out, z, mu, std

    def compute_replacement_reward(
        self,
        ob: torch.Tensor,
        ac: torch.Tensor,
        nob: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
        step: int | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            # project next state and rank it as part of previous evaluation
            proj, _, _, _ = self.proj(ob, ac, nob, d)
            obs_t_slice = proj.unsqueeze(0)

            rew = torch.zeros(1, device=obs_t_slice.device)

            for p, (sorted_proj, sorted_proj_tgt) in enumerate(
                zip(self.pi_atoms_sorted, self.exp_atoms_sorted)
            ):
                n = len(sorted_proj)

                if n > 0:
                    # idx = torch.searchsorted(sorted_proj.T.contiguous(), obs_t_slice.T.contiguous())#, right=True)
                    # determine slice index in previously sorted atoms used for SWD computation
                    if self.opt.aligned_index:
                        idx = (
                            torch.tensor(step, device=obs_t_slice.device)
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )
                    else:
                        idx = torch.searchsorted(
                            sorted_proj.T, obs_t_slice.T
                        )  # , right=True)
                        # idx[idx==0] +=1
                        # idx[idx==n] -=1

                    idxs = self.pi_atoms_sorted_idx[p]
                    idxs_e = self.exp_atoms_sorted_idx[p]

                    # idx[idx==0] +=1
                    idx[idx == n] -= 1
                    # shift extreme indices

                    # print("Number of atoms in buffer", n)
                    # calculate weight based on position in queue
                    weight = len(self.pi_atoms_sorted)

                    w = 1
                    # TODO: what if target CDF is left or mixed?
                    if self.opt.aligned_index:
                        i = torch.where(idxs == idx)[0]
                        j = 0
                        a_i = (sorted_proj_tgt[i, j] - sorted_proj[i, j]) ** 2
                        a_h = (sorted_proj_tgt[i - 1, j] - sorted_proj[i - 1, j]) ** 2
                        a_new_i = (sorted_proj_tgt[i, j] - obs_t_slice[0, j]) ** 2
                        a_new_h = (sorted_proj_tgt[i - 1, j] - obs_t_slice[0, j]) ** 2
                        rew_incr = torch.abs(sorted_proj[i, j] - obs_t_slice[0, j])
                        rew_decr = torch.abs(sorted_proj[i - 1, j] - obs_t_slice[0, j])
                        rew += torch.where(idxs == idx)[0]
                        if rew_incr > rew_decr:
                            rew += w * (a_new_h)
                        else:
                            rew += w * (a_new_i)
                    else:
                        for j, i in enumerate(idx):
                            rew_j = 0
                            # print(j,i)
                            # print(sorted_proj.shape)
                            if self.opt.swil_rew == "insertion_loss":
                                w = 1 / (n * (n + 1))
                                # new atom contribution
                                diff = sorted_proj[1:] - sorted_proj[:-1]

                                rew += w * torch.sum(
                                    sorted_proj[i, j]
                                    - obs_t_slice[0, j]
                                    + sorted_proj[i - 1, j]
                                    - obs_t_slice[0, j]
                                )
                                # reweighting of integrals
                                rew_j = diff
                                rew -= torch.sum(w * diff[: i - 1, j])
                                rew += torch.sum(w * diff[i:, j])
                            else:
                                # calculate diff when replacing atom
                                a_prev = (
                                    sorted_proj_tgt[i, j] - sorted_proj[i, j]
                                ) ** 2
                                a_prev_2 = (
                                    sorted_proj_tgt[i - 1, j] - sorted_proj[i - 1, j]
                                ) ** 2

                                a_new = (sorted_proj_tgt[i, j] - obs_t_slice[0, j]) ** 2
                                a_new_2 = (
                                    sorted_proj_tgt[i - 1, j] - obs_t_slice[0, j]
                                ) ** 2

                                a_i = (sorted_proj_tgt[i, j] - sorted_proj[i, j]) ** 2
                                a_h = (
                                    sorted_proj_tgt[i - 1, j] - sorted_proj[i - 1, j]
                                ) ** 2
                                a_new_i = (
                                    sorted_proj_tgt[i, j] - obs_t_slice[0, j]
                                ) ** 2
                                a_new_h = (
                                    sorted_proj_tgt[i - 1, j] - obs_t_slice[0, j]
                                ) ** 2

                                rew_incr = torch.abs(
                                    sorted_proj[i, j] - obs_t_slice[0, j]
                                )
                                rew_decr = torch.abs(
                                    sorted_proj[i - 1, j] - obs_t_slice[0, j]
                                )
                                # w = 1/ (n + a_prev) # more reward if closer

                                # print(sorted_proj[i,j] > sorted_proj_tgt[i,j])

                                rew_j = a_new - a_prev
                                if self.opt.repl_loss_type == "diff":
                                    rew_j = a_new - a_prev
                                    rew += w * (rew_j)
                                elif self.opt.repl_loss_type == "diffmax0":
                                    rew_j = a_new - a_prev
                                    if rew_j > 0:  # > 0 bc we flip it later
                                        rew_j = 0
                                    rew += w * (rew_j)
                                elif self.opt.repl_loss_type == "diff2":
                                    # if sorted_proj[i,j] > sorted_proj_tgt[i,j]:
                                    # print("api>ae")
                                    if rew_incr > rew_decr:
                                        chosen_idx = i - 1
                                        rew += w * (a_new_h - a_h)
                                        rew_j = a_new_h - a_h
                                    else:
                                        chosen_idx = i
                                        rew += w * (a_new_i - a_i)
                                        rew_j = a_new_i - a_i
                                        # rew += w*(a_new - a_prev)
                                elif self.opt.repl_loss_type == "diff2max0":
                                    # print(sorted_proj[i,j] > sorted_proj_tgt[i,j])
                                    if rew_incr > rew_decr:
                                        rew_j = a_new_h - a_h
                                    else:
                                        rew_j = a_new_i - a_i
                                    if rew_j > 0:  # > 0 bc we flip it later
                                        rew_j = 0
                                    # XXX: sign problems!!!???
                                    rew += w * (rew_j)

                                elif self.opt.repl_loss_type == "diff3":
                                    if sorted_proj[i, j] > sorted_proj_tgt[i, j]:
                                        rew += w * (a_new - a_prev)
                                    else:
                                        rew -= w * (a_new - a_prev)
                                else:
                                    if rew_incr > rew_decr:
                                        rew += w * (a_new_h)
                                    else:
                                        rew += w * (a_new_i)

                            # replace stored atoms in queue if we're getting closer
                            if self.opt.replace_atoms and rew_j < 0:
                                self.pi_atoms_sorted[p][chosen_idx] = obs_t_slice

                else:
                    self.buffer_empty_cnt += 1
                    print("Atom buffer empty", self.buffer_empty_cnt)
                    self.pi_atoms_sorted = copy.deepcopy(self.pi_atoms_sorted_bkp)
                    self.exp_atoms_sorted = copy.deepcopy(self.exp_atoms_sorted_bkp)

            rew = -rew

        return rew

    def swil_reward(
        self,
        ob: torch.Tensor,
        ac: torch.Tensor,
        nob: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ## TODO: replacement loss here..
        if self.opt.gsw_df == "nn":
            base_out = self.base_fwd(self.base, ob, ac, nob, d)
            if self.reward_type == "linear":
                if self.opt.proj_layer:
                    return self.proj_layer(self.reward(base_out))
                else:
                    return torch.mean(self.reward(base_out), -1)
                    # return torch.min(self.reward(base_out),-1)[0]
            else:
                return torch.mean(self.reward(base_out, base_out, base_out)[0], -1)

        elif self.opt.swil_loss == "approx_sw":
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            return approximate_sw(
                torch.cat([ob, ac], -1),
                torch.cat([self.expert_obs, self.expert_acs], -1),
            )

        elif self.opt.swil_loss == "atom_gsw":
            return self.atom_gsw(ob, ac, self.expert_obs, self.expert_acs, random=False)

        else:
            # compute distance for single ob,ac and batch of experts via projections
            if self.gsw_df == "nn":
                gsw_dist = self.gsw_dist_nn(
                    ob, ac, self.expert_obs, self.expert_acs, random=False
                )
            else:
                gsw_dist = self.gsw_dist(ob, ac, self.expert_obs, self.expert_acs)

            # rew, v, v_n, d_out = self.forward(ob, next_ob, ac, lprobs) TODO??
            # potentially use more sophisticated mechanism for averaging out projections
            # Self-Attention sliced wasserstein distances is a weighted sum?
            return gsw_dist

    def get_reward(
        self,
        ob: torch.Tensor,
        ac: torch.Tensor,
        nob: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
        step: int | None = None,
    ) -> torch.Tensor:
        if self.opt.swil_rew == "old_swil":
            return self.swil_reward(ob, ac, nob, d)
        elif self.opt.swil_rew == "replacement_nn":
            return self.rew_replnn(self.base_fwd(self.base_rew_replnn, ob, ac, nob))
        else:
            return self.compute_replacement_reward(ob, ac, nob, d, step)

    def compute_replacement_reward_loss(
        self, update_dict: TensorDict, update_dict_2: TensorDict
    ) -> torch.Tensor:
        obs_pi = update_dict["policy_obs"]
        acs_pi = update_dict["policy_acs"]
        nobs_pi = update_dict["policy_obs_next"]
        d_pi = update_dict["policy_dones"]
        exp_obs = update_dict["expert_obs"]
        exp_acs = update_dict["expert_acs"]
        exp_obs_next = update_dict["expert_obs_next"]
        exp_dones = update_dict["expert_dones"]

        obs_pi_2 = update_dict_2["policy_obs"]
        acs_pi_2 = update_dict_2["policy_acs"]
        nobs_pi_2 = update_dict_2["policy_obs_next"]
        d_pi_2 = update_dict_2["policy_dones"]

        # project atoms
        pi_slices, _, _, _ = self.proj(obs_pi, acs_pi, nobs_pi, d_pi)
        pi_slices_2, _, _, _ = self.proj(obs_pi_2, acs_pi_2, nobs_pi_2, d_pi_2)

        pred_diffs = self.get_reward(obs_pi_2, acs_pi_2, nobs_pi_2, d_pi_2)

        exp_slices, _, _, _ = self.proj(obs_exp, acs_exp, nobs_exp, d_exp)

        # sort slices
        pi_slices_sorted, pi_slices_sorted_idx = torch.sort(
            pi_slices, dim=0, stable=True
        )
        exp_slices_sorted, exp_slices_sorted_idx = torch.sort(
            exp_slices, dim=0, stable=True
        )

        # sort and insert using torch.searchsorted
        idx = torch.searchsorted(pi_slices_sorted, pi_slices_2)

        # compute distances for all indices
        diffs = pi_slices_sorted[idx] - pi_slices_2

        # sum up loss and return it
        l2_pred_diff_loss = torch.sum(torch.nn.functional.mse_loss(pred_diffs, diffs))

        return l2_pred_diff_loss

    def get_slice(
        self, ob: torch.Tensor, ac: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """Slices samples from distribution X~P_X
        Inputs:
            X:  Nxd matrix of N data samples
            theta: parameters of g (e.g., a d vector in the linear case)
        """

        if self.use_actions:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            X = torch.cat([ob, ac], axis=-1)
        else:
            X = ob

        if self.ftype == "linear":
            return self.gsw_module.linear(X, theta)
        elif self.ftype == "poly":
            return self.gsw_module.poly(X, theta)
        elif self.ftype == "circular":
            return self.gsw_module.circular(X, theta)
        else:
            raise Exception("Defining function not implemented")

    def prepare_x(self, ob: torch.Tensor, ac: torch.Tensor) -> torch.Tensor:
        if self.use_actions:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            X = torch.cat([ob, ac], axis=-1)
        else:
            X = ob

        return X

    def gsw_dist(
        self,
        obs_pi: torch.Tensor,
        acs_pi: torch.Tensor,
        obs_exp: torch.Tensor,
        acs_exp: torch.Tensor,
        nobs_pi: torch.Tensor | None = None,
        d_pi: torch.Tensor | None = None,
        nobs_exp: torch.Tensor | None = None,
        d_exp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        X_exp = self.prepare_x(obs_exp, acs_exp)
        X_pi = self.prepare_x(obs_pi, acs_pi)

        return self.gsw_module.gsw(X_exp, X_pi)

    def max_gsw_dist(
        self,
        obs_pi: torch.Tensor,
        acs_pi: torch.Tensor,
        nobs_pi: torch.Tensor,
        d_pi: torch.Tensor,
        obs_exp: torch.Tensor,
        acs_exp: torch.Tensor,
        nobs_exp: torch.Tensor,
        d_exp: torch.Tensor,
    ) -> torch.Tensor:
        X_exp = self.prepare_x(obs_exp, acs_exp)
        X_pi = self.prepare_x(obs_pi, acs_pi)

        return self.gsw_module.max_gsw(X_exp, X_pi)

    def gsw_dist_nn(
        self,
        obs_pi: torch.Tensor,
        acs_pi: torch.Tensor,
        nobs_pi: torch.Tensor,
        d_pi: torch.Tensor,
        obs_exp: torch.Tensor,
        acs_exp: torch.Tensor,
        nobs_exp: torch.Tensor,
        d_exp: torch.Tensor,
        random: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates GSW between two empirical state-action distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        """
        # N,dn = X.shape
        # M,dm = Y.shape
        # assert dn==dm and M==N

        if random:
            self.base.reset()
            self.reward.reset()

        # project slices
        pi_slices, _, _, _ = self.proj(obs_pi, acs_pi, nobs_pi, d_pi)
        exp_slices, _, _, _ = self.proj(obs_exp, acs_exp, nobs_exp, d_exp)

        # sort slices
        pi_slices_sorted, pi_slices_sorted_idx = torch.sort(
            pi_slices, dim=0, stable=True
        )
        exp_slices_sorted, exp_slices_sorted_idx = torch.sort(
            exp_slices, dim=0, stable=True
        )

        self.pi_atoms_sorted.append(pi_slices_sorted)
        self.exp_atoms_sorted.append(exp_slices_sorted)
        self.pi_atoms_sorted_idx.append(pi_slices_sorted_idx)
        self.exp_atoms_sorted_idx.append(exp_slices_sorted_idx)

        return (
            torch.sqrt(torch.sum((pi_slices_sorted - exp_slices_sorted) ** 2)),
            torch.norm(pi_slices),
            torch.norm(exp_slices),
        )

    def atom_gsw(
        self,
        obs_pi: torch.Tensor,
        acs_pi: torch.Tensor,
        obs_exp: torch.Tensor,
        acs_exp: torch.Tensor,
        random: bool = False,
    ) -> torch.Tensor:
        """
        Calculates distance for single atom in 1d by identifying expert neighbours
        """
        # N,dn = X.shape
        # M,dm = Y.shape
        # assert dn==dm and M==N

        if random:
            self.reward.reset()

        # project slices
        pi_slices, _, _, _ = self.proj(obs_pi, acs_pi)
        exp_slices, _, _, _ = self.proj(obs_exp, acs_exp)
        all_slices = torch.cat([pi_slices.unsqueeze(0), exp_slices], 0)

        # sort slices
        slices_sorted_idx = torch.argsort(all_slices, dim=0)
        idx0 = torch.where(slices_sorted_idx == 0)[0]
        idx_exp0 = idx0 - 1
        d0 = torch.sqrt(torch.sum(pi_slices - exp_slices[idx_exp0]) ** 2)

        if torch.all(idx0 > 0):
            idx_exp0 = idx0 - 1
            d0 = torch.sqrt(torch.sum(pi_slices - exp_slices[idx_exp0]) ** 2)
        else:
            d0 = torch.tensor(0.0)
        if torch.all(idx0 < len(exp_slices) - 1):
            idx_exp1 = idx0 + 1
            d1 = torch.sqrt(torch.sum(pi_slices - exp_slices[idx_exp1]) ** 2)
        else:
            d1 = torch.tensor(0.0)

        return d0 + d1

    # Knothe-Rosenblatt transport -> need a way to design partitioning?
    def kr(
        self,
        obs_pi: torch.Tensor,
        acs_pi: torch.Tensor,
        obs_exp: torch.Tensor,
        acs_exp: torch.Tensor,
        random: bool = False,
    ) -> int:
        return 0

    def max_gsw(
        self,
        obs_pi: torch.Tensor,
        acs_pi: torch.Tensor,
        obs_exp: torch.Tensor,
        acs_exp: torch.Tensor,
        iterations: int = 10,
        lr: float = 1e-4,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # N,dn = X.shape
        # M,dm = Y.shape
        # assert dn==dm and M==N

        self.weight_reset()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        total_loss = np.zeros((iterations,))
        for i in tqdm(range(iterations)):
            optimizer.zero_grad()
            loss = -self.gsw(obs_pi, acs_pi, obs_exp, acs_exp, random=False)
            total_loss[i] = loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()

        return self.gsw_dist_nn(obs_pi, acs_pi, obs_exp, acs_exp, random=False)

    def irm_penalty(
        self, logits: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = torch.tensor(1.0).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        # safe_norm = (torch.sum(grad ** 2, dim=-1) + 1e-8).sqrt()
        # L1
        # gradient_mag = torch.mean((safe_norm - 1) ** 2)
        # return gradient_mag, grad
        return torch.sum(grad**2), grad

    def lip_penalty(
        self, update_dict: TensorDict, p: float = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_obs = update_dict["policy_obs"]
        policy_acs = update_dict["policy_acs"]
        policy_obs_next = update_dict["policy_obs_next"]
        policy_dones = update_dict["policy_dones"]
        exp_obs = update_dict["expert_obs"]
        exp_acs = update_dict["expert_acs"]
        exp_obs_next = update_dict["expert_obs_next"]
        exp_dones = update_dict["expert_dones"]

        if len(exp_obs.shape) != len(exp_dones.shape):
            exp_dones = torch.unsqueeze(exp_dones, -1)

        if len(policy_obs.shape) != len(policy_dones.shape):
            policy_dones = torch.unsqueeze(policy_dones, -1)

        obs_epsilon = torch.rand(
            policy_obs.shape, device=policy_obs.device, dtype=policy_obs.dtype
        )
        interp_obs = obs_epsilon * policy_obs + (1 - obs_epsilon) * exp_obs
        interp_obs.requires_grad = True  # For gradient calculation

        input_ = [interp_obs]
        if self.use_actions:
            acs_epsilon = torch.rand(
                policy_acs.shape, device=policy_acs.device, dtype=policy_acs.dtype
            )
            interp_acs = acs_epsilon * policy_acs + (1 - acs_epsilon) * exp_acs
            interp_acs.requires_grad = True  # For gradient calculation
            input_.append(interp_acs)

        if self.use_next_obs:
            nobs_epsilon = torch.rand(
                policy_obs_next.shape,
                device=policy_obs_next.device,
                dtype=policy_obs_next.dtype,
            )
            interp_next_obs = (
                nobs_epsilon * policy_obs_next + (1 - nobs_epsilon) * exp_obs_next
            )
            interp_next_obs.requires_grad = True  # For gradient calculation
            input_.append(interp_next_obs)
        if self.use_dones:
            d_epsilon = torch.rand(
                policy_dones.shape, device=policy_dones.device, dtype=policy_dones.dtype
            )
            interp_d = d_epsilon * policy_dones + (1 - d_epsilon) * exp_dones
            interp_d.requires_grad = True  # For gradient calculation
            input_.append(interp_d)

        estimate, _, mu, std = self.proj(*input_)

        # policy_obs = update_dict['policy_obs']
        # expert_obs = update_dict['expert_obs']
        # policy_obs.requires_grad = True
        # expert_obs.requires_grad = True

        gradient_mix = torch.autograd.grad(estimate.sum(), input_, create_graph=True)[0]
        # gradient_p = torch.autograd.grad(
        #     estimate.sum(), policy_obs, create_graph=True)[0]
        # gradient_e = torch.autograd.grad(
        #     estimate.sum(), expert_obs, create_graph=True)[0]

        # Norm's gradient could be NaN at 0. Use our own safe_norm
        safe_norm = (torch.sum(gradient_mix**2, dim=1) + 1e-8).sqrt()
        # L1
        gradient_mag = torch.mean((safe_norm - p) ** 2)

        return gradient_mag, gradient_mix  # , gradient_e, gradient_p

    def compute_loss(self, update_dict: TensorDict) -> Dict[str, torch.Tensor | float]:
        # compute sliced Wasserstein distance here
        self.policy_obs = copy.deepcopy(update_dict["policy_obs"])
        self.policy_acs = update_dict["policy_acs"]
        policy_obs_next = update_dict["policy_obs_next"]
        policy_dones = update_dict["policy_dones"]
        self.buffer_empty_cnt = 0

        exp_obs = update_dict["expert_obs"]
        exp_acs = update_dict["expert_acs"]
        exp_obs_next = update_dict["expert_obs_next"]
        exp_dones = update_dict["expert_dones"]

        if self.opt.swil_loss == "surr_loss":
            # surrogate loss from maxSWGAN paper
            policy_out, _, p_mu, p_std = self.proj(
                self.policy_obs,
                self.policy_acs,
                policy_obs_next,
                policy_dones,
                noise=True,
            )
            expert_out, _, e_mu, e_std = self.proj(
                exp_obs, exp_acs, exp_obs_next, exp_dones, noise=True
            )
            # ensure contiguous/cloned to avoid as_strided/inplace issues in autograd
            policy_out = policy_out.contiguous().clone()
            expert_out = expert_out.contiguous().clone()

            device = expert_out.device
            labels = torch.cat(
                [
                    torch.zeros(expert_out.size(), device=device),
                    torch.ones(policy_out.size(), device=device),
                ]
            )

            # sort slices for policy repl loss on detached tensors to avoid autograd
            pi_slices_sorted, pi_slices_sorted_idx = torch.sort(
                policy_out.detach(), dim=0, stable=True
            )
            exp_slices_sorted, exp_slices_sorted_idx = torch.sort(
                expert_out.detach(), dim=0, stable=True
            )

            # save to FIFO queue of sorted atoms
            self.pi_atoms_sorted.append(pi_slices_sorted.detach())
            self.exp_atoms_sorted.append(exp_slices_sorted.detach())
            self.pi_atoms_sorted_idx.append(pi_slices_sorted_idx)
            self.exp_atoms_sorted_idx.append(exp_slices_sorted_idx)
            self.pi_atoms_sorted_bkp = copy.deepcopy(self.pi_atoms_sorted)
            self.exp_atoms_sorted_bkp = copy.deepcopy(self.exp_atoms_sorted)

            # self.pi_atoms_sorted_ = copy.deepcopy(self.pi_atoms_sorted)

            d_out = torch.cat([expert_out, policy_out], dim=0).contiguous()
            bce_loss = F.binary_cross_entropy_with_logits(d_out, labels)
            if self.opt.irm_coeff > 0:
                irm_pen, _ = self.irm_penalty(d_out, labels)
            else:
                irm_pen = 0
            # surr_loss = torch.sum(self.proj(self.policy_obs, self.policy_acs)) - torch.sum(self.proj(exp_obs,exp_acs))
            d_loss = bce_loss
            # print(d_loss)
        elif self.opt.swil_loss == "approx_sw":
            obs = self.policy_obs
            acs = self.policy_acs
            if len(obs.shape) != len(acs.shape):
                acs = torch.unsqueeze(acs, -1)
            if len(exp_obs.shape) != len(exp_acs.shape):
                exp_acs = torch.unsqueeze(exp_acs, -1)
            d_loss = approximate_sw(
                torch.cat([obs, acs], -1), torch.cat([exp_obs, exp_acs], -1)
            )
            irm_pen = 0
        else:
            if self.opt.max_gsw:
                gsw_dist, pi_proj_norm, exp_proj_norm = self.max_gsw(
                    self.policy_obs, self.policy_acs, exp_obs, exp_acs
                )
            else:
                gsw_dist, pi_proj_norm, exp_proj_norm = self.gsw_dist_nn(
                    self.policy_obs,
                    self.policy_acs,
                    policy_obs_next,
                    policy_dones,
                    exp_obs,
                    exp_acs,
                    exp_obs_next,
                    exp_dones,
                    random=False,
                )

            d_loss = -gsw_dist + self.proj_norm_coeff * (pi_proj_norm + exp_proj_norm)
            irm_pen = 0

        output_dict = {}
        if self.opt.swil_ae:
            # meaningful reconstruction: action prediction?
            base_out = self.base_fwd(self.base, exp_obs, exp_acs)
            recon = self.recon_net(self.reward(base_out))
            if self.use_actions:
                if len(exp_acs.shape) == 1:
                    exp_acs = exp_acs.unsqueeze(-1)
                output_dict["recon_loss"] = F.mse_loss(
                    recon, torch.cat([exp_obs, exp_acs], -1)
                )
            else:
                output_dict["recon_loss"] = F.mse_loss(recon, exp_obs)
        else:
            output_dict["recon_loss"] = 0

        if self.swil_vb > 0:
            l_kld = gaussian_kld(p_mu, p_std)
            l_kld = l_kld.mean()
            e_kld = gaussian_kld(e_mu, e_std)
            e_kld = e_kld.mean()
            kld = 0.5 * (l_kld + e_kld)
            bottleneck_loss = kld - self.i_c
        else:
            bottleneck_loss = 0

        if self.opt.lip_coeff > 0:
            lip_penalty, grad_mix = self.lip_penalty(update_dict, self.opt.lip_p)
            grad_mix_norm = torch.norm(grad_mix)
        else:
            lip_penalty, grad_mix_norm = 0, 0

        # with torch.no_grad():
        #    self.beta = torch.max(torch.tensor(0.0), self.beta + self.alpha_beta * bottleneck_loss)

        output_dict["d_loss"] = d_loss
        output_dict["grad_penalty"] = irm_pen
        output_dict["ib_loss"] = bottleneck_loss
        output_dict["beta"] = self.beta
        output_dict["lip_penalty"] = lip_penalty

        # TODO: classification or reconstruction loss for

        return output_dict

    def update(self, loss: torch.Tensor) -> None:
        self.d_optimizer.zero_grad()
        if self.opt.swil_loss != "approx_sw":
            loss.backward()
        self.d_optimizer.step()
        self.d_scheduler.step()

    def update_rew(self, loss: torch.Tensor) -> None:
        self.r_optimizer.zero_grad()
        loss.backward()
        self.r_optimizer.step()
        self.r_scheduler.step()


class FlowModel(nn.Module):
    def __init__(self, env: gym.Env, opt: Namespace) -> None:
        super(FlowModel, self).__init__()

        self.env = env
        self.opt = opt
        self.gsw_df = opt.gsw_df
        self.layer_dims = opt.d_layer_dims
        self.lr = opt.disc_lr
        self.use_actions = opt.use_actions
        self.use_dones = opt.use_dones
        self.use_next_obs = opt.use_next_obs
        self.bias = opt.use_disc_bias
        bias = self.bias
        self.model_type = opt.swil_reward_type

        if opt.disc_nonlin == "relu":
            nonlin = nn.ReLU()
        elif opt.disc_nonlin == "leakyrelu":
            nonlin = nn.LeakyReLU()
        elif opt.disc_nonlin == "tanh":
            nonlin = nn.Tanh()
        elif opt.disc_nonlin == "id":
            nonlin = nn.Identity()
        else:
            nonlin = nn.PReLU()

        ob_shapes = list(env.observation_space.shape)
        ac_shapes = list(env.action_space.shape)
        if not ac_shapes:
            ac_shapes = [1]

        dim0 = ob_shapes[-1]
        if self.use_actions:
            dim0 = dim0 + ac_shapes[-1]
        if opt.use_dones:
            dim0 = dim0 + 1
        if opt.use_next_obs:
            dim0 = dim0 + ob_shapes[-1]

        self.layer_dims = [dim0] + self.layer_dims

        ac_len = ac_shapes[0]

        if opt.linear_proj:
            self.base = linlayer(self.layer_dims[0], ob_shapes[-1], bias)
        else:
            self.base = nn.Sequential(
                linlayer(self.layer_dims[0], self.layer_dims[1], bias), nonlin
            )

        if opt.linear_proj:
            self.model = nn.Identity()
        else:
            if self.model_type == "linear":
                self.model_layers = []
                for i in range(2, len(self.layer_dims)):
                    self.model_layers += [
                        linlayer(self.layer_dims[i - 1], self.layer_dims[i], bias),
                        nonlin,
                    ]

                self.model_layers += [
                    linlayer(self.layer_dims[-1], ob_shapes[-1], bias)
                ]
                self.model = nn.Sequential(*self.model_layers)
            else:
                self.model = nn.MultiheadAttention(
                    embed_dim=self.layer_dims[1], num_heads=1
                )

        if torch.cuda.is_available():
            self.base.cuda()
            self.model.cuda()
            if self.opt.proj_layer:
                self.proj_layer.cuda()

        self.d_optimizer = Adam(
            self.parameters(), lr=self.lr, weight_decay=self.opt.l2_coeff
        )  # , eps=1e-5)
        self.d_scheduler = ExponentialLR(self.d_optimizer, gamma=opt.scheduler_gamma)

    def forward(
        self,
        ob: torch.Tensor,
        ac: torch.Tensor | None = None,
        nob: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
        lprobs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # forward the nn models
        base_out = self.base_fwd(self.base, ob, ac, nob, d)
        grads = self.model(base_out)

        # get random projections based on reward
        return grads

    def base_fwd(
        self,
        base_fn: nn.Module,
        ob: torch.Tensor,
        ac: torch.Tensor | None = None,
        nob: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_actions:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = base_fn(torch.cat([ob, ac], axis=-1))
        else:
            base_out = base_fn(ob)

        return base_out

    def weight_reset(self) -> None:
        reset_parameters = getattr(self.reward, "reset_parameters", None)
        if callable(reset_parameters):
            self.reward.reset_parameters()

    def get_reward(
        self,
        ob: torch.Tensor,
        ac: torch.Tensor,
        nob: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_out = self.base_fwd(self.base, ob, ac, nob, d)
        grads = self.model(base_out)

        d_ob = nob - ob
        d_ob = d_ob / torch.norm(d_ob, -1)
        grads = grads / torch.norm(grads, -1)
        # dot product
        return torch.sum((nob - ob) * grads, -1)

    def prepare_x(self, ob: torch.Tensor, ac: torch.Tensor) -> torch.Tensor:
        if self.use_actions:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            X = torch.cat([ob, ac], axis=-1)
        else:
            X = ob

        return X

    def update(self, loss: torch.Tensor) -> None:
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()
        self.d_scheduler.step()
