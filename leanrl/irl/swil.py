import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import copy

from collections import deque
from sklearn.utils import shuffle

from torch.nn.utils import spectral_norm, weight_norm
from irl.utils import MiniGridCNN, AtariCNNBase, gaussian_kld
from irl.gsw_utils import gsw
from tqdm import tqdm


def get_random_projections(
    latent_dim: int, num_samples: int, proj_dist="normal"
) -> torch.Tensor:
    """
    Returns random samples from latent distribution's (Gaussian)
    unit sphere for projecting the encoded samples and the
    distribution samples.

    :param latent_dim: (Int) Dimensionality of the latent space (D)
    :param num_samples: (Int) Number of samples required (S)
    :return: Random projections from the latent unit sphere
    """
    if proj_dist == "normal":
        rand_samples = torch.randn(num_samples, latent_dim)
    elif proj_dist == "cauchy":
        rand_samples = (
            torch.distributions.Cauchy(torch.tensor([0.0]), torch.tensor([1.0]))
            .sample((num_samples, latent_dim))
            .squeeze()
        )
    else:
        raise ValueError("Unknown projection distribution.")

    rand_proj = rand_samples / rand_samples.norm(dim=1).view(-1, 1)
    return rand_proj  # [S x D]


def get_generalized_random_projections(
    projection_model, latent_dim: int, num_samples: int, proj_dist: str
) -> torch.Tensor:
    """
    Returns random samples from latent distribution's (Gaussian)
    unit sphere for projecting the encoded samples and the
    distribution samples.

    :param latent_dim: (Int) Dimensionality of the latent space (D)
    :param num_samples: (Int) Number of samples required (S)
    :return: Random projections from the latent unit sphere
    """

    # flows?
    rand_samples = projection_model(torch.randn(num_samples, latent_dim))

    rand_proj = rand_samples / rand_samples.norm(dim=1).view(-1, 1)
    return rand_proj  # [S x D]


def approximate_sw(X, Y, centering=True):
    """
    Approximates SW with Wasserstein distance between Gaussian approximations
    """
    if len(X.shape) == 1:
        X = X.unsqueeze(0)
    d = X.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Convert numpy arrays to torch tensors
    X = torch.tensor(X, dtype=torch.float, device=device)
    Y = torch.tensor(Y, dtype=torch.float, device=device)
    if centering:
        # Center the data
        mean_X = torch.mean(X, dim=0)
        mean_Y = torch.mean(Y, dim=0)
        X = X - mean_X
        Y = Y - mean_Y
    # Approximate SW
    m2_Xc = torch.mean(torch.linalg.norm(X, dim=1) ** 2) / d
    m2_Yc = torch.mean(torch.linalg.norm(Y, dim=1) ** 2) / d
    sw = torch.abs(m2_Xc ** (1 / 2) - m2_Yc ** (1 / 2))
    return sw


def compute_swd(
    z_expert: torch.Tensor,
    z_policy: torch.Tensor,
    p: float,
    reg_weight: float,
    n_proj: int,
    latent_dim: int,
) -> torch.Tensor:
    """
    Computes the Sliced Wasserstein Distance (SWD) - which consists of
    randomly projecting the encoded and prior vectors and computing
    their Wasserstein distance along those projections.

    :param z: Latent samples # [N  x D]
    :param p: Value for the p^th Wasserstein distance
    :param reg_weight:
    :return:
    """
    prior_z = z_policy
    device = z_expert.device

    proj_matrix = get_random_projections(latent_dim, n_proj).transpose(0, 1).to(device)
    # gen_proj_matrix = get_generalized_random_projections(latent_dim, ns_proj).transpose(0,1).to(device)

    latent_projections = z_expert.matmul(proj_matrix)  # [N x S]
    prior_projections = prior_z.matmul(proj_matrix)  # [N x S]

    # The Wasserstein distance is computed by sorting the two projections
    # across the batches and computing their element-wise l2 distance
    w_dist = (
        torch.sort(latent_projections.t(), dim=1)[0]
        - torch.sort(prior_projections.t(), dim=1)[0]
    )
    w_dist = w_dist.pow(p)
    return reg_weight * w_dist.mean()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    if isinstance(layer, nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, std)
    # torch.nn.init.constant_(layer.bias, bias_const)
    # return layer


def linlayer(in_dim, out_dim, bias=True, wnorm=False, snorm=False):
    # return layer_init(nn.Linear(in_dim, out_dim, bias=bias))
    if wnorm:
        return weight_norm(nn.Linear(in_dim, out_dim, bias=bias), "weight")
    elif snorm:
        return spectral_norm(nn.Linear(in_dim, out_dim, bias=bias), "weight")
    else:
        return nn.Linear(in_dim, out_dim, bias=bias)


class SWILDiscriminator(nn.Module):
    def __init__(self, env, opt):
        super(SWILDiscriminator, self).__init__()

        self.env = env
        self.opt = opt
        self.gsw_df = opt.gsw_df
        self.layer_dims = opt.d_layer_dims
        self.lr = opt.disc_lr
        self.use_actions = opt.use_actions
        self.use_dones = opt.use_dones
        self.use_next_obs = opt.use_next_obs
        self.use_cnn_base = opt.use_cnn_base
        use_cnn_base = self.use_cnn_base
        self.bias = opt.use_disc_bias
        bias = self.bias
        self.n_proj = opt.n_proj
        self.reward_type = opt.swil_reward_type
        self.is_atari = "atari" in self.opt.exp_name
        self.swil_vb = opt.swil_vb
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
        elif opt.disc_nonlin == "tanh":
            nonlin = nn.Tanh()
        elif opt.disc_nonlin == "id":
            nonlin = nn.Identity()
        else:
            nonlin = nn.PReLU()

        # if opt.swil_ae:
        # constrain the dimension if using an autoencoder
        #    self.layer_dims[1] = 10
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
            if use_cnn_base:
                self.base = MiniGridCNN(self.layer_dims, self.use_actions)
            elif self.is_atari:
                self.base = AtariCNNBase(opt, env, self.use_actions)
            elif opt.linear_proj:
                self.base = linlayer(
                    self.layer_dims[0],
                    self.opt.n_proj,
                    bias,
                    opt.use_weight_norm,
                    opt.use_spectral_norm,
                )
            else:
                # self.base = nn.Sequential(torch.nn.Linear(self.layer_dims[0],
                #                                          self.layer_dims[1], bias), nonlin)
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

            if self.is_atari:
                self.reward = nn.Linear(512, self.opt.n_proj)
            elif opt.linear_proj:
                self.reward = nn.Identity()
            else:
                if self.reward_type == "linear":
                    self.reward_layers = []
                    for i in range(2, len(self.layer_dims)):
                        # self.reward_layers += [torch.nn.Linear(in_features=self.layer_dims[i - 1],
                        #                                     out_features=self.layer_dims[i],
                        #                                     bias=bias), nonlin]
                        self.reward_layers += [
                            linlayer(
                                self.layer_dims[i - 1],
                                self.layer_dims[i],
                                bias,
                                opt.use_weight_norm,
                                opt.use_spectral_norm,
                            ),
                            nonlin,
                        ]

                    # self.reward_layers += [torch.nn.Linear(in_features=self.layer_dims[-1],
                    #                                     out_features=self.n_proj,
                    #                                     bias=bias)]
                    self.reward_layers += [
                        linlayer(
                            self.layer_dims[-1],
                            self.opt.n_proj,
                            bias,
                            opt.use_ll_weight_norm,
                            opt.use_spectral_norm,
                        ),
                        nonlin,
                    ]
                    # linlayer(self.layer_dims[-1], self.n_proj, bias, opt.use_ll_weight_norm, opt.use_spectral_norm), nn.Sigmoid()]
                    self.reward = nn.Sequential(*self.reward_layers)
                else:
                    self.reward = nn.MultiheadAttention(
                        embed_dim=self.layer_dims[1], num_heads=1
                    )

            if self.opt.proj_layer:
                # self.proj_layer = nn.Sequential(nn.Linear(in_features=self.n_proj,
                #                                         out_features=1,
                #                                         bias=bias), torch.nn.Tanh())
                self.proj_layer = nn.Sequential(
                    linlayer(
                        self.opt.n_proj,
                        1,
                        bias,
                        opt.use_weight_norm,
                        opt.use_spectral_norm,
                    ),
                    torch.nn.Tanh(),
                )

            if self.opt.swil_rew == "replacement_nn":
                self.base_rew_replnn = copy.deepcopy(self.base)
                self.rew_replnn = copy.deepcopy(self.reward)

            if self.opt.swil_ae:
                self.ae_layers = []
                for i in reversed(range(2, len(self.layer_dims))):
                    self.ae_layers += [
                        torch.nn.Linear(
                            in_features=self.opt.n_proj,
                            out_features=self.layer_dims[i],
                            bias=bias,
                        ),
                        nonlin,
                    ]
                self.ae_layers += [
                    torch.nn.Linear(
                        in_features=self.layer_dims[-1],
                        out_features=self.layer_dims[0],
                        bias=bias,
                    )
                ]
                self.recon_net = nn.Sequential(*self.ae_layers)

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

        if self.swil_vb > 0:
            self.encoder_z = nn.Sequential(
                linlayer(
                    self.layer_dims[1],
                    self.swil_vb * 2,
                    bias,
                    opt.use_weight_norm,
                    opt.use_spectral_norm,
                ),
                nonlin,
            )
            self.decoder_z = nn.Sequential(
                linlayer(
                    self.swil_vb,
                    self.layer_dims[2],
                    bias,
                    opt.use_weight_norm,
                    opt.use_spectral_norm,
                ),
                nonlin,
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

    def forward(self, ob, ac, nob=None, d=None, lprobs=None):
        # forward the nn models
        gsw_dist = self.gsw_dist(
            ob, ac, nob, d, self.expert_obs, self.expert_acs, random=False
        )

        # get random projections based on reward
        return gsw_dist

    def proj(self, ob, ac, nob=None, d=None, noise=False):
        if self.opt.n_proj > 1 and not self.opt.linear_proj:
            with torch.no_grad():
                self.base.apply(layer_init)
                self.reward.apply(layer_init)

        base_out = self.base_fwd(self.base, ob, ac, nob, d)

        if self.swil_vb > 0:
            vb_out, z, mu, std = self.vb(base_out, noise=noise)
        else:
            vb_out = base_out
            z, mu, std = 0, 0, 0
        # rew, v, v_n, d_out = self.forward(ob, next_ob, ac, lprobs) TODO??
        # potentially use more sophisticated mechanism with multiple projections
        if self.opt.proj_layer:
            return self.proj_layer(self.reward(vb_out)), z, mu, std
        else:
            # XXX: additional noise?
            rew = self.reward(vb_out)
            # perturb with noise -> TODO: sample from stochastic (e.g. Gaussian) process?
            if self.opt.n_proj > 1:
                rew + torch.randn_like(rew) * 0.01
            if self.opt.add_proj_noise:
                rew + torch.randn_like(rew) * 0.01

            # out = rew/rew.norm(-1)
            return rew, z, mu, std  # + torch.randn_like(self.reward(base_out))

    def base_fwd(self, base_fn, ob, ac, nob=None, d=None):
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

        if self.use_cnn_base or self.is_atari:
            base_out = base_fn(*input_)

        else:
            base_out = base_fn(torch.cat(input_, axis=-1))

        return base_out

    def weight_reset(self):
        reset_parameters = getattr(self.reward, "reset_parameters", None)
        if callable(reset_parameters):
            self.reward.reset_parameters()

    def reset_all_weights(self):
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

    def vb(self, base_out, noise=True):
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

    def compute_replacement_reward(self, ob, ac, nob=None, d=None, step=None):
        with torch.no_grad():
            # project next state and rank it as part of previous evaluation
            base_out = self.base_fwd(self.base, ob, ac, nob, d)
            if self.swil_vb > 0:
                vb_out, z, mu, std = self.vb(base_out, noise=False)
            else:
                vb_out = base_out

            obs_t_slice = self.reward(vb_out).unsqueeze(0)

            rew = torch.zeros(1, device=obs_t_slice.device)

            for p, (sorted_proj, sorted_proj_tgt) in enumerate(
                zip(self.pi_atoms_sorted, self.exp_atoms_sorted)
            ):
                # sorted_proj = self.pi_atoms_sorted
                # sorted_proj_tgt = self.exp_atoms_sorted

                # if sorted_proj is None:
                #     return torch.zeros(1)

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

                    # print(idxs, idxs_e, idx, torch.where(idxs==idx)[0])

                    ## if idx is not in sorted list, use nearest neighbours

                    # if self.opt.repl_loss_type == 'diff':
                    #    idx -= 1
                    # idx[idx==0] +=1
                    idx[idx == n] -= 1
                    # shift extreme indices
                    # print(n)

                    # print("Number of atoms in buffer", n)
                    # calculate weight based on position in queue
                    weight = len(self.pi_atoms_sorted)
                    ### TODO: this is important?
                    if self.opt.atom_weight == "1/n":
                        w = 1 / n
                    else:
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
                                        rew += w * (a_new_h - a_h)
                                        rew_j = a_new_h - a_h
                                    else:
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
                                self.pi_atoms_sorted[p][idx] = obs_t_slice

                            if self.opt.replace_atoms_ouo:
                                # assert self.opt.swil_period == '-1.0', 'full trajectories required'
                                self.pi_atoms_sorted[p] = torch.cat(
                                    [
                                        self.pi_atoms_sorted[p][:idx],
                                        self.pi_atoms_sorted[p][idx + 1 : -1],
                                    ]
                                )
                                self.exp_atoms_sorted[p] = torch.cat(
                                    [
                                        self.exp_atoms_sorted[p][:idx],
                                        self.exp_atoms_sorted[p][idx + 1 : -1],
                                    ]
                                )

                            ## TODO: hierarchy of multiple batches?
                else:
                    self.buffer_empty_cnt += 1
                    print("Atom buffer empty", self.buffer_empty_cnt)
                    self.pi_atoms_sorted = copy.deepcopy(self.pi_atoms_sorted_bkp)
                    self.exp_atoms_sorted = copy.deepcopy(self.exp_atoms_sorted_bkp)

            rew = -rew

        return rew

    def swil_reward(self, ob, ac, nob=None, d=None):
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

    def get_reward(self, ob, ac, nob=None, d=None, step=None):
        if self.opt.swil_rew == "old_swil":
            return self.swil_reward(ob, ac, nob, d)
        elif self.opt.swil_rew == "replacement_nn":
            return self.rew_replnn(self.base_fwd(self.base_rew_replnn, ob, ac, nob))
        else:
            return self.compute_replacement_reward(ob, ac, nob, d, step)

    def compute_replacement_reward_loss(self, update_dict, update_dict_2):
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

    def get_slice(self, ob, ac, theta):
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

    def prepare_x(self, ob, ac):
        if self.use_actions:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            X = torch.cat([ob, ac], axis=-1)
        else:
            X = ob

        return X

    def gsw_dist(
        self,
        obs_pi,
        acs_pi,
        obs_exp,
        acs_exp,
        nobs_pi=None,
        d_pi=None,
        nobs_exp=None,
        d_exp=None,
    ):
        X_exp = self.prepare_x(obs_exp, acs_exp)
        X_pi = self.prepare_x(obs_pi, acs_pi)

        return self.gsw_module.gsw(X_exp, X_pi)

    def max_gsw_dist(
        self, obs_pi, acs_pi, nobs_pi, d_pi, obs_exp, acs_exp, nobs_exp, d_exp
    ):
        X_exp = self.prepare_x(obs_exp, acs_exp)
        X_pi = self.prepare_x(obs_pi, acs_pi)

        return self.gsw_module.max_gsw(X_exp, X_pi)

    def gsw_dist_nn(
        self,
        obs_pi,
        acs_pi,
        nobs_pi,
        d_pi,
        obs_exp,
        acs_exp,
        nobs_exp,
        d_exp,
        random=False,
    ):
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

        if self.opt.shuffle_atom_batches:
            if len(self.pi_atoms_sorted) == 0:
                self.pi_atoms_sorted.append(pi_slices_sorted)
                self.exp_atoms_sorted.append(exp_slices_sorted)
                self.pi_atoms_sorted_idx.append(pi_slices_sorted_idx)
                self.exp_atoms_sorted_idx.append(exp_slices_sorted_idx)
            elif np.random.rand() > 0.5:
                self.pi_atoms_sorted.append(pi_slices_sorted)
                self.exp_atoms_sorted.append(exp_slices_sorted)
                self.pi_atoms_sorted_idx.append(pi_slices_sorted_idx)
                self.exp_atoms_sorted_idx.append(exp_slices_sorted_idx)
        else:
            self.pi_atoms_sorted.append(pi_slices_sorted)
            self.exp_atoms_sorted.append(exp_slices_sorted)
            self.pi_atoms_sorted_idx.append(pi_slices_sorted_idx)
            self.exp_atoms_sorted_idx.append(exp_slices_sorted_idx)

        # shuffle atom batches
        if self.opt.shuffle_atom_batches:
            self.pi_atoms_sorted, self.pi_atoms_sorted_idx = shuffle(
                self.pi_atoms_sorted, self.pi_atoms_sorted_idx, random_state=0
            )
            self.exp_atoms_sorted, self.exp_atoms_sorted_idx = shuffle(
                self.exp_atoms_sorted, self.exp_atoms_sorted_idx, random_state=0
            )

        return torch.sqrt(torch.sum((pi_slices_sorted - exp_slices_sorted) ** 2))

    def atom_gsw(self, obs_pi, acs_pi, obs_exp, acs_exp, random=False):
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
    def kr(self, obs_pi, acs_pi, obs_exp, acs_exp, random=False):
        return 0

    def max_gsw(self, obs_pi, acs_pi, obs_exp, acs_exp, iterations=10, lr=1e-4):
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

    def irm_penalty(self, logits, y):
        scale = torch.tensor(1.0).requires_grad_()
        loss = F.binary_cross_entropy_with_logits(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        # safe_norm = (torch.sum(grad ** 2, dim=-1) + 1e-8).sqrt()
        # L1
        # gradient_mag = torch.mean((safe_norm - 1) ** 2)
        # return gradient_mag, grad
        return torch.sum(grad**2), grad

    def lip_penalty(self, update_dict, p=1):
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

    def compute_loss(self, update_dict):
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
                gsw_dist = self.max_gsw(
                    self.policy_obs, self.policy_acs, exp_obs, exp_acs
                )
            else:
                gsw_dist = -self.gsw_dist_nn(
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
            d_loss = gsw_dist
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

    def update(self, loss):
        self.d_optimizer.zero_grad()
        if self.opt.swil_loss != "approx_sw":
            loss.backward()
        self.d_optimizer.step()
        self.d_scheduler.step()

    def update_rew(self, loss):
        self.r_optimizer.zero_grad()
        loss.backward()
        self.r_optimizer.step()
        self.r_scheduler.step()


class FlowModel(nn.Module):
    def __init__(self, env, opt):
        super(FlowModel, self).__init__()

        self.env = env
        self.opt = opt
        self.gsw_df = opt.gsw_df
        self.layer_dims = opt.d_layer_dims
        self.lr = opt.disc_lr
        self.use_actions = opt.use_actions
        self.use_dones = opt.use_dones
        self.use_next_obs = opt.use_next_obs
        self.use_cnn_base = opt.use_cnn_base
        use_cnn_base = self.use_cnn_base
        self.bias = opt.use_disc_bias
        bias = self.bias
        self.model_type = opt.swil_reward_type
        self.is_atari = "atari" in self.opt.exp_name

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

        if use_cnn_base:
            self.base = MiniGridCNN(self.layer_dims, self.use_actions)
        elif self.is_atari:
            self.base = AtariCNNBase(opt, env, self.use_actions)
        elif opt.linear_proj:
            self.base = linlayer(self.layer_dims[0], ob_shapes[-1], bias)
        else:
            self.base = nn.Sequential(
                linlayer(self.layer_dims[0], self.layer_dims[1], bias), nonlin
            )

        if self.is_atari:
            self.model = nn.Linear(512, ob_shapes[-1])
        elif opt.linear_proj:
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

    def forward(self, ob, ac=None, nob=None, d=None, lprobs=None):
        # forward the nn models
        base_out = self.base_fwd(self.base, ob, ac, nob, d)
        grads = self.model(base_out)

        # get random projections based on reward
        return grads

    def base_fwd(self, base_fn, ob, ac=None, nob=None, d=None):
        if self.use_actions and (self.use_cnn_base or self.is_atari):
            base_out = base_fn(ob, ac)
        elif self.use_actions and not (self.use_cnn_base or self.is_atari):
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            base_out = base_fn(torch.cat([ob, ac], axis=-1))
        else:
            base_out = base_fn(ob)

        return base_out

    def weight_reset(self):
        reset_parameters = getattr(self.reward, "reset_parameters", None)
        if callable(reset_parameters):
            self.reward.reset_parameters()

    def get_reward(self, ob, ac, nob=None, d=None):
        base_out = self.base_fwd(self.base, ob, ac, nob, d)
        grads = self.model(base_out)

        d_ob = nob - ob
        d_ob = d_ob / torch.norm(d_ob, -1)
        grads = grads / torch.norm(grads, -1)
        # dot product
        return torch.sum((nob - ob) * grads, -1)

    def prepare_x(self, ob, ac):
        if self.use_actions:
            if len(ob.shape) != len(ac.shape):
                ac = torch.unsqueeze(ac, -1)
            X = torch.cat([ob, ac], axis=-1)
        else:
            X = ob

        return X

    def update(self, loss):
        self.d_optimizer.zero_grad()
        loss.backward()
        self.d_optimizer.step()
        self.d_scheduler.step()


class SwilReward(gym.Wrapper):
    def __init__(self, env, opt, disc):
        super().__init__(env=env)
        # self.reward_fn = make_network(**reward_fn_spec)
        self.use_actions = opt.use_actions
        self.use_next_obs = opt.use_next_obs
        self.use_dones = opt.use_dones
        self.opt = opt

        self.discriminator = disc
        self.obs = None
        # calculate reward at the end of trajecotiry
        self.traj = []
        self.traj_ = []
        self.cnt = 0

    def reset(self, seed=0, options=None):
        obs, info = super().reset(seed=self.opt.seed)
        self.cnt = 0
        return obs, info

    def step(self, action):
        next_obs, gt_reward, term, trunc, info = self.env.step(action)
        done = term or trunc
        info["gt_reward"] = gt_reward
        info["step"] = self.cnt
        self.cnt += 1
        self.traj.append(gt_reward)
        device = next(self.discriminator.parameters()).device
        dtype = torch.get_default_dtype()
        if self.obs is not None:
            obs_t = torch.tensor(self.obs, dtype=dtype, device=device)
        else:
            obs_t = torch.tensor(next_obs, dtype=dtype, device=device)

        acs_t = torch.tensor(action, dtype=dtype, device=device)
        next_obs_t = torch.tensor(next_obs, dtype=dtype, device=device)
        done_t = torch.tensor(done, dtype=dtype, device=device)

        with torch.no_grad():
            # get discriminator reward and train on that
            # print([p.norm(2) for p in self.discriminator.parameters()])
            irl_reward = (
                self.discriminator.get_reward(
                    obs_t, acs_t, next_obs_t, done_t, self.cnt
                )
                .detach()
                .cpu()
                .numpy()
            )
            # irl_reward = self.discriminator.get_reward(obs_t, acs_t, next_obs_t).cpu().numpy()
        self.obs = next_obs
        self.traj_.append(irl_reward)
        # print("GT: ", gt_reward, "IRL: ", irl_reward)
        if done:
            print("episode reward gt:", np.sum(np.array(self.traj)))
            print("episode reward irl:", np.sum(np.array(self.traj_)))
            info["episode"]["ep_rew_irl"] = np.sum(np.array(self.traj_))
            self.traj = []
            self.traj_ = []

        return next_obs, float(irl_reward), term, trunc, info


class SwilFlowReward(gym.Wrapper):
    def __init__(self, env, opt):
        super().__init__(env=env)
        # self.reward_fn = make_network(**reward_fn_spec)
        self.use_actions = opt.use_actions
        self.gradient_model = FlowModel(env, opt)
        self.obs = None
        self.sw_poly_deg = opt.sw_poly_deg
        self.radon_df_type = opt.radon_df_type
        # calculate reward at the end of trajecotiry
        self.traj = []

    def compute_sw_grad(self, policy_atoms, expert_atoms):
        # loss = ot.sliced_wasserstein_distance(x1_torch, x2_torch, n_projections=100, seed=gen)
        policy_atoms.requires_grad_(True)
        sw_dist, _, Xsorted, Ysorted = gsw(
            policy_atoms,
            expert_atoms,
            ftype=self.radon_df_type,
            n_proj=self.opt.n_proj,
            degree=self.sw_poly_deg,
        )
        sw_dist.backward()

        return sw_dist, policy_atoms.grad, Xsorted, Ysorted

    def update_gradient_estimator(self, policy_atoms, expert_atoms):
        sw_dist, policy_atoms_grad, Xsorted, Ysorted = self.compute_sw_grad(
            policy_atoms, expert_atoms
        )
        estimated_gradients = self.gradient_model.forward(policy_atoms)
        loss = F.mse_loss(estimated_gradients, policy_atoms_grad)
        self.gradient_model.update(loss)

        return sw_dist, Xsorted, Ysorted

    def step(self, action):
        next_obs, gt_reward, done, info = self.env.step(action)
        info["gt_reward"] = gt_reward
        if self.obs is not None:
            obs_t = torch.tensor(self.obs, dtype=torch.get_default_dtype())
        else:
            obs_t = torch.tensor(next_obs, dtype=torch.get_default_dtype())

        acs_t = torch.tensor(action, dtype=torch.get_default_dtype())
        next_obs_t = torch.tensor(next_obs, dtype=torch.get_default_dtype())
        done_t = torch.tensor(done, dtype=torch.get_default_dtype())

        if done:
            self.traj = []

        with torch.no_grad():
            # get discriminator reward and train on that
            # print([p.norm(2) for p in self.discriminator.parameters()])
            d_obs = next_obs_t - obs_t
            # sw_grad_obs =
            irl_reward = (
                self.gradient_model.get_reward(obs_t, acs_t, next_obs_t, done_t)
                .cpu()
                .numpy()
            )
            # irl_reward = self.discriminator.get_reward(obs_t, acs_t, next_obs_t).cpu().numpy()
        self.obs = next_obs

        return next_obs, irl_reward, done, info


"""
SWIL reward class based on difference of increasing arrangement maps
"""


class SwilIRDiffReward(gym.Wrapper):
    def __init__(self, env, opt):
        super().__init__(env=env)
        # self.reward_fn = make_network(**reward_fn_spec)
        self.use_actions = opt.use_actions
        self.obs = None
        self.opt = opt
        self.sw_poly_deg = opt.sw_poly_deg
        self.radon_df_type = opt.radon_df_type
        # TODO: makes GSW module shared across vec envs?
        self.gsw = GSW(
            ftype=self.radon_df_type,
            nofprojections=self.n_proj,
            degree=self.sw_poly_deg,
        )
        # calculate reward at the end of trajecotiry
        self.traj = []

    def compute_gsw(self, policy_atoms, expert_atoms):
        dist, theta, sorted_proj_pi, sorted_proj_exp = self.gsw.gsw(
            policy_atoms,
            expert_atoms,
            ftype=self.radon_df_type,
            n_proj=self.opt.n_proj,
            degree=self.sw_poly_deg,
        )
        self.sorted_proj_pi = sorted_proj_pi
        self.sorted_proj_exp = sorted_proj_exp
        self.theta = theta

        return dist

    def step(self, action):
        next_obs, gt_reward, done, info = self.env.step(action)
        info["gt_reward"] = gt_reward
        if self.obs is not None:
            obs_t = torch.tensor(self.obs, dtype=torch.get_default_dtype())
        else:
            obs_t = torch.tensor(next_obs, dtype=torch.get_default_dtype())

        acs_t = torch.tensor(action, dtype=torch.get_default_dtype())
        next_obs_t = torch.tensor(next_obs, dtype=torch.get_default_dtype())
        done_t = torch.tensor(done, dtype=torch.get_default_dtype())

        if done:
            self.traj = []

        rew = torch.tensor([0.0])
        if self.gsw.theta is not None:
            with torch.no_grad():
                # project next state and rank it as part of previous evaluation
                if self.opt.measurable_space == "sxs":
                    obs_nextobs = torch.cat([obs_t, next_obs_t], -1)
                    obs_t_slice = self.gsw.get_slice(
                        torch.unsqueeze(obs_nextobs, 0), self.gsw.theta
                    )
                elif self.opt.measurable_space == "sxa":
                    obs_nextobs = torch.cat([obs_t, acs_t], -1)
                    obs_t_slice = self.gsw.get_slice(
                        torch.unsqueeze(obs_nextobs, 0), self.gsw.theta
                    )
                else:
                    obs_t_slice = self.gsw.get_slice(
                        torch.unsqueeze(next_obs_t, 0), self.gsw.theta
                    )

                sorted_proj = self.gsw.Xslices_sorted
                sorted_proj_tgt = self.gsw.Yslices_sorted
                n = len(sorted_proj)

                # idx = torch.searchsorted(sorted_proj.T.contiguous(), obs_t_slice.T.contiguous())#, right=True)
                idx = torch.searchsorted(sorted_proj.T, obs_t_slice.T)  # , right=True)

                # shift extreme indices
                idx[idx == 0] += 1
                idx[idx == n] -= 1

                # TODO: what if target CDF is left or mixed?
                for j, i in enumerate(idx):
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
                        rew -= torch.sum(w * diff[: i - 1, j])
                        rew += torch.sum(w * diff[i:, j])
                    else:
                        # calculate diff when replacing atom
                        w = 1 / n
                        a_prev = (sorted_proj_tgt[i, j] - sorted_proj[i, j]) ** 2
                        a_new = (sorted_proj_tgt[i, j] - obs_t_slice[0, j]) ** 2

                        rew += w * (a_new - a_prev)

                        # replace atom in original array with new slice
                        # if (a_new - a_prev) < 0:
                        #    self.gsw.Xslices_sorted[i, j] = obs_t_slice[0, j]
                    # else:
                    # replace atom with obs_t_slice and calculate integral difference

                    # rew += torch.sum((sorted_proj[idx]-sorted_proj_tgt[idx])**2)

                rew = -rew
            # irl_reward = self.discriminator.get_reward(obs_t, acs_t, next_obs_t).cpu().numpy()
        self.obs = next_obs

        return next_obs, float(rew.detach().numpy()), done, info


class SwilRewardNew(gym.Wrapper):
    """Buffer-based SWIL rewards with three variants: SR, DUAL, RPL.
    Uses linear random projections over concatenated features of (obs, action, next_obs, done)
    controlled by opt.use_actions/use_next_obs/use_dones.
    """

    def __init__(self, env, opt, demos):
        super().__init__(env=env)
        self.opt = opt
        self.use_actions = opt.use_actions
        self.use_next_obs = opt.use_next_obs
        self.use_dones = opt.use_dones
        self.variant = getattr(opt, "swil_variant", "SR").upper()
        self.agg = getattr(opt, "swil_agg", "mean").lower()
        self.tau = float(getattr(opt, "swil_tau", 0.5))
        self.obs = None
        self.traj, self.traj_ = [], []
        self.cnt = 0

        # feature dimension
        if isinstance(env.observation_space, gym.spaces.Dict):
            ob_shapes = list(env.observation_space["observation"].shape)
        else:
            ob_shapes = list(env.observation_space.shape)
        ac_shapes = list(env.action_space.shape)
        if not ac_shapes:
            ac_shapes = [1]
        dim0 = ob_shapes[-1]
        if self.use_actions:
            dim0 += ac_shapes[-1]
        if self.use_dones:
            dim0 += 1
        if self.use_next_obs:
            dim0 += ob_shapes[-1]
        self.K = int(getattr(self.opt, "n_proj", 10))
        # directions
        self.dirs = torch.randn(self.K, dim0, dtype=torch.get_default_dtype())
        self.dirs = torch.nn.functional.normalize(self.dirs, dim=-1)
        # expert projections
        exp_obs = torch.tensor(demos["obs"], dtype=torch.get_default_dtype())
        parts = [exp_obs]
        if self.use_actions and "acs" in demos:
            parts.append(torch.tensor(demos["acs"], dtype=torch.get_default_dtype()))
        if self.use_dones and "done" in demos:
            d = torch.tensor(demos["done"], dtype=torch.get_default_dtype())
            if d.ndim == 1:
                d = d.unsqueeze(-1)
            parts.append(d)
        if self.use_next_obs:
            next_obs = np.concatenate([demos["obs"][1:], demos["obs"][-1:]], 0)
            parts.append(torch.tensor(next_obs, dtype=torch.get_default_dtype()))
        Xexp = torch.cat(parts, dim=-1)
        Zexp = Xexp @ self.dirs.t()  # [N,K]
        Zexp = Zexp.transpose(0, 1).contiguous()  # [K,N]
        # store expert projections as a single sorted matrix [K, N]
        self.exp_sorted = torch.sort(Zexp, dim=1, stable=True).values
        # optional quantile LUT over expert distribution per projection
        self.use_lut = bool(getattr(self.opt, "swil_use_lut", True))
        self.qgrid = int(getattr(self.opt, "swil_qgrid", 1025))
        if self.use_lut:
            self._build_exp_lut()
        # policy buffer: maintain per-projection sorted arrays incrementally
        self.cap = 8192
        self.pol_sorted = torch.full(
            (self.K, self.cap), float("inf"), dtype=torch.get_default_dtype()
        )
        self.counts = torch.zeros(self.K, dtype=torch.long)
        self.J = torch.arange(self.cap, dtype=torch.long)
        self.rpl_cursor = 0

    def reset(self, seed=0, options=None):
        obs, info = super().reset(seed=self.opt.seed)
        self.cnt = 0
        return obs, info

    def _feats(self, ob, ac, nob, d):
        feats = [ob]
        if self.use_actions:
            if len(ob.shape) != len(ac.shape):
                ac = ac.unsqueeze(-1)
            feats.append(ac)
        if self.use_dones:
            if len(ob.shape) != len(d.shape):
                d = d.unsqueeze(-1)
            feats.append(d.float())
        if self.use_next_obs:
            feats.append(nob)
        return torch.cat(feats, dim=-1)

    def _policy_sorted(self):
        """Return current sorted policy projections and counts (no re-sorting)."""
        return self.pol_sorted, self.counts

    def _qmap(self, z_sorted, y_sorted, z):
        """Scalar version kept for compatibility (single projection)."""
        N = max(z_sorted.numel(), 1)
        Ny = max(y_sorted.numel(), 1)
        pos = torch.searchsorted(z_sorted, z, right=False)
        q = (pos.to(z.dtype) + 0.5) / float(N)
        idxf = q * float(Ny) - 0.5
        i0 = torch.clamp(idxf.floor().to(torch.long), 0, Ny - 1)
        i1 = torch.clamp(i0 + 1, 0, Ny - 1)
        t = torch.clamp(idxf - i0.to(idxf.dtype), 0.0, 1.0)
        y0 = y_sorted[i0]
        y1 = y_sorted[i1]
        return y0 * (1.0 - t) + y1 * t

    def _qmap_vec(self, z_sorted, counts, y_sorted, z):
        """Vectorized quantile mapping.
        Inputs:
          - z_sorted: [K, Nz] sorted ascending (trailing filled)
          - counts: [K] valid counts per row
          - y_sorted: [K, Ny] sorted ascending
          - z: [K] or [K, Nzq] query values
        Returns: yhat with same trailing dims as z
        """
        K = z_sorted.shape[0]
        device = z_sorted.device
        Ny = y_sorted.shape[1]
        # ensure z has a last dim for batching
        if z.dim() == 1:
            zq = z.unsqueeze(1)  # [K,1]
        else:
            zq = z
        # positions along each row
        pos = torch.searchsorted(z_sorted, zq, right=False)  # [K, ...]
        # clamp positions to counts
        c = counts.unsqueeze(1).expand_as(pos)
        pos = torch.minimum(pos, c)
        denom = (
            torch.clamp(counts.to(z_sorted.dtype), min=1.0).unsqueeze(1).expand_as(pos)
        )
        q = (pos.to(z_sorted.dtype) + 0.5) / denom
        idxf = q * float(Ny) - 0.5
        i0 = torch.clamp(idxf.floor().to(torch.long), 0, Ny - 1)
        i1 = torch.clamp(i0 + 1, 0, Ny - 1)
        t = torch.clamp(idxf - i0.to(idxf.dtype), 0.0, 1.0)
        # gather along dim=1
        y0 = y_sorted.gather(1, i0)
        y1 = y_sorted.gather(1, i1)
        yhat = y0 * (1.0 - t) + y1 * t
        if z.dim() == 1:
            return yhat.squeeze(1)
        return yhat

    def _build_exp_lut(self):
        """Precompute quantile->value LUT for expert projections per direction.
        Produces self.exp_lut of shape [K, G], where G=self.qgrid.
        """
        K, Ny = self.exp_sorted.shape
        G = max(int(self.qgrid), 2)
        device = self.exp_sorted.device
        qgrid = torch.linspace(0.0, 1.0, G, dtype=self.exp_sorted.dtype, device=device)
        idxf = qgrid * float(Ny) - 0.5
        i0 = torch.clamp(idxf.floor().to(torch.long), 0, Ny - 1)
        i1 = torch.clamp(i0 + 1, 0, Ny - 1)
        t = torch.clamp(idxf - i0.to(idxf.dtype), 0.0, 1.0).view(1, -1)
        i0e = i0.view(1, -1).expand(K, -1)
        i1e = i1.view(1, -1).expand(K, -1)
        y0 = self.exp_sorted.gather(1, i0e)
        y1 = self.exp_sorted.gather(1, i1e)
        self.exp_lut = y0 * (1.0 - t) + y1 * t  # [K, G]

    def _qmap_from_q(self, q):
        """Map per-row quantiles q in [0,1] to expert values using LUT if enabled.
        q: [K] or [K, L]
        Returns yhat with same trailing dims as q.
        """
        if q.dim() == 1:
            qv = q.unsqueeze(1)
            squeeze = True
        else:
            qv = q
            squeeze = False
        qv = torch.clamp(qv, 0.0, 1.0)
        if self.use_lut and hasattr(self, "exp_lut"):
            G = self.exp_lut.shape[1]
            idxf = qv * float(G - 1)
            i0 = torch.clamp(idxf.floor().to(torch.long), 0, G - 1)
            i1 = torch.clamp(i0 + 1, 0, G - 1)
            t = torch.clamp(idxf - i0.to(idxf.dtype), 0.0, 1.0)
            y0 = self.exp_lut.gather(1, i0)
            y1 = self.exp_lut.gather(1, i1)
            yhat = y0 * (1.0 - t) + y1 * t
        else:
            # exact mapping via expert sorted values
            Ny = self.exp_sorted.shape[1]
            idxf = qv * float(Ny) - 0.5
            i0 = torch.clamp(idxf.floor().to(torch.long), 0, Ny - 1)
            i1 = torch.clamp(i0 + 1, 0, Ny - 1)
            t = torch.clamp(idxf - i0.to(idxf.dtype), 0.0, 1.0)
            y0 = self.exp_sorted.gather(1, i0)
            y1 = self.exp_sorted.gather(1, i1)
            yhat = y0 * (1.0 - t) + y1 * t
        if squeeze:
            return yhat.squeeze(1)
        return yhat

    def _agg(self, per):
        if self.agg == "mean":
            return per.mean()
        if self.agg == "max":
            return per.max()
        x = per / max(self.tau, 1e-6)
        x = x - x.max()
        w = torch.softmax(x, dim=0)
        return (w * per).sum()

    def _insert(self, z):
        """Insert a new sample z into per-projection sorted buffers (vectorized)."""
        Z = z @ self.dirs.t()  # [K]
        s = self.pol_sorted
        c = self.counts
        K, C = s.shape
        device = s.device
        # find insertion positions w.r.t. current sorted rows (trailing +inf)
        pos = torch.searchsorted(s, Z.unsqueeze(1), right=False).squeeze(1)
        pos = torch.minimum(pos, c)
        idx = torch.arange(C, device=device).unsqueeze(0).expand(K, -1)
        # shift right region [pos, c) by one
        mask_shift = (idx > pos.unsqueeze(1)) & (idx <= c.unsqueeze(1))
        src_shift = s.gather(1, torch.clamp(idx - 1, min=0))
        new_s = torch.where(mask_shift, src_shift, s)
        # place Z at pos
        new_s.scatter_(1, pos.unsqueeze(1), Z.unsqueeze(1))
        self.pol_sorted = new_s
        self.counts = torch.minimum(c + 1, torch.full_like(c, self.cap))

    def _replace_effect_sorted(self, Z):
        """Return the per-row sorted arrays after replacing one element with Z.
        Does not mutate buffers; used to compute RPL effect efficiently.
        """
        s = self.pol_sorted
        c = self.counts
        K, C = s.shape
        device = s.device
        if K == 0:
            return s
        # index to remove per row (based on current counts)
        nused = torch.clamp(c, min=1)
        idx_rem = torch.remainder(self.rpl_cursor, nused)
        idx = torch.arange(C, device=device).unsqueeze(0).expand(K, -1)
        cm1 = torch.clamp(c - 1, min=0)
        # remove by shifting left on valid range
        mask_leftshift = (idx >= idx_rem.unsqueeze(1)) & (idx < cm1.unsqueeze(1))
        src_left = s.gather(1, torch.clamp(idx + 1, max=C - 1))
        s_removed = torch.where(mask_leftshift, src_left, s)
        # compute insertion position before removal, adjust if it was after removed index
        pos_new = torch.searchsorted(s, Z.unsqueeze(1), right=False).squeeze(1)
        pos_new = torch.minimum(pos_new, c)
        pos_ins = pos_new - (pos_new > idx_rem)
        # insert Z into s_removed with counts decreased by 1
        mask_shift = (idx > pos_ins.unsqueeze(1)) & (idx <= cm1.unsqueeze(1))
        src_shift = s_removed.gather(1, torch.clamp(idx - 1, min=0))
        new_s = torch.where(mask_shift, src_shift, s_removed)
        new_s.scatter_(1, pos_ins.unsqueeze(1), Z.unsqueeze(1))
        return new_s

    def step(self, action):
        next_obs, gt_reward, term, trunc, info = self.env.step(action)
        done = term or trunc
        info["gt_reward"] = gt_reward
        info["step"] = self.cnt
        self.cnt += 1
        self.traj.append(gt_reward)

        if self.obs is not None:
            obs_t = torch.tensor(self.obs, dtype=torch.get_default_dtype())
        else:
            obs_t = torch.tensor(next_obs, dtype=torch.get_default_dtype())
        acs_t = torch.tensor(action, dtype=torch.get_default_dtype())
        next_obs_t = torch.tensor(next_obs, dtype=torch.get_default_dtype())
        done_t = torch.tensor(done, dtype=torch.get_default_dtype())

        z = self._feats(obs_t, acs_t, next_obs_t, done_t)
        Znew = z @ self.dirs.t()  # [K]
        pol_sorted_mat, counts = self._policy_sorted()
        K = self.K
        cap = self.cap
        device = pol_sorted_mat.device
        # masks for valid entries per row
        J = self.J.to(device).unsqueeze(0).expand(K, -1)
        mask_valid = J < counts.unsqueeze(1)
        if self.variant == "SR":
            # vectorized SR
            if self.exp_sorted.numel() == 0:
                rew = torch.tensor(0.0)
            else:
                # compute quantiles of Znew relative to policy buffer per row
                pos = torch.searchsorted(
                    pol_sorted_mat, Znew.unsqueeze(1), right=False
                ).squeeze(1)
                pos = torch.minimum(pos, counts)
                denom = torch.clamp(counts.to(Znew.dtype), min=1.0)
                q = (pos.to(Znew.dtype) + 0.5) / denom
                yhat = self._qmap_from_q(q)
                per = (Znew - yhat) ** 2
                # zero-out rows with no policy samples
                per = torch.where(counts > 0, per, torch.zeros_like(per))
                rew = -self._agg(per)
        elif self.variant == "DUAL":
            # vectorized DUAL
            if self.exp_sorted.numel() == 0:
                rew = torch.tensor(0.0)
            else:
                # per-row quantiles for sorted policy entries
                denom = torch.clamp(counts.to(pol_sorted_mat.dtype), min=1.0).unsqueeze(
                    1
                )
                q_rows = ((J.to(pol_sorted_mat.dtype) + 0.5) / denom).clamp(0.0, 1.0)
                # y_on_z for each row over its own support via LUT
                y_on_z = self._qmap_from_q(q_rows)
                g = (pol_sorted_mat - y_on_z) * mask_valid.to(pol_sorted_mat.dtype)
                # cumulative trapezoidal integration per row
                dz = pol_sorted_mat[:, 1:] - pol_sorted_mat[:, :-1]
                trap = 0.5 * (g[:, 1:] + g[:, :-1]) * dz
                phi = torch.zeros_like(pol_sorted_mat)
                phi[:, 1:] = torch.cumsum(trap, dim=1)
                # center per row
                denom = torch.clamp(counts.to(phi.dtype), min=1.0)
                phi = phi - (phi.sum(dim=1) / denom).unsqueeze(1)
                # evaluate phi at Znew via linear interpolation
                q = torch.searchsorted(
                    pol_sorted_mat, Znew.unsqueeze(1), right=False
                ).squeeze(1)
                hi = torch.maximum(counts - 1, torch.ones_like(counts))
                lo = torch.ones_like(q)
                q = torch.clamp(q, lo, hi)
                x0 = pol_sorted_mat.gather(1, (q - 1).unsqueeze(1)).squeeze(1)
                x1 = pol_sorted_mat.gather(1, q.unsqueeze(1)).squeeze(1)
                y0 = phi.gather(1, (q - 1).unsqueeze(1)).squeeze(1)
                y1 = phi.gather(1, q.unsqueeze(1)).squeeze(1)
                t = (Znew - x0) / (x1 - x0 + 1e-12)
                val = y0 * (1.0 - t) + y1 * t
                # zero rows with insufficient samples
                val = torch.where(counts > 1, val, torch.zeros_like(val))
                rew = -self._agg(val)
        else:
            # RPL
            if self.exp_sorted.numel() == 0:
                rew = torch.tensor(0.0)
            else:
                # baseline SW on current buffer (use LUT)
                denom = torch.clamp(counts.to(pol_sorted_mat.dtype), min=1.0).unsqueeze(
                    1
                )
                q_rows = ((J.to(pol_sorted_mat.dtype) + 0.5) / denom).clamp(0.0, 1.0)
                y_on_z = self._qmap_from_q(q_rows)
                diff = (pol_sorted_mat - y_on_z) ** 2
                diff = diff * mask_valid.to(diff.dtype)
                denom_b = torch.clamp(counts.to(diff.dtype), min=1.0)
                vb = diff.sum(dim=1) / denom_b
                sw_b = self._agg(vb)

                # after replacement (efficient remove+insert without full re-sort)
                pol_repl_sorted = self._replace_effect_sorted(Znew)
                Nused = torch.clamp(counts, min=1)
                denom_a = torch.clamp(
                    Nused.to(pol_repl_sorted.dtype), min=1.0
                ).unsqueeze(1)
                q_rows_a = ((J.to(pol_repl_sorted.dtype) + 0.5) / denom_a).clamp(
                    0.0, 1.0
                )
                y_on_z_a = self._qmap_from_q(q_rows_a)
                diff_a = (pol_repl_sorted - y_on_z_a) ** 2
                mask_a = J < Nused.unsqueeze(1)
                diff_a = diff_a * mask_a.to(diff_a.dtype)
                va = diff_a.sum(dim=1) / Nused.to(diff_a.dtype)
                sw_a = self._agg(va)
                rew = sw_b - sw_a

        self._insert(z)
        self.rpl_cursor = (self.rpl_cursor + 1) % self.cap
        self.obs = next_obs
        self.traj_.append(float(rew))
        if done:
            print("episode reward gt:", np.sum(np.array(self.traj)))
            print("episode reward irl:", np.sum(np.array(self.traj_)))
            info["episode"]["ep_rew_irl"] = np.sum(np.array(self.traj_))
            self.traj = []
            self.traj_ = []

        return next_obs, float(rew.item()), term, trunc, info


class GSW:
    def __init__(
        self, ftype="poly", nofprojections=10, degree=2, radius=2.0, use_cuda=False
    ):
        self.ftype = ftype
        self.nofprojections = nofprojections
        self.degree = degree
        self.radius = radius
        if torch.cuda.is_available() and use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.theta = None  # Thi is for max-GSW
        self.Y = None

    def update_theta(self):
        self.theta = self.random_slice(self.theta_shape)

    def gsw(self, X, Y, new_theta=False):
        """
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        """
        # print(X.shape)
        # N,dn = X.shape
        # M,dm = Y.shape
        # assert dn==dm and M==N
        self.Y = Y
        self.theta_shape = X.shape[-1]
        if self.ftype == "index":
            self.theta = torch.randint(0, X.shape[-1], (self.nofprojections,))
        else:
            if self.theta is None or new_theta:
                self.theta = self.random_slice(X.shape[-1])

        Xslices = self.get_slice(X, self.theta)
        Yslices = self.get_slice(Y, self.theta)

        self.Xslices_sorted = torch.sort(Xslices, dim=0)[0]
        self.Yslices_sorted = torch.sort(Yslices, dim=0)[0]

        return torch.sqrt(torch.sum((self.Xslices_sorted - self.Yslices_sorted) ** 2))

    def max_gsw(self, X, Y, iterations=50, lr=1e-4):
        # N, dn = X.shape
        # M, dm = Y.shape
        dn = X.shape[-1]
        device = self.device
        # assert dn==dm and M==N
        if self.ftype == "linear":
            theta = torch.randn((1, dn), device=device, requires_grad=True)
            theta.data /= torch.sqrt(torch.sum((theta.data) ** 2))
        elif self.ftype == "poly":
            dpoly = self.homopoly(dn, self.degree)
            theta = torch.randn((1, dpoly), device=device, requires_grad=True)
            theta.data /= torch.sqrt(torch.sum((theta.data) ** 2))
        elif self.ftype == "circular":
            theta = torch.randn((1, dn), device=device, requires_grad=True)
            theta.data /= torch.sqrt(torch.sum((theta.data) ** 2))
            theta.data *= self.radius
        self.theta = theta

        optimizer = torch.optim.Adam([self.theta], lr=lr)
        total_loss = np.zeros((iterations,))
        for i in range(iterations):
            optimizer.zero_grad()
            loss = -self.gsw(X.to(self.device), Y.to(self.device))
            total_loss[i] = loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            self.theta.data /= torch.sqrt(torch.sum(self.theta.data**2))

        return self.gsw(X.to(self.device), Y.to(self.device))

    def gsl2(self, X, Y, theta=None):
        """
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        """
        N, dn = X.shape
        M, dm = Y.shape
        assert dn == dm and M == N
        if theta is None:
            theta = self.random_slice(dn)

        Xslices = self.get_slice(X, theta)
        Yslices = self.get_slice(Y, theta)

        Yslices_sorted = torch.sort(Yslices, dim=0)

        return torch.sqrt(torch.sum((Xslices - Yslices) ** 2))

    def get_slice(self, X, theta):
        """Slices samples from distribution X~P_X
        Inputs:
            X:  Nxd matrix of N data samples
            theta: parameters of g (e.g., a d vector in the linear case)
        """
        if theta is None:
            theta = self.random_slice(X.shape[-1])
            self.theta = theta

        if self.ftype == "linear":
            return self.linear(X, theta)
        elif self.ftype == "poly":
            return self.poly(X, theta)
        elif self.ftype == "circular":
            return self.circular(X, theta)
        elif self.ftype == "index":
            return self.random_data_index(X, theta)
        else:
            raise Exception("Defining function not implemented")

    def random_slice(self, dim):
        if self.ftype == "linear":
            theta = torch.randn((self.nofprojections, dim))
            theta = torch.stack([th / torch.sqrt((th**2).sum()) for th in theta])
        elif self.ftype == "poly":
            dpoly = self.homopoly(dim, self.degree)
            theta = torch.randn((self.nofprojections, dpoly))
            theta = torch.stack([th / torch.sqrt((th**2).sum()) for th in theta])
        elif self.ftype == "circular":
            theta = torch.randn((self.nofprojections, dim))
            theta = torch.stack(
                [self.radius * th / torch.sqrt((th**2).sum()) for th in theta]
            )
        return theta.to(self.device)

    def random_data_index(self, X, theta):
        return X[..., theta]

    def linear(self, X, theta):
        if len(theta.shape) == 1:
            return torch.matmul(X, theta)
        else:
            return torch.matmul(X, theta.t())

    def poly(self, X, theta):
        """The polynomial defining function for generalized Radon transform
        Inputs
        X:  Nxd matrix of N data samples
        theta: Lxd vector that parameterizes for L projections
        degree: degree of the polynomial
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)
        N, d = X.shape
        assert theta.shape[1] == self.homopoly(d, self.degree)
        # vectorize polynomial feature map using exponent matrix
        powers = torch.tensor(
            list(self.get_powers(d, self.degree)), device=X.device, dtype=X.dtype
        )  # [P, d]
        HX = torch.ones((N, powers.shape[0]), device=X.device, dtype=X.dtype)
        # multiply across dimensions with broadcasted powers
        for i in range(d):
            HX = HX * (X[:, i : i + 1] ** powers[:, i].unsqueeze(0))
        if len(theta.shape) == 1:
            return torch.matmul(HX, theta)
        else:
            return torch.matmul(HX, theta.t())

    def circular(self, X, theta):
        """The circular defining function for generalized Radon transform
        Inputs
        X:  Nxd matrix of N data samples
        theta: Lxd vector that parameterizes for L projections
        """
        N, d = X.shape
        if len(theta.shape) == 1:
            return torch.sqrt(torch.sum((X - theta) ** 2, dim=1))
        else:
            return torch.stack(
                [torch.sqrt(torch.sum((X - th) ** 2, dim=1)) for th in theta], 1
            )

    def get_powers(self, dim, degree):
        """
        This function calculates the powers of a homogeneous polynomial
        e.g.

        list(get_powers(dim=2,degree=3))
        [(0, 3), (1, 2), (2, 1), (3, 0)]

        list(get_powers(dim=3,degree=2))
        [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
        """
        if dim == 1:
            yield (degree,)
        else:
            for value in range(degree + 1):
                for permutation in self.get_powers(dim - 1, degree - value):
                    yield (value,) + permutation

    def homopoly(self, dim, degree):
        """
        calculates the number of elements in a homogeneous polynomial
        """
        return len(list(self.get_powers(dim, degree)))
