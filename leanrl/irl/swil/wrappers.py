from __future__ import annotations

from argparse import Namespace
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwilReward(gym.Wrapper):
    def __init__(self, env: gym.Env, opt: Namespace, disc: nn.Module) -> None:
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

    def reset(
        self, seed: int = 0, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = super().reset(seed=self.opt.seed)
        self.cnt = 0
        return obs, info

    def step(
        self, action: np.ndarray | torch.Tensor | float
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
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
            # get discriminator reward on device to avoid CPU<->GPU transfers
            irl_reward = self.discriminator.get_reward(
                obs_t, acs_t, next_obs_t, done_t, self.cnt
            ).detach()

        self.obs = next_obs
        self.traj_.append(irl_reward.item())
        # print("GT: ", gt_reward, "IRL: ", irl_reward)
        if done:
            info["episode"]["ep_rew_irl"] = np.sum(np.array(self.traj_))
            self.traj = []
            self.traj_ = []

        return next_obs, float(irl_reward.item()), term, trunc, info


class SwilFlowReward(gym.Wrapper):
    def __init__(self, env: gym.Env, opt: Namespace):
        super().__init__(env=env)
        # self.reward_fn = make_network(**reward_fn_spec)
        self.use_actions = opt.use_actions
        self.gradient_model: Any = FlowModel(env, opt)
        self.obs = None
        self.sw_poly_deg = opt.sw_poly_deg
        self.radon_df_type = opt.radon_df_type
        # calculate reward at the end of trajecotiry
        self.traj = []

    def compute_sw_grad(
        self, policy_atoms: torch.Tensor, expert_atoms: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def update_gradient_estimator(
        self, policy_atoms: torch.Tensor, expert_atoms: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sw_dist, policy_atoms_grad, Xsorted, Ysorted = self.compute_sw_grad(
            policy_atoms, expert_atoms
        )
        estimated_gradients = self.gradient_model.forward(policy_atoms)
        loss = F.mse_loss(estimated_gradients, policy_atoms_grad)
        self.gradient_model.update(loss)

        return sw_dist, Xsorted, Ysorted

    def step(
        self, action: np.ndarray | torch.Tensor | float
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        next_obs, gt_reward, done, info = self.env.step(action)
        info["gt_reward"] = gt_reward
        device = next(self.gradient_model.parameters()).device
        dtype = torch.get_default_dtype()
        if self.obs is not None:
            obs_t = torch.tensor(self.obs, dtype=dtype, device=device)
        else:
            obs_t = torch.tensor(next_obs, dtype=dtype, device=device)

        acs_t = torch.tensor(action, dtype=dtype, device=device)
        next_obs_t = torch.tensor(next_obs, dtype=dtype, device=device)
        done_t = torch.tensor(done, dtype=dtype, device=device)

        if done:
            self.traj = []

        with torch.no_grad():
            # get discriminator reward and train on that
            # print([p.norm(2) for p in self.discriminator.parameters()])
            d_obs = next_obs_t - obs_t
            # sw_grad_obs =
            irl_reward = self.gradient_model.get_reward(obs_t, acs_t, next_obs_t, done_t)
            # irl_reward = self.discriminator.get_reward(obs_t, acs_t, next_obs_t).cpu().numpy()
        self.obs = next_obs

        return next_obs, float(irl_reward.item()), done, info


"""
SWIL reward class based on difference of increasing arrangement maps
"""


class SwilIRDiffReward(gym.Wrapper):
    def __init__(self, env: gym.Env, opt: Namespace):
        super().__init__(env=env)
        # self.reward_fn = make_network(**reward_fn_spec)
        self.use_actions = opt.use_actions
        self.obs = None
        self.opt = opt
        self.sw_poly_deg = opt.sw_poly_deg
        self.radon_df_type = opt.radon_df_type
        self.n_proj = opt.n_proj
        # TODO: makes GSW module shared across vec envs?
        self.gsw = GSW(
            ftype=self.radon_df_type,
            nofprojections=self.n_proj,
            degree=self.sw_poly_deg,
        )
        # calculate reward at the end of trajecotiry
        self.traj = []

    def compute_gsw(
        self, policy_atoms: torch.Tensor, expert_atoms: torch.Tensor
    ) -> torch.Tensor:
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

    def step(
        self, action: np.ndarray | torch.Tensor | float
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        next_obs, gt_reward, done, info = self.env.step(action)
        info["gt_reward"] = gt_reward
        device = getattr(self.gsw, "device", torch.device("cpu"))
        dtype = torch.get_default_dtype()
        if self.obs is not None:
            obs_t = torch.as_tensor(self.obs, dtype=dtype, device=device)
        else:
            obs_t = torch.as_tensor(next_obs, dtype=dtype, device=device)
        acs_t = torch.as_tensor(action, dtype=dtype, device=device)
        next_obs_t = torch.as_tensor(next_obs, dtype=dtype, device=device)
        done_t = torch.as_tensor(done, dtype=dtype, device=device)

        if done:
            self.traj = []

        rew = torch.tensor([0.0], device=device, dtype=dtype)
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

        return next_obs, float(rew.item()), done, info


class SwilRewardNew(gym.Wrapper):
    """Buffer-based SWIL rewards with three variants: SR, DUAL, RPL.
    Uses linear random projections over concatenated features of (obs, action, next_obs, done)
    controlled by opt.use_actions/use_next_obs/use_dones.
    """

    def __init__(
        self, env: gym.Env, opt: Namespace, demos: Dict[str, np.ndarray]
    ) -> None:
        super().__init__(env=env)
        self.opt = opt
        self.use_actions = opt.use_actions
        self.use_next_obs = opt.use_next_obs
        self.use_dones = opt.use_dones
        self.variant = getattr(opt, "swil_variant", "DUAL").upper()
        self.agg = getattr(opt, "swil_agg", "mean").lower()
        self.tau = float(getattr(opt, "swil_tau", 0.5))
        self.feature_norm = bool(getattr(opt, "swil_feature_norm", True))
        self.feature_sigma_min = float(getattr(opt, "swil_feature_sigma_min", 1e-2))
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
        self.dirs = self._make_directions(self.K, dim0, int(getattr(self.opt, "seed", 0)))
        # expert features/projections
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
        self._fit_feature_stats(Xexp)
        Xexp = self._normalize_features(Xexp)
        Zexp = Xexp @ self.dirs.t()  # [N,K]
        Zexp = Zexp.transpose(0, 1).contiguous()  # [K,N]
        # store expert projections as a single sorted matrix [K, N]
        self.exp_sorted = torch.sort(Zexp, dim=1, stable=True).values
        # optional quantile LUT over expert distribution per projection
        self.use_lut = bool(getattr(self.opt, "swil_use_lut", True))
        self.qgrid = int(getattr(self.opt, "swil_qgrid", 1025))
        if self.use_lut:
            self._build_exp_lut()
        # Current-policy occupancy buffer. This is intentionally separate from
        # SAC replay and uses FIFO eviction by sample age, not by projected value.
        self.cap = int(getattr(self.opt, "swil_occ_buffer_size", 32768))
        self.pol_sorted = torch.full(
            (self.K, self.cap), float("inf"), dtype=torch.get_default_dtype()
        )
        self.counts = torch.zeros(self.K, dtype=torch.long)
        self.occ_proj = torch.empty(
            (self.cap, self.K), dtype=torch.get_default_dtype()
        )
        self.occ_count = 0
        self.occ_cursor = 0
        self.J = torch.arange(self.cap, dtype=torch.long)
        self.rpl_cursor = 0
        self.last_slice_weights = torch.full(
            (self.K,), 1.0 / max(self.K, 1), dtype=torch.get_default_dtype()
        )
        self.last_k_eff = float(self.K)

    @staticmethod
    def _make_directions(K: int, dim: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        blocks = []
        remaining = K
        while remaining > 0:
            block = min(dim, remaining)
            mat = torch.randn(dim, block, generator=generator, dtype=torch.get_default_dtype())
            q, _ = torch.linalg.qr(mat, mode="reduced")
            blocks.append(q[:, :block].transpose(0, 1).contiguous())
            remaining -= block
        dirs = torch.cat(blocks, dim=0)
        return torch.nn.functional.normalize(dirs, dim=-1)

    def _fit_feature_stats(self, x: torch.Tensor) -> None:
        if not self.feature_norm:
            self.feature_mean = torch.zeros(
                x.shape[-1], dtype=x.dtype, device=x.device
            )
            self.feature_scale = torch.ones(
                x.shape[-1], dtype=x.dtype, device=x.device
            )
            return
        self.feature_mean = x.mean(dim=0)
        scale = x.std(dim=0, unbiased=False)
        self.feature_scale = torch.clamp(scale, min=self.feature_sigma_min)

    def _normalize_features(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feature_mean.to(x.device)) / self.feature_scale.to(x.device)

    def reset(
        self, seed: int = 0, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = super().reset(seed=self.opt.seed)
        self.cnt = 0
        return obs, info

    def _feats(
        self,
        ob: torch.Tensor,
        ac: torch.Tensor,
        nob: torch.Tensor,
        d: torch.Tensor,
    ) -> torch.Tensor:
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

    def _policy_sorted(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return current sorted policy projections and counts (no re-sorting)."""
        return self.pol_sorted, self.counts

    def _qmap(
        self, z_sorted: torch.Tensor, y_sorted: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
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

    def _qmap_vec(
        self,
        z_sorted: torch.Tensor,
        counts: torch.Tensor,
        y_sorted: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
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

    def _build_exp_lut(self) -> None:
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

    def _qmap_from_q(self, q: torch.Tensor) -> torch.Tensor:
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

    def _agg(
        self, values: torch.Tensor, scores: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Aggregate per-projection values.

        For smooth/max sliced objectives, weights are functions of per-slice
        distances, not necessarily of the values being combined.
        """
        if self.agg == "mean":
            return values.mean()
        scores = values if scores is None else scores
        if self.agg == "max":
            return values[torch.argmax(scores)]
        x = scores / max(self.tau, 1e-6)
        x = x - x.max()
        w = torch.softmax(x, dim=0)
        self.last_slice_weights = w.detach()
        entropy = -(w * torch.log(torch.clamp(w, min=1e-12))).sum()
        self.last_k_eff = float(torch.exp(entropy).detach().cpu())
        return (w * values).sum()

    def _insert_sorted_projection(self, Z: torch.Tensor) -> None:
        """Insert projected sample Z into per-projection sorted buffers."""
        s = self.pol_sorted
        c = self.counts
        K, C = s.shape
        device = s.device
        # find insertion positions w.r.t. current sorted rows (trailing +inf)
        pos = torch.searchsorted(s, Z.unsqueeze(1), right=False).squeeze(1)
        pos = torch.minimum(pos, c)
        # when full, searchsorted can return cap (out of bounds); clamp to last slot
        max_pos = torch.where(c >= self.cap, c - 1, c)
        pos = torch.minimum(pos, max_pos)
        idx = torch.arange(C, device=device).unsqueeze(0).expand(K, -1)
        # shift right region [pos, c) by one
        mask_shift = (idx > pos.unsqueeze(1)) & (idx <= c.unsqueeze(1))
        src_shift = s.gather(1, torch.clamp(idx - 1, min=0))
        new_s = torch.where(mask_shift, src_shift, s)
        # place Z at pos
        new_s.scatter_(1, pos.unsqueeze(1), Z.unsqueeze(1))
        self.pol_sorted = new_s
        self.counts = torch.minimum(c + 1, torch.full_like(c, self.cap))

    def _remove_sorted_projection(self, Z: torch.Tensor) -> None:
        """Remove one projected sample from per-projection sorted buffers."""
        s = self.pol_sorted
        c = self.counts
        K, C = s.shape
        device = s.device
        if torch.any(c <= 0):
            return
        pos = torch.searchsorted(s, Z.unsqueeze(1), right=False).squeeze(1)
        pos = torch.minimum(pos, torch.clamp(c - 1, min=0))
        idx = torch.arange(C, device=device).unsqueeze(0).expand(K, -1)
        cm1 = torch.clamp(c - 1, min=0)
        mask_leftshift = (idx >= pos.unsqueeze(1)) & (idx < cm1.unsqueeze(1))
        src_left = s.gather(1, torch.clamp(idx + 1, max=C - 1))
        new_s = torch.where(mask_leftshift, src_left, s)
        inf_col = torch.full((K, 1), float("inf"), dtype=s.dtype, device=device)
        new_s.scatter_(1, cm1.unsqueeze(1), inf_col)
        self.pol_sorted = new_s
        self.counts = cm1

    def _insert(self, z: torch.Tensor) -> None:
        """Insert a new normalized feature sample into the current-policy FIFO buffer."""
        Z = z @ self.dirs.t()  # [K]
        if self.occ_count >= self.cap:
            self._remove_sorted_projection(self.occ_proj[self.occ_cursor])
        else:
            self.occ_count += 1
        self.occ_proj[self.occ_cursor] = Z
        self.occ_cursor = (self.occ_cursor + 1) % self.cap
        self._insert_sorted_projection(Z)

    def _replace_effect_sorted(self, Z: torch.Tensor) -> torch.Tensor:
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

    def step(
        self, action: np.ndarray | torch.Tensor | float
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        next_obs, gt_reward, term, trunc, info = self.env.step(action)
        done = term or trunc
        info["gt_reward"] = gt_reward
        info["step"] = self.cnt
        self.cnt += 1
        self.traj.append(gt_reward)

        device = self.exp_sorted.device
        dtype = self.exp_sorted.dtype
        if self.obs is not None:
            obs_t = torch.as_tensor(self.obs, dtype=dtype, device=device)
        else:
            obs_t = torch.as_tensor(next_obs, dtype=dtype, device=device)
        acs_t = torch.as_tensor(action, dtype=dtype, device=device)
        next_obs_t = torch.as_tensor(next_obs, dtype=dtype, device=device)
        done_t = torch.as_tensor(done, dtype=dtype, device=device)

        z = self._normalize_features(self._feats(obs_t, acs_t, next_obs_t, done_t))
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
                g = torch.where(
                    mask_valid,
                    pol_sorted_mat - y_on_z,
                    torch.zeros_like(pol_sorted_mat),
                )
                per_slice_dist = g.pow(2).sum(dim=1) / torch.clamp(
                    counts.to(g.dtype), min=1.0
                )
                per_slice_dist = torch.where(
                    counts > 1, per_slice_dist, torch.zeros_like(per_slice_dist)
                )
                # cumulative trapezoidal integration per row
                pair_valid = mask_valid[:, 1:] & mask_valid[:, :-1]
                dz = torch.where(
                    pair_valid,
                    pol_sorted_mat[:, 1:] - pol_sorted_mat[:, :-1],
                    torch.zeros_like(pol_sorted_mat[:, 1:]),
                )
                trap = 0.5 * (g[:, 1:] + g[:, :-1]) * dz
                phi = torch.zeros_like(pol_sorted_mat)
                phi[:, 1:] = torch.cumsum(trap, dim=1)
                phi = torch.where(mask_valid, phi, torch.zeros_like(phi))
                # center per row
                denom = torch.clamp(counts.to(phi.dtype), min=1.0)
                phi_mean = (phi.sum(dim=1) / denom).unsqueeze(1)
                phi = torch.where(mask_valid, phi - phi_mean, torch.zeros_like(phi))
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
                rew = -self._agg(val, per_slice_dist)
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

        info["swil_k_eff"] = self.last_k_eff
        info["swil_occ_count"] = self.occ_count
        return next_obs, float(rew.item()), term, trunc, info
