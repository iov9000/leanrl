from __future__ import annotations

import copy
import os
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

SUPPORTED_PROJECTION_TYPES = {
    "linear_random",
    "poly_random",
    "circular_random",
    "nn_random",
    "mixed_linear_nn",
}

SUPPORTED_REWARD_MODES = {"dual", "dual_raw", "dual_centered", "rpl"}
SUPPORTED_CENTER_MODES = {"none", "batch", "per_bank"}


def _first_present(data: Any, keys: list[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _as_feature_array(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D after reshape, got shape {arr.shape}")
    return arr.astype(np.float32)


def load_expert_dataset(expert_path: str) -> dict[str, np.ndarray]:
    if not os.path.isfile(expert_path):
        raise FileNotFoundError(f"expert_path not found: {expert_path}")

    ext = os.path.splitext(expert_path)[1].lower()
    if ext == ".npz":
        data = np.load(expert_path, allow_pickle=True)
    elif ext == ".pkl":
        with open(expert_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "all" in data and isinstance(data["all"], dict):
            nested = data["all"]
            if _first_present(data, ["obs", "observations"]) is None:
                data = nested
    else:
        raise ValueError(f"Unsupported expert dataset extension {ext}; use .npz or .pkl")

    obs_raw = _first_present(data, ["observations", "obs"])
    act_raw = _first_present(data, ["actions", "acs", "act"])
    if obs_raw is None or act_raw is None:
        raise ValueError("Expert dataset must contain observations/obs and actions/acs")

    observations = _as_feature_array(obs_raw, "observations")
    actions = _as_feature_array(act_raw, "actions")
    if observations.shape[0] != actions.shape[0]:
        raise ValueError(
            f"Expert observations/actions length mismatch: {observations.shape[0]} vs {actions.shape[0]}"
        )

    out = {"observations": observations, "actions": actions}
    rewards = _first_present(data, ["rewards", "rew"])
    terminals = _first_present(data, ["terminals", "dones", "done"])
    if rewards is not None:
        out["rewards"] = np.asarray(rewards, dtype=np.float32).reshape(-1)
    if terminals is not None:
        out["terminals"] = (np.asarray(terminals).reshape(-1) > 0.5).astype(np.bool_)
    return out


def state_action_samples(observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError(
            f"observations and actions must be 2D, got {tuple(observations.shape)} and {tuple(actions.shape)}"
        )
    if observations.shape[0] != actions.shape[0]:
        raise ValueError("observations and actions must have matching batch sizes")
    return torch.cat([observations, actions], dim=1)


@dataclass
class StateActionNormalizer:
    enabled: bool
    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-6

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return (x - self.mean) / (self.std + self.eps)


def fit_state_action_normalizer(
    expert_sa: torch.Tensor,
    learner_sa: torch.Tensor | None = None,
    *,
    normalize_sa: bool = True,
    norm_source: str = "expert_only",
    eps: float = 1e-6,
) -> StateActionNormalizer:
    if norm_source not in {"expert_only", "expert_and_initial_policy"}:
        raise ValueError(f"Unsupported norm_source={norm_source}")
    if not normalize_sa:
        zeros = torch.zeros(expert_sa.shape[1], device=expert_sa.device, dtype=expert_sa.dtype)
        ones = torch.ones_like(zeros)
        return StateActionNormalizer(False, zeros, ones, eps=eps)
    if norm_source == "expert_and_initial_policy":
        if learner_sa is None:
            raise ValueError("learner_sa is required when norm_source='expert_and_initial_policy'")
        samples = torch.cat([expert_sa, learner_sa.to(expert_sa.device)], dim=0)
    else:
        samples = expert_sa
    return StateActionNormalizer(
        True,
        samples.mean(dim=0),
        samples.std(dim=0, unbiased=False).clamp_min(eps),
        eps=eps,
    )


def sample_unit_projections(
    input_dim: int,
    num_projections: int,
    *,
    seed: int = 0,
    projection_type: str = "linear_random",
    projection_degree: int = 2,
    projection_radius: float = 2.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if projection_type not in {"linear_random", "poly_random", "circular_random"}:
        raise ValueError(
            "Unsupported projection_type for sample_unit_projections; "
            "expected one of ['circular_random', 'linear_random', 'poly_random'], "
            f"got {projection_type}"
        )
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if num_projections <= 0:
        raise ValueError(f"num_projections must be positive, got {num_projections}")
    if projection_degree <= 0:
        raise ValueError(f"projection_degree must be positive, got {projection_degree}")
    if projection_radius <= 0:
        raise ValueError(f"projection_radius must be positive, got {projection_radius}")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    if projection_type == "linear_random":
        theta_dim = input_dim
        theta = torch.randn(num_projections, theta_dim, generator=gen, dtype=dtype)
        theta = theta / theta.norm(dim=1, keepdim=True).clamp_min(1e-12)
    elif projection_type == "poly_random":
        theta_dim = len(_homogeneous_powers(input_dim, projection_degree))
        theta = torch.randn(num_projections, theta_dim, generator=gen, dtype=dtype)
        theta = theta / theta.norm(dim=1, keepdim=True).clamp_min(1e-12)
    else:
        theta = torch.randn(num_projections, input_dim, generator=gen, dtype=dtype)
        theta = projection_radius * theta / theta.norm(dim=1, keepdim=True).clamp_min(1e-12)

    return theta.to(device=device)


class ErfActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.erf(x)


class FixedSliceMLP(nn.Module):
    """Frozen two-hidden-layer slice network matching the SAC MLP layout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "relu",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.eps = eps
        fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        fc3 = nn.Linear(self.hidden_dim, self.output_dim)
        self.net = nn.Sequential(
            fc1,
            self._activation_module(),
            fc2,
            self._activation_module(),
            fc3,
        )
        self.register_buffer("cal_mean", torch.zeros(self.output_dim))
        self.register_buffer("cal_std", torch.ones(self.output_dim))
        self.register_buffer("cal_inv_std", torch.ones(self.output_dim))
        self._calibrated = False
        self.requires_grad_(False)

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def fc1(self) -> nn.Linear:
        return self.net[0]

    @property
    def fc2(self) -> nn.Linear:
        return self.net[2]

    @property
    def fc3(self) -> nn.Linear:
        return self.net[4]

    def _activation_module(self) -> nn.Module:
        if self.activation == "relu":
            return nn.ReLU()
        if self.activation == "silu":
            return nn.SiLU()
        if self.activation == "tanh":
            return nn.Tanh()
        if self.activation == "erf":
            return ErfActivation()
        raise ValueError(f"Unsupported nn_activation={self.activation}")

    def raw_project(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.raw_project(x)
        return (raw - self.cal_mean) * self.cal_inv_std

    @torch.no_grad()
    def calibrate(self, x: torch.Tensor) -> None:
        raw = self.raw_project(x)
        self.cal_mean.copy_(raw.mean(dim=0))
        self.cal_std.copy_(raw.std(dim=0, unbiased=False).clamp_min(self.eps))
        self.cal_inv_std.copy_(self.cal_std.add(self.eps).reciprocal())
        self._calibrated = True


@dataclass
class SliceProjector:
    projection_type: str
    input_dim: int
    num_projections: int
    directions: torch.Tensor | None = None
    projection_degree: int = 2
    nn_projector: FixedSliceMLP | None = None
    nn_activation: str = "silu"
    nn_linear_count: int = 0
    eps: float = 1e-6

    @property
    def needs_calibration(self) -> bool:
        return self.projection_type in {"nn_random", "mixed_linear_nn"} and (
            self.nn_projector is None or not self.nn_projector.calibrated
        )

    def _nn_project(self, x: torch.Tensor) -> torch.Tensor:
        if self.nn_projector is None:
            raise RuntimeError("NN slice projector is missing frozen MLP")
        return self.nn_projector(x)

    @property
    def nn_w1(self) -> torch.Tensor | None:
        return None if self.nn_projector is None else self.nn_projector.fc1.weight

    @property
    def nn_b1(self) -> torch.Tensor | None:
        return None if self.nn_projector is None else self.nn_projector.fc1.bias

    @property
    def nn_readouts(self) -> torch.Tensor | None:
        return None if self.nn_projector is None else self.nn_projector.fc3.weight

    def project(self, x: torch.Tensor) -> torch.Tensor:
        if self.projection_type in {"linear_random", "poly_random", "circular_random"}:
            if self.directions is None:
                raise RuntimeError("Slice projector is missing directions")
            return _project_slices(
                x,
                self.directions,
                projection_type=self.projection_type,
                projection_degree=self.projection_degree,
            )
        if self.projection_type == "nn_random":
            return self._nn_project(x)
        if self.projection_type == "mixed_linear_nn":
            if self.directions is None:
                raise RuntimeError("Mixed slice projector is missing linear directions")
            linear = _project_slices(
                x,
                self.directions,
                projection_type="linear_random",
                projection_degree=self.projection_degree,
            )
            return torch.cat([linear, self._nn_project(x)], dim=1)
        raise ValueError(f"Unsupported projection_type={self.projection_type}")

    @torch.no_grad()
    def with_calibration(self, calibration_x: torch.Tensor) -> "SliceProjector":
        if self.projection_type not in {"nn_random", "mixed_linear_nn"}:
            return self
        nn_projector = None
        if self.nn_projector is not None:
            nn_projector = copy.deepcopy(self.nn_projector)
            nn_projector.calibrate(calibration_x)
        return SliceProjector(
            projection_type=self.projection_type,
            input_dim=self.input_dim,
            num_projections=self.num_projections,
            directions=self.directions,
            projection_degree=self.projection_degree,
            nn_projector=nn_projector,
            nn_activation=self.nn_activation,
            nn_linear_count=self.nn_linear_count,
            eps=self.eps,
        )


class LearnedSliceProjector(nn.Module):
    """Trainable projection g_ψ for MGSW with Danskin-valid subgradient rewards.

    Optimizing ψ to maximize projected W₂² gives the Danskin inner problem;
    the KP potential at the resulting g* is a valid subgradient of the MGSW distance.
    """

    def __init__(
        self,
        input_dim: int,
        num_projections: int,
        hidden_dim: int = 256,
        activation: str = "silu",
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_projections = num_projections
        self._hidden_dim = hidden_dim

        trunk = nn.Linear(input_dim, hidden_dim)
        readout = nn.Linear(hidden_dim, num_projections)
        if use_spectral_norm:
            trunk = nn.utils.parametrizations.spectral_norm(trunk)
            readout = nn.utils.parametrizations.spectral_norm(readout)
        self.trunk = trunk
        self.readout = readout

        _acts = {"silu": nn.functional.silu, "relu": nn.functional.relu, "tanh": torch.tanh, "erf": torch.erf}
        self.act_fn = _acts.get(activation, nn.functional.silu)

        self.register_buffer("cal_mean", torch.zeros(num_projections))
        self.register_buffer("cal_std", torch.ones(num_projections))
        self.register_buffer("cal_inv_std", torch.ones(num_projections))
        self._calibrated = False
        self.eps = 1e-6

    @property
    def directions(self) -> None:
        return None

    @property
    def projection_type(self) -> str:
        return "nn_learned"

    @property
    def projection_degree(self) -> int:
        return 1

    def _raw_project(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act_fn(self.trunk(x))
        return self.readout(h / (self._hidden_dim**0.5))

    def project(self, x: torch.Tensor) -> torch.Tensor:
        raw = self._raw_project(x)
        return (raw - self.cal_mean) * self.cal_inv_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)

    @torch.no_grad()
    def reset_calibration(self) -> None:
        self.cal_mean.zero_()
        self.cal_std.fill_(1.0)
        self.cal_inv_std.fill_(1.0)
        self._calibrated = False

    @torch.no_grad()
    def calibrate(self, x: torch.Tensor) -> None:
        raw = self._raw_project(x)
        self.cal_mean.copy_(raw.mean(dim=0))
        self.cal_std.copy_(raw.std(dim=0, unbiased=False).clamp_min(self.eps))
        self.cal_inv_std.copy_(self.cal_std.add(self.eps).reciprocal())
        self._calibrated = True


@torch.enable_grad()
def maximize_projected_w2(
    projector: LearnedSliceProjector,
    learner_sa: torch.Tensor,
    expert_sa: torch.Tensor,
    n_steps: int = 50,
    lr: float = 1e-3,
    optimizer: torch.optim.Optimizer | None = None,
    capturable: bool = False,
) -> tuple[float, torch.optim.Optimizer]:
    """Solve the Danskin inner problem: maximize projected W₂² over projector ψ.

    When optimizer is provided, it is reused (incremental training).
    Returns (w2_value, optimizer) so the caller can persist the optimizer.
    """
    projector.reset_calibration()
    projector.train()
    if optimizer is None:
        optimizer = torch.optim.Adam(projector.parameters(), lr=lr, capturable=capturable)

    w2_val = torch.zeros((), device=learner_sa.device, dtype=learner_sa.dtype)
    for _ in range(n_steps):
        optimizer.zero_grad()
        z_pol = projector.project(learner_sa)
        z_exp = projector.project(expert_sa)

        z_pol_sorted = torch.sort(z_pol, dim=0).values
        z_exp_sorted = torch.sort(z_exp, dim=0).values

        n_pol, n_exp = z_pol_sorted.shape[0], z_exp_sorted.shape[0]
        if n_pol != n_exp:
            n = min(n_pol, n_exp)
            positions = torch.linspace(0, 1, n, device=z_pol.device)
            z_pol_t = z_pol_sorted.transpose(0, 1).contiguous()
            z_exp_t = z_exp_sorted.transpose(0, 1).contiguous()
            if n_pol > n:
                z_pol_t = _quantile_values(z_pol_t, positions)
            if n_exp > n:
                z_exp_t = _quantile_values(z_exp_t, positions)
            w2_sq = (z_pol_t - z_exp_t).pow(2).mean()
        else:
            w2_sq = (z_pol_sorted - z_exp_sorted).pow(2).mean()

        (-w2_sq).backward()
        optimizer.step()
        w2_val = w2_sq.detach()

    projector.eval()
    with torch.no_grad():
        projector.calibrate(torch.cat([learner_sa, expert_sa], dim=0))

    return float(w2_val.detach().cpu()), optimizer


def sample_slice_projector(
    input_dim: int,
    num_projections: int,
    *,
    seed: int = 0,
    projection_type: str = "linear_random",
    projection_degree: int = 2,
    projection_radius: float = 2.0,
    nn_feature_dim: int = 256,
    nn_activation: str = "silu",
    nn_linear_count: int = 0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> SliceProjector:
    if projection_type not in SUPPORTED_PROJECTION_TYPES:
        raise ValueError(
            f"Unsupported projection_type={projection_type}; expected one of {sorted(SUPPORTED_PROJECTION_TYPES)}"
        )
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if num_projections <= 0:
        raise ValueError(f"num_projections must be positive, got {num_projections}")
    if nn_feature_dim <= 0:
        raise ValueError(f"nn_feature_dim must be positive, got {nn_feature_dim}")

    if projection_type in {"linear_random", "poly_random", "circular_random"}:
        directions = sample_unit_projections(
            input_dim,
            num_projections,
            seed=seed,
            projection_type=projection_type,
            projection_degree=projection_degree,
            projection_radius=projection_radius,
            device=device,
            dtype=dtype,
        )
        return SliceProjector(
            projection_type=projection_type,
            input_dim=input_dim,
            num_projections=num_projections,
            directions=directions,
            projection_degree=projection_degree,
        )

    if projection_type == "mixed_linear_nn":
        if nn_linear_count <= 0:
            nn_linear_count = num_projections // 2
        if nn_linear_count >= num_projections:
            raise ValueError("--nn-slice-linear-count must be smaller than --num-projections")
    else:
        nn_linear_count = 0
    nn_count = num_projections - nn_linear_count
    directions = None
    if nn_linear_count > 0:
        directions = sample_unit_projections(
            input_dim,
            nn_linear_count,
            seed=seed + 10_003,
            projection_type="linear_random",
            device=device,
            dtype=dtype,
        )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        nn_projector = FixedSliceMLP(
            input_dim=input_dim,
            hidden_dim=nn_feature_dim,
            output_dim=nn_count,
            activation=nn_activation,
        )
    nn_projector = nn_projector.to(device=device, dtype=dtype)
    return SliceProjector(
        projection_type=projection_type,
        input_dim=input_dim,
        num_projections=num_projections,
        directions=directions,
        projection_degree=projection_degree,
        nn_projector=nn_projector,
        nn_activation=nn_activation,
        nn_linear_count=nn_linear_count,
    )


def _homogeneous_powers(input_dim: int, degree: int) -> list[tuple[int, ...]]:
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if degree < 0:
        raise ValueError(f"degree must be nonnegative, got {degree}")
    if input_dim == 1:
        return [(degree,)]
    powers: list[tuple[int, ...]] = []
    for value in range(degree + 1):
        for suffix in _homogeneous_powers(input_dim - 1, degree - value):
            powers.append((value,) + suffix)
    return powers


def _polynomial_features(x: torch.Tensor, degree: int) -> torch.Tensor:
    powers = torch.tensor(_homogeneous_powers(x.shape[1], degree), device=x.device, dtype=x.dtype)
    return x.unsqueeze(1).pow(powers.unsqueeze(0)).prod(dim=2)


def _project_slices(
    x: torch.Tensor,
    directions: torch.Tensor,
    *,
    projection_type: str,
    projection_degree: int,
) -> torch.Tensor:
    if projection_type == "linear_random":
        expected_dim = x.shape[1]
        if directions.shape[1] != expected_dim:
            raise ValueError(f"linear directions must have feature dim {expected_dim}, got {directions.shape[1]}")
        return x @ directions.transpose(0, 1)
    if projection_type == "poly_random":
        features = _polynomial_features(x, projection_degree)
        if directions.shape[1] != features.shape[1]:
            raise ValueError(f"poly directions must have feature dim {features.shape[1]}, got {directions.shape[1]}")
        return features @ directions.transpose(0, 1)
    if projection_type == "circular_random":
        expected_dim = x.shape[1]
        if directions.shape[1] != expected_dim:
            raise ValueError(f"circular centers must have feature dim {expected_dim}, got {directions.shape[1]}")
        return torch.linalg.norm(x.unsqueeze(1) - directions.unsqueeze(0), dim=2)
    raise ValueError(f"Unsupported projection_type={projection_type}; expected one of {sorted(SUPPORTED_PROJECTION_TYPES)}")


def _quantile_values(sorted_values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if sorted_values.ndim != 2:
        raise ValueError("sorted_values must have shape [M, N]")
    n = sorted_values.shape[1]
    if n <= 0:
        raise ValueError("Cannot interpolate an empty sorted tensor")
    if n == 1:
        return sorted_values[:, :1].expand(-1, positions.numel())
    idx = positions.clamp(0.0, 1.0) * float(n - 1)
    idx_lo = idx.floor().to(torch.long).clamp(0, n - 1)
    idx_hi = idx.ceil().to(torch.long).clamp(0, n - 1)
    t = (idx - idx_lo.to(idx.dtype)).view(1, -1)
    lo = sorted_values.gather(1, idx_lo.view(1, -1).expand(sorted_values.shape[0], -1))
    hi = sorted_values.gather(1, idx_hi.view(1, -1).expand(sorted_values.shape[0], -1))
    return lo * (1.0 - t) + hi * t


def _interp_rows(x_grid: torch.Tensor, y_grid: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
    if x_grid.shape != y_grid.shape:
        raise ValueError("x_grid and y_grid must have matching shapes")
    if x_grid.ndim != 2 or x_query.ndim != 2:
        raise ValueError("x_grid, y_grid, and x_query must be 2D")
    if x_query.shape[1] != x_grid.shape[0]:
        raise ValueError(f"x_query has {x_query.shape[1]} rows, expected {x_grid.shape[0]}")

    xq_t = x_query.transpose(0, 1).contiguous()
    idx_hi = torch.searchsorted(x_grid.contiguous(), xq_t, right=False)
    q = x_grid.shape[1]
    idx_hi = idx_hi.clamp(0, q - 1)
    idx_lo = (idx_hi - 1).clamp(0, q - 1)

    x_lo = x_grid.gather(1, idx_lo)
    x_hi = x_grid.gather(1, idx_hi)
    y_lo = y_grid.gather(1, idx_lo)
    y_hi = y_grid.gather(1, idx_hi)
    denom = x_hi - x_lo
    t = torch.where(denom.abs() > 1e-12, (xq_t - x_lo) / denom.clamp_min(1e-12), torch.zeros_like(xq_t))
    y = y_lo * (1.0 - t) + y_hi * t
    return y.transpose(0, 1).contiguous()


@dataclass
class PotentialBank:
    projector: SliceProjector | LearnedSliceProjector
    z_grid: torch.Tensor
    target_grid: torch.Tensor
    phi_grid: torch.Tensor
    projected_w2: torch.Tensor
    policy_reward_mean: torch.Tensor
    expert_reward_mean: torch.Tensor
    rpl_base_sq: torch.Tensor
    rpl_left_prefix: torch.Tensor
    rpl_right_prefix: torch.Tensor
    @property
    def directions(self) -> torch.Tensor | None:
        return self.projector.directions

    @property
    def projection_type(self) -> str:
        return self.projector.projection_type

    @property
    def projection_degree(self) -> int:
        return self.projector.projection_degree

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector.project(x)

    def potentials(self, x: torch.Tensor) -> torch.Tensor:
        z = self.project(x)
        return _interp_rows(self.z_grid, self.phi_grid, z)

    def raw_rewards(self, x: torch.Tensor) -> torch.Tensor:
        return -self.potentials(x).mean(dim=1, keepdim=True)

    def rpl_finite_difference_rewards(
        self,
        x: torch.Tensor,
        *,
        remove_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """N-scaled finite replacement decrease in projected W2 for a frozen bank."""
        z = self.project(x)
        if z.ndim != 2:
            raise ValueError(f"projected queries must be 2D, got {tuple(z.shape)}")
        if z.shape[1] != self.z_grid.shape[0]:
            raise ValueError(f"projected query width {z.shape[1]} does not match bank width {self.z_grid.shape[0]}")

        z_t = z.transpose(0, 1).contiguous()
        num_projections, num_support = self.z_grid.shape
        if remove_indices is None:
            idx_hi = torch.searchsorted(self.z_grid.contiguous(), z_t, right=False).clamp(0, num_support - 1)
            idx_lo = (idx_hi - 1).clamp(0, num_support - 1)
            z_lo = self.z_grid.gather(1, idx_lo)
            z_hi = self.z_grid.gather(1, idx_hi)
            use_hi = (z_hi - z_t).abs() < (z_t - z_lo).abs()
            remove_idx = torch.where(use_hi, idx_hi, idx_lo)
        else:
            remove_idx = remove_indices.to(device=z.device, dtype=torch.long)
            if remove_idx.ndim == 1:
                if remove_idx.shape[0] != z.shape[0]:
                    raise ValueError(f"remove_indices length {remove_idx.shape[0]} does not match batch {z.shape[0]}")
                remove_idx = remove_idx.unsqueeze(0).expand(num_projections, -1)
            elif remove_idx.shape != z_t.shape:
                raise ValueError(
                    f"remove_indices must have shape {(z.shape[0],)} or {tuple(z_t.shape)}, got {tuple(remove_idx.shape)}"
                )
            remove_idx = torch.remainder(remove_idx, num_support)

        insert_pos = torch.searchsorted(self.z_grid.contiguous(), z_t, right=False)
        insert_pos = insert_pos - (insert_pos > remove_idx).to(insert_pos.dtype)
        insert_pos = insert_pos.clamp(0, num_support - 1)

        replace_delta = (z_t - self.target_grid.gather(1, insert_pos)).pow(2) - self.rpl_base_sq.gather(
            1, insert_pos
        )
        left_shift = self.rpl_left_prefix.gather(1, insert_pos) - self.rpl_left_prefix.gather(1, remove_idx)
        right_shift = self.rpl_right_prefix.gather(1, remove_idx + 1) - self.rpl_right_prefix.gather(
            1, insert_pos + 1
        )
        shift_delta = torch.where(insert_pos >= remove_idx, left_shift, right_shift)
        delta = replace_delta + shift_delta
        return (-0.5 * delta.mean(dim=0, keepdim=True)).transpose(0, 1).contiguous()

    def fw_duality_gap(self, learner_x: torch.Tensor | None = None) -> torch.Tensor:
        if learner_x is None:
            learner_reward_mean = self.policy_reward_mean
        else:
            learner_reward_mean = self.raw_rewards(learner_x).mean()
        return self.expert_reward_mean - learner_reward_mean

    def rewards(
        self,
        x: torch.Tensor,
        *,
        reward_scale: float = 1.0,
        mode: str = "dual",
        center: bool = True,
        center_mode: str | None = None,
        std_norm: bool = True,
        eps: float = 1e-6,
        potential_shift: float = 0.0,
        rpl_remove_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mode not in SUPPORTED_REWARD_MODES:
            raise ValueError(f"Unsupported reward mode {mode}; expected one of {sorted(SUPPORTED_REWARD_MODES)}")
        explicit_center_mode = center_mode is not None
        if center_mode is None:
            center_mode = "batch" if center else "none"
        if center_mode not in SUPPORTED_CENTER_MODES:
            raise ValueError(f"Unsupported center_mode {center_mode}; expected one of {sorted(SUPPORTED_CENTER_MODES)}")

        if mode == "dual_centered":
            mode = "dual"
            center_mode = "per_bank"
        elif mode == "dual_raw":
            mode = "dual"
            center_mode = "none"
        elif mode == "rpl" and not explicit_center_mode:
            center_mode = "none"

        if mode == "rpl":
            reward = self.rpl_finite_difference_rewards(x, remove_indices=rpl_remove_indices)
        else:
            reward = self.raw_rewards(x)
        if potential_shift and center_mode == "none":
            reward = reward - potential_shift
        if center_mode == "batch":
            reward = reward - reward.mean()
        elif center_mode == "per_bank":
            reward = reward - self.policy_reward_mean
        if std_norm:
            reward = reward / (reward.std(unbiased=False) + eps)
        return reward_scale * reward


@torch.no_grad()
def build_potential_bank(
    learner_sa: torch.Tensor,
    expert_sa: torch.Tensor,
    projector: torch.Tensor | SliceProjector | LearnedSliceProjector,
    *,
    projection_type: str = "linear_random",
    projection_degree: int = 2,
) -> PotentialBank:
    if learner_sa.ndim != 2 or expert_sa.ndim != 2:
        raise ValueError("learner_sa and expert_sa must be 2D")
    if learner_sa.shape[1] != expert_sa.shape[1]:
        raise ValueError("learner_sa and expert_sa feature dimensions must match")
    if learner_sa.shape[0] < 2 or expert_sa.shape[0] < 2:
        raise ValueError("Need at least two learner and expert samples to build potentials")

    if isinstance(projector, torch.Tensor):
        projector = SliceProjector(
            projection_type=projection_type,
            input_dim=learner_sa.shape[1],
            num_projections=projector.shape[0],
            directions=projector,
            projection_degree=projection_degree,
        )

    policy_atoms = projector.project(learner_sa)
    expert_atoms = projector.project(expert_sa)
    policy_sorted = torch.sort(policy_atoms, dim=0).values.transpose(0, 1).contiguous()
    expert_sorted = torch.sort(expert_atoms, dim=0).values.transpose(0, 1).contiguous()

    n_policy = policy_sorted.shape[1]
    positions = torch.linspace(0.0, 1.0, steps=n_policy, device=policy_sorted.device, dtype=policy_sorted.dtype)
    if expert_sorted.shape[1] == n_policy:
        target_grid = expert_sorted
    else:
        target_grid = _quantile_values(expert_sorted, positions)

    grad = policy_sorted - target_grid
    dz = policy_sorted[:, 1:] - policy_sorted[:, :-1]
    trap = 0.5 * (grad[:, 1:] + grad[:, :-1]) * dz
    phi = torch.zeros_like(policy_sorted)
    phi[:, 1:] = torch.cumsum(trap, dim=1)

    projected_w2 = 0.5 * (policy_sorted - target_grid).pow(2).mean()
    raw_policy_reward = -phi.mean(dim=0, keepdim=True).transpose(0, 1)
    raw_expert_reward = -_interp_rows(policy_sorted, phi, expert_atoms).mean(dim=1, keepdim=True)
    rpl_base_sq = (policy_sorted - target_grid).pow(2)
    rpl_left_delta = torch.zeros_like(rpl_base_sq)
    rpl_right_delta = torch.zeros_like(rpl_base_sq)
    rpl_left_delta[:, :-1] = (policy_sorted[:, 1:] - target_grid[:, :-1]).pow(2) - rpl_base_sq[:, :-1]
    rpl_right_delta[:, 1:] = (policy_sorted[:, :-1] - target_grid[:, 1:]).pow(2) - rpl_base_sq[:, 1:]
    prefix_pad = torch.zeros(rpl_base_sq.shape[0], 1, device=policy_sorted.device, dtype=policy_sorted.dtype)
    return PotentialBank(
        projector=projector,
        z_grid=policy_sorted,
        target_grid=target_grid,
        phi_grid=phi,
        projected_w2=projected_w2,
        policy_reward_mean=raw_policy_reward.mean(),
        expert_reward_mean=raw_expert_reward.mean(),
        rpl_base_sq=rpl_base_sq,
        rpl_left_prefix=torch.cat([prefix_pad, torch.cumsum(rpl_left_delta, dim=1)], dim=1),
        rpl_right_prefix=torch.cat([prefix_pad, torch.cumsum(rpl_right_delta, dim=1)], dim=1),
    )


@torch.no_grad()
def normalized_swil_rewards(
    bank: PotentialBank,
    x: torch.Tensor,
    *,
    reward_scale: float = 1.0,
    mode: str = "dual",
    center: bool = True,
    center_mode: str | None = None,
    std_norm: bool = True,
    eps: float = 1e-6,
    potential_shift: float = 0.0,
    rpl_remove_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    return bank.rewards(
        x,
        reward_scale=reward_scale,
        mode=mode,
        center=center,
        center_mode=center_mode,
        std_norm=std_norm,
        eps=eps,
        potential_shift=potential_shift,
        rpl_remove_indices=rpl_remove_indices,
    )
