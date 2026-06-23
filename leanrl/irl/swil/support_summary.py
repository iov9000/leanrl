from __future__ import annotations

from dataclasses import dataclass

import torch


def _as_2d(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return values.unsqueeze(-1)
    if values.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D tensor, got shape {tuple(values.shape)}")
    return values


def _interp_rows(x_grid: torch.Tensor, y_grid: torch.Tensor, x_query: torch.Tensor) -> torch.Tensor:
    """Row-wise linear interpolation.

    Args:
        x_grid: [K, Q] ascending coordinates per row.
        y_grid: [K, Q] values per row.
        x_query: [B, K] query points.
    Returns:
        [B, K] interpolated values.
    """
    if x_grid.shape != y_grid.shape:
        raise ValueError("x_grid and y_grid must have the same shape")
    if x_grid.ndim != 2:
        raise ValueError("x_grid and y_grid must be 2D")
    if x_query.ndim != 2:
        raise ValueError("x_query must be 2D")
    if x_query.shape[1] != x_grid.shape[0]:
        raise ValueError(
            f"x_query second dim {x_query.shape[1]} must match row count {x_grid.shape[0]}"
        )

    xq_t = x_query.transpose(0, 1).contiguous()  # [K, B]
    idx_hi = torch.searchsorted(x_grid, xq_t, right=False)
    q = x_grid.shape[1]
    idx_hi = idx_hi.clamp(0, q - 1)
    idx_lo = (idx_hi - 1).clamp(0, q - 1)

    x_lo = x_grid.gather(1, idx_lo)
    x_hi = x_grid.gather(1, idx_hi)
    y_lo = y_grid.gather(1, idx_lo)
    y_hi = y_grid.gather(1, idx_hi)

    denom = (x_hi - x_lo).abs()
    same = denom < 1e-12
    t = torch.where(
        same,
        torch.zeros_like(xq_t),
        (xq_t - x_lo) / (x_hi - x_lo + 1e-12),
    )
    y = y_lo * (1.0 - t) + y_hi * t
    return y.transpose(0, 1).contiguous()


@dataclass
class QuantileSupportSummary:
    quantiles: torch.Tensor
    q_grid: torch.Tensor
    sample_count: int
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    stat_eps: float = 1e-6

    @property
    def n_slices(self) -> int:
        return int(self.quantiles.shape[0])

    @property
    def n_quantiles(self) -> int:
        return int(self.quantiles.shape[1])

    def normalize_queries(self, values: torch.Tensor) -> torch.Tensor:
        values_2d = _as_2d(values)
        if self.mean is None or self.std is None:
            return values_2d
        return (values_2d - self.mean.unsqueeze(0)) / (self.std.unsqueeze(0) + self.stat_eps)

    def quantile_positions(self, values: torch.Tensor) -> torch.Tensor:
        """Approximate row-wise CDF values for queries, in [0, 1]."""
        values_2d = self.normalize_queries(values)
        xq_t = values_2d.transpose(0, 1).contiguous()  # [K, B]
        pos = torch.searchsorted(self.quantiles, xq_t, right=False)
        q = self.n_quantiles
        if q <= 1:
            out = torch.zeros_like(values_2d)
        else:
            out = (pos.to(values_2d.dtype) / float(q - 1)).transpose(0, 1).contiguous()
        return out.clamp(0.0, 1.0)

    def inverse_cdf(self, q_values: torch.Tensor) -> torch.Tensor:
        q_2d = _as_2d(q_values).clamp(0.0, 1.0)
        q_t = q_2d.transpose(0, 1).contiguous()  # [K, B]
        q_grid = self.q_grid.view(1, -1).expand(self.n_slices, -1)
        y = _interp_rows(q_grid, self.quantiles, q_2d)
        return y

    def interp_on_support(self, values: torch.Tensor, y_grid: torch.Tensor) -> torch.Tensor:
        values_2d = self.normalize_queries(values)
        return _interp_rows(self.quantiles, y_grid, values_2d)

    def state_dict(self) -> dict[str, torch.Tensor | int | float | None]:
        return {
            "quantiles": self.quantiles,
            "q_grid": self.q_grid,
            "sample_count": int(self.sample_count),
            "mean": self.mean,
            "std": self.std,
            "stat_eps": float(self.stat_eps),
        }

    @classmethod
    def from_state_dict(
        cls, state: dict[str, torch.Tensor | int | float | None]
    ) -> "QuantileSupportSummary":
        return cls(
            quantiles=state["quantiles"],  # type: ignore[arg-type]
            q_grid=state["q_grid"],  # type: ignore[arg-type]
            sample_count=int(state["sample_count"]),  # type: ignore[arg-type]
            mean=state.get("mean"),  # type: ignore[arg-type]
            std=state.get("std"),  # type: ignore[arg-type]
            stat_eps=float(state.get("stat_eps", 1e-6)),
        )


def build_quantile_summary(
    values: torch.Tensor,
    num_quantiles: int,
    normalize_mode: str = "none",
    stat_eps: float = 1e-6,
) -> QuantileSupportSummary:
    values_2d = _as_2d(values)
    if values_2d.shape[0] <= 0:
        raise ValueError("Cannot build a support summary from an empty tensor")
    if num_quantiles <= 1:
        raise ValueError("num_quantiles must be at least 2")

    mean = None
    std = None
    work = values_2d
    if normalize_mode == "snapshot_zscore":
        mean = work.mean(dim=0)
        std = work.std(dim=0, unbiased=False)
        work = (work - mean.unsqueeze(0)) / (std.unsqueeze(0) + stat_eps)
    elif normalize_mode != "none":
        raise ValueError(f"Unsupported normalize_mode={normalize_mode}")

    sorted_values = torch.sort(work, dim=0).values.transpose(0, 1).contiguous()  # [K, N]
    n = sorted_values.shape[1]
    q_grid = torch.linspace(
        0.0,
        1.0,
        steps=num_quantiles,
        device=sorted_values.device,
        dtype=sorted_values.dtype,
    )
    idxf = q_grid * float(max(n - 1, 1))
    idx_lo = idxf.floor().to(torch.long).clamp(0, n - 1)
    idx_hi = idxf.ceil().to(torch.long).clamp(0, n - 1)
    t = (idxf - idx_lo.to(idxf.dtype)).view(1, -1)
    idx_lo_e = idx_lo.view(1, -1).expand(sorted_values.shape[0], -1)
    idx_hi_e = idx_hi.view(1, -1).expand(sorted_values.shape[0], -1)
    q_lo = sorted_values.gather(1, idx_lo_e)
    q_hi = sorted_values.gather(1, idx_hi_e)
    quantiles = q_lo * (1.0 - t) + q_hi * t

    return QuantileSupportSummary(
        quantiles=quantiles,
        q_grid=q_grid,
        sample_count=int(values_2d.shape[0]),
        mean=mean,
        std=std,
        stat_eps=stat_eps,
    )


def build_dual_potential_grid(
    policy_summary: QuantileSupportSummary,
    expert_summary: QuantileSupportSummary,
) -> torch.Tensor:
    """Approximate 1D OT potential on the policy summary support.

    The derivative is approximated by the quantile-matched residual
    between policy and expert projected summaries.
    """
    z_pi = policy_summary.quantiles
    z_exp = expert_summary.quantiles
    if z_pi.shape != z_exp.shape:
        raise ValueError("policy and expert summaries must have matching shapes")

    grad = z_pi - z_exp
    dz = z_pi[:, 1:] - z_pi[:, :-1]
    trap = 0.5 * (grad[:, 1:] + grad[:, :-1]) * dz
    phi = torch.zeros_like(z_pi)
    phi[:, 1:] = torch.cumsum(trap, dim=1)
    return phi - phi.mean(dim=1, keepdim=True)
