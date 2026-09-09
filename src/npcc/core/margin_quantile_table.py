"""Conditional margins reconstructed from predicted quantile tables.

Given a backend that predicts a conditional quantile function

    Q(alpha | x)

the complete conditional margin is reconstructed numerically:

- PDF: ``f(y | x) = 1 / Q'(F(y | x))``;
- CDF: interpolation in the predicted ``(Q, alpha)`` table;
- inverse CDF: interpolation in the ``(alpha, Q)`` table.

Concrete backends implement only :meth:`_predict_quantiles`. Table prediction
is chunked over conditioning rows, and grid evaluation predicts at most once
per conditioning row.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch

from npcc.core._interp import (
  gradient_1d,
  interp_batched_fp,
  interp_batched_xp,
)
from npcc.core.margin import ConditionalMargin


@dataclass(frozen=True)
class QuantileTableConfig:
  """Configuration for the quantile-table reconstruction.

  Attributes
  ----------
  n_quantiles
    Number of equally spaced probability levels.
  alpha_min
    Smallest predicted probability level.
  alpha_max
    Largest predicted probability level.
  min_qprime
    Lower bound for the estimated quantile derivative.
  """

  n_quantiles: int = 101
  alpha_min: float = 1e-3
  alpha_max: float = 1.0 - 1e-3
  min_qprime: float = 1e-6

  def __post_init__(self) -> None:
    """Validate the quantile-table configuration."""
    if not (0.0 < self.alpha_min < self.alpha_max < 1.0):
      raise ValueError("Require 0 < alpha_min < alpha_max < 1.")

    if self.n_quantiles < 5:
      raise ValueError("n_quantiles must be at least 5.")

    if not self.min_qprime > 0.0:
      raise ValueError("min_qprime must be positive.")

  def alphas(
    self,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float64,
  ) -> torch.Tensor:
    """Return the validated probability grid."""
    return torch.linspace(
      self.alpha_min,
      self.alpha_max,
      self.n_quantiles,
      dtype=dtype,
      device=device,
    )


class QuantileTableDistribution1D(ConditionalMargin):
  """Conditional margin reconstructed from predicted quantile tables.

  Concrete backends implement :meth:`_predict_quantiles`, which predicts one
  complete quantile table for every conditioning row.
  """

  quantile_table_config: QuantileTableConfig

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    quantile_table_config: QuantileTableConfig | None = None,
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
  ) -> None:
    cfg = quantile_table_config or QuantileTableConfig()

    super().__init__(
      transform=transform,
      eps=eps,
      device=device,
      batch_size=batch_size,
    )

    self.quantile_table_config = cfg

  @abstractmethod
  def _predict_quantiles(
    self,
    x: torch.Tensor,
    alphas: torch.Tensor,
  ) -> torch.Tensor:
    """Predict conditional quantiles for one chunk of conditioning rows.

    Parameters
    ----------
    x
      Conditioning matrix with shape ``(n_chunk, n_features)`` on the
      configured device.
    alphas
      Ascending probability levels with shape ``(n_alphas,)``.

    Returns
    -------
    torch.Tensor
      Predicted quantiles with shape ``(n_chunk, n_alphas)`` on the configured
      device. Rows need not be monotone because this class sorts them.
    """

  def _predict_quantile_table(
    self,
    x: torch.Tensor,
    *,
    batch_size: int | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict and monotonically rearrange conditional quantile tables."""
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    alphas = self.quantile_table_config.alphas(
      device=x.device,
      dtype=x.dtype,
    )
    n_alphas = alphas.shape[0]
    n_obs = x.shape[0]

    parts: list[torch.Tensor] = []

    for start in range(0, n_obs, effective_batch_size):
      end = min(start + effective_batch_size, n_obs)

      quantiles = self._predict_quantiles(x[start:end], alphas)

      n_chunk = end - start
      expected_shape = (n_chunk, n_alphas)

      if quantiles.shape != (n_chunk, n_alphas):
        raise RuntimeError(
          "_predict_quantiles must return shape (n_chunk, n_alphas); "
          f"got {tuple(quantiles.shape)}, expected {expected_shape}."
        )

      parts.append(quantiles)

    quantile_table = (
      torch.cat(parts, dim=0)
      if parts
      else torch.empty(
        (0, n_alphas),
        dtype=torch.float64,
        device=self._device,
      )
    )

    sorted_quantiles, _ = torch.sort(
      quantile_table,
      dim=1,
    )

    return sorted_quantiles, alphas

  def pdf(
    self,
    y: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate ``f(y_i | x_i)`` for every observation."""
    y_t = y.reshape(-1)

    x_t = self._conditioning(y_t, x)

    if y_t.shape[0] != x_t.shape[0]:
      raise ValueError("x and y have incompatible lengths.")

    z = self._transform_y(y_t)

    sorted_quantiles, alphas = self._predict_quantile_table(
      x_t,
      batch_size=batch_size,
    )

    quantile_derivative = gradient_1d(
      sorted_quantiles,
      alphas,
    )
    quantile_derivative = torch.clamp(
      quantile_derivative,
      min=self.quantile_table_config.min_qprime,
    )

    density_at_quantiles = 1.0 / quantile_derivative

    alpha_at_z = interp_batched_xp(
      z,
      sorted_quantiles,
      alphas.expand_as(sorted_quantiles),
    )

    transformed_density = interp_batched_fp(
      alpha_at_z,
      alphas,
      density_at_quantiles,
    )

    out_of_support = (z <= sorted_quantiles[:, 0]) | (
      z >= sorted_quantiles[:, -1]
    )

    transformed_density = torch.where(
      out_of_support,
      torch.zeros_like(transformed_density),
      transformed_density,
    )

    return transformed_density * self._jacobian_inverse(y_t)

  def pdf_grid(
    self,
    y_grid: torch.Tensor,
    /,
    *,
    x: torch.Tensor,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate density on the Cartesian product of ``x`` and ``y_grid``."""
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    y_grid_t = y_grid.reshape(-1)

    if y_grid_t.numel() == 0:
      raise ValueError("y_grid must contain at least one value.")

    x_t = x.reshape(-1, 1) if x.ndim == 1 else x

    n_y = y_grid_t.shape[0]
    z_grid = self._transform_y(y_grid_t)
    jacobian = self._jacobian_inverse(y_grid_t)

    chunks: list[torch.Tensor] = []

    for start in range(0, x_t.shape[0], effective_batch_size):
      end = min(start + effective_batch_size, x_t.shape[0])

      sorted_quantiles, alphas = self._predict_quantile_table(
        x_t[start:end],
        batch_size=effective_batch_size,
      )

      quantile_derivative = torch.clamp(
        gradient_1d(sorted_quantiles, alphas),
        min=self.quantile_table_config.min_qprime,
      )
      density_at_quantiles = 1.0 / quantile_derivative

      n_chunk = sorted_quantiles.shape[0]
      z_flat = z_grid.repeat(n_chunk)

      quantiles_repeated = sorted_quantiles.repeat_interleave(n_y, dim=0)
      densities_repeated = density_at_quantiles.repeat_interleave(n_y, dim=0)

      alpha_at_z = interp_batched_xp(
        z_flat,
        quantiles_repeated,
        alphas.expand_as(quantiles_repeated),
      )

      density = interp_batched_fp(
        alpha_at_z,
        alphas,
        densities_repeated,
      ).reshape(n_chunk, n_y)

      out_of_support = (z_grid.unsqueeze(0) <= sorted_quantiles[:, 0:1]) | (
        z_grid.unsqueeze(0) >= sorted_quantiles[:, -1:]
      )

      density = torch.where(
        out_of_support,
        torch.zeros_like(density),
        density,
      )

      chunks.append(density * jacobian.unsqueeze(0))

    return (
      torch.cat(chunks, dim=0)
      if chunks
      else torch.empty(
        (0, n_y),
        dtype=x.dtype,
        device=x.device,
      )
    )

  def cdf(
    self,
    y: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate ``F(y_i | x_i)`` for every observation."""
    y_t = y.reshape(-1)
    x_t = self._conditioning(y_t, x=x)

    if y_t.shape[0] != x_t.shape[0]:
      raise ValueError("x and y have incompatible lengths.")

    z = self._transform_y(y_t)

    sorted_quantiles, alphas = self._predict_quantile_table(
      x_t,
      batch_size=batch_size,
    )

    return interp_batched_xp(
      z,
      sorted_quantiles,
      alphas.expand_as(sorted_quantiles),
    )

  def cdf_grid(
    self,
    y_grid: torch.Tensor,
    /,
    *,
    x: torch.Tensor,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate CDFs on the Cartesian product of ``x`` and ``y_grid``."""
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    y_grid_t = y_grid.reshape(-1)

    if y_grid_t.numel() == 0:
      raise ValueError("y_grid must contain at least one value.")

    x_t = x.reshape(-1, 1) if x.ndim == 1 else x

    n_y = y_grid_t.shape[0]
    z_grid = self._transform_y(y_grid_t)

    chunks: list[torch.Tensor] = []

    for start in range(0, x_t.shape[0], effective_batch_size):
      end = min(start + effective_batch_size, x_t.shape[0])

      sorted_quantiles, alphas = self._predict_quantile_table(
        x_t[start:end],
        batch_size=effective_batch_size,
      )

      n_chunk = sorted_quantiles.shape[0]
      z_flat = z_grid.repeat(n_chunk)

      quantiles_repeated = sorted_quantiles.repeat_interleave(n_y, dim=0)

      cdf_flat = interp_batched_xp(
        z_flat,
        quantiles_repeated,
        alphas.expand_as(quantiles_repeated),
      )

      chunks.append(cdf_flat.reshape(n_chunk, n_y))

    return (
      torch.cat(chunks, dim=0)
      if chunks
      else torch.empty(
        (0, n_y),
        dtype=x.dtype,
        device=x.device,
      )
    )

  def icdf(
    self,
    p: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate ``F^-1(p_i | x_i)`` on the response scale."""
    p_t = p.reshape(-1)
    x_t = self._conditioning(p_t, x)

    if p_t.shape[0] != x_t.shape[0]:
      raise ValueError("x and p have incompatible lengths.")

    if torch.any((p_t <= 0.0) | (p_t >= 1.0)):
      raise ValueError("p must lie strictly inside (0, 1).")

    sorted_quantiles, table_alphas = self._predict_quantile_table(
      x_t,
      batch_size=batch_size,
    )

    transformed_quantiles = interp_batched_fp(
      p_t,
      table_alphas,
      sorted_quantiles,
    )

    return self._inverse_transform(transformed_quantiles)
