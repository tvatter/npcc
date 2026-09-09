"""
NGBoost parametric conditional margin.

NGBoost fits a parametric conditional distribution using natural-gradient
boosting. ``pred_dist(x).dist`` provides a vectorized SciPy distribution with
PDF, CDF, and inverse-CDF operations.

This backend therefore implements the conditional-margin operations directly
instead of reconstructing them from a quantile table.

NumPy and SciPy conversions are isolated inside this adapter. All public inputs
and outputs remain torch tensors.

This backend requires the ``ngboost`` optional dependency.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from ngboost import NGBRegressor
from ngboost.distns import Normal
from pyvinecopulib.core.extend import to_numpy

from npcc.core.margin import ConditionalMargin


class NGBoostBackend(ConditionalMargin):
  """Conditional margin using an NGBoost parametric distribution.

  Parameters
  ----------
  transform
    Response transformation inherited from
    :class:`~npcc.core.margin.ConditionalMargin`.
  eps
    Boundary clipping distance for logit and probit transformations.
  device
    Device used for returned tensors.
  batch_size
    Maximum number of conditioning rows evaluated in one prediction call.
  dist
    NGBoost distribution class. The default is ``Normal``.
  **ngb_kwargs
    Additional arguments passed to ``NGBRegressor``.
  """

  model_: NGBRegressor | None

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    dist: type | None = None,
    **ngb_kwargs: object,
  ) -> None:
    super().__init__(
      transform=transform,
      eps=eps,
      device=device,
      batch_size=batch_size,
    )
    self._dist = dist if dist is not None else Normal
    self.ngb_kwargs = dict(ngb_kwargs)
    self.model_ = None

  def _fit_model(
    self,
    x: torch.Tensor,
    z: torch.Tensor,
  ) -> None:
    """Fit the NGBoost model on transformed responses."""
    model = NGBRegressor(Dist=self._dist, verbose=False, **self.ngb_kwargs)

    model.fit(
      to_numpy(x),
      to_numpy(z),
    )

    self.model_ = model

  def _frozen(
    self,
    x: torch.Tensor,
  ) -> Any:  # noqa: ANN401 - scipy frozen dist
    """Return the vectorized SciPy distribution for conditioning rows."""
    assert self.model_ is not None

    return self.model_.pred_dist(to_numpy(x)).dist

  @staticmethod
  def _to_tensor(
    values: object,
    *,
    like: torch.Tensor,
  ) -> torch.Tensor:
    """Convert SciPy output to match a reference tensor."""
    return torch.as_tensor(
      values,
      dtype=like.dtype,
      device=like.device,
    )

  def pdf(
    self,
    y: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate ``f(y_i | x_i)`` for every observation."""
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    y_t = y.reshape(-1)
    x_t = self._conditioning(y_t, x=x)

    if y_t.shape[0] != x_t.shape[0]:
      raise ValueError("x and y have incompatible lengths.")

    z = self._transform_y(y_t)
    parts: list[torch.Tensor] = []

    for start in range(0, y_t.shape[0], effective_batch_size):
      end = min(start + effective_batch_size, y_t.shape[0])

      frozen = self._frozen(x_t[start:end])
      density = frozen.pdf(
        to_numpy(z[start:end]),
      )

      parts.append(self._to_tensor(density, like=y_t))

    density_z = (
      torch.cat(parts)
      if parts
      else torch.empty(0, dtype=y_t.dtype, device=y_t.device)
    )
    return density_z * self._jacobian_inverse(y_t)

  def cdf(
    self,
    y: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate ``F(y_i | x_i)`` for every observation."""
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    y_t = y.reshape(-1)
    x_t = self._conditioning(y_t, x=x)

    if y_t.shape[0] != x_t.shape[0]:
      raise ValueError("x and y have incompatible lengths.")

    z = self._transform_y(y_t)
    parts: list[torch.Tensor] = []

    for start in range(0, y_t.shape[0], effective_batch_size):
      end = min(start + effective_batch_size, y_t.shape[0])

      frozen = self._frozen(x_t[start:end])
      probabilities = frozen.cdf(to_numpy(z[start:end]))

      parts.append(self._to_tensor(probabilities, like=y_t))

    return (
      torch.cat(parts)
      if parts
      else torch.empty(0, dtype=y_t.dtype, device=y_t.device)
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
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    p_t = p.reshape(-1)
    x_t = self._conditioning(p_t, x=x)

    if p_t.shape[0] != x_t.shape[0]:
      raise ValueError("x and p have incompatible lengths.")

    if torch.any((p_t <= 0.0) | (p_t >= 1.0)):
      raise ValueError("p must lie strictly inside (0, 1).")

    parts: list[torch.Tensor] = []

    for start in range(0, p_t.shape[0], effective_batch_size):
      end = min(start + effective_batch_size, p_t.shape[0])

      frozen = self._frozen(x_t[start:end])
      transformed_quantiles = frozen.ppf(to_numpy(p_t[start:end]))

      parts.append(self._to_tensor(transformed_quantiles, like=p_t))

    quantiles_z = (
      torch.cat(parts)
      if parts
      else torch.empty(0, dtype=p_t.dtype, device=p_t.device)
    )

    return self._inverse_transform(quantiles_z)

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

    y_grid_t = self._prep(y_grid).reshape(-1)
    if y_grid_t.numel() == 0:
      raise ValueError("y_grid must contain at least one value.")

    x_t = self._grid_covariates(x)
    z_grid = self._transform_y(y_grid_t)
    jacobian = self._jacobian_inverse(y_grid_t)
    z_host = to_numpy(z_grid)

    chunks: list[torch.Tensor] = []

    for start in range(0, x_t.shape[0], effective_batch_size):
      end = min(start + effective_batch_size, x_t.shape[0])

      frozen = self._frozen(x_t[start:end])
      density = frozen.pdf(z_host[:, None]).T

      density_tensor = self._to_tensor(density, like=y_grid_t)

      chunks.append(density_tensor * jacobian.unsqueeze(0))

    return (
      torch.cat(chunks, dim=0)
      if chunks
      else torch.empty(
        (0, y_grid_t.shape[0]), dtype=y_grid_t.dtype, device=y_grid_t.device
      )
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

    y_grid_t = self._prep(y_grid).reshape(-1)

    if y_grid_t.numel() == 0:
      raise ValueError("y_grid must contain at least one value.")

    x_t = self._grid_covariates(x)
    z_grid = self._transform_y(y_grid_t)
    z_host = to_numpy(z_grid)

    chunks: list[torch.Tensor] = []

    for start in range(0, x_t.shape[0], effective_batch_size):
      end = min(start + effective_batch_size, x_t.shape[0])

      frozen = self._frozen(x_t[start:end])
      probabilities = frozen.cdf(z_host[:, None]).T

      chunks.append(self._to_tensor(probabilities, like=y_grid_t))

    return (
      torch.cat(chunks, dim=0)
      if chunks
      else torch.empty(
        (0, y_grid_t.shape[0]), dtype=y_grid_t.dtype, device=y_grid_t.device
      )
    )
