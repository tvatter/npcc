"""
quantile_table_distribution1d.py — universal, fast conditional
predictive distribution via numerical inversion of a conditional
quantile table.

Approach
--------
Given a backend that can predict the conditional quantile function on a
fine grid of cumulative probabilities,

    Q(alpha | w),    alpha in {alpha_1, ..., alpha_K},

every primitive of the predictive distribution is derived from that
table:

- **PDF.**  ``f(y | w) = 1 / Q'(alpha)`` at ``alpha = F(y | w)``:
  sort the predicted quantiles per row (monotone rearrangement), take
  ``dQ/dalpha`` (clipped to ``min_qprime``), locate ``alpha(y)`` by
  interpolating ``y`` in the ``(Q, alpha)`` table, and read off
  ``1 / Q'`` there.
- **CDF.**  Linear interpolation in the ``(Q_sorted, alphas)`` table at
  ``transform(y)``; flat extrapolation outside the empirical range
  yields ``alpha_min`` / ``alpha_max``.
- **iCDF.**  Linear interpolation in the ``(alphas, Q_sorted)`` table,
  then mapped back to the y-scale via the inverse support transform.

This recovery is purely numerical and works with **any** conditional
quantile regressor.  Concrete backends implement the single abstract
hook :py:meth:`_predict_quantiles`; everything else — the chunked,
memory-safe table prediction, the monotone sort, and the
predict-once-per-row grid fast paths — lives here and is inherited by
every quantile-based backend.

Performance
-----------
The base owns the chunk loop in :py:meth:`_predict_quantile_table`, so a
backend cannot accidentally run a single huge forward pass (the
``workers>1`` OOM this design removed).  :py:meth:`pdf_grid` /
:py:meth:`cdf_grid` predict the table **once per conditioning row** and
evaluate it against every grid point by interpolation, chunked so the
``(chunk * n_y, n_alphas)`` transient stays bounded by ``batch_size``.

Support transforms
------------------
``U`` and ``V`` are copula scores in ``(0, 1)``.  With
``transform="logit"`` the backend is fit on ``Z = logit(Y)``; with
``transform="probit"`` on ``Z = Phi^{-1}(Y)``.  Densities convert back
via the matching Jacobian (CDFs and quantiles need no correction).
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from npcc.core._common import (
  TensorLike,
  _as_2d,
  _normalize_inputs,
  _torch_gradient_1d,
  _torch_interp_batched_fp,
  _torch_interp_batched_xp,
  _wrap_output,
)
from npcc.core.conditional_distribution1d import ConditionalDistribution1D


@dataclass
class QuantileGridConfig:
  """Grid configuration for the quantile-based conditional density.

  Attributes
  ----------
  n_quantiles
      Number of equally spaced ``alpha`` values in ``(alpha_min,
      alpha_max)``.  More points reduce interpolation error in the
      derivative step but increase inference cost linearly.
  alpha_min, alpha_max
      Tail trimming.  Defaults give the inner 99.8% of the distribution;
      pushing closer to ``0`` / ``1`` hurts numerical stability of
      ``Q'``.
  min_qprime
      Floor applied to ``dQ/dalpha`` before inversion, to avoid division
      by tiny numbers when the quantile curve is locally flat.
  eps
      Clip distance from the boundary of ``(0, 1)`` used by the support
      transforms; matches :func:`npcc.core._common._check_uv`.
  """

  n_quantiles: int = 101
  alpha_min: float = 1e-3
  alpha_max: float = 1.0 - 1e-3
  min_qprime: float = 1e-6
  eps: float = 1e-6

  def alphas(self) -> np.ndarray:
    """Return the validated ``alpha`` grid as a NumPy array.

    NumPy is convenient for forwarding to quantile-prediction APIs that
    expect a Python list of levels.
    """
    if not (0.0 < self.alpha_min < self.alpha_max < 1.0):
      raise ValueError("Require 0 < alpha_min < alpha_max < 1.")
    if self.n_quantiles < 5:
      raise ValueError("n_quantiles must be at least 5.")
    return np.linspace(self.alpha_min, self.alpha_max, self.n_quantiles)


class QuantileTableDistribution1D(ConditionalDistribution1D):
  """Conditional predictive distribution via inversion of a quantile table.

  Concrete backends implement only :py:meth:`_predict_quantiles`, which
  returns raw conditional quantiles for a chunk of conditioning rows.
  See the module docstring for the algorithm.

  Parameters
  ----------
  transform
      Forwarded to the base class.
  config
      Quantile-grid configuration.  Defaults to
      :class:`QuantileGridConfig`.  Its ``eps`` controls the boundary
      clipping used by the support transforms.
  device, batch_size
      Forwarded to the base class.
  """

  config: QuantileGridConfig

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    config: QuantileGridConfig | None = None,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
  ) -> None:
    cfg = config or QuantileGridConfig()
    super().__init__(
      transform=transform,
      eps=cfg.eps,
      device=device,
      batch_size=batch_size,
    )
    self.config = cfg

  # ------------------------------------------------------------------
  # Abstract backend hook.
  # ------------------------------------------------------------------

  @abstractmethod
  def _predict_quantiles(
    self, w: torch.Tensor, alphas: np.ndarray
  ) -> torch.Tensor:
    """Raw conditional quantiles for one chunk of rows.

    Parameters
    ----------
    w
        Feature matrix for the chunk, shape ``(n_chunk, p)``.
    alphas
        Ascending cumulative-probability levels, i.e.
        ``self.config.alphas()``.

    Returns
    -------
    torch.Tensor
        Shape ``(n_chunk, len(alphas))``, float64 on ``self._device``,
        with columns aligned to ``alphas``.  Monotonicity across columns
        is **not** required — the base rearranges (sorts) each row.
    """

  # ------------------------------------------------------------------
  # Internal: chunked, memory-safe quantile-table prediction (Q1).
  # ------------------------------------------------------------------

  def _predict_quantile_table(
    self, w_t: torch.Tensor, *, batch_size: int | None = None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(q_sorted [n_obs, n_alphas], alphas [n_alphas])``.

    Inference is chunked over rows of ``w_t`` to bound memory: each chunk
    is a single :py:meth:`_predict_quantiles` call and the per-chunk
    tables are concatenated.  Rows of ``q_sorted`` are then sorted to
    enforce monotonicity (rearrangement), which is backend-agnostic.
    """
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    alphas_np = self.config.alphas()
    n_alphas = len(alphas_np)
    n_obs = w_t.shape[0]

    parts: list[torch.Tensor] = []
    for start in range(0, n_obs, effective_batch_size):
      end = min(start + effective_batch_size, n_obs)
      q = self._predict_quantiles(w_t[start:end], alphas_np)
      n_chunk = end - start
      if q.shape != (n_chunk, n_alphas):
        raise RuntimeError(
          "_predict_quantiles must return shape (n_chunk, n_alphas); "
          f"got {tuple(q.shape)}, expected {(n_chunk, n_alphas)}."
        )
      parts.append(q.to(dtype=torch.float64, device=self._device))

    q_t = (
      torch.cat(parts, dim=0)
      if parts
      else torch.empty((0, n_alphas), dtype=torch.float64, device=self._device)
    )
    alphas_t = torch.as_tensor(
      alphas_np, dtype=torch.float64, device=self._device
    )
    q_sorted, _ = torch.sort(q_t, dim=1)
    return q_sorted, alphas_t

  # ------------------------------------------------------------------
  # Public API: pdf / cdf / icdf + the *_grid fast paths (Q2).
  # ------------------------------------------------------------------

  def pdf(
    self, w: TensorLike, y: TensorLike, *, batch_size: int | None = None
  ) -> TensorLike:
    """Return ``f(y_i | w_i)`` for each row ``i``.

    Out-of-support queries (``z`` outside the predicted quantile range)
    return ``0``.
    """
    return_as_torch, (w_in, y_in) = _normalize_inputs(w, y, device=self._device)
    assert w_in is not None and y_in is not None
    w_t = _as_2d(w_in, device=self._device)
    y_t = y_in.reshape(-1)
    if y_t.shape[0] != w_t.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_t)
    q_sorted, alphas = self._predict_quantile_table(w_t, batch_size=batch_size)

    dq_da = _torch_gradient_1d(q_sorted, alphas)
    dq_da = torch.clamp(dq_da, min=self.config.min_qprime)
    f_at_q = 1.0 / dq_da

    alpha_at_z = _torch_interp_batched_xp(
      z, q_sorted, alphas.expand_as(q_sorted)
    )
    dens_z = _torch_interp_batched_fp(alpha_at_z, alphas, f_at_q)

    out_of_support = (z <= q_sorted[:, 0]) | (z >= q_sorted[:, -1])
    dens_z = torch.where(out_of_support, torch.zeros_like(dens_z), dens_z)

    out = dens_z * self._jacobian_inverse(y_t)
    return _wrap_output(out, return_as_torch=return_as_torch)

  def pdf_grid(
    self,
    w: TensorLike,
    y_grid: TensorLike,
    *,
    batch_size: int | None = None,
  ) -> TensorLike:
    """Density on the Cartesian product of ``w`` rows and ``y_grid``.

    Returns shape ``(n_w, n_y)`` with ``out[i, j] = f(y_grid[j] | w[i])``.
    The quantile table is predicted **once per ``w`` row** (chunked by
    ``batch_size``); the same table is then evaluated against every ``y``
    value by interpolation, so the ``(chunk * n_y, n_alphas)`` transient
    stays bounded.  Numerically identical to calling :py:meth:`pdf` on
    the explicit tile.
    """
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    return_as_torch, (w_in, y_in) = _normalize_inputs(
      w, y_grid, device=self._device
    )
    assert w_in is not None and y_in is not None
    w_t = _as_2d(w_in, device=self._device)
    y_grid_t = y_in.reshape(-1)
    if y_grid_t.numel() == 0:
      raise ValueError("y_grid must contain at least one value.")

    n_y = y_grid_t.shape[0]
    z_grid = self._transform_y(y_grid_t)
    jac = self._jacobian_inverse(y_grid_t)

    chunks: list[torch.Tensor] = []
    for start in range(0, w_t.shape[0], effective_batch_size):
      end = min(start + effective_batch_size, w_t.shape[0])
      q_sorted, alphas = self._predict_quantile_table(
        w_t[start:end], batch_size=effective_batch_size
      )

      dq_da = torch.clamp(
        _torch_gradient_1d(q_sorted, alphas), min=self.config.min_qprime
      )
      f_at_q = 1.0 / dq_da

      n_chunk = q_sorted.shape[0]
      z_flat = z_grid.repeat(n_chunk)
      q_rep = q_sorted.repeat_interleave(n_y, dim=0)
      f_rep = f_at_q.repeat_interleave(n_y, dim=0)

      alpha_at_z = _torch_interp_batched_xp(
        z_flat, q_rep, alphas.expand_as(q_rep)
      )
      dens = _torch_interp_batched_fp(alpha_at_z, alphas, f_rep).reshape(
        n_chunk, n_y
      )

      out_of_support = (z_grid.unsqueeze(0) <= q_sorted[:, 0:1]) | (
        z_grid.unsqueeze(0) >= q_sorted[:, -1:]
      )
      dens = torch.where(out_of_support, torch.zeros_like(dens), dens)
      chunks.append(dens * jac.unsqueeze(0))

    out = (
      torch.cat(chunks, dim=0)
      if chunks
      else torch.empty((0, n_y), dtype=torch.float64, device=self._device)
    )
    return _wrap_output(out, return_as_torch=return_as_torch)

  def cdf(
    self, w: TensorLike, y: TensorLike, *, batch_size: int | None = None
  ) -> TensorLike:
    """Return ``F(y_i | w_i) = P(Y <= y_i | W = w_i)`` per row.

    Inverts the quantile table: ``F(y | w) = alpha`` such that
    ``Q(alpha | w) = transform(y)``.  Linear interpolation between grid
    alphas; flat extrapolation outside the empirical range yields
    ``alpha_min`` / ``alpha_max``.  No Jacobian (monotone transforms
    preserve the CDF).
    """
    return_as_torch, (w_in, y_in) = _normalize_inputs(w, y, device=self._device)
    assert w_in is not None and y_in is not None
    w_t = _as_2d(w_in, device=self._device)
    y_t = y_in.reshape(-1)
    if y_t.shape[0] != w_t.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_t)
    q_sorted, alphas = self._predict_quantile_table(w_t, batch_size=batch_size)

    out = _torch_interp_batched_xp(z, q_sorted, alphas.expand_as(q_sorted))
    return _wrap_output(out, return_as_torch=return_as_torch)

  def cdf_grid(
    self,
    w: TensorLike,
    y_grid: TensorLike,
    *,
    batch_size: int | None = None,
  ) -> TensorLike:
    """CDF on the Cartesian product of ``w`` rows and ``y_grid``.

    Returns shape ``(n_w, n_y)`` with ``out[i, j] = F(y_grid[j] | w[i])``.
    Predicts the quantile table once per ``w`` row (chunked by
    ``batch_size``) and interpolates the CDF at every grid point;
    numerically identical to :py:meth:`cdf` on the explicit tile.  Flat
    extrapolation outside the empirical range yields ``alpha_min`` /
    ``alpha_max`` (no Jacobian correction).
    """
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    return_as_torch, (w_in, y_in) = _normalize_inputs(
      w, y_grid, device=self._device
    )
    assert w_in is not None and y_in is not None
    w_t = _as_2d(w_in, device=self._device)
    y_grid_t = y_in.reshape(-1)
    if y_grid_t.numel() == 0:
      raise ValueError("y_grid must contain at least one value.")

    n_y = y_grid_t.shape[0]
    z_grid = self._transform_y(y_grid_t)

    chunks: list[torch.Tensor] = []
    for start in range(0, w_t.shape[0], effective_batch_size):
      end = min(start + effective_batch_size, w_t.shape[0])
      q_sorted, alphas = self._predict_quantile_table(
        w_t[start:end], batch_size=effective_batch_size
      )
      n_chunk = q_sorted.shape[0]
      z_flat = z_grid.repeat(n_chunk)
      q_rep = q_sorted.repeat_interleave(n_y, dim=0)
      cdf_flat = _torch_interp_batched_xp(
        z_flat, q_rep, alphas.expand_as(q_rep)
      )
      chunks.append(cdf_flat.reshape(n_chunk, n_y))

    out = (
      torch.cat(chunks, dim=0)
      if chunks
      else torch.empty((0, n_y), dtype=torch.float64, device=self._device)
    )
    return _wrap_output(out, return_as_torch=return_as_torch)

  def icdf(
    self, w: TensorLike, alphas: TensorLike, *, batch_size: int | None = None
  ) -> TensorLike:
    """Per-row conditional quantile ``F^{-1}(alphas_i | w_i)`` on the y-scale.

    Linear interpolation in the predicted quantile table ``Q(alphas_i |
    w_i)``, then mapped back through the inverse support transform.
    """
    return_as_torch, (w_in, a_in) = _normalize_inputs(
      w, alphas, device=self._device
    )
    assert w_in is not None and a_in is not None
    w_t = _as_2d(w_in, device=self._device)
    alpha_t = a_in.reshape(-1)
    if alpha_t.shape[0] != w_t.shape[0]:
      raise ValueError("w and alphas have incompatible lengths.")
    if torch.any((alpha_t <= 0.0) | (alpha_t >= 1.0)):
      raise ValueError("alphas must lie strictly inside (0, 1).")

    q_sorted, table_alphas = self._predict_quantile_table(
      w_t, batch_size=batch_size
    )
    z_out = _torch_interp_batched_fp(alpha_t, table_alphas, q_sorted)
    return _wrap_output(
      self._inverse_transform(z_out), return_as_torch=return_as_torch
    )
