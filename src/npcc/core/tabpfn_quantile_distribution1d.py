"""
tabpfn_quantile_distribution1d.py — univariate conditional predictive
distribution via numerical inversion of TabPFN conditional quantiles.

Approach
--------
Given a TabPFN regressor trained to predict ``Y`` given features ``W``,
ask the regressor for the conditional quantile function on a fine grid
of cumulative probabilities

    Q(alpha | w),    alpha in {alpha_1, ..., alpha_K},

then derive each primitive of the predictive distribution from this
table:

- **PDF.**  Use the change-of-variables formula
  ``f(y | w) = 1 / Q'(alpha)`` at ``alpha = F(y | w)``.  Numerically:

  1. Sort the predicted quantiles per observation (monotone
     rearrangement; TabPFN's quantile output is not guaranteed monotone
     for tightly spaced ``alpha``).
  2. Compute ``dQ/dalpha`` with a torch port of :func:`numpy.gradient`,
     clipped to ``min_qprime`` to avoid the singular ``1 / 0`` at
     constant plateaus.
  3. Locate ``alpha(y) = F(y | w)`` by interpolating ``y`` in the
     ``(Q, alpha)`` table.
  4. Read off ``f_at_q = 1 / Q'`` at ``alpha(y)``.

- **CDF.**  Linear interpolation in the ``(Q_sorted, alphas)`` table at
  ``transform(y)``; flat extrapolation outside the empirical range
  yields ``alpha_min`` / ``alpha_max``, matching the support of the
  alpha grid.

- **iCDF.**  Linear interpolation in the ``(alphas, Q_sorted)`` table
  at the requested ``alpha``.  Then map back to the y-scale via the
  inverse support transform.

This recovery is purely numerical and works with any quantile
regressor; it does not require access to TabPFN's internal distribution
head.  The counterpart in
:mod:`npcc.tabpfn_criterion_distribution1d` calls TabPFN's native
``criterion.{pdf,cdf,icdf}`` directly, which is faster and avoids the
slope inversion, but is specific to TabPFN's "full" output.

Support transforms
------------------
``U`` and ``V`` are copula scores in ``(0, 1)`` and quantile estimation
on a bounded interval is awkward.  With ``transform="logit"`` we fit
the regressor on ``Z = logit(Y)``. With ``transform="probit"`` we use
``Z = Phi^{-1}(Y)``. For either choice, convert back via the matching
Jacobian for densities (CDFs and quantiles need no correction):

  f_Y(y | w) = f_Z(T(y) | w) * |dT(y)/dy|.

For example, ``T(y) = logit(y)`` gives

  f_Y(y | w) = f_Z(logit(y) | w) / (y * (1 - y)),

while ``T(y) = Phi^{-1}(y)`` gives

  f_Y(y | w) = f_Z(Phi^{-1}(y) | w) / phi(Phi^{-1}(y)).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from tabpfn.constants import ModelVersion

from npcc.core._common import (
  TensorLike,
  _as_2d,
  _normalize_inputs,
  _torch_gradient_1d,
  _torch_interp_batched_fp,
  _torch_interp_batched_xp,
  _wrap_output,
)
from npcc.core.tabpfn_distribution1d import (
  _DEFAULT_MODEL_VERSION,
  TabPFNDistribution1D,
)

logger = logging.getLogger(__name__)


@dataclass
class QuantileGridConfig:
  """Grid configuration for quantile-based conditional density.

  Attributes
  ----------
  n_quantiles
      Number of equally spaced ``alpha`` values in ``(alpha_min,
      alpha_max)``.  More points reduce interpolation error in the
      derivative step but increase TabPFN inference cost linearly.
  alpha_min, alpha_max
      Tail trimming.  Defaults give the inner 99.8% of the distribution;
      pushing closer to ``0`` / ``1`` hurts numerical stability of
      ``Q'``.
  min_qprime
      Floor applied to ``dQ/dalpha`` before inversion, to avoid division
      by tiny numbers when the quantile curve is locally flat.
  eps
      Clip distance from the boundary of ``(0, 1)`` used by the logit
      transform; matches :func:`npcc._common._check_uv`.
  """

  n_quantiles: int = 101
  alpha_min: float = 1e-3
  alpha_max: float = 1.0 - 1e-3
  min_qprime: float = 1e-6
  eps: float = 1e-6

  def alphas(self) -> np.ndarray:
    """Return the validated ``alpha`` grid as a NumPy array.

    NumPy here is convenient for forwarding to TabPFN's
    ``predict(quantiles=...)`` API, which expects a Python list.
    """
    if not (0.0 < self.alpha_min < self.alpha_max < 1.0):
      raise ValueError("Require 0 < alpha_min < alpha_max < 1.")
    if self.n_quantiles < 5:
      raise ValueError("n_quantiles must be at least 5.")
    return np.linspace(self.alpha_min, self.alpha_max, self.n_quantiles)


class TabPFNQuantileDistribution1D(TabPFNDistribution1D):
  """Univariate conditional predictive distribution via TabPFN quantiles.

  See the module-level docstring for the algorithm.  The ``fit`` /
  ``pdf`` / ``cdf`` / ``icdf`` API mirrors
  :class:`TabPFNCriterionDistribution1D` so the two classes are drop-in
  interchangeable inside :class:`PFNRBicop`.

  Parameters
  ----------
  transform
      Forwarded to the base class.
  config
      Quantile-grid configuration.  Defaults to
      :class:`QuantileGridConfig`.  Its ``eps`` controls the boundary
      clipping used by the support transforms.
  device
      Forwarded to the base class.
  model_kwargs
      Forwarded to
      :py:meth:`TabPFNRegressor.create_default_for_version`.
  """

  config: QuantileGridConfig

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    config: QuantileGridConfig | None = None,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
    model_version: ModelVersion | None = _DEFAULT_MODEL_VERSION,
  ) -> None:
    cfg = config or QuantileGridConfig()
    super().__init__(
      transform=transform,
      eps=cfg.eps,
      device=device,
      batch_size=batch_size,
      model_kwargs=model_kwargs,
      model_version=model_version,
    )
    self.config = cfg

  # ------------------------------------------------------------------
  # Internal: shared quantile-table prediction.
  # ------------------------------------------------------------------

  def _predict_quantile_table(
    self, w_t: torch.Tensor, *, batch_size: int | None = None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(q_sorted [n_obs, n_alphas], alphas [n_alphas])``.

    Inference is chunked over rows of ``w_t`` to bound GPU memory: each
    chunk is a single ``predict`` forward pass and the per-chunk tables
    are concatenated.  Rows of ``q_sorted`` are then sorted to enforce
    monotonicity (rearrangement) regardless of TabPFN's output
    orientation.
    """
    if self.model_ is None:
      raise RuntimeError("The model is not fitted.")
    effective_batch_size = self._resolve_batch_size(batch_size)

    alphas_np = self.config.alphas()
    n_alphas = len(alphas_np)
    w_cpu = w_t.detach().cpu()
    n_obs = w_cpu.shape[0]

    parts: list[torch.Tensor] = []
    for start in range(0, n_obs, effective_batch_size):
      end = min(start + effective_batch_size, n_obs)
      q_pred = self.model_.predict(
        w_cpu[start:end],
        output_type="quantiles",
        quantiles=alphas_np.tolist(),
      )
      q = np.asarray(q_pred, dtype=float)
      n_chunk = end - start

      # TabPFN's quantile output may be (n_chunk, n_quantiles) or its
      # transpose, depending on the version. Prefer the documented
      # (n_chunk, n_alphas) layout so a square chunk (n_chunk == n_alphas)
      # is never spuriously transposed.
      if q.shape == (n_chunk, n_alphas):
        pass
      elif q.shape == (n_alphas, n_chunk):
        logger.debug(
          "Reshaping TabPFN quantile output from (n_q, n_obs) to (n_obs, n_q)."
        )
        q = q.T
      else:
        raise RuntimeError(
          "Unexpected quantile output shape. "
          f"Got {q.shape}, expected {(n_chunk, n_alphas)} "
          f"or {(n_alphas, n_chunk)}."
        )
      parts.append(torch.as_tensor(q, dtype=torch.float64, device=self._device))

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
  # Public API: pdf / cdf / icdf.
  # ------------------------------------------------------------------

  def pdf(
    self, w: TensorLike, y: TensorLike, *, batch_size: int | None = None
  ) -> TensorLike:
    """Return ``f(y_i | w_i)`` for each row ``i``.

    ``w`` and ``y`` must have the same number of rows.  Out-of-support
    queries (``z`` outside the predicted quantile range) return ``0``.
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
    ``batch_size``); the same table is then evaluated against every
    ``y`` value by interpolation.  This mirrors
    :py:meth:`TabPFNCriterionDistribution1D.pdf_grid` and is the fast
    path for grid-based Sinkhorn projection — it avoids the explicit
    ``n_w * n_y`` tile that would predict one table per grid cell.

    Numerically identical to calling :py:meth:`pdf` on the explicit
    tile: the per-row inversion, out-of-support masking and inverse
    Jacobian match exactly.
    """
    if self.model_ is None:
      raise RuntimeError("The model is not fitted.")
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

    # Chunk over conditioning rows so the (chunk * n_y, n_alphas)
    # interpolation transient stays bounded by batch_size (mirrors the
    # criterion pdf_grid loop).
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

      # Cartesian (n_chunk x n_y) interpolation, reusing the per-row
      # helpers: row i contributes its table to every grid column j.
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

      # Out-of-support per (row, column): z outside the row's quantile range.
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

    Computed by inverting the quantile table: ``F(y | w) = α`` such
    that ``Q(α | w) = transform(y)``.  Linear interpolation between
    grid alphas; flat extrapolation outside the empirical quantile
    range yields ``alpha_min`` / ``alpha_max``, matching the support
    of the alpha grid.  ``batch_size`` chunks the underlying quantile
    forward pass to bound GPU memory.

    No Jacobian correction: monotone transforms preserve the CDF, so
    ``F_Y(y | w) = F_Z(transform(y) | w)``.
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

  def icdf(
    self, w: TensorLike, alphas: TensorLike, *, batch_size: int | None = None
  ) -> TensorLike:
    """Per-row conditional quantile ``F^{-1}(alphas_i | w_i)`` on the y-scale.

    For each row ``i``, returns ``y`` such that
    ``F(y | w_i) = alphas_i``.  Implemented as linear interpolation in
    the predicted quantile table: ``Q(alphas_i | w_i)``.  Used by the
    Rosenblatt simulation recipe behind :py:meth:`PFNRBicop.tau`.
    ``batch_size`` chunks the underlying quantile forward pass to bound
    GPU memory.
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
