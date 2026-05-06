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

Logit support transform
-----------------------
``U`` and ``V`` are copula scores in ``(0, 1)`` and quantile estimation
on a bounded interval is awkward.  When ``transform="logit"`` we fit
the regressor on ``Z = logit(Y)`` (the unbounded image) and convert
back via the standard Jacobian for densities (CDFs and quantiles need
no correction):

    f_Y(y | w) = f_Z(logit(y) | w) / (y * (1 - y)).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from npcc._common import (
  TensorLike,
  _as_2d,
  _normalize_inputs,
  _torch_gradient_1d,
  _torch_interp_batched_fp,
  _torch_interp_batched_xp,
  _wrap_output,
)
from npcc.tabpfn_distribution1d import TabPFNDistribution1D

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

  n_quantiles: int = 201
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
      clipping used by the logit transform.
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
    transform: Literal["identity", "logit"] = "logit",
    config: QuantileGridConfig | None = None,
    device: str | torch.device | None = None,
    model_kwargs: dict[str, Any] | None = None,
  ) -> None:
    cfg = config or QuantileGridConfig()
    super().__init__(
      transform=transform,
      eps=cfg.eps,
      device=device,
      model_kwargs=model_kwargs,
    )
    self.config = cfg

  # ------------------------------------------------------------------
  # Internal: shared quantile-table prediction.
  # ------------------------------------------------------------------

  def _predict_quantile_table(
    self, w_t: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(q_sorted [n_obs, n_alphas], alphas [n_alphas])``.

    Single TabPFN forward pass; rows of ``q_sorted`` are sorted to
    enforce monotonicity (rearrangement) regardless of TabPFN's
    output orientation.
    """
    if self.model_ is None:
      raise RuntimeError("The model is not fitted.")

    alphas_np = self.config.alphas()
    q_pred = self.model_.predict(
      w_t.detach().cpu(),
      output_type="quantiles",
      quantiles=alphas_np.tolist(),
    )

    q = np.asarray(q_pred, dtype=float)
    n_obs = w_t.shape[0]

    # TabPFN's quantile output may be (n_obs, n_quantiles) or its
    # transpose, depending on the version. Normalise here.
    if q.shape == (len(alphas_np), n_obs):
      logger.debug(
        "Reshaping TabPFN quantile output from (n_q, n_obs) to (n_obs, n_q)."
      )
      q = q.T
    elif q.shape != (n_obs, len(alphas_np)):
      raise RuntimeError(
        "Unexpected quantile output shape. "
        f"Got {q.shape}, expected {(len(alphas_np), n_obs)} "
        f"or {(n_obs, len(alphas_np))}."
      )

    q_t = torch.as_tensor(q, dtype=torch.float64, device=self._device)
    alphas_t = torch.as_tensor(
      alphas_np, dtype=torch.float64, device=self._device
    )
    q_sorted, _ = torch.sort(q_t, dim=1)
    return q_sorted, alphas_t

  # ------------------------------------------------------------------
  # Public API: pdf / cdf / icdf.
  # ------------------------------------------------------------------

  def pdf(self, w: TensorLike, y: TensorLike) -> TensorLike:
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
    q_sorted, alphas = self._predict_quantile_table(w_t)

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

  def cdf(self, w: TensorLike, y: TensorLike) -> TensorLike:
    """Return ``F(y_i | w_i) = P(Y <= y_i | W = w_i)`` per row.

    Computed by inverting the quantile table: ``F(y | w) = α`` such
    that ``Q(α | w) = transform(y)``.  Linear interpolation between
    grid alphas; flat extrapolation outside the empirical quantile
    range yields ``alpha_min`` / ``alpha_max``, matching the support
    of the alpha grid.

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
    q_sorted, alphas = self._predict_quantile_table(w_t)

    out = _torch_interp_batched_xp(z, q_sorted, alphas.expand_as(q_sorted))
    return _wrap_output(out, return_as_torch=return_as_torch)

  def icdf(self, w: TensorLike, alphas: TensorLike) -> TensorLike:
    """Per-row conditional quantile ``F^{-1}(alphas_i | w_i)`` on the y-scale.

    For each row ``i``, returns ``y`` such that
    ``F(y | w_i) = alphas_i``.  Implemented as linear interpolation in
    the predicted quantile table: ``Q(alphas_i | w_i)``.  Used by the
    Rosenblatt simulation recipe behind :py:meth:`PFNRBicop.tau`.
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

    q_sorted, table_alphas = self._predict_quantile_table(w_t)
    z_out = _torch_interp_batched_fp(alpha_t, table_alphas, q_sorted)
    return _wrap_output(
      self._inverse_transform(z_out), return_as_torch=return_as_torch
    )
