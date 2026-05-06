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
  2. Compute ``dQ/dalpha`` with :func:`numpy.gradient`, clipped to
     ``min_qprime`` to avoid the singular ``1 / 0`` at constant
     plateaus.
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
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from npcc._common import _as_2d
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
    """Return the validated ``alpha`` grid."""
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
    model_kwargs: dict[str, Any] | None = None,
  ) -> None:
    cfg = config or QuantileGridConfig()
    super().__init__(
      transform=transform, eps=cfg.eps, model_kwargs=model_kwargs
    )
    self.config = cfg

  def fit(self, w: ArrayLike, y: ArrayLike) -> Self:
    """Fit a TabPFN-v2.5 regressor on ``(w, transform(y))``.

    The estimator is locked to TabPFN-v2.5 because v2.6 has reported
    regressions on tabular regression tasks.  Override via
    ``model_kwargs`` if needed.
    """
    w_arr = _as_2d(w)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if len(y_arr) != w_arr.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_arr)

    self.model_ = TabPFNRegressor.create_default_for_version(
      ModelVersion.V2_5, **self.model_kwargs
    )
    self.model_.fit(w_arr, z)
    return self

  # ------------------------------------------------------------------
  # Internal: shared quantile-table prediction.
  # ------------------------------------------------------------------

  def _predict_quantile_table(
    self, w_arr: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(q_sorted [n_obs, n_alphas], alphas [n_alphas])``.

    Single TabPFN forward pass; rows of ``q_sorted`` are sorted to
    enforce monotonicity (rearrangement) regardless of TabPFN's
    output orientation.
    """
    if self.model_ is None:
      raise RuntimeError("The model is not fitted.")

    alphas = self.config.alphas()
    q_pred = self.model_.predict(
      w_arr,
      output_type="quantiles",
      quantiles=alphas.tolist(),
    )

    q = np.asarray(q_pred, dtype=float)
    n_obs = w_arr.shape[0]

    # TabPFN's quantile output may be (n_obs, n_quantiles) or its
    # transpose, depending on the version. Normalise here.
    if q.shape == (len(alphas), n_obs):
      logger.debug(
        "Reshaping TabPFN quantile output from (n_q, n_obs) to (n_obs, n_q)."
      )
      q = q.T
    elif q.shape != (n_obs, len(alphas)):
      raise RuntimeError(
        "Unexpected quantile output shape. "
        f"Got {q.shape}, expected {(len(alphas), n_obs)} "
        f"or {(n_obs, len(alphas))}."
      )

    return np.sort(q, axis=1), alphas

  # ------------------------------------------------------------------
  # Public API: pdf / cdf / icdf.
  # ------------------------------------------------------------------

  def pdf(self, w: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Return ``f(y_i | w_i)`` for each row ``i``.

    ``w`` and ``y`` must have the same number of rows.  Out-of-support
    queries (``z`` outside the predicted quantile range) return ``0``.
    """
    w_arr = _as_2d(w)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if len(y_arr) != w_arr.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_arr)
    q_sorted, alphas = self._predict_quantile_table(w_arr)
    n_obs = w_arr.shape[0]

    dens_z = np.empty(n_obs, dtype=float)

    for i in range(n_obs):
      qi = q_sorted[i]
      dq_da = np.gradient(qi, alphas)
      dq_da = np.maximum(dq_da, self.config.min_qprime)
      f_at_q = 1.0 / dq_da

      z_i = z[i]
      if z_i <= qi[0] or z_i >= qi[-1]:
        dens_z[i] = 0.0
      else:
        alpha_i = np.interp(z_i, qi, alphas)
        dens_z[i] = np.interp(alpha_i, alphas, f_at_q)

    return dens_z * self._jacobian_inverse(y_arr)

  def cdf(self, w: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Return ``F(y_i | w_i) = P(Y <= y_i | W = w_i)`` per row.

    Computed by inverting the quantile table: ``F(y | w) = α`` such
    that ``Q(α | w) = transform(y)``.  Linear interpolation between
    grid alphas; ``np.interp``'s flat extrapolation outside the
    empirical quantile range yields ``alpha_min`` / ``alpha_max``,
    matching the support of the alpha grid.

    No Jacobian correction: monotone transforms preserve the CDF, so
    ``F_Y(y | w) = F_Z(transform(y) | w)``.
    """
    w_arr = _as_2d(w)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if len(y_arr) != w_arr.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_arr)
    q_sorted, alphas = self._predict_quantile_table(w_arr)
    n_obs = w_arr.shape[0]

    cdf_z = np.empty(n_obs, dtype=float)
    for i in range(n_obs):
      cdf_z[i] = np.interp(z[i], q_sorted[i], alphas)

    return cdf_z

  def icdf(self, w: ArrayLike, alphas: ArrayLike) -> np.ndarray:
    """Per-row conditional quantile ``F^{-1}(alphas_i | w_i)`` on the y-scale.

    For each row ``i``, returns ``y`` such that
    ``F(y | w_i) = alphas_i``.  Implemented as linear interpolation in
    the predicted quantile table: ``Q(alphas_i | w_i)``.  Used by the
    Rosenblatt simulation recipe behind :py:meth:`PFNRBicop.tau`.
    """
    w_arr = _as_2d(w)
    alpha_arr = np.asarray(alphas, dtype=float).reshape(-1)
    if len(alpha_arr) != w_arr.shape[0]:
      raise ValueError("w and alphas have incompatible lengths.")
    if np.any((alpha_arr <= 0.0) | (alpha_arr >= 1.0)):
      raise ValueError("alphas must lie strictly inside (0, 1).")

    q_sorted, table_alphas = self._predict_quantile_table(w_arr)
    n_obs = w_arr.shape[0]

    z_out = np.empty(n_obs, dtype=float)
    for i in range(n_obs):
      z_out[i] = np.interp(alpha_arr[i], table_alphas, q_sorted[i])

    return self._inverse_transform(z_out)
