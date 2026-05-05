"""
tabpfn_quantile_density1d.py — univariate conditional density via TabPFN
conditional quantiles.

Approach
--------
Given a TabPFN regressor trained to predict ``Y`` given features ``W``,
we ask the regressor for the conditional quantile function on a fine
grid of cumulative probabilities

    Q(alpha | w),    alpha in {alpha_1, ..., alpha_K},

then recover the conditional density by inverting the slope of ``Q``::

    f(y | w) = 1 / Q'(alpha)    evaluated at    alpha = F(y | w),

where ``F`` is the conditional CDF.  Numerically:

  1. **Monotone rearrangement.**  Sort the predicted quantiles per
     observation so they are non-decreasing.  TabPFN's quantile output
     is not guaranteed to be monotone for tightly spaced ``alpha``.
  2. **Numerical derivative.**  Compute ``dQ/dalpha`` with
     :func:`numpy.gradient`, clipped to ``min_qprime`` to avoid the
     singular ``1 / 0`` that would otherwise appear at constant
     plateaus of the quantile curve.
  3. **Locate alpha.**  For a query point ``y``, interpolate
     ``alpha(y) = F(y | w)`` from the (Q, alpha) table.
  4. **Read off the density.**  Interpolate ``f_at_q = 1 / Q'`` at
     ``alpha(y)``.

Logit support transform
-----------------------
``U`` and ``V`` are copula scores in ``(0, 1)`` and quantile estimation
on a bounded interval is awkward.  When ``transform="logit"`` we instead
fit the regressor on ``Z = logit(Y)`` (the unbounded image) and convert
back to the ``Y`` scale via the standard Jacobian::

    f_Y(y | w) = f_Z(logit(y) | w) / (y * (1 - y)).

Trade-offs
----------
This recovery is purely numerical and works with any quantile regressor;
it does not require access to TabPFN's internal distribution head.  The
counterpart in :mod:`npcc.tabpfn_density1d` calls TabPFN's native
``criterion.pdf`` directly, which is faster and avoids the slope
inversion, but is specific to TabPFN's "full" output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from npcc._common import _as_2d, _logit

logger = logging.getLogger(__name__)


@dataclass
class QuantileDensityConfig:
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


class TabPFNQuantileDensity1D:
  """Univariate conditional density ``f(Y | W=w)`` from TabPFN quantiles.

  See the module-level docstring for the full algorithm.  The ``fit`` /
  ``density`` API mirrors :class:`TabPFNDensity1D` so the two classes are
  drop-in interchangeable inside :class:`PFNRBicop`.

  Parameters
  ----------
  transform
      ``"identity"`` fits TabPFN directly on ``Y``.  ``"logit"`` fits on
      ``Z = logit(Y)`` and applies the inverse Jacobian on the way out;
      this is the only sensible choice when ``Y`` is bounded in
      ``(0, 1)``, which is always the case for copula scores.
  config
      Quantile-grid configuration.  Defaults to :class:`QuantileDensityConfig`.
  model_kwargs
      Forwarded to :py:meth:`TabPFNRegressor.create_default_for_version`.
      Useful for ``device=...``, ``n_estimators=...``, etc.
  """

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit"] = "logit",
    config: QuantileDensityConfig | None = None,
    model_kwargs: dict[str, Any] | None = None,
  ) -> None:
    self.transform = transform
    self.config = config or QuantileDensityConfig()
    self.model_kwargs = model_kwargs or {}
    self.model_: TabPFNRegressor | None = None

  def _transform_y(self, y: np.ndarray) -> np.ndarray:
    if self.transform == "identity":
      return y
    if self.transform == "logit":
      y_clip = np.clip(y, self.config.eps, 1.0 - self.config.eps)
      return _logit(y_clip)
    raise ValueError(f"Unknown transform: {self.transform}")

  def _jacobian_inverse(self, y: np.ndarray) -> np.ndarray:
    if self.transform == "identity":
      return np.ones_like(y)
    if self.transform == "logit":
      y_clip = np.clip(y, self.config.eps, 1.0 - self.config.eps)
      return 1.0 / (y_clip * (1.0 - y_clip))
    raise ValueError(f"Unknown transform: {self.transform}")

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

  def density(self, w: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Return ``f(y_i | w_i)`` for each row ``i``.

    ``w`` and ``y`` must have the same number of rows.  Out-of-support
    queries (``z`` outside the predicted quantile range) return ``0``.
    """
    if self.model_ is None:
      raise RuntimeError("The model is not fitted.")

    w_arr = _as_2d(w)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if len(y_arr) != w_arr.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_arr)
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

    dens_z = np.empty(n_obs, dtype=float)

    for i in range(n_obs):
      qi = np.sort(q[i])
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
