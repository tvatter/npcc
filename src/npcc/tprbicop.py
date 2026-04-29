"""
tprbicop.py — TabPFN-Rosenblatt conditional bivariate copula estimator.

This module provides a conditional bivariate copula density estimator
c(u, v | x) built from a Rosenblatt factorisation,

    c(u, v | x) = f_{V | U, X}(v | u, x),

where the scalar conditional density is read off TabPFN quantile
predictions.  A symmetrised variant averages with the reverse direction
f_{U | V, X}(u | v, x) to reduce ordering bias.

The density at y is recovered from the conditional quantile function
Q(alpha | w), with w = (u, x) or (v, x), via

    f(y | w) = 1 / Q'(alpha)   evaluated at   alpha = F(y | w).

Since U, V live in (0, 1), it is convenient to model a logit-transformed
response Z = logit(Y) and map back via the standard Jacobian
1 / (y (1 - y)).

The estimator wraps the local `tabpfn` package (not `tabpfn-client`).
The local package authenticates once via the TABPFN_TOKEN environment
variable; inference itself runs locally.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike
from tabpfn import TabPFNRegressor

logger = logging.getLogger(__name__)


def _as_2d(x: ArrayLike) -> np.ndarray:
  arr = np.asarray(x, dtype=float)
  if arr.ndim == 1:
    arr = arr.reshape(-1, 1)
  if arr.ndim != 2:
    raise ValueError("Expected a 1D or 2D array.")
  return arr


def _check_uv(
  u: ArrayLike, v: ArrayLike, eps: float
) -> tuple[np.ndarray, np.ndarray]:
  u_arr = np.asarray(u, dtype=float).reshape(-1)
  v_arr = np.asarray(v, dtype=float).reshape(-1)
  if u_arr.shape != v_arr.shape:
    raise ValueError("u and v must have the same shape.")
  if np.any((u_arr <= 0.0) | (u_arr >= 1.0)) or np.any(
    (v_arr <= 0.0) | (v_arr >= 1.0)
  ):
    raise ValueError("u and v must lie strictly inside (0, 1).")
  return (
    np.clip(u_arr, eps, 1.0 - eps),
    np.clip(v_arr, eps, 1.0 - eps),
  )


def _logit(p: np.ndarray) -> np.ndarray:
  return np.log(p) - np.log1p(-p)


def _sigmoid(z: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-z))


@dataclass
class QuantileDensityConfig:
  """Grid configuration for quantile-based conditional density."""

  n_quantiles: int = 201
  alpha_min: float = 1e-3
  alpha_max: float = 1.0 - 1e-3
  min_qprime: float = 1e-6
  eps: float = 1e-6

  def alphas(self) -> np.ndarray:
    if not (0.0 < self.alpha_min < self.alpha_max < 1.0):
      raise ValueError("Require 0 < alpha_min < alpha_max < 1.")
    if self.n_quantiles < 5:
      raise ValueError("n_quantiles must be at least 5.")
    return np.linspace(self.alpha_min, self.alpha_max, self.n_quantiles)


class TabPFNQuantileDensity1D:
  """
  Univariate conditional density f(Y | W = w) wrapping TabPFNRegressor.

  Estimation proceeds by

    1. requesting conditional quantiles Q(alpha | w),
    2. enforcing monotonicity by sorting (rearrangement),
    3. using f(Q(alpha) | w) = 1 / dQ(alpha | w)/dalpha.

  When ``transform="logit"``, the model is fitted to Z = logit(Y), with
  Y in (0, 1), and the density is mapped back to the Y-scale via

    f_Y(y | w) = f_Z(logit(y) | w) / (y (1 - y)).
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
    w_arr = _as_2d(w)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if len(y_arr) != w_arr.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_arr)

    self.model_ = TabPFNRegressor(**self.model_kwargs)
    self.model_.fit(w_arr, z)
    return self

  def density(self, w: ArrayLike, y: ArrayLike) -> np.ndarray:
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


class TPRBicop:
  """
  TabPFN-Rosenblatt conditional bivariate copula estimator (TPRBicop).

  Estimates a conditional bivariate copula density c(u, v | x) using
  the Rosenblatt factorisation

    c(u, v | x) = f_{V | U, X}(v | u, x).

  Symmetric variant (``symmetric=True``) averages with the reverse
  direction:

    c_sym(u, v | x)
      = 0.5 f_{V | U, X}(v | u, x) + 0.5 f_{U | V, X}(u | v, x).

  Notes
  -----
  - The asymmetric estimator enforces int c(u, v | x) dv = 1 by
    construction.  It does not generally enforce int c(u, v | x) du = 1.
  - The symmetric estimator reduces ordering bias but still does not
    impose exact uniform copula margins.  Add a grid IPF / Sinkhorn
    projection if that is required.
  - Authentication for the local ``tabpfn`` package is read from the
    ``TABPFN_TOKEN`` environment variable.  Set it before calling
    :py:meth:`fit`.
  """

  def __init__(
    self,
    *,
    symmetric: bool = True,
    density_config: QuantileDensityConfig | None = None,
    model_kwargs: dict[str, Any] | None = None,
  ) -> None:
    self.symmetric = symmetric
    self.density_config = density_config or QuantileDensityConfig()
    self.model_kwargs = model_kwargs or {}

    self.v_given_ux_ = TabPFNQuantileDensity1D(
      transform="logit",
      config=self.density_config,
      model_kwargs=self.model_kwargs,
    )
    self.u_given_vx_: TabPFNQuantileDensity1D | None = (
      TabPFNQuantileDensity1D(
        transform="logit",
        config=self.density_config,
        model_kwargs=self.model_kwargs,
      )
      if symmetric
      else None
    )

  @staticmethod
  def _features(first_coord: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.column_stack([first_coord, x])

  @staticmethod
  def _default_x(n: int) -> np.ndarray:
    return np.ones((n, 1), dtype=float)

  def fit(
    self,
    u: ArrayLike,
    v: ArrayLike,
    x: ArrayLike | None = None,
  ) -> Self:
    u_arr, v_arr = _check_uv(u, v, self.density_config.eps)
    x_arr = self._default_x(len(u_arr)) if x is None else _as_2d(x)

    if x_arr.shape[0] != len(u_arr):
      raise ValueError("x, u, and v must have the same number of observations.")

    self.v_given_ux_.fit(self._features(u_arr, x_arr), v_arr)

    if self.symmetric:
      assert self.u_given_vx_ is not None
      self.u_given_vx_.fit(self._features(v_arr, x_arr), u_arr)

    return self

  def density(
    self,
    u: ArrayLike,
    v: ArrayLike,
    x: ArrayLike | None = None,
  ) -> np.ndarray:
    u_arr, v_arr = _check_uv(u, v, self.density_config.eps)
    x_arr = self._default_x(len(u_arr)) if x is None else _as_2d(x)

    if x_arr.shape[0] != len(u_arr):
      raise ValueError("x, u, and v must have the same number of observations.")

    c_v_given_u = self.v_given_ux_.density(self._features(u_arr, x_arr), v_arr)

    if not self.symmetric:
      return c_v_given_u

    assert self.u_given_vx_ is not None
    c_u_given_v = self.u_given_vx_.density(self._features(v_arr, x_arr), u_arr)
    return 0.5 * (c_v_given_u + c_u_given_v)

  def log_density(
    self,
    u: ArrayLike,
    v: ArrayLike,
    x: ArrayLike | None = None,
  ) -> np.ndarray:
    c = self.density(u, v, x)
    return np.log(np.maximum(c, np.finfo(float).tiny))

  def conditional_cdf_v_given_u(
    self,
    u: ArrayLike,
    v_grid: ArrayLike,
    x: ArrayLike | None = None,
  ) -> np.ndarray:
    """
    Numerical CDF estimate for C_{V | U, X}(v | u, x) on a fixed grid.

    Mainly a diagnostic.  For production use, expose TabPFN quantiles
    directly or fit a calibrated monotone CDF smoother.
    """
    u_arr = np.asarray(u, dtype=float).reshape(-1)
    x_arr = self._default_x(len(u_arr)) if x is None else _as_2d(x)
    v_arr = np.asarray(v_grid, dtype=float).reshape(-1)

    if len(u_arr) != x_arr.shape[0]:
      raise ValueError("u and x must have the same number of observations.")
    if np.any(np.diff(v_arr) <= 0):
      raise ValueError("v_grid must be strictly increasing.")
    if v_arr[0] <= 0.0 or v_arr[-1] >= 1.0:
      raise ValueError("v_grid must lie strictly inside (0, 1).")

    out = np.empty((len(u_arr), len(v_arr)), dtype=float)

    for i in range(len(u_arr)):
      ui = np.repeat(u_arr[i], len(v_arr))
      xi = np.repeat(x_arr[i : i + 1], len(v_arr), axis=0)
      dens = self.v_given_ux_.density(self._features(ui, xi), v_arr)
      cdf = np.concatenate(
        [
          np.array([0.0]),
          np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(v_arr)),
        ]
      )
      if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
      out[i] = np.clip(cdf, 0.0, 1.0)

    return out
