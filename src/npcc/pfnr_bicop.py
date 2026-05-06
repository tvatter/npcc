"""
pfnr_bicop.py — TabPFN-Rosenblatt conditional bivariate copula.

Approach
--------
A bivariate copula density factorises through the Rosenblatt
construction.  Conditioning on covariates ``X`` and exploiting the
uniform-margin property of the copula scale ``U``::

    c(u, v | x) = f_{V | U, X}(v | u, x).

So estimating a *conditional bivariate copula density* reduces to
estimating a *univariate conditional density*, which is exactly what
:class:`TabPFNDensity1D` and :class:`TabPFNQuantileDensity1D` provide.
The features fed to the inner regressor are::

    W = [u, x]    (when predicting V | U, X)

so the inner module sees the conditioning copula score and the
covariates side by side.

Symmetric variant
-----------------
The naive estimator factorises in a single direction and is therefore
ordering-dependent: by construction it satisfies
``int c(u, v | x) dv = 1`` but generally not
``int c(u, v | x) du = 1``.  Setting ``symmetric=True`` (the default)
fits the reverse direction as well and averages::

    c_sym(u, v | x) =
        0.5 * f_{V | U, X}(v | u, x)
      + 0.5 * f_{U | V, X}(u | v, x).

This reduces the asymmetry but does not impose exact uniform copula
margins.  If exact margins are required, evaluate ``c_sym`` on a grid
and apply an iterative-proportional-fitting / Sinkhorn projection.

Density-recovery method
-----------------------
``method="criterion"`` (default) uses :class:`TabPFNDensity1D`, which
evaluates the conditional density directly via TabPFN's binned
distribution head.  ``method="quantiles"`` uses
:class:`TabPFNQuantileDensity1D`, which queries a conditional quantile
grid and inverts the slope.  The two methods are interchangeable and
share the same outer ``fit`` / ``density`` API, but only the criterion
method exposes the ``density_grid`` Cartesian-product fast path.

Plotting
--------
:class:`PFNRBicop` doubles as a duck-typed bivariate copula via
:py:meth:`as_bicop`, which returns an object exposing
``var_types = ["c", "c"]`` and ``pdf(uv)`` — exactly what
``pyvinecopulib`` plotting helpers expect.  The convenience
:py:meth:`plot` method wraps that adapter and calls
``pyvinecopulib._python_helpers.bicop.bicop_plot`` to render contour or
surface plots in the same style used elsewhere in the copula
ecosystem.  Cartesian-grid queries (which is what plotting always
produces) automatically take the fast :py:meth:`density_grid` path.

Authentication
--------------
The local ``tabpfn`` package authenticates once via the
``TABPFN_TOKEN`` environment variable and then runs locally.  Set the
token before calling :py:meth:`fit`.  Model weights for TabPFN-v2.5 are
pulled from HuggingFace on first use into the platform cache directory.
"""

from __future__ import annotations

from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike

from npcc._common import _as_2d, _check_uv
from npcc.tabpfn_density1d import TabPFNDensity1D
from npcc.tabpfn_quantile_density1d import (
  QuantileDensityConfig,
  TabPFNQuantileDensity1D,
)

_DensityModule = TabPFNQuantileDensity1D | TabPFNDensity1D


class PFNRBicop:
  """TabPFN-based Rosenblatt conditional bivariate copula estimator.

  Parameters
  ----------
  symmetric
      If ``True`` (default), also fit the reverse Rosenblatt direction
      and return the average of the two density estimates.
  method
      ``"criterion"`` (default) → :class:`TabPFNDensity1D` (direct PDF
      via TabPFN's binned head).  ``"quantiles"`` →
      :class:`TabPFNQuantileDensity1D` (numerical inversion of the
      conditional quantile slope).
  density_config
      :class:`QuantileDensityConfig` instance.  Only its ``eps`` field
      is used by the criterion method (for clipping).  The quantile
      method additionally uses the alpha-grid fields.
  model_kwargs
      Forwarded into the inner ``TabPFNRegressor`` (via
      :py:meth:`TabPFNRegressor.create_default_for_version`).  Useful
      for ``device=...``, ``n_estimators=...``, etc.

  Notes
  -----
  - The asymmetric estimator (``symmetric=False``) enforces
    ``int c(u, v | x) dv = 1`` by construction and is the cheaper
    option (one direction fit, one direction at evaluation).
  - The symmetric estimator reduces the directional bias but doubles
    both fit and inference cost.
  """

  def __init__(
    self,
    *,
    symmetric: bool = True,
    method: Literal["criterion", "quantiles"] = "criterion",
    density_config: QuantileDensityConfig | None = None,
    model_kwargs: dict[str, Any] | None = None,
  ) -> None:
    self.symmetric = symmetric
    self.method = method
    self.density_config = density_config or QuantileDensityConfig()
    self.model_kwargs = model_kwargs or {}

    self.v_given_ux_: _DensityModule = self._make_density_module()
    self.u_given_vx_: _DensityModule | None = (
      self._make_density_module() if symmetric else None
    )

  def _make_density_module(self) -> _DensityModule:
    if self.method == "quantiles":
      return TabPFNQuantileDensity1D(
        transform="logit",
        config=self.density_config,
        model_kwargs=self.model_kwargs,
      )
    return TabPFNDensity1D(
      transform="logit",
      eps=self.density_config.eps,
      model_kwargs=self.model_kwargs,
    )

  @staticmethod
  def _features(first_coord: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Build the inner regressor's feature matrix ``[first_coord | x]``."""
    return np.column_stack([first_coord, x])

  @staticmethod
  def _default_x(n: int) -> np.ndarray:
    """Constant covariate column of ones (used when ``x`` is omitted)."""
    return np.ones((n, 1), dtype=float)

  def fit(
    self,
    u: ArrayLike,
    v: ArrayLike,
    x: ArrayLike | None = None,
  ) -> Self:
    """Fit the inner conditional density estimator(s).

    Fits ``f(V | U, X)``.  When ``symmetric=True`` also fits
    ``f(U | V, X)``.  ``x=None`` is shorthand for the unconditional
    case (a constant covariate column).
    """
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
    """Return the conditional copula density ``c(u_i, v_i | x_i)``."""
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
    """Log of :py:meth:`density`, floored at the smallest positive float."""
    c = self.density(u, v, x)
    return np.log(np.maximum(c, np.finfo(float).tiny))

  def density_grid(
    self,
    u_grid: ArrayLike,
    v_grid: ArrayLike,
    x_row: ArrayLike | None = None,
  ) -> np.ndarray:
    """Density on the Cartesian product ``out[i, j] = c(u_grid[i], v_grid[j] | x)``.

    Requires ``method="criterion"`` because it relies on
    :py:meth:`TabPFNDensity1D.density_grid`.  ``x_row`` is a single
    covariate row reused on both axes; when ``None`` a constant
    one-column row is used.

    For the symmetric estimator both directions are evaluated on the
    same Cartesian product (transposing the reverse one) and averaged.
    """
    if self.method != "criterion" or not isinstance(
      self.v_given_ux_, TabPFNDensity1D
    ):
      raise RuntimeError(
        "density_grid is only available when method='criterion'."
      )

    u_arr = np.asarray(u_grid, dtype=float).reshape(-1)
    v_arr = np.asarray(v_grid, dtype=float).reshape(-1)
    if np.any((u_arr <= 0.0) | (u_arr >= 1.0)) or np.any(
      (v_arr <= 0.0) | (v_arr >= 1.0)
    ):
      raise ValueError("u_grid and v_grid must lie strictly inside (0, 1).")

    eps = self.density_config.eps
    u_arr = np.clip(u_arr, eps, 1.0 - eps)
    v_arr = np.clip(v_arr, eps, 1.0 - eps)

    if x_row is None:
      x_row_arr = np.ones((1, 1), dtype=float)
    else:
      x_row_arr = _as_2d(x_row)
      if x_row_arr.shape[0] != 1:
        raise ValueError("x_row must contain exactly one row.")

    x_for_u = np.repeat(x_row_arr, len(u_arr), axis=0)
    c_v_given_u = self.v_given_ux_.density_grid(
      self._features(u_arr, x_for_u), v_arr
    )

    if not self.symmetric:
      return c_v_given_u

    assert isinstance(self.u_given_vx_, TabPFNDensity1D)
    x_for_v = np.repeat(x_row_arr, len(v_arr), axis=0)
    # density_grid returns (n_v, n_u); transpose to align with c_v_given_u.
    c_u_given_v = self.u_given_vx_.density_grid(
      self._features(v_arr, x_for_v), u_arr
    ).T

    return 0.5 * (c_v_given_u + c_u_given_v)

  # -------------------------------------------------------------------
  # h-functions (conditional CDFs along one axis)
  #
  # We follow pyvinecopulib's numbering convention: ``hfunc_i``
  # conditions on the i-th argument:
  #
  #   hfunc1(u, v | x) = P(V <= v | U = u, X = x) = F_{V | U, X}(v|u,x)
  #   hfunc2(u, v | x) = P(U <= u | V = v, X = x) = F_{U | V, X}(u|v,x)
  #
  # Equivalently, hfunc1 = ∂C/∂u and hfunc2 = ∂C/∂v.
  # -------------------------------------------------------------------

  def hfunc1(
    self,
    u: ArrayLike,
    v: ArrayLike,
    x: ArrayLike | None = None,
  ) -> np.ndarray:
    """``h_1(u, v | x) = P(V ≤ v | U = u, X = x) = F_{V | U, X}(v | u, x)``.

    Always available (the V|U regressor is always fitted).  This is a
    direct read of the inner regressor's conditional CDF — no
    integration, one batched ``criterion.cdf`` (or quantile-table
    interpolation) call.

    Convention matches :py:meth:`pyvinecopulib.Bicop.hfunc1`:
    ``hfunc1`` conditions on the first argument.
    """
    u_arr, v_arr = _check_uv(u, v, self.density_config.eps)
    x_arr = self._default_x(len(u_arr)) if x is None else _as_2d(x)

    if x_arr.shape[0] != len(u_arr):
      raise ValueError("x, u, and v must have the same number of observations.")

    return self.v_given_ux_.cdf(self._features(u_arr, x_arr), v_arr)

  def hfunc2(
    self,
    u: ArrayLike,
    v: ArrayLike,
    x: ArrayLike | None = None,
  ) -> np.ndarray:
    """``h_2(u, v | x) = P(U ≤ u | V = v, X = x) = F_{U | V, X}(u | v, x)``.

    Requires ``symmetric=True`` (the U|V regressor must have been
    fitted).  For ``symmetric=False`` this raises with a clear pointer
    to the workaround: fit a second :class:`PFNRBicop` with
    ``(u, v)`` swapped.

    Convention matches :py:meth:`pyvinecopulib.Bicop.hfunc2`:
    ``hfunc2`` conditions on the second argument.
    """
    if self.u_given_vx_ is None:
      raise RuntimeError(
        "hfunc2 requires symmetric=True. Refit with "
        "PFNRBicop(symmetric=True), or fit a second PFNRBicop with "
        "(u, v) arguments swapped."
      )

    u_arr, v_arr = _check_uv(u, v, self.density_config.eps)
    x_arr = self._default_x(len(u_arr)) if x is None else _as_2d(x)

    if x_arr.shape[0] != len(u_arr):
      raise ValueError("x, u, and v must have the same number of observations.")

    return self.u_given_vx_.cdf(self._features(v_arr, x_arr), u_arr)

  # -------------------------------------------------------------------
  # Joint CDF
  # -------------------------------------------------------------------

  def cdf(
    self,
    u: ArrayLike,
    v: ArrayLike,
    x: ArrayLike | None = None,
    *,
    n_int: int = 64,
  ) -> np.ndarray:
    """Joint CDF ``C(u_i, v_i | x_i)`` evaluated row-by-row.

    Trapezoidal integration of the inner conditional CDF over the
    Rosenblatt direction:

        C(u, v | x) = ∫_0^u F_{V | U, X}(v | s, x) ds       (asymmetric)
        C^sym(u, v | x) = 0.5 (∫_0^u F_{V|U,X}(v|s,x) ds
                               + ∫_0^v F_{U|V,X}(u|t,x) dt) (symmetric)

    ``n_int`` is the number of trapezoid steps along the integration
    axis; 64 is plenty for the typical bivariate copula.
    """
    if n_int < 2:
      raise ValueError("n_int must be at least 2.")

    u_arr, v_arr = _check_uv(u, v, self.density_config.eps)
    x_arr = self._default_x(len(u_arr)) if x is None else _as_2d(x)

    if x_arr.shape[0] != len(u_arr):
      raise ValueError("x, u, and v must have the same number of observations.")

    cdf_v_dir = self._integrate_one_direction(
      upper=u_arr,
      conditioned=v_arr,
      x=x_arr,
      module=self.v_given_ux_,
      first_arg_is_integration_var=True,
      n_int=n_int,
    )

    if not self.symmetric:
      return cdf_v_dir

    assert self.u_given_vx_ is not None
    cdf_u_dir = self._integrate_one_direction(
      upper=v_arr,
      conditioned=u_arr,
      x=x_arr,
      module=self.u_given_vx_,
      first_arg_is_integration_var=True,
      n_int=n_int,
    )
    return 0.5 * (cdf_v_dir + cdf_u_dir)

  def _integrate_one_direction(
    self,
    *,
    upper: np.ndarray,
    conditioned: np.ndarray,
    x: np.ndarray,
    module: _DensityModule,
    first_arg_is_integration_var: bool,
    n_int: int,
  ) -> np.ndarray:
    """Compute ∫_eps^{upper_i} F(conditioned_i | s, x_i) ds for each row.

    ``first_arg_is_integration_var=True`` builds features as
    ``[s | x_i]``, matching the inner regressor's expectation that the
    first feature column is the conditioning copula coordinate.
    """
    eps = self.density_config.eps
    n = len(upper)
    upper_safe = np.maximum(upper, eps + 1e-12)

    # s_grids[i, k] = linspace(eps, upper_safe[i], n_int+1)[k]
    s_grids = np.linspace(eps, upper_safe, n_int + 1).T  # (n, n_int+1)

    s_flat = s_grids.flatten()
    cond_flat = np.repeat(conditioned, n_int + 1)
    x_flat = np.repeat(x, n_int + 1, axis=0)

    if first_arg_is_integration_var:
      feats = self._features(s_flat, x_flat)
    else:  # pragma: no cover - kept for future asymmetric h₁ extension
      feats = self._features(cond_flat, x_flat)

    F_flat = module.cdf(feats, cond_flat)
    F_grid = F_flat.reshape(n, n_int + 1)

    # Per-row trapezoidal integral over the s axis.
    ds = np.diff(s_grids, axis=1)  # (n, n_int)
    avgs = 0.5 * (F_grid[:, :-1] + F_grid[:, 1:])  # (n, n_int)
    return np.sum(avgs * ds, axis=1)

  def cdf_grid(
    self,
    u_grid: ArrayLike,
    v_grid: ArrayLike,
    x_row: ArrayLike | None = None,
    *,
    n_int: int = 64,
  ) -> np.ndarray:
    """Cartesian-grid joint CDF ``out[i, j] = C(u_grid[i], v_grid[j] | x_row)``.

    Requires ``method="criterion"`` (uses the inner ``cdf_grid`` fast
    path).  Builds a single shared fine ``s``-grid covering
    ``[eps, max(u_grid)]``, evaluates the inner CDF on the
    Cartesian product ``(s_fine × v_grid)`` in one TabPFN forward
    pass per row of ``s_fine``, then for each ``u_grid[i]`` reads off
    the cumulative trapezoidal integral up to ``u_grid[i]`` via
    interpolation. Symmetric case averages the analogous ``v``-axis
    integral.
    """
    if self.method != "criterion" or not isinstance(
      self.v_given_ux_, TabPFNDensity1D
    ):
      raise RuntimeError("cdf_grid is only available when method='criterion'.")
    if n_int < 2:
      raise ValueError("n_int must be at least 2.")

    u_arr = np.asarray(u_grid, dtype=float).reshape(-1)
    v_arr = np.asarray(v_grid, dtype=float).reshape(-1)
    if np.any((u_arr <= 0.0) | (u_arr >= 1.0)) or np.any(
      (v_arr <= 0.0) | (v_arr >= 1.0)
    ):
      raise ValueError("u_grid and v_grid must lie strictly inside (0, 1).")

    eps = self.density_config.eps
    u_arr = np.clip(u_arr, eps, 1.0 - eps)
    v_arr = np.clip(v_arr, eps, 1.0 - eps)

    if x_row is None:
      x_row_arr = np.ones((1, 1), dtype=float)
    else:
      x_row_arr = _as_2d(x_row)
      if x_row_arr.shape[0] != 1:
        raise ValueError("x_row must contain exactly one row.")

    cdf_v_dir = self._integrate_grid_one_direction(
      upper_grid=u_arr,
      conditioned_grid=v_arr,
      x_row=x_row_arr,
      module=self.v_given_ux_,
      n_int=n_int,
    )

    if not self.symmetric:
      return cdf_v_dir

    assert isinstance(self.u_given_vx_, TabPFNDensity1D)
    # Returns (n_v, n_u); transpose to align with cdf_v_dir.
    cdf_u_dir = self._integrate_grid_one_direction(
      upper_grid=v_arr,
      conditioned_grid=u_arr,
      x_row=x_row_arr,
      module=self.u_given_vx_,
      n_int=n_int,
    ).T

    return 0.5 * (cdf_v_dir + cdf_u_dir)

  def _integrate_grid_one_direction(
    self,
    *,
    upper_grid: np.ndarray,
    conditioned_grid: np.ndarray,
    x_row: np.ndarray,
    module: TabPFNDensity1D,
    n_int: int,
  ) -> np.ndarray:
    """Compute ∫_0^{upper_grid[i]} F(conditioned_grid[j] | s, x_row) ds.

    Returns shape ``(len(upper_grid), len(conditioned_grid))``.
    """
    eps = self.density_config.eps
    n_u, n_v = len(upper_grid), len(conditioned_grid)

    # Shared fine s-grid covering [eps, max(upper_grid)].
    s_fine = np.linspace(eps, max(upper_grid.max(), eps + 1e-12), n_int + 1)

    x_for_s = np.repeat(x_row, len(s_fine), axis=0)
    feats = self._features(s_fine, x_for_s)
    # F_table[j, k] = F(conditioned_grid[k] | s_fine[j], x_row)
    F_table = module.cdf_grid(feats, conditioned_grid)  # (n_int+1, n_v)

    # Cumulative trapezoid along axis=0.
    ds = np.diff(s_fine)  # (n_int,)
    avgs = 0.5 * (F_table[:-1] + F_table[1:])  # (n_int, n_v)
    cum = np.zeros((len(s_fine), n_v))
    cum[1:] = np.cumsum(avgs * ds[:, None], axis=0)

    # For each upper_grid[i], interpolate cum at s = upper_grid[i].
    out = np.empty((n_u, n_v), dtype=float)
    for j in range(n_v):
      out[:, j] = np.interp(upper_grid, s_fine, cum[:, j])
    return out

  # -------------------------------------------------------------------
  # Kendall's tau (sample-based, mirroring pyvinecopulib's KernelBicop)
  # -------------------------------------------------------------------

  # Default seeds used by pyvinecopulib's
  # ``KernelBicop::parameters_to_tau``.  Reusing them gives
  # byte-identical reproducibility against vinecopulib.
  _GHALTON_DEFAULT_SEEDS: tuple[int, ...] = (
    204967043,
    733593603,
    184618802,
    399707801,
    290266245,
  )

  def tau(
    self,
    x_row: ArrayLike | None = None,
    *,
    n: int = 1000,
    seeds: list[int] | None = None,
  ) -> float:
    """Kendall's τ via the recipe used by ``pv.KernelBicop::parameters_to_tau``.

    1. Draw a deterministic 2-D Generalised-Halton quasi-random sample
       ``(u_i, alpha_i)`` of size ``n`` via :func:`pyvinecopulib.ghalton`.
    2. Apply the inverse Rosenblatt transform along the first axis:
       ``v_i = F_{V | U, X}^{-1}(alpha_i | u_i, x_row)``.  The
       resulting ``(u_i, v_i)`` pairs are distributed according to the
       fitted copula.
    3. Return the weighted (rank-)Kendall ``τ`` of the sample via
       :func:`pyvinecopulib.wdm`.

    Quasi-random sampling and the closed-form ``criterion.icdf`` (or
    interpolation in the quantile table) make this much more accurate
    than a grid-based integral, especially when the copula has heavy
    tail dependence (e.g. Clayton near the lower-left corner).
    Available for both density-recovery methods.
    """
    if n < 10:
      raise ValueError("n must be at least 10.")

    if seeds is None:
      seeds_list = list(self._GHALTON_DEFAULT_SEEDS)
    else:
      seeds_list = list(seeds)

    from pyvinecopulib import ghalton, wdm

    quasi = np.asarray(ghalton(n, 2, seeds_list), dtype=float)
    u = np.clip(
      quasi[:, 0], self.density_config.eps, 1.0 - self.density_config.eps
    )
    alpha = np.clip(
      quasi[:, 1], self.density_config.eps, 1.0 - self.density_config.eps
    )

    if x_row is None:
      x_arr = self._default_x(n)
    else:
      x_row_arr = _as_2d(x_row)
      if x_row_arr.shape[0] != 1:
        raise ValueError("x_row must contain exactly one row.")
      x_arr = np.repeat(x_row_arr, n, axis=0)

    # Inverse Rosenblatt: v = F_{V | U, X}^{-1}(alpha | u, x).
    v = self.v_given_ux_.icdf(self._features(u, x_arr), alpha)

    return float(wdm(u, v, "tau"))

  # -------------------------------------------------------------------
  # Diagnostic CDF (kept for backward compatibility)
  # -------------------------------------------------------------------

  def conditional_cdf_v_given_u(
    self,
    u: ArrayLike,
    v_grid: ArrayLike,
    x: ArrayLike | None = None,
  ) -> np.ndarray:
    """``C_{V | U, X}(v_grid[j] | u_i, x_i)`` on a grid of ``v`` values.

    Thin wrapper over :py:meth:`hfunc1` (which conditions on ``u`` per
    pyvinecopulib convention) that broadcasts each ``u_i`` against
    ``v_grid``.  Inputs are validated to lie strictly inside
    ``(0, 1)`` and ``v_grid`` must be strictly increasing.
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

    n_u, n_v = len(u_arr), len(v_arr)
    u_flat = np.repeat(u_arr, n_v)
    v_flat = np.tile(v_arr, n_u)
    x_flat = np.repeat(x_arr, n_v, axis=0)
    return self.hfunc1(u_flat, v_flat, x_flat).reshape(n_u, n_v)

  # -------------------------------------------------------------------
  # pyvinecopulib-compatible plotting interface.
  # -------------------------------------------------------------------

  def as_bicop(self, x_row: ArrayLike | None = None) -> _BicopAdapter:
    """Return a duck-typed bivariate-copula adapter bound to ``x_row``.

    The returned object exposes ``var_types = ["c", "c"]`` and a
    ``pdf(uv)`` method matching the interface that pyvinecopulib
    plotting and CDF utilities expect from a bivariate copula.  Pass it
    directly to ``pyvinecopulib._python_helpers.bicop.bicop_plot``.

    Parameters
    ----------
    x_row
        Single covariate row (shape ``(1, p)``) reused on every queried
        ``(u, v)`` pair.  ``None`` keeps the constant-one default used
        when fitting without covariates.
    """
    return _BicopAdapter(self, x_row=x_row)

  def plot(
    self,
    *,
    x_row: ArrayLike | None = None,
    plot_type: Literal["contour", "surface"] = "contour",
    margin_type: Literal["unif", "norm", "exp"] = "norm",
    grid_size: int | None = None,
    xylim: tuple[float, float] | None = None,
  ) -> None:
    """Render a copula contour or surface plot of the fitted density.

    Lazily imports ``pyvinecopulib._python_helpers.bicop.bicop_plot``
    (which itself pulls in ``matplotlib``) so the rest of npcc can run
    in a headless / matplotlib-free environment.

    Parameters
    ----------
    x_row
        Single covariate row to condition on.  ``None`` uses the
        constant-one default — appropriate when the model was fit
        without covariates.
    plot_type
        ``"contour"`` (default) or ``"surface"``.
    margin_type
        ``"unif"``, ``"norm"`` (default), or ``"exp"``.  Selects which
        margin transform pyvinecopulib uses for the axes.
    grid_size, xylim
        Forwarded to pyvinecopulib's ``bicop_plot``.
    """
    from pyvinecopulib._python_helpers.bicop import bicop_plot

    bicop_plot(
      self.as_bicop(x_row=x_row),
      plot_type=plot_type,
      margin_type=margin_type,
      xylim=xylim,
      grid_size=grid_size,
    )


class _BicopAdapter:
  """pyvinecopulib-compatible adapter around a fitted :class:`PFNRBicop`.

  pyvinecopulib's plotting helpers call ``cop.pdf(uv)`` on a flattened
  Cartesian grid and inspect ``cop.var_types``.  This class exposes
  those, while delegating density evaluation to the wrapped
  :class:`PFNRBicop`.  An optional bound covariate row ``x_row`` is
  reused across all queried points, which is what makes a fixed
  conditional copula plottable as if it were unconditional.

  When the underlying model uses ``method="criterion"`` and the
  queried ``(u, v)`` pairs span a Cartesian product (``len(unique_u) *
  len(unique_v) == len(uv)``), :py:meth:`pdf` takes the much faster
  :py:meth:`PFNRBicop.density_grid` path under the hood.  This is
  always the case for the regular grids built by pyvinecopulib's
  plotter.
  """

  var_types: list[str] = ["c", "c"]

  def __init__(self, model: PFNRBicop, x_row: ArrayLike | None = None) -> None:
    self._model = model
    if x_row is None:
      self._x_row: np.ndarray | None = None
    else:
      x_row_arr = _as_2d(x_row)
      if x_row_arr.shape[0] != 1:
        raise ValueError("x_row must contain exactly one row.")
      self._x_row = x_row_arr

  def pdf(self, uv: ArrayLike) -> np.ndarray:
    """Evaluate ``c(u_i, v_i | x_row)`` for each row of ``uv``.

    ``uv`` must have shape ``(n, 2)``.  When the underlying model uses
    the ``criterion`` method and ``uv`` happens to span a Cartesian
    product (the typical plotting case), the call is rerouted through
    :py:meth:`PFNRBicop.density_grid` for speed.
    """
    uv_arr = np.asarray(uv, dtype=float)
    if uv_arr.ndim != 2 or uv_arr.shape[1] != 2:
      raise ValueError("uv must have shape (n, 2).")
    n = uv_arr.shape[0]

    if self._model.method == "criterion":
      unique_u, inv_u = np.unique(uv_arr[:, 0], return_inverse=True)
      unique_v, inv_v = np.unique(uv_arr[:, 1], return_inverse=True)
      if len(unique_u) * len(unique_v) == n:
        grid = self._model.density_grid(unique_u, unique_v, x_row=self._x_row)
        return grid[inv_u, inv_v]

    x = None if self._x_row is None else np.repeat(self._x_row, n, axis=0)
    return self._model.density(uv_arr[:, 0], uv_arr[:, 1], x)
