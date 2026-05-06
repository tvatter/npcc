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
:class:`TabPFNCriterionDistribution1D` and :class:`TabPFNQuantileDistribution1D` provide.
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
``method="criterion"`` (default) uses :class:`TabPFNCriterionDistribution1D`, which
evaluates the conditional density directly via TabPFN's binned
distribution head.  ``method="quantiles"`` uses
:class:`TabPFNQuantileDistribution1D`, which queries a conditional quantile
grid and inverts the slope.  The two methods are interchangeable and
share the same outer ``fit`` / ``density`` API, but only the criterion
method exposes the ``pdf_grid`` Cartesian-product fast path.

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
produces) automatically take the fast :py:meth:`pdf_grid` path.

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
import torch

from npcc._common import (
  TensorLike,
  _as_2d,
  _check_uv,
  _normalize_inputs,
  _resolve_device,
  _to_tensor,
  _torch_interp,
  _wrap_output,
)
from npcc.tabpfn_criterion_distribution1d import TabPFNCriterionDistribution1D
from npcc.tabpfn_quantile_distribution1d import (
  QuantileGridConfig,
  TabPFNQuantileDistribution1D,
)

_Distribution1D = TabPFNQuantileDistribution1D | TabPFNCriterionDistribution1D


class PFNRBicop:
  """TabPFN-based Rosenblatt conditional bivariate copula estimator.

  Parameters
  ----------
  symmetric
      If ``True`` (default), also fit the reverse Rosenblatt direction
      and return the average of the two density estimates.
  method
      ``"criterion"`` (default) → :class:`TabPFNCriterionDistribution1D`
      (direct PDF via TabPFN's binned head).  ``"quantiles"`` →
      :class:`TabPFNQuantileDistribution1D` (numerical inversion of
      the conditional quantile slope).
  quantile_config
      :class:`QuantileGridConfig` instance.  Only its ``eps`` field is
      used by the criterion method (for clipping).  The quantile
      method additionally uses the alpha-grid fields.
  device
      Device for internal tensors and TabPFN inference.  ``None``
      (default) auto-selects ``cuda`` if available, else ``cpu``.
      Forwarded into the inner distributions and into TabPFN via
      ``model_kwargs["device"]``.
  model_kwargs
      Forwarded into the inner ``TabPFNRegressor`` (via
      :py:meth:`TabPFNRegressor.create_default_for_version`).  Useful
      for ``n_estimators=...``, etc.

  Notes
  -----
  - The asymmetric estimator (``symmetric=False``) enforces
    ``int c(u, v | x) dv = 1`` by construction and is the cheaper
    option (one direction fit, one direction at evaluation).
  - The symmetric estimator reduces the directional bias but doubles
    both fit and inference cost.
  - Public methods accept either NumPy arrays or torch tensors.  When
    any positional numeric input is a torch tensor, the return value
    is a torch tensor on ``device``; otherwise it is a NumPy array.
  """

  def __init__(
    self,
    *,
    symmetric: bool = True,
    method: Literal["criterion", "quantiles"] = "criterion",
    quantile_config: QuantileGridConfig | None = None,
    device: str | torch.device | None = None,
    model_kwargs: dict[str, Any] | None = None,
  ) -> None:
    self.symmetric = symmetric
    self.method = method
    self.quantile_config = quantile_config or QuantileGridConfig()
    self._device = _resolve_device(device)
    self.model_kwargs = dict(model_kwargs or {})

    self.v_given_ux_: _Distribution1D = self._make_distribution()
    self.u_given_vx_: _Distribution1D | None = (
      self._make_distribution() if symmetric else None
    )

  def _make_distribution(self) -> _Distribution1D:
    if self.method == "quantiles":
      return TabPFNQuantileDistribution1D(
        transform="logit",
        config=self.quantile_config,
        device=self._device,
        model_kwargs=self.model_kwargs,
      )
    return TabPFNCriterionDistribution1D(
      transform="logit",
      eps=self.quantile_config.eps,
      device=self._device,
      model_kwargs=self.model_kwargs,
    )

  def _features(
    self, first_coord: torch.Tensor, x: torch.Tensor
  ) -> torch.Tensor:
    """Build the inner regressor's feature matrix ``[first_coord | x]``."""
    return torch.column_stack([first_coord, x])

  def _default_x(self, n: int) -> torch.Tensor:
    """Constant covariate column of ones (used when ``x`` is omitted)."""
    return torch.ones((n, 1), dtype=torch.float64, device=self._device)

  def fit(
    self,
    u: TensorLike,
    v: TensorLike,
    x: TensorLike | None = None,
  ) -> Self:
    """Fit the inner conditional density estimator(s).

    Fits ``f(V | U, X)``.  When ``symmetric=True`` also fits
    ``f(U | V, X)``.  ``x=None`` is shorthand for the unconditional
    case (a constant covariate column).
    """
    u_t, v_t = _check_uv(u, v, self.quantile_config.eps, device=self._device)
    x_t = (
      self._default_x(u_t.shape[0])
      if x is None
      else _as_2d(x, device=self._device)
    )

    if x_t.shape[0] != u_t.shape[0]:
      raise ValueError("x, u, and v must have the same number of observations.")

    self.v_given_ux_.fit(self._features(u_t, x_t), v_t)

    if self.symmetric:
      assert self.u_given_vx_ is not None
      self.u_given_vx_.fit(self._features(v_t, x_t), u_t)

    return self

  def pdf(
    self,
    u: TensorLike,
    v: TensorLike,
    x: TensorLike | None = None,
  ) -> TensorLike:
    """Return the conditional copula density ``c(u_i, v_i | x_i)``."""
    return_as_torch, _ = _normalize_inputs(u, v, x, device=self._device)
    out = self._pdf_torch(u, v, x)
    return _wrap_output(out, return_as_torch=return_as_torch)

  def _pdf_torch(
    self,
    u: TensorLike,
    v: TensorLike,
    x: TensorLike | None,
  ) -> torch.Tensor:
    u_t, v_t = _check_uv(u, v, self.quantile_config.eps, device=self._device)
    x_t = (
      self._default_x(u_t.shape[0])
      if x is None
      else _as_2d(x, device=self._device)
    )

    if x_t.shape[0] != u_t.shape[0]:
      raise ValueError("x, u, and v must have the same number of observations.")

    c_v_given_u = self.v_given_ux_.pdf(self._features(u_t, x_t), v_t)
    assert isinstance(c_v_given_u, torch.Tensor)

    if not self.symmetric:
      return c_v_given_u

    assert self.u_given_vx_ is not None
    c_u_given_v = self.u_given_vx_.pdf(self._features(v_t, x_t), u_t)
    assert isinstance(c_u_given_v, torch.Tensor)
    return 0.5 * (c_v_given_u + c_u_given_v)

  def log_pdf(
    self,
    u: TensorLike,
    v: TensorLike,
    x: TensorLike | None = None,
  ) -> TensorLike:
    """Log of :py:meth:`pdf`, floored at the smallest positive float."""
    return_as_torch, _ = _normalize_inputs(u, v, x, device=self._device)
    c = self._pdf_torch(u, v, x)
    out = torch.log(torch.clamp(c, min=torch.finfo(c.dtype).tiny))
    return _wrap_output(out, return_as_torch=return_as_torch)

  def pdf_grid(
    self,
    u_grid: TensorLike,
    v_grid: TensorLike,
    x_row: TensorLike | None = None,
  ) -> TensorLike:
    """Density on the Cartesian product ``out[i, j] = c(u_grid[i], v_grid[j] | x)``.

    Requires ``method="criterion"`` because it relies on
    :py:meth:`TabPFNCriterionDistribution1D.pdf_grid`.  ``x_row`` is a single
    covariate row reused on both axes; when ``None`` a constant
    one-column row is used.

    For the symmetric estimator both directions are evaluated on the
    same Cartesian product (transposing the reverse one) and averaged.
    """
    if self.method != "criterion" or not isinstance(
      self.v_given_ux_, TabPFNCriterionDistribution1D
    ):
      raise RuntimeError("pdf_grid is only available when method='criterion'.")

    return_as_torch, (u_in, v_in, x_in) = _normalize_inputs(
      u_grid, v_grid, x_row, device=self._device
    )
    assert u_in is not None and v_in is not None
    u_t = u_in.reshape(-1)
    v_t = v_in.reshape(-1)
    if torch.any((u_t <= 0.0) | (u_t >= 1.0)) or torch.any(
      (v_t <= 0.0) | (v_t >= 1.0)
    ):
      raise ValueError("u_grid and v_grid must lie strictly inside (0, 1).")

    eps = self.quantile_config.eps
    u_t = torch.clamp(u_t, eps, 1.0 - eps)
    v_t = torch.clamp(v_t, eps, 1.0 - eps)

    if x_in is None:
      x_row_t = torch.ones((1, 1), dtype=torch.float64, device=self._device)
    else:
      x_row_t = _as_2d(x_in, device=self._device)
      if x_row_t.shape[0] != 1:
        raise ValueError("x_row must contain exactly one row.")

    x_for_u = x_row_t.repeat_interleave(u_t.shape[0], dim=0)
    c_v_given_u = self.v_given_ux_.pdf_grid(self._features(u_t, x_for_u), v_t)
    assert isinstance(c_v_given_u, torch.Tensor)

    if not self.symmetric:
      return _wrap_output(c_v_given_u, return_as_torch=return_as_torch)

    assert isinstance(self.u_given_vx_, TabPFNCriterionDistribution1D)
    x_for_v = x_row_t.repeat_interleave(v_t.shape[0], dim=0)
    # pdf_grid returns (n_v, n_u); transpose to align with c_v_given_u.
    c_u_given_v = self.u_given_vx_.pdf_grid(self._features(v_t, x_for_v), u_t)
    assert isinstance(c_u_given_v, torch.Tensor)

    out = 0.5 * (c_v_given_u + c_u_given_v.T)
    return _wrap_output(out, return_as_torch=return_as_torch)

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
    u: TensorLike,
    v: TensorLike,
    x: TensorLike | None = None,
  ) -> TensorLike:
    """``h_1(u, v | x) = P(V ≤ v | U = u, X = x) = F_{V | U, X}(v | u, x)``.

    Always available (the V|U regressor is always fitted).  This is a
    direct read of the inner regressor's conditional CDF — no
    integration, one batched ``criterion.cdf`` (or quantile-table
    interpolation) call.

    Convention matches :py:meth:`pyvinecopulib.Bicop.hfunc1`:
    ``hfunc1`` conditions on the first argument.
    """
    return_as_torch, _ = _normalize_inputs(u, v, x, device=self._device)
    u_t, v_t = _check_uv(u, v, self.quantile_config.eps, device=self._device)
    x_t = (
      self._default_x(u_t.shape[0])
      if x is None
      else _as_2d(x, device=self._device)
    )
    if x_t.shape[0] != u_t.shape[0]:
      raise ValueError("x, u, and v must have the same number of observations.")

    out = self.v_given_ux_.cdf(self._features(u_t, x_t), v_t)
    assert isinstance(out, torch.Tensor)
    return _wrap_output(out, return_as_torch=return_as_torch)

  def hfunc2(
    self,
    u: TensorLike,
    v: TensorLike,
    x: TensorLike | None = None,
  ) -> TensorLike:
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

    return_as_torch, _ = _normalize_inputs(u, v, x, device=self._device)
    u_t, v_t = _check_uv(u, v, self.quantile_config.eps, device=self._device)
    x_t = (
      self._default_x(u_t.shape[0])
      if x is None
      else _as_2d(x, device=self._device)
    )
    if x_t.shape[0] != u_t.shape[0]:
      raise ValueError("x, u, and v must have the same number of observations.")

    out = self.u_given_vx_.cdf(self._features(v_t, x_t), u_t)
    assert isinstance(out, torch.Tensor)
    return _wrap_output(out, return_as_torch=return_as_torch)

  # -------------------------------------------------------------------
  # Joint CDF
  # -------------------------------------------------------------------

  def cdf(
    self,
    u: TensorLike,
    v: TensorLike,
    x: TensorLike | None = None,
    *,
    n_int: int = 64,
  ) -> TensorLike:
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

    return_as_torch, _ = _normalize_inputs(u, v, x, device=self._device)
    u_t, v_t = _check_uv(u, v, self.quantile_config.eps, device=self._device)
    x_t = (
      self._default_x(u_t.shape[0])
      if x is None
      else _as_2d(x, device=self._device)
    )
    if x_t.shape[0] != u_t.shape[0]:
      raise ValueError("x, u, and v must have the same number of observations.")

    cdf_v_dir = self._integrate_one_direction(
      upper=u_t,
      conditioned=v_t,
      x=x_t,
      module=self.v_given_ux_,
      n_int=n_int,
    )

    if not self.symmetric:
      return _wrap_output(cdf_v_dir, return_as_torch=return_as_torch)

    assert self.u_given_vx_ is not None
    cdf_u_dir = self._integrate_one_direction(
      upper=v_t,
      conditioned=u_t,
      x=x_t,
      module=self.u_given_vx_,
      n_int=n_int,
    )
    return _wrap_output(
      0.5 * (cdf_v_dir + cdf_u_dir), return_as_torch=return_as_torch
    )

  def _integrate_one_direction(
    self,
    *,
    upper: torch.Tensor,
    conditioned: torch.Tensor,
    x: torch.Tensor,
    module: _Distribution1D,
    n_int: int,
  ) -> torch.Tensor:
    """Compute ∫_eps^{upper_i} F(conditioned_i | s, x_i) ds for each row."""
    eps = self.quantile_config.eps
    n = upper.shape[0]
    upper_safe = torch.clamp(upper, min=eps + 1e-12)

    # s_grids[i, k] = linspace(eps, upper_safe[i], n_int+1)[k]
    t = torch.linspace(0.0, 1.0, n_int + 1, device=self._device)
    s_grids = eps + (upper_safe.unsqueeze(1) - eps) * t.unsqueeze(0)

    s_flat = s_grids.reshape(-1)
    cond_flat = conditioned.repeat_interleave(n_int + 1)
    x_flat = x.repeat_interleave(n_int + 1, dim=0)

    feats = self._features(s_flat, x_flat)
    F_flat = module.cdf(feats, cond_flat)
    assert isinstance(F_flat, torch.Tensor)
    F_grid = F_flat.reshape(n, n_int + 1)

    # Per-row trapezoidal integral over the s axis.
    ds = torch.diff(s_grids, dim=1)
    avgs = 0.5 * (F_grid[:, :-1] + F_grid[:, 1:])
    return torch.sum(avgs * ds, dim=1)

  def cdf_grid(
    self,
    u_grid: TensorLike,
    v_grid: TensorLike,
    x_row: TensorLike | None = None,
    *,
    n_int: int = 64,
  ) -> TensorLike:
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
      self.v_given_ux_, TabPFNCriterionDistribution1D
    ):
      raise RuntimeError("cdf_grid is only available when method='criterion'.")
    if n_int < 2:
      raise ValueError("n_int must be at least 2.")

    return_as_torch, (u_in, v_in, x_in) = _normalize_inputs(
      u_grid, v_grid, x_row, device=self._device
    )
    assert u_in is not None and v_in is not None
    u_t = u_in.reshape(-1)
    v_t = v_in.reshape(-1)
    if torch.any((u_t <= 0.0) | (u_t >= 1.0)) or torch.any(
      (v_t <= 0.0) | (v_t >= 1.0)
    ):
      raise ValueError("u_grid and v_grid must lie strictly inside (0, 1).")

    eps = self.quantile_config.eps
    u_t = torch.clamp(u_t, eps, 1.0 - eps)
    v_t = torch.clamp(v_t, eps, 1.0 - eps)

    if x_in is None:
      x_row_t = torch.ones((1, 1), dtype=torch.float64, device=self._device)
    else:
      x_row_t = _as_2d(x_in, device=self._device)
      if x_row_t.shape[0] != 1:
        raise ValueError("x_row must contain exactly one row.")

    cdf_v_dir = self._integrate_grid_one_direction(
      upper_grid=u_t,
      conditioned_grid=v_t,
      x_row=x_row_t,
      module=self.v_given_ux_,
      n_int=n_int,
    )

    if not self.symmetric:
      return _wrap_output(cdf_v_dir, return_as_torch=return_as_torch)

    assert isinstance(self.u_given_vx_, TabPFNCriterionDistribution1D)
    cdf_u_dir = self._integrate_grid_one_direction(
      upper_grid=v_t,
      conditioned_grid=u_t,
      x_row=x_row_t,
      module=self.u_given_vx_,
      n_int=n_int,
    )
    return _wrap_output(
      0.5 * (cdf_v_dir + cdf_u_dir.T), return_as_torch=return_as_torch
    )

  def _integrate_grid_one_direction(
    self,
    *,
    upper_grid: torch.Tensor,
    conditioned_grid: torch.Tensor,
    x_row: torch.Tensor,
    module: TabPFNCriterionDistribution1D,
    n_int: int,
  ) -> torch.Tensor:
    """Compute ∫_0^{upper_grid[i]} F(conditioned_grid[j] | s, x_row) ds.

    Returns shape ``(len(upper_grid), len(conditioned_grid))``.
    """
    eps = self.quantile_config.eps
    n_u, n_v = upper_grid.shape[0], conditioned_grid.shape[0]

    # Shared fine s-grid covering [eps, max(upper_grid)].
    s_max = torch.clamp(upper_grid.max(), min=eps + 1e-12)
    s_fine = torch.linspace(
      eps, float(s_max.item()), n_int + 1, device=self._device
    )

    x_for_s = x_row.repeat_interleave(s_fine.shape[0], dim=0)
    feats = self._features(s_fine, x_for_s)
    F_table = module.cdf_grid(feats, conditioned_grid)
    assert isinstance(F_table, torch.Tensor)

    # Cumulative trapezoid along axis=0.
    ds = torch.diff(s_fine)
    avgs = 0.5 * (F_table[:-1] + F_table[1:])
    cum = torch.zeros((s_fine.shape[0], n_v), device=self._device)
    cum[1:] = torch.cumsum(avgs * ds.unsqueeze(1), dim=0)

    # For each upper_grid[i], interpolate cum at s = upper_grid[i].
    out = torch.empty((n_u, n_v), dtype=torch.float64, device=self._device)
    for j in range(n_v):
      out[:, j] = _torch_interp(upper_grid, s_fine, cum[:, j])
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
    x_row: TensorLike | None = None,
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
    eps = self.quantile_config.eps
    u_np = np.clip(quasi[:, 0], eps, 1.0 - eps)
    alpha_np = np.clip(quasi[:, 1], eps, 1.0 - eps)
    u_t = torch.as_tensor(u_np, dtype=torch.float64, device=self._device)
    alpha_t = torch.as_tensor(
      alpha_np, dtype=torch.float64, device=self._device
    )

    if x_row is None:
      x_t = self._default_x(n)
    else:
      x_row_t = _as_2d(x_row, device=self._device)
      if x_row_t.shape[0] != 1:
        raise ValueError("x_row must contain exactly one row.")
      x_t = x_row_t.repeat_interleave(n, dim=0)

    # Inverse Rosenblatt: v = F_{V | U, X}^{-1}(alpha | u, x).
    v_t = self.v_given_ux_.icdf(self._features(u_t, x_t), alpha_t)
    assert isinstance(v_t, torch.Tensor)

    return float(wdm(u_np, v_t.detach().cpu().numpy(), "tau"))

  # -------------------------------------------------------------------
  # Diagnostic CDF (kept for backward compatibility)
  # -------------------------------------------------------------------

  def conditional_cdf_v_given_u(
    self,
    u: TensorLike,
    v_grid: TensorLike,
    x: TensorLike | None = None,
  ) -> TensorLike:
    """``C_{V | U, X}(v_grid[j] | u_i, x_i)`` on a grid of ``v`` values.

    Thin wrapper over :py:meth:`hfunc1` (which conditions on ``u`` per
    pyvinecopulib convention) that broadcasts each ``u_i`` against
    ``v_grid``.  Inputs are validated to lie strictly inside
    ``(0, 1)`` and ``v_grid`` must be strictly increasing.
    """
    return_as_torch, (u_in, v_in, x_in) = _normalize_inputs(
      u, v_grid, x, device=self._device
    )
    assert u_in is not None and v_in is not None
    u_t = u_in.reshape(-1)
    v_t = v_in.reshape(-1)
    x_t = (
      self._default_x(u_t.shape[0])
      if x_in is None
      else _as_2d(x_in, device=self._device)
    )

    if u_t.shape[0] != x_t.shape[0]:
      raise ValueError("u and x must have the same number of observations.")
    if torch.any(torch.diff(v_t) <= 0):
      raise ValueError("v_grid must be strictly increasing.")
    if v_t[0] <= 0.0 or v_t[-1] >= 1.0:
      raise ValueError("v_grid must lie strictly inside (0, 1).")

    n_u, n_v = u_t.shape[0], v_t.shape[0]
    u_flat = u_t.repeat_interleave(n_v)
    v_flat = v_t.tile(n_u)
    x_flat = x_t.repeat_interleave(n_v, dim=0)
    out = self.hfunc1(u_flat, v_flat, x_flat)
    assert isinstance(out, torch.Tensor)
    return _wrap_output(out.reshape(n_u, n_v), return_as_torch=return_as_torch)

  # -------------------------------------------------------------------
  # pyvinecopulib-compatible plotting interface.
  # -------------------------------------------------------------------

  def as_bicop(self, x_row: TensorLike | None = None) -> _BicopAdapter:
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
    x_row: TensorLike | None = None,
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
  :py:meth:`PFNRBicop.pdf_grid` path under the hood.  This is
  always the case for the regular grids built by pyvinecopulib's
  plotter.

  The adapter is the pyvinecopulib boundary, so its ``pdf`` always
  returns a NumPy array regardless of input type.
  """

  var_types: list[str] = ["c", "c"]

  def __init__(self, model: PFNRBicop, x_row: TensorLike | None = None) -> None:
    self._model = model
    if x_row is None:
      self._x_row: torch.Tensor | None = None
    else:
      x_row_t = _as_2d(x_row, device=model._device)
      if x_row_t.shape[0] != 1:
        raise ValueError("x_row must contain exactly one row.")
      self._x_row = x_row_t

  def pdf(self, uv: TensorLike) -> np.ndarray:
    """Evaluate ``c(u_i, v_i | x_row)`` for each row of ``uv``.

    ``uv`` must have shape ``(n, 2)``.  When the underlying model uses
    the ``criterion`` method and ``uv`` happens to span a Cartesian
    product (the typical plotting case), the call is rerouted through
    :py:meth:`PFNRBicop.pdf_grid` for speed.
    """
    uv_t = _to_tensor(uv, device=self._model._device)
    if uv_t.ndim != 2 or uv_t.shape[1] != 2:
      raise ValueError("uv must have shape (n, 2).")
    n = uv_t.shape[0]

    if self._model.method == "criterion":
      unique_u, inv_u = torch.unique(uv_t[:, 0], return_inverse=True)
      unique_v, inv_v = torch.unique(uv_t[:, 1], return_inverse=True)
      if unique_u.shape[0] * unique_v.shape[0] == n:
        grid = self._model.pdf_grid(unique_u, unique_v, x_row=self._x_row)
        assert isinstance(grid, torch.Tensor)
        return grid[inv_u, inv_v].detach().cpu().numpy()

    x = None if self._x_row is None else self._x_row.repeat_interleave(n, dim=0)
    out = self._model.pdf(uv_t[:, 0], uv_t[:, 1], x)
    assert isinstance(out, torch.Tensor)
    return out.detach().cpu().numpy()
