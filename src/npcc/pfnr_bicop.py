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
margins. If exact margins are required, enable the optional Sinkhorn
projection via ``sinkhorn_iters``. For ``method="criterion"``, the
projection grid is a uniform grid of ``projection_grid_size`` points on
the copula scale ``(0, 1)``; for ``method="quantiles"``, it is given by
the predefined quantile alpha grid. For :py:meth:`pdf_grid`, the
projection is applied directly on the evaluated grid; for pointwise
:py:meth:`pdf`, the correction is computed on the internal projection
grid and interpolated back to the queried points.


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
token before calling :py:meth:`fit`.  Model weights for TabPFN-v3 are
pulled from HuggingFace on first use into the platform cache directory.
"""

from __future__ import annotations

from typing import Any, Literal, Self

import numpy as np
import torch

from tabpfn.constants import ModelVersion

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
from npcc.tabpfn_distribution1d import _DEFAULT_MODEL_VERSION
from npcc.tabpfn_quantile_distribution1d import (
  QuantileGridConfig,
  TabPFNQuantileDistribution1D,
)

_Distribution1D = TabPFNQuantileDistribution1D | TabPFNCriterionDistribution1D


def _sinkhorn_project(
  density: torch.Tensor,
  wu: torch.Tensor,
  wv: torch.Tensor,
  n_iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute Sinkhorn row and column scaling factors in log domain.

  Performs iterative proportional fitting (IPF) to project the density
  matrix onto the space of densities with marginal constraints
  ``int c(u, v) dv = 1`` and ``int c(u, v) du = 1`` under the trapezoidal
  rule with weights ``wu`` and ``wv``.

  The output scalings ``r`` and ``s`` satisfy:
  - Row constraint: ``(density * s[j] * wv[j]).sum() ≈ 1 / wu[i]`` for row i.
  - Col constraint: ``(density * r[i] * wu[i]).sum() ≈ 1 / wv[j]`` for col j.

  Parameters
  ----------
  density
      Shape ``(m, n)`` — raw density matrix at grid points.
  wu
      Shape ``(m,)`` — trapezoidal weights for u-axis (row dimension).
  wv
      Shape ``(n,)`` — trapezoidal weights for v-axis (column dimension).
  n_iters
      Number of alternating normalization iterations.

  Returns
  -------
  r : torch.Tensor
      Shape ``(m,)`` — row scalings.
  s : torch.Tensor
      Shape ``(n,)`` — column scalings.
  """
  if n_iters <= 0:
    raise ValueError("n_iters must be positive.")

  m, n = density.shape
  tiny = torch.finfo(density.dtype).tiny

  # Work in log space to avoid underflow/overflow in repeated updates.
  log_density = torch.clamp(density, min=tiny).log()
  log_wu = torch.clamp(wu, min=tiny).log()
  log_wv = torch.clamp(wv, min=tiny).log()

  log_r = torch.zeros(m, dtype=density.dtype, device=density.device)
  log_s = torch.zeros(n, dtype=density.dtype, device=density.device)

  for _ in range(n_iters):
    # Row normalization: enforce sum_j density_ij * s_j * wv_j = 1.
    log_row_sums = torch.logsumexp(
      log_density + log_s[None, :] + log_wv[None, :],
      dim=1,
    )
    log_r = -log_row_sums

    # Column normalization: enforce sum_i density_ij * r_i * wu_i = 1.
    log_col_sums = torch.logsumexp(
      log_density + log_r[:, None] + log_wu[:, None],
      dim=0,
    )
    log_s = -log_col_sums

  r = torch.exp(log_r)
  s = torch.exp(log_s)

  return r, s


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
  transform
      Support transform used by the inner TabPFN distribution models.
      ``"logit"`` (default) maps copula values in ``(0, 1)`` to
      ``R`` before fitting; ``"probit"`` applies ``Phi^{-1}``; and
      ``"identity"`` keeps the original scale.
  device
      Device for internal tensors and TabPFN inference.  ``None``
      (default) auto-selects ``cuda`` if available, else ``cpu``.
      Forwarded into the inner distributions and into TabPFN via
      ``model_kwargs["device"]``.
  batch_size
      Default chunk size used by criterion-based inner ``pdf`` / ``cdf``
      calls.  If ``None`` (default), uses 400 on CPU and 2000 on CUDA.
      A positive value overrides this device-based default.
  model_kwargs
      Forwarded into the inner ``TabPFNRegressor`` (via
      :py:meth:`TabPFNRegressor.create_default_for_version`).  Useful
      for ``n_estimators=...``, etc.
  sinkhorn_iters
      Default number of Sinkhorn / iterative-proportional-fitting
      iterations used to project the estimated density onto the space of
      bivariate copula densities with approximately uniform margins.
      ``None`` (default) disables projection. A positive integer enables
      projection by default for :py:meth:`pdf`, :py:meth:`log_pdf`, and
      :py:meth:`pdf_grid`, unless overridden per call.

      The projection is carried out on a 2-D grid derived from the fitted
      inner univariate conditional density models. For
      ``method="criterion"``, the grid of size ``projection_grid_size``
      per axis is used. For ``method="quantiles"``, the projection grid is
      given by the predefined quantile alpha grid.
  projection_grid_size
      Number of points per axis in the uniform copula-scale grid used for
      the optional Sinkhorn projection in ``method="criterion"``; the
      default is 101.

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
    transform: Literal["identity", "logit", "probit"] = "logit",
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
    model_version: ModelVersion | None = _DEFAULT_MODEL_VERSION,
    sinkhorn_iters: int | None = None,
    projection_grid_size: int = 101,
  ) -> None:
    if sinkhorn_iters is not None and sinkhorn_iters <= 0:
      raise ValueError("sinkhorn_iters must be None or a positive integer.")

    self.symmetric = symmetric
    self.method = method
    self.quantile_config = quantile_config or QuantileGridConfig()
    self.transform = transform
    self._device = _resolve_device(device)
    if batch_size is None:
      self.batch_size = 2000 if self._device.type == "cuda" else 400
    else:
      if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
      self.batch_size = batch_size
    self.model_kwargs = dict(model_kwargs or {})
    self.model_version = model_version
    self.sinkhorn_iters = sinkhorn_iters
    if projection_grid_size < 2:
      raise ValueError("projections_grid_size must be at least 2.")
    self.projection_grid_size = projection_grid_size

    self.v_given_ux_: _Distribution1D = self._make_distribution()
    self.u_given_vx_: _Distribution1D | None = (
      self._make_distribution() if symmetric else None
    )

    # Grid borders (cached after fit)
    self._v_grid_borders_: torch.Tensor | None = None
    self._u_grid_borders_: torch.Tensor | None = None

  def _make_distribution(self) -> _Distribution1D:
    if self.method == "quantiles":
      return TabPFNQuantileDistribution1D(
        transform=self.transform,
        config=self.quantile_config,
        device=self._device,
        model_kwargs=self.model_kwargs,
        model_version=self.model_version,
      )
    return TabPFNCriterionDistribution1D(
      transform=self.transform,
      eps=self.quantile_config.eps,
      device=self._device,
      batch_size=self.batch_size,
      model_kwargs=self.model_kwargs,
      model_version=self.model_version,
    )

  def _get_grid_borders(self) -> None:
    """Cache the 1-D projection grids used for Sinkhorn projection.

    For ``method="criterion"``, the Sinkhorn correction uses a separate
    uniform projection grid on the copula scale.
    For ``method="quantiles"``, use the alpha grid.
    """
    if self.method == "criterion":
      eps = self.quantile_config.eps
      borders = torch.linspace(
        eps,
        1 - eps,
        steps=self.projection_grid_size,
        dtype=torch.float64,
        device=self._device,
      )

      self._v_grid_borders_ = borders
      self._u_grid_borders_ = borders
      return

    alphas = torch.as_tensor(
      self.quantile_config.alphas(),
      dtype=torch.float64,
      device=self._device,
    )
    self._v_grid_borders_ = alphas
    self._u_grid_borders_ = alphas

  def _resolve_batch_size(self, batch_size: int | None) -> int:
    effective = self.batch_size if batch_size is None else batch_size
    if effective <= 0:
      raise ValueError("batch_size must be positive.")
    return effective

  def _resolve_sinkhorn_iters(self, sinkhorn_iters: int | None) -> int | None:
    effective = (
      self.sinkhorn_iters if sinkhorn_iters is None else sinkhorn_iters
    )
    if effective is not None and effective <= 0:
      raise ValueError("sinkhorn_iters must be None or a positive integer.")
    return effective

  def _features(
    self, first_coord: torch.Tensor, x: torch.Tensor
  ) -> torch.Tensor:
    """Build the inner regressor's feature matrix ``[first_coord | x]``."""
    return torch.column_stack([first_coord, x])

  def _default_x(self, n: int) -> torch.Tensor:
    """Constant covariate column of ones (used when ``x`` is omitted)."""
    return torch.ones((n, 1), dtype=torch.float64, device=self._device)

  def _prepare_joint_inputs(
    self, u: TensorLike, v: TensorLike, x: TensorLike | None
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u_t, v_t = _check_uv(u, v, self.quantile_config.eps, device=self._device)
    x_t = (
      self._default_x(u_t.shape[0])
      if x is None
      else _as_2d(x, device=self._device)
    )

    if x_t.shape[0] != u_t.shape[0]:
      raise ValueError("x, u, and v must have the same number of observations.")

    return u_t, v_t, x_t

  def _prepare_grid_inputs(
    self, u_grid: TensorLike, v_grid: TensorLike, x_row: TensorLike | None
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, (u_in, v_in, x_in) = _normalize_inputs(
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

    return u_t, v_t, x_row_t

  @staticmethod
  def _trapezoidal_weights(grid: torch.Tensor) -> torch.Tensor:
    """Compute trapezoidal rule weights for a sorted 1-D grid.

    For a grid ``g`` of length ``m``, the weight at position ``i`` is the
    average distance to neighbors:

        w[i] = (g[min(i+1, m-1)] - g[max(i-1, 0)]) / 2

    This is the standard weight in the composite trapezoidal rule.
    """
    m = grid.shape[0]
    if m == 1:
      return torch.ones(1, dtype=grid.dtype, device=grid.device)

    weights = torch.zeros(m, dtype=grid.dtype, device=grid.device)
    weights[0] = (grid[1] - grid[0]) / 2.0
    weights[-1] = (grid[-1] - grid[-2]) / 2.0
    if m > 2:
      weights[1:-1] = (grid[2:] - grid[:-2]) / 2.0

    return weights

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

    Notes
    -----
    When ``self.sinkhorn_iters`` is not ``None``, the 1-D grids used for the
    optional Sinkhorn projection are initialized and cached during fit. For
    ``method="criterion"`` they are a uniform grid of ``projection_grid_size``
    points on the copula scale ``(0, 1)``; for ``method="quantiles"`` they are
    the configured quantile alpha grid.
    """
    u_t, v_t, x_t = self._prepare_joint_inputs(u, v, x)

    self.v_given_ux_.fit(self._features(u_t, x_t), v_t)

    if self.symmetric:
      assert self.u_given_vx_ is not None
      self.u_given_vx_.fit(self._features(v_t, x_t), u_t)

    # Cache grid borders for Sinkhorn projection (if enabled)
    if self.sinkhorn_iters is not None:
      self._get_grid_borders()

    return self

  def pdf(
    self,
    u: TensorLike,
    v: TensorLike,
    x: TensorLike | None = None,
    *,
    batch_size: int | None = None,
    sinkhorn_iters: int | None = None,
  ) -> TensorLike:
    """Return the conditional copula density ``c(u_i, v_i | x_i)``.

    ``batch_size`` overrides the model-level default chunk size for
    this call when using ``method="criterion"``.
    sinkhorn_iters
        Overrides the model-level default Sinkhorn / iterative-proportional-
        fitting iteration count for this call. ``None`` means “use
        ``self.sinkhorn_iters``”. If the effective value is ``None``, no
        projection is applied. If it is a positive integer, the estimated
        density is projected toward the space of bivariate copula densities
        with approximately uniform margins using a grid-based Sinkhorn
        correction.
    """
    return_as_torch, _ = _normalize_inputs(u, v, x, device=self._device)
    with torch.inference_mode():
      out = self._pdf_torch(
        u,
        v,
        x,
        batch_size=self._resolve_batch_size(batch_size),
        sinkhorn_iters=self._resolve_sinkhorn_iters(sinkhorn_iters),
      )
    return _wrap_output(out, return_as_torch=return_as_torch)

  def _pdf_torch(
    self,
    u: TensorLike,
    v: TensorLike,
    x: TensorLike | None,
    *,
    batch_size: int,
    sinkhorn_iters: int | None,
  ) -> torch.Tensor:
    u_t, v_t, x_t = self._prepare_joint_inputs(u, v, x)
    c_raw = self._raw_pdf_torch(u_t, v_t, x_t, batch_size=batch_size)

    if sinkhorn_iters is None:
      return c_raw

    # Apply Sinkhorn projection if enabled
    return self._project_points_by_x(
      c_raw,
      u_t,
      v_t,
      x_t,
      batch_size=batch_size,
      sinkhorn_iters=sinkhorn_iters,
    )

  def _project_points_by_x(
    self,
    c_raw: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    *,
    batch_size: int,
    sinkhorn_iters: int,
  ) -> torch.Tensor:
    if self._u_grid_borders_ is None or self._v_grid_borders_ is None:
      self._get_grid_borders()

    assert self._u_grid_borders_ is not None
    assert self._v_grid_borders_ is not None

    u_grid = self._u_grid_borders_
    v_grid = self._v_grid_borders_

    wu = self._trapezoidal_weights(u_grid)
    wv = self._trapezoidal_weights(v_grid)

    x_unique, x_inverse = torch.unique(x, dim=0, return_inverse=True)
    out = torch.empty_like(c_raw)

    for x_idx in range(x_unique.shape[0]):
      mask = x_inverse == x_idx
      if not torch.any(mask):
        continue

      x_row = x_unique[x_idx : x_idx + 1]

      density_grid = self._raw_pdf_grid_torch(
        u_grid,
        v_grid,
        x_row,
        batch_size=batch_size,
      )

      r, s = _sinkhorn_project(density_grid, wu, wv, sinkhorn_iters)

      r_interp = _torch_interp(u[mask], u_grid, r)
      s_interp = _torch_interp(v[mask], v_grid, s)

      out[mask] = c_raw[mask] * r_interp * s_interp

    return out

  def _project_grid(
    self,
    c_grid_raw: torch.Tensor,
    u_grid: torch.Tensor,
    v_grid: torch.Tensor,
    *,
    sinkhorn_iters: int,
  ) -> torch.Tensor:
    wu = self._trapezoidal_weights(u_grid)
    wv = self._trapezoidal_weights(v_grid)
    r, s = _sinkhorn_project(c_grid_raw, wu, wv, sinkhorn_iters)
    return r[:, None] * c_grid_raw * s[None, :]

  def _raw_pdf_torch(
    self,
    u: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    *,
    batch_size: int,
  ) -> torch.Tensor:

    c_v_given_u = self.v_given_ux_.pdf(
      self._features(u, x), v, batch_size=batch_size
    )
    assert isinstance(c_v_given_u, torch.Tensor)

    if not self.symmetric:
      c_raw = c_v_given_u
    else:
      assert self.u_given_vx_ is not None
      c_u_given_v = self.u_given_vx_.pdf(
        self._features(v, x), u, batch_size=batch_size
      )
      assert isinstance(c_u_given_v, torch.Tensor)
      c_raw = 0.5 * (c_v_given_u + c_u_given_v)

    return c_raw

  def _grid_one_direction(
    self,
    module: _Distribution1D,
    first_grid: torch.Tensor,
    second_grid: torch.Tensor,
    x_row: torch.Tensor,
    *,
    batch_size: int,
  ) -> torch.Tensor:
    """Density grid ``out[i, j] = f(second_grid[j] | first_grid[i], x_row)``.

    ``module`` predicts the *second* coordinate conditioned on
    ``[first_grid, x_row]``.  The criterion method reuses one forward
    pass per ``first`` row via
    :py:meth:`TabPFNCriterionDistribution1D.pdf_grid`; the quantile
    method has no such shortcut and evaluates the explicit Cartesian
    tile (``first`` slow, ``second`` fast) before reshaping.

    ponytail: one direction helper, called once per Rosenblatt direction
    (forward V|U, reverse U|V) — the reverse caller transposes the result.
    """
    n_first, n_second = first_grid.shape[0], second_grid.shape[0]

    if isinstance(module, TabPFNCriterionDistribution1D):
      grid = module.pdf_grid(
        self._features(first_grid, x_row.repeat(n_first, 1)),
        second_grid,
        batch_size=batch_size,
      )
    else:
      first_tiled = first_grid.repeat_interleave(n_second)
      second_tiled = second_grid.tile(n_first)
      x_tiled = x_row.repeat(n_first * n_second, 1)
      flat = module.pdf(
        self._features(first_tiled, x_tiled),
        second_tiled,
        batch_size=batch_size,
      )
      assert isinstance(flat, torch.Tensor)
      grid = flat.reshape(n_first, n_second)

    assert isinstance(grid, torch.Tensor)
    return grid

  def _raw_pdf_grid_torch(
    self,
    u_grid: torch.Tensor,
    v_grid: torch.Tensor,
    x_row: torch.Tensor,
    *,
    batch_size: int,
  ) -> torch.Tensor:
    density_grid = self._grid_one_direction(
      self.v_given_ux_, u_grid, v_grid, x_row, batch_size=batch_size
    )

    if not self.symmetric:
      return density_grid

    # Reverse direction returns f_{U|V}(u_i | v_j) at [j, i]; transpose
    # before averaging so both grids index [u, v].
    assert self.u_given_vx_ is not None
    density_grid_u = self._grid_one_direction(
      self.u_given_vx_, v_grid, u_grid, x_row, batch_size=batch_size
    )
    return 0.5 * (density_grid + density_grid_u.T)

  def log_pdf(
    self,
    u: TensorLike,
    v: TensorLike,
    x: TensorLike | None = None,
    *,
    batch_size: int | None = None,
    sinkhorn_iters: int | None = None,
  ) -> TensorLike:
    """Log of the optionally projected :py:meth:`pdf`, floored at the smallest positive float.

    ``batch_size`` matches :py:meth:`pdf` and is forwarded to the same
    underlying criterion calls when ``method="criterion"``.
    sinkhorn_iters
        Overrides the model-level default Sinkhorn / iterative-proportional-
        fitting iteration count for this call. ``None`` means “use
        ``self.sinkhorn_iters``”. If the effective value is ``None``, no
        projection is applied. If it is a positive integer, the estimated
        density is projected toward the space of bivariate copula densities
        with approximately uniform margins using a grid-based Sinkhorn
        correction.
    """
    return_as_torch, _ = _normalize_inputs(u, v, x, device=self._device)
    with torch.inference_mode():
      c = self._pdf_torch(
        u,
        v,
        x,
        batch_size=self._resolve_batch_size(batch_size),
        sinkhorn_iters=self._resolve_sinkhorn_iters(sinkhorn_iters),
      )
    out = torch.log(torch.clamp(c, min=torch.finfo(c.dtype).tiny))
    return _wrap_output(out, return_as_torch=return_as_torch)

  def pdf_grid(
    self,
    u_grid: TensorLike,
    v_grid: TensorLike,
    x_row: TensorLike | None = None,
    *,
    batch_size: int | None = None,
    sinkhorn_iters: int | None = None,
  ) -> TensorLike:
    """Density on the Cartesian product ``out[i, j] = c(u_grid[i], v_grid[j] | x)``.

    Requires ``method="criterion"`` because it relies on
    :py:meth:`TabPFNCriterionDistribution1D.pdf_grid`.  ``x_row`` is a single
    covariate row reused on both axes; when ``None`` a constant
    one-column row is used.

    For the symmetric estimator both directions are evaluated on the
    same Cartesian product (transposing the reverse one) and averaged.

    batch_size
        Overrides the model-level batch size used by the inner criterion
        ``pdf_grid`` calls for this Cartesian-grid evaluation.

    sinkhorn_iters
        Overrides the model-level default Sinkhorn / iterative-proportional-
        fitting iteration count for this call. ``None`` means “use
        ``self.sinkhorn_iters``”. If the effective value is ``None``, no
        projection is applied. If it is a positive integer, the estimated
        density is projected toward the space of bivariate copula densities
        with approximately uniform margins using a grid-based Sinkhorn
        correction.
    """
    if self.method != "criterion" or not isinstance(
      self.v_given_ux_, TabPFNCriterionDistribution1D
    ):
      raise RuntimeError("pdf_grid is only available when method='criterion'.")

    return_as_torch, _ = _normalize_inputs(
      u_grid, v_grid, x_row, device=self._device
    )
    with torch.inference_mode():
      out = self._pdf_grid_torch(
        u_grid,
        v_grid,
        x_row,
        batch_size=self._resolve_batch_size(batch_size),
        sinkhorn_iters=self._resolve_sinkhorn_iters(sinkhorn_iters),
      )
    return _wrap_output(out, return_as_torch=return_as_torch)

  def _pdf_grid_torch(
    self,
    u_grid: TensorLike,
    v_grid: TensorLike,
    x_row: TensorLike | None,
    *,
    batch_size: int,
    sinkhorn_iters: int | None,
  ) -> torch.Tensor:
    u_t, v_t, x_row_t = self._prepare_grid_inputs(u_grid, v_grid, x_row)

    c_raw = self._raw_pdf_grid_torch(
      u_t,
      v_t,
      x_row_t,
      batch_size=batch_size,
    )

    if sinkhorn_iters is None:
      return c_raw

    return self._project_grid(
      c_raw,
      u_t,
      v_t,
      sinkhorn_iters=sinkhorn_iters,
    )

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
    u_t, v_t, x_t = self._prepare_joint_inputs(u, v, x)

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
    u_t, v_t, x_t = self._prepare_joint_inputs(u, v, x)

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
    n_int: int = 12,
    batch_size: int | None = None,
  ) -> TensorLike:
    """Joint CDF ``C(u_i, v_i | x_i)`` evaluated row-by-row.

    Trapezoidal integration of the inner conditional CDF over the
    Rosenblatt direction:

        C(u, v | x) = ∫_0^u F_{V | U, X}(v | s, x) ds       (asymmetric)
        C^sym(u, v | x) = 0.5 (∫_0^u F_{V|U,X}(v|s,x) ds
                               + ∫_0^v F_{U|V,X}(u|t,x) dt) (symmetric)

    ``n_int`` is the number of trapezoid steps along the integration
    axis; the default 12 trades a little accuracy for speed on this
    per-row path (each step is a separate inner CDF evaluation).
    :py:meth:`cdf_grid` shares one fine grid across the whole Cartesian
    product, so it can afford a finer default (64).  ``batch_size``
    overrides the model-level default chunk size for the inner
    criterion CDF calls used during integration.
    """
    if n_int < 2:
      raise ValueError("n_int must be at least 2.")
    effective_batch_size = self._resolve_batch_size(batch_size)

    return_as_torch, _ = _normalize_inputs(u, v, x, device=self._device)
    u_t, v_t, x_t = self._prepare_joint_inputs(u, v, x)

    cdf_v_dir = self._integrate_one_direction(
      upper=u_t,
      conditioned=v_t,
      x=x_t,
      module=self.v_given_ux_,
      n_int=n_int,
      batch_size=effective_batch_size,
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
      batch_size=effective_batch_size,
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
    batch_size: int,
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
    if isinstance(module, TabPFNCriterionDistribution1D):
      F_flat = module.cdf(feats, cond_flat, batch_size=batch_size)
    else:
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

    return_as_torch, _ = _normalize_inputs(
      u_grid, v_grid, x_row, device=self._device
    )
    u_t, v_t, x_row_t = self._prepare_grid_inputs(u_grid, v_grid, x_row)

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
