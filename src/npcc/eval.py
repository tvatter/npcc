"""
eval.py — evaluation-only diagnostics for copula density estimates.

Pure numpy, estimator-free (never imported by the estimators): it takes arrays
or duck-typed objects exposing ``pdf`` / ``hfunc1`` / ``hfunc2``. It promotes the
ad-hoc ``grid_metrics`` used in the demo notebooks into a tested library surface
and adds the conditional-grid and margin-calibration helpers needed to benchmark
fine-tuned vs base TabPFN weights.

Metrics on a ``(u, v)`` grid (``c_true``, ``c_hat`` are 2-D, shape
``(len(u_grid), len(v_grid))``):

- ``ISE = ∫∫ (ĉ - c)²``,  ``IAE = ∫∫ |ĉ - c|``,
- ``KL  = ∫∫ c · (log c - log ĉ)``  (forward KL, both densities floored at
  ``eps`` before the log).

``reduction="mean"`` reproduces the notebooks exactly (unweighted mean on an
interior grid); ``reduction="trapezoid"`` integrates with the grid spacing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

ArrayLike = npt.ArrayLike
Reduction = Literal["mean", "trapezoid"]


def unit_grid(m: int, *, interior: bool = True) -> np.ndarray:
  """A 1-D grid on ``[0, 1]``.

  ``interior`` (default) returns the midpoint grid ``(i + 0.5) / m`` used by the
  demo notebooks (avoids the copula boundary); otherwise ``linspace(0, 1, m)``.
  """
  if m < 2:
    raise ValueError("m must be >= 2.")
  if interior:
    return (np.arange(m) + 0.5) / m
  return np.linspace(0.0, 1.0, m)


def _reduce(
  integrand: np.ndarray,
  reduction: Reduction,
  u_grid: np.ndarray | None,
  v_grid: np.ndarray | None,
) -> float:
  if reduction == "mean":
    return float(np.mean(integrand))
  if reduction == "trapezoid":
    if u_grid is None or v_grid is None:
      raise ValueError("trapezoid reduction requires u_grid and v_grid.")
    return float(np.trapezoid(np.trapezoid(integrand, v_grid, axis=1), u_grid))
  raise ValueError(f"Unknown reduction: {reduction}")


def grid_metrics_density(
  c_true: ArrayLike,
  c_hat: ArrayLike,
  *,
  u_grid: ArrayLike | None = None,
  v_grid: ArrayLike | None = None,
  reduction: Reduction = "trapezoid",
  eps: float = 1e-12,
) -> dict[str, float]:
  """ISE / IAE / KL between a true and an estimated density on a grid."""
  c_true = np.asarray(c_true, float)
  c_hat = np.asarray(c_hat, float)
  if c_true.shape != c_hat.shape:
    raise ValueError("c_true and c_hat must have the same shape.")
  ug = None if u_grid is None else np.asarray(u_grid, float)
  vg = None if v_grid is None else np.asarray(v_grid, float)
  diff = c_hat - c_true
  log_ratio = np.log(np.maximum(c_true, eps)) - np.log(np.maximum(c_hat, eps))
  return {
    "ISE": _reduce(diff**2, reduction, ug, vg),
    "IAE": _reduce(np.abs(diff), reduction, ug, vg),
    "KL": _reduce(c_true * log_ratio, reduction, ug, vg),
  }


def grid_metrics_hfunc(
  h_true: ArrayLike,
  h_hat: ArrayLike,
  *,
  u_grid: ArrayLike | None = None,
  v_grid: ArrayLike | None = None,
  reduction: Reduction = "trapezoid",
) -> dict[str, float]:
  """ISE / IAE between a true and an estimated h-function on a grid."""
  h_true = np.asarray(h_true, float)
  h_hat = np.asarray(h_hat, float)
  if h_true.shape != h_hat.shape:
    raise ValueError("h_true and h_hat must have the same shape.")
  ug = None if u_grid is None else np.asarray(u_grid, float)
  vg = None if v_grid is None else np.asarray(v_grid, float)
  diff = h_hat - h_true
  return {
    "ISE": _reduce(diff**2, reduction, ug, vg),
    "IAE": _reduce(np.abs(diff), reduction, ug, vg),
  }


DensityFn = Callable[[np.ndarray, np.ndarray, np.ndarray | None], np.ndarray]
"""A conditional density evaluator ``f(u, v, x) -> (n,)`` (e.g. ``PFNRBicop.pdf``
or a pyvinecopulib closure). Bake Sinkhorn / direction options into the closure
to keep this module estimator-agnostic."""


@dataclass(frozen=True)
class ConditionalGridSpec:
  """A grid of ``(u, v)`` levels swept across covariate rows ``x_values``.

  Attributes
  ----------
  u_levels, v_levels
      The ``(u, v)`` evaluation pairs (Cartesian product).
  x_values
      Covariate rows, shape ``(n_x, p)`` (a 1-D array is treated as ``(n_x, 1)``;
      ``None`` for the unconditional case).
  """

  u_levels: tuple[float, ...]
  v_levels: tuple[float, ...]
  x_values: np.ndarray | None = None

  @property
  def uv_pairs(self) -> list[tuple[float, float]]:
    return [(u, v) for u in self.u_levels for v in self.v_levels]


def conditional_density_grids(
  density_fn: DensityFn, spec: ConditionalGridSpec
) -> np.ndarray:
  """Evaluate ``density_fn`` over the spec; returns shape ``(n_uv_pairs, n_x)``.

  Mirrors the notebook ``true_pdf_grid`` / ``estimated_pdf_grid`` pattern in one
  batched call. ``density_fn`` receives flat ``(u, v, x)`` arrays.
  """
  pairs = spec.uv_pairs
  u_pair = np.array([u for u, _ in pairs], float)
  v_pair = np.array([v for _, v in pairs], float)
  if spec.x_values is None:
    c = density_fn(u_pair, v_pair, None)
    return np.asarray(c, float).reshape(len(pairs), 1)

  x = np.asarray(spec.x_values, float)
  if x.ndim == 1:
    x = x[:, None]
  n_x = x.shape[0]
  u_flat = np.repeat(u_pair[:, None], n_x, axis=1).reshape(-1)
  v_flat = np.repeat(v_pair[:, None], n_x, axis=1).reshape(-1)
  x_flat = np.repeat(x[None, :, :], len(pairs), axis=0).reshape(-1, x.shape[1])
  c = density_fn(u_flat, v_flat, x_flat)
  return np.asarray(c, float).reshape(len(pairs), n_x)


def conditional_metrics(
  true_grid: ArrayLike,
  est_grid: ArrayLike,
  *,
  reduce_over: Literal["all", "x", "uv"] = "all",
  eps: float = 1e-12,
) -> dict[str, np.ndarray]:
  """ISE / IAE / KL between conditional density grids of shape ``(n_uv, n_x)``.

  ``reduce_over="all"`` averages over the whole grid (0-d arrays); ``"x"`` returns
  a per-x array; ``"uv"`` returns a per-uv-pair array.
  """
  t = np.asarray(true_grid, float)
  e = np.asarray(est_grid, float)
  if t.shape != e.shape:
    raise ValueError("true_grid and est_grid must have the same shape.")
  diff = e - t
  kl = t * (np.log(np.maximum(t, eps)) - np.log(np.maximum(e, eps)))
  stacks = {"ISE": diff**2, "IAE": np.abs(diff), "KL": kl}
  axis = {"all": None, "x": 0, "uv": 1}[reduce_over]
  return {k: np.asarray(np.mean(v, axis=axis)) for k, v in stacks.items()}


class _HasHFunc(Protocol):
  def hfunc1(
    self, u: np.ndarray, v: np.ndarray, x: np.ndarray | None
  ) -> np.ndarray: ...


def _ks_uniform(samples: np.ndarray) -> float:
  """Kolmogorov-Smirnov distance of ``samples`` to Uniform(0, 1)."""
  s = np.sort(np.asarray(samples, float))
  n = len(s)
  grid = np.arange(1, n + 1) / n
  d_plus = np.max(grid - s)
  d_minus = np.max(s - (grid - 1.0 / n))
  return float(max(d_plus, d_minus))


def margin_calibration(
  estimator: _HasHFunc,
  u: ArrayLike,
  v: ArrayLike,
  x: ArrayLike | None = None,
) -> dict[str, float]:
  """KS distance of the PIT ``hfunc_i(u, v | x)`` to Uniform(0, 1).

  ``hfunc1`` (always available) PIT-checks the ``V | U, X`` margin; ``hfunc2``
  (symmetric estimators only) the ``U | V, X`` margin. A well-calibrated copula
  estimate yields small KS values.
  """
  u_arr = np.asarray(u, float)
  v_arr = np.asarray(v, float)
  x_arr = None if x is None else np.asarray(x, float)
  out = {
    "ks_h1": _ks_uniform(np.asarray(estimator.hfunc1(u_arr, v_arr, x_arr)))
  }
  hfunc2 = getattr(estimator, "hfunc2", None)
  if hfunc2 is not None:
    try:
      out["ks_h2"] = _ks_uniform(np.asarray(hfunc2(u_arr, v_arr, x_arr)))
    except (ValueError, RuntimeError):
      pass
  return out
