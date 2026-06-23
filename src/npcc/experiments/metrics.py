"""Error metrics for the simulation study.

- :func:`curve_metrics` — IAE / ISE / KL of an estimated curve against truth,
  integrated over the covariate axis (conditional scenarios).
- :func:`grid_metrics` — mean IAE / ISE / KL over a flattened uv grid
  (unconditional scenarios).
- :func:`marginal_diagnostics` — how far a density grid's margins are from the
  uniform-copula constraint ``int c du = int c dv = 1`` (quantifies the
  normalization axis).
"""

from __future__ import annotations

import numpy as np

_EPS: float = 1e-12


def curve_metrics(
  y_true: np.ndarray,
  y_hat: np.ndarray,
  x_grid: np.ndarray,
  *,
  include_kl: bool = False,
) -> dict[str, float]:
  """Integrated absolute / squared error (and optional KL) over ``x_grid``.

  KL normalises both curves to unit mass before comparing, so it is only
  meaningful for densities (pass ``include_kl=True`` for ``pdf``).
  """
  y_true = np.asarray(y_true, dtype=np.float64)
  y_hat = np.asarray(y_hat, dtype=np.float64)
  err = y_hat - y_true

  iae = float(np.trapezoid(np.abs(err), x_grid))
  ise = float(np.trapezoid(err**2, x_grid))

  kl = np.nan
  if include_kl:
    y_true_pos = np.clip(y_true, _EPS, None)
    y_hat_pos = np.clip(y_hat, _EPS, None)
    true_mass = float(np.trapezoid(y_true_pos, x_grid))
    hat_mass = float(np.trapezoid(y_hat_pos, x_grid))
    p = y_true_pos / max(true_mass, _EPS)
    q = y_hat_pos / max(hat_mass, _EPS)
    kl = float(np.trapezoid(p * np.log(p / q), x_grid))

  return {"IAE": iae, "ISE": ise, "KL": kl}


def grid_metrics(
  y_true: np.ndarray,
  y_hat: np.ndarray,
  *,
  include_kl: bool = False,
) -> dict[str, float]:
  """Mean absolute / squared error (and optional KL) over a flattened grid."""
  y_true = np.asarray(y_true, dtype=np.float64)
  y_hat = np.asarray(y_hat, dtype=np.float64)
  err = y_hat - y_true

  iae = float(np.mean(np.abs(err)))
  ise = float(np.mean(err**2))

  kl = np.nan
  if include_kl:
    p = np.clip(y_true, _EPS, None)
    q = np.clip(y_hat, _EPS, None)
    kl = float(np.mean(p * (np.log(p) - np.log(q))))

  return {"IAE": iae, "ISE": ise, "KL": kl}


def marginal_diagnostics(
  c: np.ndarray, u_grid: np.ndarray, v_grid: np.ndarray
) -> dict[str, float]:
  """Absolute deviation of a density grid's margins from the constraint = 1.

  ``c`` has shape ``(len(u_grid), len(v_grid))``.  Returns the mean/max
  absolute error of the row integrals ``int c(u, .) dv`` and column integrals
  ``int c(., v) du`` from 1.
  """
  int_over_v = np.trapezoid(c, x=v_grid, axis=1)
  int_over_u = np.trapezoid(c, x=u_grid, axis=0)
  err_rows = np.abs(int_over_v - 1.0)
  err_cols = np.abs(int_over_u - 1.0)
  return {
    "row_mean_abs_err": float(err_rows.mean()),
    "row_max_abs_err": float(err_rows.max()),
    "col_mean_abs_err": float(err_cols.mean()),
    "col_max_abs_err": float(err_cols.max()),
  }
