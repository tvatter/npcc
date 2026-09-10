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

import torch

_EPS: float = 1e-12


def curve_metrics(
  y_true: torch.Tensor,
  y_hat: torch.Tensor,
  x_grid: torch.Tensor,
  *,
  include_kl: bool = False,
) -> dict[str, float]:
  """Integrated absolute / squared error (and optional KL) over ``x_grid``.

  KL normalizes both curves to unit mass before comparing, so it is only
  meaningful for densities (pass ``include_kl=True`` for ``pdf``).
  """
  err = y_hat - y_true

  iae = float(torch.trapezoid(err.abs(), x_grid))
  ise = float(torch.trapezoid(err.square(), x_grid))

  kl = float("nan")
  if include_kl:
    y_true_pos = y_true.clamp_min(_EPS)
    y_hat_pos = y_hat.clamp_min(_EPS)
    true_mass = float(torch.trapezoid(y_true_pos, x_grid))
    hat_mass = float(torch.trapezoid(y_hat_pos, x_grid))
    p = y_true_pos / max(true_mass, _EPS)
    q = y_hat_pos / max(hat_mass, _EPS)
    kl = float(torch.trapezoid(p * torch.log(p / q), x_grid))

  return {"IAE": iae, "ISE": ise, "KL": kl}


def grid_metrics(
  y_true: torch.Tensor,
  y_hat: torch.Tensor,
  *,
  include_kl: bool = False,
) -> dict[str, float]:
  """Mean absolute / squared error (and optional KL) over a flattened grid."""
  err = y_hat - y_true

  iae = float(err.abs().mean())
  ise = float(err.square().mean())

  kl = float("nan")
  if include_kl:
    p = y_true.clamp_min(_EPS)
    q = y_hat.clamp_min(_EPS)
    kl = float((p * (p.log() - q.log())).mean())

  return {"IAE": iae, "ISE": ise, "KL": kl}


def marginal_diagnostics(
  c: torch.Tensor, u_grid: torch.Tensor, v_grid: torch.Tensor
) -> dict[str, float]:
  """Absolute deviation of a density grid's margins from the constraint = 1.

  ``u_grid`` and ``v_grid`` are midpoint grids covering the unit interval, and
  ``c`` has shape ``(len(u_grid), len(v_grid))``. Midpoint quadrature avoids
  evaluating potentially unstable copula densities at zero or one while still
  integrating over the full unit interval.
  """
  if c.shape != (u_grid.numel(), v_grid.numel()):
    raise ValueError(
      "c must have shape (len(u_grid), len(v_grid)); "
      f"got {c.shape} for ({u_grid.numel()}, {v_grid.numel()})."
    )
  int_over_v = c.mean(dim=1)
  int_over_u = c.mean(dim=0)
  err_rows = (int_over_v - 1.0).abs()
  err_cols = (int_over_u - 1.0).abs()
  return {
    "row_mean_abs_err": float(err_rows.mean()),
    "row_max_abs_err": float(err_rows.max()),
    "col_mean_abs_err": float(err_cols.mean()),
    "col_max_abs_err": float(err_cols.max()),
  }
