"""Tests for ``npcc.experiments.metrics``."""

from __future__ import annotations

import numpy as np

from npcc.experiments import metrics


def test_curve_metrics_zero_error() -> None:
  x = np.linspace(0.0, 1.0, 11)
  y = np.sin(x)
  out = metrics.curve_metrics(y, y, x, include_kl=True)
  assert out["IAE"] == 0.0
  assert out["ISE"] == 0.0
  assert out["KL"] == 0.0


def test_curve_metrics_constant_offset_matches_trapezoid() -> None:
  x = np.linspace(0.0, 2.0, 5)
  y_true = np.zeros_like(x)
  y_hat = np.full_like(x, 0.5)
  out = metrics.curve_metrics(y_true, y_hat, x)
  # IAE = integral of 0.5 over [0,2] = 1.0; ISE = integral of 0.25 = 0.5
  assert out["IAE"] == np.float64(1.0)
  assert out["ISE"] == np.float64(0.5)
  assert np.isnan(out["KL"])


def test_grid_metrics_means() -> None:
  y_true = np.array([1.0, 2.0, 3.0])
  y_hat = np.array([1.0, 2.0, 5.0])
  out = metrics.grid_metrics(y_true, y_hat)
  assert out["IAE"] == np.float64(2.0 / 3.0)
  assert out["ISE"] == np.float64(4.0 / 3.0)


def test_marginal_diagnostics_uniform_density_is_near_one() -> None:
  # c(u,v) == 1 has both margins integrating to 1 exactly.
  u = np.linspace(0.0, 1.0, 21)
  v = np.linspace(0.0, 1.0, 21)
  c = np.ones((u.size, v.size))
  out = metrics.marginal_diagnostics(c, u, v)
  assert out["row_mean_abs_err"] < 1e-12
  assert out["col_max_abs_err"] < 1e-12
