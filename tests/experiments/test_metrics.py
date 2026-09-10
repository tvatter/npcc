"""Tests for ``npcc.experiments.metrics``."""

from __future__ import annotations

import math

import pytest
import torch

from npcc.experiments import metrics


def test_curve_metrics_zero_error() -> None:
  x = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
  y = torch.sin(x)
  out = metrics.curve_metrics(y, y, x, include_kl=True)
  assert out["IAE"] == 0.0
  assert out["ISE"] == 0.0
  assert out["KL"] == 0.0


def test_curve_metrics_constant_offset_matches_trapezoid() -> None:
  x = torch.linspace(0.0, 2.0, 5, dtype=torch.float64)
  y_true = torch.zeros_like(x)
  y_hat = torch.full_like(x, 0.5)
  out = metrics.curve_metrics(y_true, y_hat, x)
  # IAE = integral of 0.5 over [0,2] = 1.0; ISE = integral of 0.25 = 0.5
  assert out["IAE"] == pytest.approx(1.0)
  assert out["ISE"] == pytest.approx(0.5)
  assert math.isnan(out["KL"])


def test_grid_metrics_means() -> None:
  y_true = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
  y_hat = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float64)
  out = metrics.grid_metrics(y_true, y_hat)
  assert out["IAE"] == pytest.approx(2.0 / 3.0)
  assert out["ISE"] == pytest.approx(4.0 / 3.0)


def test_marginal_diagnostics_uniform_density_is_exact() -> None:
  # c(u,v) == 1 has both margins integrating to 1 exactly.
  u = (torch.arange(20, dtype=torch.float64) + 0.5) / 20
  v = (torch.arange(20, dtype=torch.float64) + 0.5) / 20
  c = torch.ones((u.numel(), v.numel()), dtype=torch.float64)
  out = metrics.marginal_diagnostics(c, u, v)
  assert out == {
    "row_mean_abs_err": 0.0,
    "row_max_abs_err": 0.0,
    "col_mean_abs_err": 0.0,
    "col_max_abs_err": 0.0,
  }
