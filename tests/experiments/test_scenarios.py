"""Tests for ``npcc.experiments.scenarios`` (sampling + ground truth)."""

from __future__ import annotations

import numpy as np
import pyvinecopulib as pv
import pytest

from npcc.experiments import scenarios


def test_eval_grid_conditional_shapes() -> None:
  grid = scenarios.eval_grid("linear")
  n_pairs = len(scenarios.UV_PAIRS)
  assert grid.conditional is True
  assert grid.shape == (n_pairs, scenarios.N_X_EVAL)
  assert grid.u_flat.shape == (n_pairs * scenarios.N_X_EVAL,)
  assert grid.x_flat is not None
  assert grid.x_axis is not None


def test_eval_grid_unconditional_has_no_x() -> None:
  grid = scenarios.eval_grid("uncond50")
  assert grid.conditional is False
  assert grid.x_flat is None
  assert grid.shape == (grid.u_flat.shape[0],)


def test_ground_truth_conditional_matches_pyvinecopulib() -> None:
  family, scenario = "clayton", "linear"
  truth = scenarios.ground_truth(family, scenario)
  grid = scenarios.eval_grid(scenario)
  assert grid.x_axis is not None

  spec = scenarios.TAU_SCENARIOS[scenario]
  assert spec.tau_of_x is not None
  tau_x = spec.tau_of_x(grid.x_axis)
  uv = np.array(scenarios.UV_PAIRS, dtype=np.float64)

  # First x-column should equal a direct Bicop pdf at that tau.
  cop = scenarios._bicop(scenarios.FAMILIES[family], float(tau_x[0]))
  expected_pdf_col0 = np.asarray(cop.pdf(uv))
  np.testing.assert_allclose(truth["pdf"][:, 0], expected_pdf_col0, atol=1e-10)
  assert truth["pdf"].shape == (len(scenarios.UV_PAIRS), scenarios.N_X_EVAL)


def test_ground_truth_unconditional_matches_pyvinecopulib() -> None:
  family, scenario = "frank", "uncond50"
  truth = scenarios.ground_truth(family, scenario)
  grid = scenarios.eval_grid(scenario)
  cop = scenarios._bicop(scenarios.FAMILIES[family], 0.5)
  uv = np.column_stack([grid.u_flat, grid.v_flat])
  np.testing.assert_allclose(truth["cdf"], np.asarray(cop.cdf(uv)), atol=1e-10)
  assert truth["pdf"].shape == grid.shape


def test_sample_conditional_returns_x_linspace_in_unit_square() -> None:
  u, v, x = scenarios.sample("gumbel", "linear", n=200, seed=0)
  assert x is not None
  assert u.shape == v.shape == x.shape == (200,)
  np.testing.assert_allclose(
    x, np.linspace(scenarios.X_MIN, scenarios.X_MAX, 200)
  )
  assert (u > 0).all() and (u < 1).all()
  assert (v > 0).all() and (v < 1).all()


@pytest.mark.parametrize("family", ["clayton", "gumbel", "frank", "gaussian"])
def test_sample_unconditional_recovers_target_tau(family: str) -> None:
  u, v, x = scenarios.sample(family, "uncond50", n=4000, seed=1)
  assert x is None
  tau = float(pv.wdm(u, v, "tau"))
  assert abs(tau - 0.5) < 0.06


def test_is_conditional_flags() -> None:
  assert scenarios.is_conditional("linear") is True
  assert scenarios.is_conditional("uncond75") is False
