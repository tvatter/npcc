"""Tests for ``npcc.eval`` (evaluation-only density diagnostics)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from npcc.eval import (
  ConditionalGridSpec,
  conditional_density_grids,
  conditional_metrics,
  grid_metrics_density,
  grid_metrics_hfunc,
  margin_calibration,
  unit_grid,
)


def test_unit_grid_interior_vs_boundary() -> None:
  interior = unit_grid(4, interior=True)
  np.testing.assert_allclose(interior, [0.125, 0.375, 0.625, 0.875])
  boundary = unit_grid(4, interior=False)
  assert boundary[0] == 0.0 and boundary[-1] == 1.0
  with pytest.raises(ValueError):
    unit_grid(1)


def test_grid_metrics_density_identity_is_zero() -> None:
  c = np.random.default_rng(0).uniform(0.1, 2.0, (8, 8))
  m = grid_metrics_density(c, c, reduction="mean")
  assert m["ISE"] == pytest.approx(0.0)
  assert m["IAE"] == pytest.approx(0.0)
  assert m["KL"] == pytest.approx(0.0)


def test_grid_metrics_density_constant_offset() -> None:
  c_true = np.ones((5, 5))
  c_hat = c_true + 0.3
  m = grid_metrics_density(c_true, c_hat, reduction="mean")
  assert m["ISE"] == pytest.approx(0.09)
  assert m["IAE"] == pytest.approx(0.3)


def test_grid_metrics_density_kl_nonnegative_for_normalized() -> None:
  # KL >= 0 holds when both densities are normalized (integrate to 1).
  g = unit_grid(40, interior=False)
  uu, vv = np.meshgrid(g, g, indexing="ij")
  c_true = np.ones_like(uu)
  c_hat = 1.0 + 0.4 * (
    uu - 0.5
  )  # tilted but still integrates to 1 on the square
  same = grid_metrics_density(
    c_hat, c_hat, u_grid=g, v_grid=g, reduction="trapezoid"
  )
  assert same["KL"] == pytest.approx(0.0, abs=1e-9)
  diff = grid_metrics_density(
    c_true, c_hat, u_grid=g, v_grid=g, reduction="trapezoid"
  )
  assert diff["KL"] > 0.0


def test_grid_metrics_density_mean_matches_manual() -> None:
  rng = np.random.default_rng(1)
  c_true = rng.uniform(0.2, 1.5, (6, 6))
  c_hat = rng.uniform(0.2, 1.5, (6, 6))
  m = grid_metrics_density(c_true, c_hat, reduction="mean")
  assert m["ISE"] == pytest.approx(np.mean((c_hat - c_true) ** 2))
  assert m["IAE"] == pytest.approx(np.mean(np.abs(c_hat - c_true)))


def test_grid_metrics_density_trapezoid_integrates_uniform() -> None:
  g = unit_grid(50, interior=False)
  c_true = np.ones((50, 50))
  c_hat = np.ones((50, 50))
  m = grid_metrics_density(
    c_true, c_hat, u_grid=g, v_grid=g, reduction="trapezoid"
  )
  assert m["ISE"] == pytest.approx(0.0)
  # Offset by 1 -> ISE = ∫∫ 1 = 1 over the unit square.
  m2 = grid_metrics_density(
    c_true, c_hat + 1.0, u_grid=g, v_grid=g, reduction="trapezoid"
  )
  assert m2["ISE"] == pytest.approx(1.0, abs=1e-6)
  assert m2["IAE"] == pytest.approx(1.0, abs=1e-6)


def test_grid_metrics_density_trapezoid_requires_grids() -> None:
  c = np.ones((3, 3))
  with pytest.raises(ValueError):
    grid_metrics_density(c, c, reduction="trapezoid")


def test_grid_metrics_density_shape_mismatch_raises() -> None:
  with pytest.raises(ValueError):
    grid_metrics_density(np.ones((3, 3)), np.ones((3, 4)))


def test_grid_metrics_hfunc() -> None:
  h = np.linspace(0, 1, 16).reshape(4, 4)
  m = grid_metrics_hfunc(h, h, reduction="mean")
  assert m["ISE"] == pytest.approx(0.0) and m["IAE"] == pytest.approx(0.0)
  m2 = grid_metrics_hfunc(h, h + 0.2, reduction="mean")
  assert m2["IAE"] == pytest.approx(0.2)


def test_conditional_density_grids_conditional() -> None:
  spec = ConditionalGridSpec(
    u_levels=(0.25, 0.75),
    v_levels=(0.5,),
    x_values=np.array([[0.0], [1.0], [2.0]]),
  )

  # density_fn returns u + v + x[:, 0] so we can predict the layout exactly.
  def density_fn(
    u: np.ndarray, v: np.ndarray, x: np.ndarray | None
  ) -> np.ndarray:
    assert x is not None
    return u + v + x[:, 0]

  grid = conditional_density_grids(density_fn, spec)
  assert grid.shape == (2, 3)  # 2 (u,v) pairs x 3 covariate rows
  # pair (0.25, 0.5) across x in {0,1,2}
  np.testing.assert_allclose(grid[0], [0.75, 1.75, 2.75])
  np.testing.assert_allclose(grid[1], [1.25, 2.25, 3.25])


def test_conditional_density_grids_unconditional() -> None:
  spec = ConditionalGridSpec(
    u_levels=(0.2, 0.8), v_levels=(0.3, 0.7), x_values=None
  )

  def density_fn(
    u: np.ndarray, v: np.ndarray, x: np.ndarray | None
  ) -> np.ndarray:
    return u * v

  grid = conditional_density_grids(density_fn, spec)
  assert grid.shape == (4, 1)


def test_conditional_metrics_reductions() -> None:
  true_grid = np.ones((4, 3))
  est_grid = true_grid + 0.5
  assert conditional_metrics(true_grid, est_grid, reduce_over="all")[
    "ISE"
  ] == pytest.approx(0.25)
  per_x = conditional_metrics(true_grid, est_grid, reduce_over="x")["IAE"]
  assert per_x.shape == (3,) and np.allclose(per_x, 0.5)
  per_uv = conditional_metrics(true_grid, est_grid, reduce_over="uv")["IAE"]
  assert per_uv.shape == (4,)
  with pytest.raises(ValueError):
    conditional_metrics(np.ones((2, 2)), np.ones((2, 3)))


class _FakeEstimator:
  """Duck-typed estimator exposing hfunc1/hfunc2 as fixed transforms of the PIT."""

  def __init__(
    self,
    t1: Callable[[np.ndarray], np.ndarray],
    t2: Callable[[np.ndarray], np.ndarray] | None = None,
  ) -> None:
    self._t1, self._t2 = t1, t2

  def hfunc1(
    self, u: np.ndarray, v: np.ndarray, x: np.ndarray | None = None
  ) -> np.ndarray:
    return self._t1(np.asarray(v, float))

  def hfunc2(
    self, u: np.ndarray, v: np.ndarray, x: np.ndarray | None = None
  ) -> np.ndarray:
    if self._t2 is None:
      raise RuntimeError("no hfunc2")
    return self._t2(np.asarray(u, float))


def test_margin_calibration_uniform_is_small() -> None:
  rng = np.random.default_rng(7)
  u = rng.uniform(0, 1, 4000)
  v = rng.uniform(0, 1, 4000)
  cal = margin_calibration(_FakeEstimator(lambda v: v), u, v)
  assert set(cal) == {"ks_h1"}
  assert cal["ks_h1"] < 0.05


def test_margin_calibration_miscalibrated_is_larger() -> None:
  rng = np.random.default_rng(8)
  u = rng.uniform(0, 1, 4000)
  v = rng.uniform(0, 1, 4000)
  good = margin_calibration(_FakeEstimator(lambda v: v), u, v)["ks_h1"]
  bad = margin_calibration(_FakeEstimator(lambda v: v**2), u, v)["ks_h1"]
  assert bad > good + 0.1


def test_margin_calibration_includes_hfunc2_when_present() -> None:
  rng = np.random.default_rng(9)
  u = rng.uniform(0, 1, 2000)
  v = rng.uniform(0, 1, 2000)
  cal = margin_calibration(_FakeEstimator(lambda v: v, lambda u: u), u, v)
  assert "ks_h2" in cal and cal["ks_h2"] < 0.06


def test_grid_metrics_hfunc_shape_mismatch_raises() -> None:
  with pytest.raises(ValueError):
    grid_metrics_hfunc(np.ones((2, 2)), np.ones((2, 3)))


def test_conditional_density_grids_accepts_1d_x_values() -> None:
  spec = ConditionalGridSpec(
    u_levels=(0.3,), v_levels=(0.5,), x_values=np.array([0.0, 1.0])
  )

  def density_fn(
    u: np.ndarray, v: np.ndarray, x: np.ndarray | None
  ) -> np.ndarray:
    assert x is not None
    return u + v + x[:, 0]

  grid = conditional_density_grids(density_fn, spec)
  assert grid.shape == (1, 2)
  np.testing.assert_allclose(grid[0], [0.8, 1.8])
