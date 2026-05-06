"""Tests for ``TabPFNDensity1D`` (criterion-based density)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from npcc.tabpfn_density1d import TabPFNDensity1D
from tests.conftest import uniform_density_y


class TestTabPFNDensity1D:
  def test_density_before_fit_raises(self, patch_uniform: None) -> None:
    d = TabPFNDensity1D()
    with pytest.raises(RuntimeError, match="not fitted"):
      d.density(np.zeros((1, 1)), np.array([0.5]))

  def test_fit_accepts_1d_w(self, patch_uniform: None) -> None:
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.linspace(0.1, 0.9, 20), np.linspace(0.1, 0.9, 20))
    assert d.model_ is not None

  def test_fit_accepts_2d_w(self, patch_uniform: None) -> None:
    rng = np.random.default_rng(0)
    d = TabPFNDensity1D(transform="logit")
    d.fit(rng.uniform(0.1, 0.9, (20, 3)), rng.uniform(0.1, 0.9, 20))
    assert d.model_ is not None

  def test_fit_rejects_length_mismatch(self, patch_uniform: None) -> None:
    d = TabPFNDensity1D(transform="logit")
    with pytest.raises(ValueError, match="incompatible"):
      d.fit(np.zeros((5, 1)), np.zeros(6))

  def test_density_rejects_length_mismatch(self, patch_uniform: None) -> None:
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.zeros((10, 1)), np.full(10, 0.5))
    with pytest.raises(ValueError, match="incompatible"):
      d.density(np.zeros((5, 1)), np.full(6, 0.5))

  def test_density_rejects_nonpositive_batch_size(
    self, patch_uniform: None
  ) -> None:
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.zeros((10, 1)), np.full(10, 0.5))
    with pytest.raises(ValueError, match="batch_size"):
      d.density(np.zeros((2, 1)), np.array([0.3, 0.5]), batch_size=0)

  def test_density_logit_jacobian_matches_analytic(
    self, patch_uniform: None
  ) -> None:
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.zeros((10, 1)), np.full(10, 0.5))

    y = np.array([0.3, 0.5, 0.7])
    w = np.zeros((len(y), 1))
    actual = d.density(w, y)
    expected = uniform_density_y(y)
    np.testing.assert_allclose(actual, expected, atol=1e-6)

  def test_density_zero_outside_support(self, patch_uniform: None) -> None:
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.zeros((10, 1)), np.full(10, 0.5))

    y = np.array([0.02, 0.98])
    w = np.zeros((len(y), 1))
    out = d.density(w, y)
    np.testing.assert_array_equal(out, np.zeros_like(out))

  def test_density_respects_batch_size(self, patch_uniform: None) -> None:
    rng = np.random.default_rng(0)
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.zeros((10, 1)), np.full(10, 0.5))

    n = 17  # not a multiple of batch_size
    y = rng.uniform(0.15, 0.85, n)
    w = np.zeros((n, 1))
    full = d.density(w, y)
    chunked = d.density(w, y, batch_size=4)
    np.testing.assert_allclose(full, chunked, atol=1e-8)

  def test_density_grid_shape(self, patch_uniform: None) -> None:
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.zeros((10, 1)), np.full(10, 0.5))

    w = np.zeros((4, 1))
    y_grid = np.array([0.3, 0.5, 0.7])
    out = d.density_grid(w, y_grid)
    assert out.shape == (4, 3)

  def test_density_grid_matches_density(self, patch_uniform: None) -> None:
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.zeros((10, 1)), np.full(10, 0.5))

    w = np.zeros((3, 1))
    y_grid = np.array([0.3, 0.5, 0.7])
    grid = d.density_grid(w, y_grid)

    w_tiled = np.repeat(w, len(y_grid), axis=0)
    y_tiled = np.tile(y_grid, w.shape[0])
    tiled = d.density(w_tiled, y_tiled).reshape(w.shape[0], len(y_grid))
    np.testing.assert_allclose(grid, tiled, atol=1e-8)

  def test_density_grid_before_fit_raises(self, patch_uniform: None) -> None:
    d = TabPFNDensity1D()
    with pytest.raises(RuntimeError, match="not fitted"):
      d.density_grid(np.zeros((1, 1)), np.array([0.5]))

  def test_cdf_before_fit_raises(self, patch_uniform: None) -> None:
    d = TabPFNDensity1D()
    with pytest.raises(RuntimeError, match="not fitted"):
      d.cdf(np.zeros((1, 1)), np.array([0.5]))

  def test_cdf_rejects_length_mismatch(self, patch_uniform: None) -> None:
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.zeros((10, 1)), np.full(10, 0.5))
    with pytest.raises(ValueError, match="incompatible"):
      d.cdf(np.zeros((5, 1)), np.full(6, 0.5))

  def test_cdf_rejects_nonpositive_batch_size(
    self, patch_uniform: None
  ) -> None:
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.zeros((10, 1)), np.full(10, 0.5))
    with pytest.raises(ValueError, match="batch_size"):
      d.cdf(np.zeros((2, 1)), np.array([0.3, 0.5]), batch_size=0)

  def test_cdf_logit_matches_analytic(self, patch_uniform: None) -> None:
    """Under the fake (Z ~ Uniform(-2, 2)), F_Y(y) = clip((logit(y)+2)/4, 0, 1)."""
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.zeros((10, 1)), np.full(10, 0.5))

    y = np.array([0.3, 0.5, 0.7])
    w = np.zeros((len(y), 1))
    actual = d.cdf(w, y)
    expected = np.clip((np.log(y / (1 - y)) + 2.0) / 4.0, 0.0, 1.0)
    np.testing.assert_allclose(actual, expected, atol=1e-6)

  def test_cdf_clips_outside_support(self, patch_uniform: None) -> None:
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.zeros((10, 1)), np.full(10, 0.5))

    y = np.array([0.02, 0.98])
    w = np.zeros((len(y), 1))
    out = d.cdf(w, y)
    np.testing.assert_allclose(out, np.array([0.0, 1.0]), atol=1e-6)

  def test_cdf_respects_batch_size(self, patch_uniform: None) -> None:
    rng = np.random.default_rng(0)
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.zeros((10, 1)), np.full(10, 0.5))

    n = 17
    y = rng.uniform(0.15, 0.85, n)
    w = np.zeros((n, 1))
    full = d.cdf(w, y)
    chunked = d.cdf(w, y, batch_size=4)
    np.testing.assert_allclose(full, chunked, atol=1e-8)

  def test_cdf_grid_shape(self, patch_uniform: None) -> None:
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.zeros((10, 1)), np.full(10, 0.5))

    w = np.zeros((4, 1))
    y_grid = np.array([0.3, 0.5, 0.7])
    out = d.cdf_grid(w, y_grid)
    assert out.shape == (4, 3)

  def test_cdf_grid_matches_cdf(self, patch_uniform: None) -> None:
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.zeros((10, 1)), np.full(10, 0.5))

    w = np.zeros((3, 1))
    y_grid = np.array([0.3, 0.5, 0.7])
    grid = d.cdf_grid(w, y_grid)

    w_tiled = np.repeat(w, len(y_grid), axis=0)
    y_tiled = np.tile(y_grid, w.shape[0])
    tiled = d.cdf(w_tiled, y_tiled).reshape(w.shape[0], len(y_grid))
    np.testing.assert_allclose(grid, tiled, atol=1e-8)

  def test_cdf_grid_before_fit_raises(self, patch_uniform: None) -> None:
    d = TabPFNDensity1D()
    with pytest.raises(RuntimeError, match="not fitted"):
      d.cdf_grid(np.zeros((1, 1)), np.array([0.5]))

  def test_cdf_monotone_in_y(self, patch_uniform: None) -> None:
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.zeros((10, 1)), np.full(10, 0.5))

    y_sorted = np.linspace(0.05, 0.95, 20)
    w = np.zeros((len(y_sorted), 1))
    cdf_vals = d.cdf(w, y_sorted)
    assert (np.diff(cdf_vals) >= -1e-9).all()

  def test_icdf_inverts_cdf_under_uniform_fake(
    self, patch_uniform: None
  ) -> None:
    """Under Z ~ Uniform(-2, 2) + logit, Q(α) = sigmoid(-2 + 4α)."""
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.zeros((10, 1)), np.full(10, 0.5))

    alphas = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    w = np.zeros((len(alphas), 1))
    y_hat = d.icdf(w, alphas)
    z_expected = -2.0 + 4.0 * alphas
    expected = 1.0 / (1.0 + np.exp(-z_expected))
    np.testing.assert_allclose(y_hat, expected, atol=1e-6)

  def test_icdf_rejects_alpha_outside_unit(self, patch_uniform: None) -> None:
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.zeros((10, 1)), np.full(10, 0.5))
    with pytest.raises(ValueError, match="strictly inside"):
      d.icdf(np.zeros((2, 1)), np.array([0.5, 1.0]))

  def test_icdf_rejects_length_mismatch(self, patch_uniform: None) -> None:
    d = TabPFNDensity1D(transform="logit")
    d.fit(np.zeros((10, 1)), np.full(10, 0.5))
    with pytest.raises(ValueError, match="incompatible"):
      d.icdf(np.zeros((5, 1)), np.array([0.5]))

  def test_identity_transform_returns_z_density(
    self, patch_uniform: None
  ) -> None:
    d = TabPFNDensity1D(transform="identity")
    d.fit(np.zeros((10, 1)), np.zeros(10))

    z = np.array([-1.0, 0.0, 1.0])
    w = np.zeros((len(z), 1))
    actual = d.density(w, z)
    np.testing.assert_allclose(actual, np.full(3, 0.25), atol=1e-8)

  def test_unknown_transform_raises(self) -> None:
    bad: Any = "exp"
    d = TabPFNDensity1D(transform=bad)
    with pytest.raises(ValueError, match="Unknown transform"):
      d._transform_y(np.array([0.5]))
