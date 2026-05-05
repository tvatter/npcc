"""Tests for ``QuantileDensityConfig`` and ``TabPFNQuantileDensity1D``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from npcc.tabpfn_quantile_density1d import (
  QuantileDensityConfig,
  TabPFNQuantileDensity1D,
)
from tests.conftest import uniform_density_y


class TestQuantileDensityConfig:
  def test_default_alphas_shape(self) -> None:
    cfg = QuantileDensityConfig()
    assert cfg.alphas().shape == (cfg.n_quantiles,)

  def test_alphas_endpoints(self) -> None:
    cfg = QuantileDensityConfig(n_quantiles=11, alpha_min=0.05, alpha_max=0.95)
    a = cfg.alphas()
    assert a[0] == pytest.approx(0.05)
    assert a[-1] == pytest.approx(0.95)

  def test_alphas_rejects_inverted_bounds(self) -> None:
    cfg = QuantileDensityConfig(alpha_min=0.5, alpha_max=0.4)
    with pytest.raises(ValueError, match="alpha_min"):
      cfg.alphas()

  def test_alphas_rejects_alpha_min_zero(self) -> None:
    cfg = QuantileDensityConfig(alpha_min=0.0)
    with pytest.raises(ValueError, match="alpha_min"):
      cfg.alphas()

  def test_alphas_rejects_alpha_max_one(self) -> None:
    cfg = QuantileDensityConfig(alpha_max=1.0)
    with pytest.raises(ValueError, match="alpha_min"):
      cfg.alphas()

  def test_alphas_requires_at_least_5(self) -> None:
    cfg = QuantileDensityConfig(n_quantiles=4)
    with pytest.raises(ValueError, match="n_quantiles"):
      cfg.alphas()


class TestQuantileDensity1D:
  def test_density_before_fit_raises(self, patch_uniform: None) -> None:
    qd = TabPFNQuantileDensity1D()
    with pytest.raises(RuntimeError, match="not fitted"):
      qd.density(np.zeros((1, 1)), np.array([0.5]))

  def test_fit_accepts_1d_w(self, patch_uniform: None) -> None:
    qd = TabPFNQuantileDensity1D(transform="logit")
    qd.fit(np.linspace(0.1, 0.9, 20), np.linspace(0.1, 0.9, 20))
    assert qd.model_ is not None

  def test_fit_accepts_2d_w(self, patch_uniform: None) -> None:
    rng = np.random.default_rng(0)
    qd = TabPFNQuantileDensity1D(transform="logit")
    qd.fit(rng.uniform(0.1, 0.9, (20, 3)), rng.uniform(0.1, 0.9, 20))
    assert qd.model_ is not None

  def test_fit_rejects_length_mismatch(self, patch_uniform: None) -> None:
    qd = TabPFNQuantileDensity1D(transform="logit")
    with pytest.raises(ValueError, match="incompatible"):
      qd.fit(np.zeros((5, 1)), np.zeros(6))

  def test_density_rejects_length_mismatch(self, patch_uniform: None) -> None:
    qd = TabPFNQuantileDensity1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))
    with pytest.raises(ValueError, match="incompatible"):
      qd.density(np.zeros((5, 1)), np.full(6, 0.5))

  def test_density_logit_jacobian_matches_analytic(
    self, patch_uniform: None
  ) -> None:
    qd = TabPFNQuantileDensity1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))

    y = np.array([0.3, 0.5, 0.7])
    w = np.zeros((len(y), 1))
    actual = qd.density(w, y)
    expected = uniform_density_y(y)
    np.testing.assert_allclose(actual, expected, atol=1e-8)

  def test_density_zero_outside_support(self, patch_uniform: None) -> None:
    qd = TabPFNQuantileDensity1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))

    y = np.array([0.02, 0.98])
    w = np.zeros((len(y), 1))
    out = qd.density(w, y)
    np.testing.assert_array_equal(out, np.zeros_like(out))

  def test_density_handles_transposed_output(
    self, patch_transposed: None
  ) -> None:
    qd = TabPFNQuantileDensity1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))

    y = np.array([0.3, 0.5, 0.7])
    w = np.zeros((len(y), 1))
    actual = qd.density(w, y)
    expected = uniform_density_y(y)
    np.testing.assert_allclose(actual, expected, atol=1e-8)

  def test_density_rejects_bad_shape(self, patch_bad_shape: None) -> None:
    qd = TabPFNQuantileDensity1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))

    with pytest.raises(RuntimeError, match="quantile output shape"):
      qd.density(np.zeros((2, 1)), np.array([0.3, 0.5]))

  def test_identity_transform_returns_z_density(
    self, patch_uniform: None
  ) -> None:
    qd = TabPFNQuantileDensity1D(transform="identity")
    qd.fit(np.zeros((10, 1)), np.zeros(10))

    z = np.array([-1.0, 0.0, 1.0])
    w = np.zeros((len(z), 1))
    actual = qd.density(w, z)
    np.testing.assert_allclose(actual, np.full(3, 0.25), atol=1e-8)

  def test_unknown_transform_raises(self) -> None:
    bad: Any = "exp"  # bypass Literal type for test
    qd = TabPFNQuantileDensity1D(transform=bad)
    with pytest.raises(ValueError, match="Unknown transform"):
      qd._transform_y(np.array([0.5]))
