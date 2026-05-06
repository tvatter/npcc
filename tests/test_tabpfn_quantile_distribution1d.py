"""Tests for ``QuantileGridConfig`` and ``TabPFNQuantileDistribution1D``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from npcc.tabpfn_quantile_distribution1d import (
  QuantileGridConfig,
  TabPFNQuantileDistribution1D,
)
from tests.conftest import uniform_density_y


class TestQuantileGridConfig:
  def test_default_alphas_shape(self) -> None:
    cfg = QuantileGridConfig()
    assert cfg.alphas().shape == (cfg.n_quantiles,)

  def test_alphas_endpoints(self) -> None:
    cfg = QuantileGridConfig(n_quantiles=11, alpha_min=0.05, alpha_max=0.95)
    a = cfg.alphas()
    assert a[0] == pytest.approx(0.05)
    assert a[-1] == pytest.approx(0.95)

  def test_alphas_rejects_inverted_bounds(self) -> None:
    cfg = QuantileGridConfig(alpha_min=0.5, alpha_max=0.4)
    with pytest.raises(ValueError, match="alpha_min"):
      cfg.alphas()

  def test_alphas_rejects_alpha_min_zero(self) -> None:
    cfg = QuantileGridConfig(alpha_min=0.0)
    with pytest.raises(ValueError, match="alpha_min"):
      cfg.alphas()

  def test_alphas_rejects_alpha_max_one(self) -> None:
    cfg = QuantileGridConfig(alpha_max=1.0)
    with pytest.raises(ValueError, match="alpha_min"):
      cfg.alphas()

  def test_alphas_requires_at_least_5(self) -> None:
    cfg = QuantileGridConfig(n_quantiles=4)
    with pytest.raises(ValueError, match="n_quantiles"):
      cfg.alphas()


class TestQuantileDistribution1D:
  def test_pdf_before_fit_raises(self, patch_uniform: None) -> None:
    qd = TabPFNQuantileDistribution1D()
    with pytest.raises(RuntimeError, match="not fitted"):
      qd.pdf(np.zeros((1, 1)), np.array([0.5]))

  def test_fit_accepts_1d_w(self, patch_uniform: None) -> None:
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(np.linspace(0.1, 0.9, 20), np.linspace(0.1, 0.9, 20))
    assert qd.model_ is not None

  def test_fit_accepts_2d_w(self, patch_uniform: None) -> None:
    rng = np.random.default_rng(0)
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(rng.uniform(0.1, 0.9, (20, 3)), rng.uniform(0.1, 0.9, 20))
    assert qd.model_ is not None

  def test_fit_rejects_length_mismatch(self, patch_uniform: None) -> None:
    qd = TabPFNQuantileDistribution1D(transform="logit")
    with pytest.raises(ValueError, match="incompatible"):
      qd.fit(np.zeros((5, 1)), np.zeros(6))

  def test_pdf_rejects_length_mismatch(self, patch_uniform: None) -> None:
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))
    with pytest.raises(ValueError, match="incompatible"):
      qd.pdf(np.zeros((5, 1)), np.full(6, 0.5))

  def test_pdf_logit_jacobian_matches_analytic(
    self, patch_uniform: None
  ) -> None:
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))

    y = np.array([0.3, 0.5, 0.7])
    w = np.zeros((len(y), 1))
    actual = qd.pdf(w, y)
    expected = uniform_density_y(y)
    np.testing.assert_allclose(actual, expected, atol=1e-8)

  def test_pdf_zero_outside_support(self, patch_uniform: None) -> None:
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))

    y = np.array([0.02, 0.98])
    w = np.zeros((len(y), 1))
    out = qd.pdf(w, y)
    np.testing.assert_array_equal(out, np.zeros_like(out))

  def test_pdf_handles_transposed_output(self, patch_transposed: None) -> None:
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))

    y = np.array([0.3, 0.5, 0.7])
    w = np.zeros((len(y), 1))
    actual = qd.pdf(w, y)
    expected = uniform_density_y(y)
    np.testing.assert_allclose(actual, expected, atol=1e-8)

  def test_pdf_rejects_bad_shape(self, patch_bad_shape: None) -> None:
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))

    with pytest.raises(RuntimeError, match="quantile output shape"):
      qd.pdf(np.zeros((2, 1)), np.array([0.3, 0.5]))

  def test_identity_transform_returns_z_density(
    self, patch_uniform: None
  ) -> None:
    qd = TabPFNQuantileDistribution1D(transform="identity")
    qd.fit(np.zeros((10, 1)), np.zeros(10))

    z = np.array([-1.0, 0.0, 1.0])
    w = np.zeros((len(z), 1))
    actual = qd.pdf(w, z)
    np.testing.assert_allclose(actual, np.full(3, 0.25), atol=1e-8)

  def test_unknown_transform_raises(self) -> None:
    bad: Any = "exp"  # bypass Literal type for test
    qd = TabPFNQuantileDistribution1D(transform=bad)
    with pytest.raises(ValueError, match="Unknown transform"):
      qd._transform_y(torch.tensor([0.5]))

  # CDF tests ---------------------------------------------------------

  def test_cdf_rejects_length_mismatch(self, patch_uniform: None) -> None:
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))
    with pytest.raises(ValueError, match="incompatible"):
      qd.cdf(np.zeros((5, 1)), np.full(6, 0.5))

  def test_cdf_logit_matches_analytic(self, patch_uniform: None) -> None:
    """F_Y(y) = clip((logit(y)+2)/4, 0, 1) under the uniform fake."""
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))

    y = np.array([0.3, 0.5, 0.7])
    w = np.zeros((len(y), 1))
    actual = qd.cdf(w, y)
    expected = np.clip((np.log(y / (1 - y)) + 2.0) / 4.0, 0.0, 1.0)
    np.testing.assert_allclose(actual, expected, atol=1e-3)

  def test_cdf_outside_support_clips_to_alpha_bounds(
    self, patch_uniform: None
  ) -> None:
    """Below qi[0] returns alpha_min; above qi[-1] returns alpha_max."""
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))

    y = np.array([0.02, 0.98])  # logit ≈ ±3.89, outside (-2, 2)
    w = np.zeros((len(y), 1))
    out = qd.cdf(w, y)
    cfg = qd.config
    np.testing.assert_allclose(
      out, np.array([cfg.alpha_min, cfg.alpha_max]), atol=1e-12
    )

  def test_cdf_monotone_in_y(self, patch_uniform: None) -> None:
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))

    y_sorted = np.linspace(0.05, 0.95, 20)
    w = np.zeros((len(y_sorted), 1))
    cdf_vals = qd.cdf(w, y_sorted)
    assert (np.diff(cdf_vals) >= -1e-9).all()

  # icdf tests --------------------------------------------------------

  def test_icdf_matches_analytic(self, patch_uniform: None) -> None:
    """Q(α | w) = sigmoid(-2 + 4α) under Z ~ Uniform(-2, 2) + logit."""
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))

    alphas = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    w = np.zeros((len(alphas), 1))
    y_hat = qd.icdf(w, alphas)
    z_expected = -2.0 + 4.0 * alphas
    expected = 1.0 / (1.0 + np.exp(-z_expected))
    np.testing.assert_allclose(y_hat, expected, atol=1e-3)

  def test_icdf_rejects_alpha_outside_unit(self, patch_uniform: None) -> None:
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))
    with pytest.raises(ValueError, match="strictly inside"):
      qd.icdf(np.zeros((2, 1)), np.array([0.5, 0.0]))

  def test_icdf_rejects_length_mismatch(self, patch_uniform: None) -> None:
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))
    with pytest.raises(ValueError, match="incompatible"):
      qd.icdf(np.zeros((5, 1)), np.array([0.5]))
