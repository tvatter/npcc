"""Tests for ``QuantileGridConfig`` and ``TabPFNQuantileDistribution1D``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from npcc.core.tabpfn_quantile_distribution1d import (
  QuantileGridConfig,
  TabPFNQuantileDistribution1D,
)
from tests.conftest import uniform_density_y


def _stdnorm_cdf(z: np.ndarray) -> np.ndarray:
  z_t = torch.as_tensor(z, dtype=torch.float64)
  out = 0.5 * (1.0 + torch.erf(z_t / np.sqrt(2.0)))
  return out.detach().cpu().numpy()


def _stdnorm_ppf(p: np.ndarray) -> np.ndarray:
  p_t = torch.as_tensor(p, dtype=torch.float64)
  out = np.sqrt(2.0) * torch.erfinv(2.0 * p_t - 1.0)
  return out.detach().cpu().numpy()


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

  def test_probit_transform_pdf_cdf_icdf_match_analytic(
    self, patch_uniform: None
  ) -> None:
    qd = TabPFNQuantileDistribution1D(transform="probit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))

    y = np.array([0.2, 0.5, 0.8])
    w = np.zeros((len(y), 1))
    z = _stdnorm_ppf(y)
    phi_z = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)

    pdf_actual = qd.pdf(w, y)
    pdf_expected = 0.25 / phi_z
    np.testing.assert_allclose(pdf_actual, pdf_expected, atol=1e-6)

    cdf_actual = qd.cdf(w, y)
    cdf_expected = np.clip((z + 2.0) / 4.0, 0.0, 1.0)
    np.testing.assert_allclose(cdf_actual, cdf_expected, atol=1e-3)

    alphas = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    w_alpha = np.zeros((len(alphas), 1))
    icdf_actual = qd.icdf(w_alpha, alphas)
    icdf_expected = _stdnorm_cdf(-2.0 + 4.0 * alphas)
    np.testing.assert_allclose(icdf_actual, icdf_expected, atol=1e-3)

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

  # batch_size chunking ----------------------------------------------

  def test_pdf_respects_batch_size(self, patch_uniform: None) -> None:
    rng = np.random.default_rng(0)
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))

    n = 17  # not a multiple of batch_size
    y = rng.uniform(0.15, 0.85, n)
    w = np.zeros((n, 1))
    full = qd.pdf(w, y)
    chunked = qd.pdf(w, y, batch_size=4)
    np.testing.assert_allclose(full, chunked, atol=1e-10)

  def test_cdf_respects_batch_size(self, patch_uniform: None) -> None:
    rng = np.random.default_rng(1)
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))

    n = 17
    y = rng.uniform(0.15, 0.85, n)
    w = np.zeros((n, 1))
    np.testing.assert_allclose(
      qd.cdf(w, y), qd.cdf(w, y, batch_size=4), atol=1e-12
    )

  def test_pdf_chunks_with_instance_batch_size(
    self, patch_uniform: None, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    rng = np.random.default_rng(2)
    qd = TabPFNQuantileDistribution1D(transform="logit", batch_size=4)
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))
    assert qd.model_ is not None

    n = 17
    y = rng.uniform(0.15, 0.85, n)
    w = np.zeros((n, 1))
    calls = 0
    original_predict = qd.model_.predict

    def _spy_predict(
      x: np.ndarray,
      *,
      output_type: str = "mean",
      quantiles: list[float] | None = None,
    ) -> object:
      nonlocal calls
      calls += 1
      return original_predict(x, output_type=output_type, quantiles=quantiles)

    monkeypatch.setattr(qd.model_, "predict", _spy_predict)
    qd.pdf(w, y)
    assert calls == 5  # ceil(17 / 4)

  def test_square_chunk_not_transposed(self, patch_uniform: None) -> None:
    """A chunk whose row count equals n_quantiles must not be transposed.

    Regression: the per-chunk orientation check used to test the
    transposed shape first, so a square ``(n_chunk, n_alphas)`` chunk was
    silently transposed and its rows scrambled.
    """
    rng = np.random.default_rng(7)
    cfg = QuantileGridConfig(n_quantiles=11)
    qd = TabPFNQuantileDistribution1D(transform="logit", config=cfg)
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))

    n = 15  # first chunk is square (11, 11), second is (4, 11)
    y = rng.uniform(0.15, 0.85, n)
    w = np.zeros((n, 1))
    chunked = qd.cdf(w, y, batch_size=11)
    full = qd.cdf(w, y, batch_size=1000)
    np.testing.assert_allclose(chunked, full, atol=1e-12)

  # pdf_grid ----------------------------------------------------------

  def test_pdf_grid_before_fit_raises(self, patch_uniform: None) -> None:
    qd = TabPFNQuantileDistribution1D()
    with pytest.raises(RuntimeError, match="not fitted"):
      qd.pdf_grid(np.zeros((2, 1)), np.array([0.3, 0.5]))

  def test_pdf_grid_empty_y_raises(self, patch_uniform: None) -> None:
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))
    with pytest.raises(ValueError, match="at least one value"):
      qd.pdf_grid(np.zeros((2, 1)), np.array([]))

  def test_pdf_grid_shape(self, patch_uniform: None) -> None:
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))
    out = qd.pdf_grid(np.zeros((4, 1)), np.array([0.3, 0.5, 0.7]))
    assert out.shape == (4, 3)

  def test_pdf_grid_matches_pdf(self, patch_uniform: None) -> None:
    """The grid fast path must equal pdf on the explicit tile."""
    rng = np.random.default_rng(5)
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(rng.uniform(0.1, 0.9, (20, 2)), rng.uniform(0.1, 0.9, 20))

    w = rng.uniform(0.1, 0.9, (3, 2))
    y_grid = np.array([0.25, 0.4, 0.6, 0.75])
    grid = qd.pdf_grid(w, y_grid)

    w_tiled = np.repeat(w, len(y_grid), axis=0)
    y_tiled = np.tile(y_grid, w.shape[0])
    tiled = qd.pdf(w_tiled, y_tiled).reshape(w.shape[0], len(y_grid))
    np.testing.assert_allclose(grid, tiled, atol=1e-10)

  def test_pdf_grid_zero_outside_support(self, patch_uniform: None) -> None:
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(np.zeros((10, 1)), np.full(10, 0.5))
    out = qd.pdf_grid(np.zeros((2, 1)), np.array([0.02, 0.98]))
    np.testing.assert_array_equal(out, np.zeros_like(out))

  def test_pdf_grid_chunked_matches_full(self, patch_uniform: None) -> None:
    rng = np.random.default_rng(6)
    qd = TabPFNQuantileDistribution1D(transform="logit")
    qd.fit(rng.uniform(0.1, 0.9, (20, 1)), rng.uniform(0.1, 0.9, 20))

    w = rng.uniform(0.1, 0.9, (17, 1))  # not a multiple of batch_size
    y_grid = np.array([0.3, 0.5, 0.7])
    np.testing.assert_allclose(
      qd.pdf_grid(w, y_grid),
      qd.pdf_grid(w, y_grid, batch_size=4),
      atol=1e-10,
    )
