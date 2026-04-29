"""Tests for TPRBicop and TabPFNQuantileDensity1D."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

pytest.importorskip("tabpfn")

from npcc.tprbicop import (  # noqa: E402
  QuantileDensityConfig,
  TabPFNQuantileDensity1D,
  TPRBicop,
  _as_2d,
  _check_uv,
  _logit,
  _sigmoid,
)


# ===========================================================================
# Fake TabPFN regressors
# ===========================================================================


class _UniformQuantileRegressor:
  """
  Mimics ``TabPFNRegressor`` with predictable outputs.

  For any input ``X``, returns the quantiles of Z ~ Uniform(-2, 2).
  Used to test the quantile-derivative density recovery analytically:

    Q(alpha) = -2 + 4 * alpha,
    f_Z(z)   = 1/4 for z in (-2, 2),
    f_Y(y)   = (1/4) / (y * (1 - y)) when transform="logit".
  """

  Q_LO: float = -2.0
  Q_HI: float = 2.0

  def __init__(self, **_: object) -> None:
    self.fitted_: bool = False

  def fit(self, X: np.ndarray, y: np.ndarray) -> _UniformQuantileRegressor:
    self.fitted_ = True
    return self

  def predict(
    self,
    X: np.ndarray,
    *,
    output_type: str = "mean",
    quantiles: list[float] | None = None,
  ) -> np.ndarray:
    if output_type != "quantiles" or quantiles is None:
      raise ValueError("Fake regressor only supports output_type='quantiles'.")
    alphas = np.asarray(quantiles, dtype=float)
    qrow = self.Q_LO + (self.Q_HI - self.Q_LO) * alphas
    n = X.shape[0]
    return np.broadcast_to(qrow[None, :], (n, len(alphas))).copy()


class _UniformQuantileRegressorTransposed(_UniformQuantileRegressor):
  """Same distribution, but returns shape ``(n_q, n_obs)``."""

  def predict(
    self,
    X: np.ndarray,
    *,
    output_type: str = "mean",
    quantiles: list[float] | None = None,
  ) -> np.ndarray:
    out = super().predict(X, output_type=output_type, quantiles=quantiles)
    return out.T


class _BadShapeQuantileRegressor(_UniformQuantileRegressor):
  """Returns a 3-D array to trigger the unexpected-shape error."""

  def predict(
    self,
    X: np.ndarray,
    *,
    output_type: str = "mean",
    quantiles: list[float] | None = None,
  ) -> np.ndarray:
    return np.zeros((2, 3, 4))


@pytest.fixture
def patch_uniform(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(
    "npcc.tprbicop.TabPFNRegressor", _UniformQuantileRegressor
  )


@pytest.fixture
def patch_transposed(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(
    "npcc.tprbicop.TabPFNRegressor",
    _UniformQuantileRegressorTransposed,
  )


@pytest.fixture
def patch_bad_shape(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(
    "npcc.tprbicop.TabPFNRegressor", _BadShapeQuantileRegressor
  )


def _uniform_density_y(y: np.ndarray) -> np.ndarray:
  """Analytic f_Y(y) under logit + Z~Uniform(-2,2)."""
  return 0.25 / (y * (1.0 - y))


def make_tpr(symmetric: bool = True) -> TPRBicop:
  return TPRBicop(symmetric=symmetric)


# ===========================================================================
# Helpers
# ===========================================================================


class TestHelpers:
  def test_logit_sigmoid_roundtrip(self) -> None:
    p = np.array([0.1, 0.4, 0.6, 0.9])
    np.testing.assert_allclose(_sigmoid(_logit(p)), p, atol=1e-12)

  def test_as_2d_reshapes_1d(self) -> None:
    out = _as_2d(np.arange(5, dtype=float))
    assert out.shape == (5, 1)

  def test_as_2d_keeps_2d(self) -> None:
    out = _as_2d(np.zeros((3, 4)))
    assert out.shape == (3, 4)

  def test_as_2d_rejects_3d(self) -> None:
    with pytest.raises(ValueError, match="1D or 2D"):
      _as_2d(np.zeros((2, 3, 4)))

  def test_check_uv_rejects_outside_unit(self) -> None:
    with pytest.raises(ValueError, match="strictly inside"):
      _check_uv(np.array([0.5, 0.0]), np.array([0.5, 0.5]), 1e-6)
    with pytest.raises(ValueError, match="strictly inside"):
      _check_uv(np.array([0.5, 1.0]), np.array([0.5, 0.5]), 1e-6)

  def test_check_uv_rejects_shape_mismatch(self) -> None:
    with pytest.raises(ValueError, match="same shape"):
      _check_uv(np.array([0.5]), np.array([0.5, 0.5]), 1e-6)

  def test_check_uv_clips(self) -> None:
    u, v = _check_uv(
      np.array([1e-9, 0.5]), np.array([0.5, 1.0 - 1e-9]), eps=1e-6
    )
    assert u[0] == pytest.approx(1e-6)
    assert v[1] == pytest.approx(1.0 - 1e-6)


# ===========================================================================
# QuantileDensityConfig
# ===========================================================================


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


# ===========================================================================
# TabPFNQuantileDensity1D
# ===========================================================================


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
    expected = _uniform_density_y(y)
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
    expected = _uniform_density_y(y)
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


# ===========================================================================
# TPRBicop init
# ===========================================================================


class TestTPRInit:
  def test_default_is_symmetric(self) -> None:
    m = TPRBicop()
    assert m.symmetric is True
    assert m.u_given_vx_ is not None

  def test_asymmetric_has_no_reverse_density(self) -> None:
    m = TPRBicop(symmetric=False)
    assert m.symmetric is False
    assert m.u_given_vx_ is None


# ===========================================================================
# TPRBicop input validation
# ===========================================================================


class TestTPRValidation:
  def test_fit_rejects_x_length_mismatch(self, patch_uniform: None) -> None:
    rng = np.random.default_rng(0)
    u = rng.uniform(0.1, 0.9, 10)
    v = rng.uniform(0.1, 0.9, 10)
    x = rng.normal(size=(5, 2))
    with pytest.raises(ValueError, match="same number"):
      make_tpr().fit(u, v, x)

  def test_density_rejects_x_length_mismatch(self, patch_uniform: None) -> None:
    rng = np.random.default_rng(0)
    u = rng.uniform(0.1, 0.9, 10)
    v = rng.uniform(0.1, 0.9, 10)
    m = make_tpr().fit(u, v)
    with pytest.raises(ValueError, match="same number"):
      m.density(u, v, rng.normal(size=(5, 2)))

  def test_fit_rejects_uv_outside_unit(self, patch_uniform: None) -> None:
    u = np.array([0.5, 0.0])
    v = np.array([0.3, 0.4])
    with pytest.raises(ValueError, match="strictly inside"):
      make_tpr().fit(u, v)


# ===========================================================================
# TPRBicop density
# ===========================================================================


class TestTPRDensity:
  def test_density_shape_unconditional(self, patch_uniform: None) -> None:
    rng = np.random.default_rng(0)
    u = rng.uniform(0.15, 0.85, 30)
    v = rng.uniform(0.15, 0.85, 30)
    m = make_tpr().fit(u, v)
    out = m.density(u, v)
    assert out.shape == (30,)

  def test_density_nonnegative(self, patch_uniform: None) -> None:
    rng = np.random.default_rng(1)
    u = rng.uniform(0.15, 0.85, 30)
    v = rng.uniform(0.15, 0.85, 30)
    m = make_tpr().fit(u, v)
    out = m.density(u, v)
    assert (out >= 0).all()

  def test_density_x_default_is_constant(self, patch_uniform: None) -> None:
    rng = np.random.default_rng(2)
    u = rng.uniform(0.15, 0.85, 20)
    v = rng.uniform(0.15, 0.85, 20)
    m = make_tpr().fit(u, v)

    out_default = m.density(u, v)
    out_explicit = m.density(u, v, x=np.ones((len(u), 1)))
    np.testing.assert_allclose(out_default, out_explicit)

  def test_density_with_2d_x(self, patch_uniform: None) -> None:
    rng = np.random.default_rng(3)
    u = rng.uniform(0.15, 0.85, 25)
    v = rng.uniform(0.15, 0.85, 25)
    x = rng.normal(size=(25, 3))
    m = make_tpr().fit(u, v, x)
    out = m.density(u, v, x)
    assert out.shape == (25,)
    assert (out >= 0).all()

  def test_symmetric_is_average_of_two_halves(
    self, patch_uniform: None
  ) -> None:
    rng = np.random.default_rng(4)
    u = rng.uniform(0.15, 0.85, 20)
    v = rng.uniform(0.15, 0.85, 20)
    m = make_tpr(symmetric=True).fit(u, v)

    x_default = np.ones((len(u), 1))
    c_v = m.v_given_ux_.density(np.column_stack([u, x_default]), v)
    assert m.u_given_vx_ is not None
    c_u = m.u_given_vx_.density(np.column_stack([v, x_default]), u)
    expected = 0.5 * (c_v + c_u)

    np.testing.assert_allclose(m.density(u, v), expected, atol=1e-12)

  def test_asymmetric_equals_v_given_ux(self, patch_uniform: None) -> None:
    rng = np.random.default_rng(5)
    u = rng.uniform(0.15, 0.85, 20)
    v = rng.uniform(0.15, 0.85, 20)
    m = make_tpr(symmetric=False).fit(u, v)

    x_default = np.ones((len(u), 1))
    expected = m.v_given_ux_.density(np.column_stack([u, x_default]), v)
    np.testing.assert_allclose(m.density(u, v), expected, atol=1e-12)

  def test_log_density_matches_log_of_density(
    self, patch_uniform: None
  ) -> None:
    rng = np.random.default_rng(6)
    u = rng.uniform(0.15, 0.85, 15)
    v = rng.uniform(0.15, 0.85, 15)
    m = make_tpr().fit(u, v)
    expected = np.log(np.maximum(m.density(u, v), np.finfo(float).tiny))
    np.testing.assert_allclose(m.log_density(u, v), expected)


# ===========================================================================
# TPRBicop conditional CDF
# ===========================================================================


class TestTPRConditionalCdf:
  def _fit(self) -> TPRBicop:
    rng = np.random.default_rng(7)
    u = rng.uniform(0.15, 0.85, 20)
    v = rng.uniform(0.15, 0.85, 20)
    return make_tpr().fit(u, v)

  def test_cdf_shape(self, patch_uniform: None) -> None:
    m = self._fit()
    u_q = np.array([0.3, 0.7])
    v_grid = np.linspace(0.05, 0.95, 11)
    cdf = m.conditional_cdf_v_given_u(u_q, v_grid)
    assert cdf.shape == (2, 11)

  def test_cdf_in_unit_interval(self, patch_uniform: None) -> None:
    m = self._fit()
    u_q = np.array([0.3, 0.7])
    v_grid = np.linspace(0.05, 0.95, 11)
    cdf = m.conditional_cdf_v_given_u(u_q, v_grid)
    assert (cdf >= 0.0).all()
    assert (cdf <= 1.0).all()

  def test_cdf_monotone_nondecreasing(self, patch_uniform: None) -> None:
    m = self._fit()
    u_q = np.array([0.3, 0.5, 0.7])
    v_grid = np.linspace(0.05, 0.95, 21)
    cdf = m.conditional_cdf_v_given_u(u_q, v_grid)
    diffs = np.diff(cdf, axis=1)
    assert (diffs >= -1e-12).all()

  def test_cdf_rejects_non_increasing_grid(self, patch_uniform: None) -> None:
    m = self._fit()
    with pytest.raises(ValueError, match="strictly increasing"):
      m.conditional_cdf_v_given_u(np.array([0.5]), np.array([0.5, 0.4]))

  def test_cdf_rejects_grid_outside_unit(self, patch_uniform: None) -> None:
    m = self._fit()
    with pytest.raises(ValueError, match="strictly inside"):
      m.conditional_cdf_v_given_u(np.array([0.5]), np.array([0.0, 0.5, 0.95]))

  def test_cdf_rejects_u_x_length_mismatch(self, patch_uniform: None) -> None:
    m = self._fit()
    with pytest.raises(ValueError, match="same number"):
      m.conditional_cdf_v_given_u(
        np.array([0.3, 0.7]),
        np.linspace(0.1, 0.9, 5),
        x=np.zeros((1, 1)),
      )


# ===========================================================================
# Integration test (real TabPFN, opt-in via TABPFN_TOKEN)
# ===========================================================================


def test_real_tabpfn_smoke() -> None:
  dotenv = pytest.importorskip("dotenv")
  dotenv.load_dotenv()
  if not os.getenv("TABPFN_TOKEN"):
    pytest.skip("TABPFN_TOKEN not set; skipping integration test.")

  pv = pytest.importorskip("pyvinecopulib")
  from tabpfn.errors import TabPFNLicenseError

  cop = pv.Bicop(
    family=pv.BicopFamily.gaussian,
    parameters=np.array([[0.6]], dtype=np.float64),
  )
  rng = np.random.default_rng(42)
  uv = cop.simulate(200, seeds=[int(rng.integers(1, 1_000_000))])
  u, v = uv[:, 0], uv[:, 1]

  try:
    m = TPRBicop(symmetric=True).fit(u, v)
  except TabPFNLicenseError as exc:
    pytest.skip(f"TabPFN authentication unavailable: {exc}")

  dens = m.density(np.array([0.3, 0.5, 0.7]), np.array([0.3, 0.5, 0.7]))

  assert dens.shape == (3,)
  assert np.all(np.isfinite(dens))
  assert np.all(dens > 0)
