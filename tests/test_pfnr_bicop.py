"""Tests for ``PFNRBicop`` (TabPFN-Rosenblatt conditional bicop)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from npcc.pfnr_bicop import PFNRBicop
from npcc.tabpfn_density1d import TabPFNDensity1D
from npcc.tabpfn_quantile_density1d import TabPFNQuantileDensity1D


def make_pfnr(
  symmetric: bool = True,
  method: str = "quantiles",
) -> PFNRBicop:
  """Test factory.  Defaults to ``method="quantiles"`` so legacy tests
  exercise the slope-inversion path; the criterion path is opted into
  per-test."""
  return PFNRBicop(
    symmetric=symmetric,
    method=method,  # ty: ignore[invalid-argument-type]
  )


# ===========================================================================
# Init
# ===========================================================================


class TestPFNRInit:
  def test_default_is_symmetric_and_criterion(self) -> None:
    m = PFNRBicop()
    assert m.symmetric is True
    assert m.method == "criterion"
    assert isinstance(m.v_given_ux_, TabPFNDensity1D)
    assert m.u_given_vx_ is not None

  def test_asymmetric_has_no_reverse_density(self) -> None:
    m = PFNRBicop(symmetric=False)
    assert m.symmetric is False
    assert m.u_given_vx_ is None

  def test_method_quantiles_uses_quantile_module(self) -> None:
    m = PFNRBicop(method="quantiles")
    assert isinstance(m.v_given_ux_, TabPFNQuantileDensity1D)


# ===========================================================================
# Input validation
# ===========================================================================


@pytest.mark.parametrize("method", ["quantiles", "criterion"])
class TestPFNRValidation:
  def test_fit_rejects_x_length_mismatch(
    self, patch_uniform: None, method: str
  ) -> None:
    rng = np.random.default_rng(0)
    u = rng.uniform(0.1, 0.9, 10)
    v = rng.uniform(0.1, 0.9, 10)
    x = rng.normal(size=(5, 2))
    with pytest.raises(ValueError, match="same number"):
      make_pfnr(method=method).fit(u, v, x)

  def test_density_rejects_x_length_mismatch(
    self, patch_uniform: None, method: str
  ) -> None:
    rng = np.random.default_rng(0)
    u = rng.uniform(0.1, 0.9, 10)
    v = rng.uniform(0.1, 0.9, 10)
    m = make_pfnr(method=method).fit(u, v)
    with pytest.raises(ValueError, match="same number"):
      m.density(u, v, rng.normal(size=(5, 2)))

  def test_fit_rejects_uv_outside_unit(
    self, patch_uniform: None, method: str
  ) -> None:
    u = np.array([0.5, 0.0])
    v = np.array([0.3, 0.4])
    with pytest.raises(ValueError, match="strictly inside"):
      make_pfnr(method=method).fit(u, v)


# ===========================================================================
# density()
# ===========================================================================


@pytest.mark.parametrize("method", ["quantiles", "criterion"])
class TestPFNRDensity:
  def test_density_shape_unconditional(
    self, patch_uniform: None, method: str
  ) -> None:
    rng = np.random.default_rng(0)
    u = rng.uniform(0.15, 0.85, 30)
    v = rng.uniform(0.15, 0.85, 30)
    m = make_pfnr(method=method).fit(u, v)
    out = m.density(u, v)
    assert out.shape == (30,)

  def test_density_nonnegative(self, patch_uniform: None, method: str) -> None:
    rng = np.random.default_rng(1)
    u = rng.uniform(0.15, 0.85, 30)
    v = rng.uniform(0.15, 0.85, 30)
    m = make_pfnr(method=method).fit(u, v)
    out = m.density(u, v)
    assert (out >= 0).all()

  def test_density_x_default_is_constant(
    self, patch_uniform: None, method: str
  ) -> None:
    rng = np.random.default_rng(2)
    u = rng.uniform(0.15, 0.85, 20)
    v = rng.uniform(0.15, 0.85, 20)
    m = make_pfnr(method=method).fit(u, v)

    out_default = m.density(u, v)
    out_explicit = m.density(u, v, x=np.ones((len(u), 1)))
    np.testing.assert_allclose(out_default, out_explicit)

  def test_density_with_2d_x(self, patch_uniform: None, method: str) -> None:
    rng = np.random.default_rng(3)
    u = rng.uniform(0.15, 0.85, 25)
    v = rng.uniform(0.15, 0.85, 25)
    x = rng.normal(size=(25, 3))
    m = make_pfnr(method=method).fit(u, v, x)
    out = m.density(u, v, x)
    assert out.shape == (25,)
    assert (out >= 0).all()

  def test_symmetric_is_average_of_two_halves(
    self, patch_uniform: None, method: str
  ) -> None:
    rng = np.random.default_rng(4)
    u = rng.uniform(0.15, 0.85, 20)
    v = rng.uniform(0.15, 0.85, 20)
    m = make_pfnr(symmetric=True, method=method).fit(u, v)

    x_default = np.ones((len(u), 1))
    c_v = m.v_given_ux_.density(np.column_stack([u, x_default]), v)
    assert m.u_given_vx_ is not None
    c_u = m.u_given_vx_.density(np.column_stack([v, x_default]), u)
    expected = 0.5 * (c_v + c_u)

    np.testing.assert_allclose(m.density(u, v), expected, atol=1e-12)

  def test_asymmetric_equals_v_given_ux(
    self, patch_uniform: None, method: str
  ) -> None:
    rng = np.random.default_rng(5)
    u = rng.uniform(0.15, 0.85, 20)
    v = rng.uniform(0.15, 0.85, 20)
    m = make_pfnr(symmetric=False, method=method).fit(u, v)

    x_default = np.ones((len(u), 1))
    expected = m.v_given_ux_.density(np.column_stack([u, x_default]), v)
    np.testing.assert_allclose(m.density(u, v), expected, atol=1e-12)

  def test_log_density_matches_log_of_density(
    self, patch_uniform: None, method: str
  ) -> None:
    rng = np.random.default_rng(6)
    u = rng.uniform(0.15, 0.85, 15)
    v = rng.uniform(0.15, 0.85, 15)
    m = make_pfnr(method=method).fit(u, v)
    expected = np.log(np.maximum(m.density(u, v), np.finfo(float).tiny))
    np.testing.assert_allclose(m.log_density(u, v), expected)


# ===========================================================================
# Both methods agree under the uniform fake
# ===========================================================================


class TestMethodsAgree:
  def test_criterion_matches_quantiles_under_uniform_fake(
    self, patch_uniform: None
  ) -> None:
    rng = np.random.default_rng(11)
    u = rng.uniform(0.2, 0.8, 30)
    v = rng.uniform(0.2, 0.8, 30)

    m_q = make_pfnr(method="quantiles").fit(u, v)
    m_c = make_pfnr(method="criterion").fit(u, v)

    np.testing.assert_allclose(m_q.density(u, v), m_c.density(u, v), atol=1e-6)


# ===========================================================================
# density_grid (criterion only)
# ===========================================================================


class TestPFNRDensityGrid:
  def _fit_criterion(self) -> PFNRBicop:
    rng = np.random.default_rng(12)
    u = rng.uniform(0.2, 0.8, 25)
    v = rng.uniform(0.2, 0.8, 25)
    return make_pfnr(method="criterion").fit(u, v)

  def test_grid_shape(self, patch_uniform: None) -> None:
    m = self._fit_criterion()
    u_g = np.linspace(0.2, 0.8, 5)
    v_g = np.linspace(0.2, 0.8, 7)
    out = m.density_grid(u_g, v_g)
    assert out.shape == (5, 7)

  def test_grid_matches_pointwise_density(self, patch_uniform: None) -> None:
    m = self._fit_criterion()
    u_g = np.linspace(0.25, 0.75, 4)
    v_g = np.linspace(0.25, 0.75, 5)
    grid = m.density_grid(u_g, v_g)

    u_tile = np.repeat(u_g, len(v_g))
    v_tile = np.tile(v_g, len(u_g))
    expected = m.density(u_tile, v_tile).reshape(len(u_g), len(v_g))
    np.testing.assert_allclose(grid, expected, atol=1e-6)

  def test_grid_with_x_row(self, patch_uniform: None) -> None:
    m = self._fit_criterion()
    u_g = np.linspace(0.3, 0.7, 4)
    v_g = np.linspace(0.3, 0.7, 4)
    out = m.density_grid(u_g, v_g, x_row=np.array([[1.5, -0.5]]))
    assert out.shape == (4, 4)
    assert (out >= 0).all()

  def test_grid_rejects_quantiles_method(self, patch_uniform: None) -> None:
    rng = np.random.default_rng(13)
    u = rng.uniform(0.2, 0.8, 10)
    v = rng.uniform(0.2, 0.8, 10)
    m = make_pfnr(method="quantiles").fit(u, v)
    with pytest.raises(RuntimeError, match="method='criterion'"):
      m.density_grid(np.array([0.3, 0.5]), np.array([0.4, 0.6]))

  def test_grid_rejects_grid_outside_unit(self, patch_uniform: None) -> None:
    m = self._fit_criterion()
    with pytest.raises(ValueError, match="strictly inside"):
      m.density_grid(np.array([0.0, 0.5]), np.array([0.3, 0.6]))

  def test_grid_rejects_multi_row_x(self, patch_uniform: None) -> None:
    m = self._fit_criterion()
    with pytest.raises(ValueError, match="exactly one row"):
      m.density_grid(
        np.array([0.3, 0.5]),
        np.array([0.4, 0.6]),
        x_row=np.array([[1.0], [2.0]]),
      )


# ===========================================================================
# conditional_cdf_v_given_u
# ===========================================================================


class TestPFNRConditionalCdf:
  def _fit(self) -> PFNRBicop:
    rng = np.random.default_rng(7)
    u = rng.uniform(0.15, 0.85, 20)
    v = rng.uniform(0.15, 0.85, 20)
    return make_pfnr().fit(u, v)

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
# h-functions
# ===========================================================================


def _analytic_F_y(y: np.ndarray) -> np.ndarray:
  """F_Y(y) = clip((logit(y) + 2) / 4, 0, 1) under the uniform fake."""
  return np.clip((np.log(y / (1.0 - y)) + 2.0) / 4.0, 0.0, 1.0)


@pytest.mark.parametrize("method", ["quantiles", "criterion"])
class TestPFNRHfunc1:
  """``hfunc1(u, v) = F_{V|U}(v|u)`` — conditions on the first arg.

  Always available (V|U regressor is always fitted), pyvinecopulib
  convention.
  """

  def test_shape(self, patch_uniform: None, method: str) -> None:
    m = make_pfnr(method=method).fit(
      np.linspace(0.1, 0.9, 30), np.linspace(0.1, 0.9, 30)
    )
    u = np.array([0.3, 0.5, 0.7])
    v = np.array([0.4, 0.5, 0.6])
    assert m.hfunc1(u, v).shape == (3,)

  def test_matches_analytic_uniform(
    self, patch_uniform: None, method: str
  ) -> None:
    """Under the fake, h₁(u, v) = F_Y(v) (constant in u)."""
    m = make_pfnr(method=method).fit(
      np.linspace(0.1, 0.9, 20), np.linspace(0.1, 0.9, 20)
    )
    u = np.array([0.3, 0.5, 0.7])
    v = np.array([0.3, 0.5, 0.7])
    actual = m.hfunc1(u, v)
    expected = _analytic_F_y(v)
    np.testing.assert_allclose(actual, expected, atol=1e-3)

  def test_monotone_in_v(self, patch_uniform: None, method: str) -> None:
    m = make_pfnr(method=method).fit(
      np.linspace(0.1, 0.9, 20), np.linspace(0.1, 0.9, 20)
    )
    v_sorted = np.linspace(0.1, 0.9, 15)
    u_const = np.full(len(v_sorted), 0.5)
    h = m.hfunc1(u_const, v_sorted)
    assert (np.diff(h) >= -1e-9).all()

  def test_boundary_values(self, patch_uniform: None, method: str) -> None:
    m = make_pfnr(method=method).fit(
      np.linspace(0.1, 0.9, 20), np.linspace(0.1, 0.9, 20)
    )
    # logit(0.05) = -2.94 < -2 → F → 0; logit(0.95) = 2.94 > 2 → F → 1.
    h_low = m.hfunc1(np.array([0.5]), np.array([0.05]))
    h_high = m.hfunc1(np.array([0.5]), np.array([0.95]))
    assert h_low[0] < 0.05
    assert h_high[0] > 0.95


@pytest.mark.parametrize("method", ["quantiles", "criterion"])
class TestPFNRHfunc2:
  """``hfunc2(u, v) = F_{U|V}(u|v)`` — conditions on the second arg.

  Requires ``symmetric=True`` (U|V regressor must exist).
  """

  def test_symmetric_returns_F_U_given_V(
    self, patch_uniform: None, method: str
  ) -> None:
    m = make_pfnr(symmetric=True, method=method).fit(
      np.linspace(0.1, 0.9, 20), np.linspace(0.1, 0.9, 20)
    )
    u = np.array([0.3, 0.5, 0.7])
    v = np.array([0.4, 0.5, 0.6])
    assert m.u_given_vx_ is not None
    expected = m.u_given_vx_.cdf(np.column_stack([v, np.ones(len(u))]), u)
    np.testing.assert_allclose(m.hfunc2(u, v), expected, atol=1e-12)

  def test_asymmetric_raises(self, patch_uniform: None, method: str) -> None:
    m = make_pfnr(symmetric=False, method=method).fit(
      np.linspace(0.1, 0.9, 20), np.linspace(0.1, 0.9, 20)
    )
    with pytest.raises(RuntimeError, match="symmetric"):
      m.hfunc2(np.array([0.5]), np.array([0.5]))


# ===========================================================================
# Joint CDF
# ===========================================================================


@pytest.mark.parametrize("method", ["quantiles", "criterion"])
class TestPFNRCdf:
  def test_shape(self, patch_uniform: None, method: str) -> None:
    m = make_pfnr(method=method).fit(
      np.linspace(0.1, 0.9, 20), np.linspace(0.1, 0.9, 20)
    )
    u = np.array([0.3, 0.5, 0.7])
    v = np.array([0.4, 0.5, 0.6])
    assert m.cdf(u, v).shape == (3,)

  def test_independent_factorisation(
    self, patch_uniform: None, method: str
  ) -> None:
    """Under the fake (Z independent of W), C(u, v) ≈ F_Y(u) F_Y(v)."""
    m = make_pfnr(method=method).fit(
      np.linspace(0.1, 0.9, 30), np.linspace(0.1, 0.9, 30)
    )
    u = np.array([0.3, 0.5, 0.7])
    v = np.array([0.6, 0.5, 0.4])
    actual = m.cdf(u, v, n_int=128)
    expected = _analytic_F_y(u) * _analytic_F_y(v)
    np.testing.assert_allclose(actual, expected, atol=2e-2)

  def test_rejects_uv_outside_unit(
    self, patch_uniform: None, method: str
  ) -> None:
    m = make_pfnr(method=method).fit(
      np.linspace(0.1, 0.9, 20), np.linspace(0.1, 0.9, 20)
    )
    with pytest.raises(ValueError, match="strictly inside"):
      m.cdf(np.array([0.5, 0.0]), np.array([0.3, 0.4]))

  def test_rejects_bad_n_int(self, patch_uniform: None, method: str) -> None:
    m = make_pfnr(method=method).fit(
      np.linspace(0.1, 0.9, 20), np.linspace(0.1, 0.9, 20)
    )
    with pytest.raises(ValueError, match="n_int"):
      m.cdf(np.array([0.5]), np.array([0.5]), n_int=1)


class TestPFNRCdfGrid:
  def _fit(self) -> PFNRBicop:
    return make_pfnr(method="criterion").fit(
      np.linspace(0.1, 0.9, 30), np.linspace(0.1, 0.9, 30)
    )

  def test_shape(self, patch_uniform: None) -> None:
    m = self._fit()
    u_g = np.linspace(0.1, 0.9, 5)
    v_g = np.linspace(0.1, 0.9, 7)
    out = m.cdf_grid(u_g, v_g)
    assert out.shape == (5, 7)

  def test_matches_pointwise_cdf(self, patch_uniform: None) -> None:
    m = self._fit()
    u_g = np.linspace(0.2, 0.8, 4)
    v_g = np.linspace(0.2, 0.8, 4)
    grid = m.cdf_grid(u_g, v_g, n_int=128)

    u_tile = np.repeat(u_g, len(v_g))
    v_tile = np.tile(v_g, len(u_g))
    expected = m.cdf(u_tile, v_tile, n_int=128).reshape(len(u_g), len(v_g))
    np.testing.assert_allclose(grid, expected, atol=2e-2)

  def test_rejects_quantiles_method(self, patch_uniform: None) -> None:
    m = make_pfnr(method="quantiles").fit(
      np.linspace(0.1, 0.9, 20), np.linspace(0.1, 0.9, 20)
    )
    with pytest.raises(RuntimeError, match="method='criterion'"):
      m.cdf_grid(np.array([0.3, 0.5]), np.array([0.4, 0.6]))

  def test_rejects_grid_outside_unit(self, patch_uniform: None) -> None:
    m = self._fit()
    with pytest.raises(ValueError, match="strictly inside"):
      m.cdf_grid(np.array([0.0, 0.5]), np.array([0.3, 0.6]))


# ===========================================================================
# Kendall's tau
# ===========================================================================


@pytest.mark.parametrize("method", ["quantiles", "criterion"])
class TestPFNRTau:
  """Sample-based τ via ghalton + inverse-Rosenblatt + wdm.

  Mirrors pyvinecopulib's ``KernelBicop::parameters_to_tau``.
  """

  def test_independent_components_tau_near_zero(
    self, patch_uniform: None, method: str
  ) -> None:
    """Under the fake, Z components are independent → τ ≈ 0."""
    m = make_pfnr(method=method).fit(
      np.linspace(0.1, 0.9, 30), np.linspace(0.1, 0.9, 30)
    )
    tau = m.tau(n=1000)
    # Quasi-random + wdm should give a small non-zero residual.
    assert abs(tau) < 0.05

  def test_returns_float_in_range(
    self, patch_uniform: None, method: str
  ) -> None:
    m = make_pfnr(method=method).fit(
      np.linspace(0.1, 0.9, 30), np.linspace(0.1, 0.9, 30)
    )
    tau = m.tau()
    assert isinstance(tau, float)
    assert -1.0 <= tau <= 1.0

  def test_rejects_small_n(self, patch_uniform: None, method: str) -> None:
    m = make_pfnr(method=method).fit(
      np.linspace(0.1, 0.9, 20), np.linspace(0.1, 0.9, 20)
    )
    with pytest.raises(ValueError, match="n must be at least"):
      m.tau(n=5)

  def test_seeds_make_result_deterministic(
    self, patch_uniform: None, method: str
  ) -> None:
    m = make_pfnr(method=method).fit(
      np.linspace(0.1, 0.9, 20), np.linspace(0.1, 0.9, 20)
    )
    s = [1, 2, 3, 4, 5]
    tau_a = m.tau(n=200, seeds=s)
    tau_b = m.tau(n=200, seeds=s)
    assert tau_a == tau_b


# ===========================================================================
# pyvinecopulib-compatible adapter (as_bicop) and plotting
# ===========================================================================


class TestPFNRAdapter:
  def _fit(self, method: str = "criterion") -> PFNRBicop:
    rng = np.random.default_rng(20)
    u = rng.uniform(0.2, 0.8, 25)
    v = rng.uniform(0.2, 0.8, 25)
    return make_pfnr(method=method).fit(u, v)

  def test_var_types_is_two_continuous(self, patch_uniform: None) -> None:
    m = self._fit()
    assert m.as_bicop().var_types == ["c", "c"]

  def test_pdf_returns_correct_shape(self, patch_uniform: None) -> None:
    m = self._fit()
    uv = np.array([[0.3, 0.4], [0.5, 0.6], [0.7, 0.2]])
    out = m.as_bicop().pdf(uv)
    assert out.shape == (3,)

  def test_pdf_matches_density(self, patch_uniform: None) -> None:
    m = self._fit(method="quantiles")  # avoid Cartesian fast path
    uv = np.array([[0.3, 0.4], [0.5, 0.6], [0.7, 0.2]])
    np.testing.assert_allclose(
      m.as_bicop().pdf(uv), m.density(uv[:, 0], uv[:, 1]), atol=1e-12
    )

  def test_pdf_cartesian_fast_path_matches_density(
    self, patch_uniform: None
  ) -> None:
    m = self._fit(method="criterion")
    u_g = np.linspace(0.25, 0.75, 5)
    v_g = np.linspace(0.25, 0.75, 5)
    grid_u, grid_v = np.meshgrid(u_g, v_g)
    uv = np.column_stack([grid_u.flatten(), grid_v.flatten()])

    actual = m.as_bicop().pdf(uv)
    expected = m.density(uv[:, 0], uv[:, 1])
    np.testing.assert_allclose(actual, expected, atol=1e-6)

  def test_pdf_rejects_bad_shape(self, patch_uniform: None) -> None:
    m = self._fit()
    with pytest.raises(ValueError, match="shape"):
      m.as_bicop().pdf(np.zeros((3, 3)))

  def test_pdf_with_x_row_broadcasts(self, patch_uniform: None) -> None:
    m = self._fit()
    uv = np.array([[0.3, 0.4], [0.5, 0.6]])
    out = m.as_bicop(x_row=np.array([[1.5, -0.5]])).pdf(uv)
    assert out.shape == (2,)
    assert (out >= 0).all()

  def test_as_bicop_rejects_multi_row_x(self, patch_uniform: None) -> None:
    m = self._fit()
    with pytest.raises(ValueError, match="exactly one row"):
      m.as_bicop(x_row=np.array([[1.0], [2.0]]))


class TestPFNRPlot:
  def test_plot_runs_without_error(
    self, patch_uniform: None, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    rng = np.random.default_rng(99)
    u = rng.uniform(0.15, 0.85, 20)
    v = rng.uniform(0.15, 0.85, 20)
    m = make_pfnr(method="criterion").fit(u, v)

    try:
      m.plot(grid_size=8, plot_type="contour", margin_type="norm")
    finally:
      plt.close("all")


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

  # Clayton(theta=3) — strong positive lower-tail dependence,
  # analytic Kendall's tau = theta / (theta + 2) = 0.6.
  theta = 3.0
  cop = pv.Bicop(
    family=pv.BicopFamily.clayton,
    parameters=np.array([[theta]], dtype=np.float64),
  )
  rng = np.random.default_rng(42)
  uv = cop.simulate(200, seeds=[int(rng.integers(1, 1_000_000))])
  u, v = uv[:, 0], uv[:, 1]

  try:
    m = PFNRBicop(symmetric=True).fit(u, v)
  except TabPFNLicenseError as exc:
    pytest.skip(f"TabPFN authentication unavailable: {exc}")

  # Density: finite + positive on a few points.
  query_u = np.array([0.3, 0.5, 0.7])
  query_v = np.array([0.3, 0.5, 0.7])
  dens = m.density(query_u, query_v)
  assert dens.shape == (3,)
  assert np.all(np.isfinite(dens))
  assert np.all(dens > 0)

  # h-function: hfunc1(u, v) = F_{V|U,X}(v|u,x) ∈ [0, 1], monotone in v.
  v_sorted = np.linspace(0.1, 0.9, 6)
  u_const = np.full(len(v_sorted), 0.5)
  h1 = m.hfunc1(u_const, v_sorted)
  assert h1.shape == (6,)
  assert (h1 >= 0).all() and (h1 <= 1).all()
  assert (np.diff(h1) >= -1e-6).all()

  # Joint CDF: in [0, 1] and roughly monotone.
  big_c = m.cdf(query_u, query_v, n_int=32)
  assert (big_c >= 0).all() and (big_c <= 1).all()

  # Kendall's tau via ghalton + inverse-Rosenblatt + wdm.  Under
  # Clayton(3), analytic τ = 0.6.  Quasi-random sampling on a
  # 200-row fit gets quite close.
  tau = m.tau(n=1000)
  assert -1.0 <= tau <= 1.0
  assert abs(tau - theta / (theta + 2.0)) < 0.1
