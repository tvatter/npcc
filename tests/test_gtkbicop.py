"""Tests for GTKBicop."""

import pytest
import torch

from npcc.gtkbicop import GTKBicop

from .conftest import random_uv, unit_grid


def make_gtk(m_u: int = 10, m_v: int = 10, **kwargs: object) -> GTKBicop:
  return GTKBicop(m_u=m_u, m_v=m_v, **kwargs)  # ty: ignore[invalid-argument-type]


# ===========================================================================
# Initialisation
# ===========================================================================


class TestGTKInit:
  def test_m_u_must_be_at_least_1(self) -> None:
    with pytest.raises(ValueError, match="m_u"):
      GTKBicop(m_u=0)

  def test_m_v_must_be_at_least_1(self) -> None:
    with pytest.raises(ValueError, match="m_v"):
      GTKBicop(m_v=0)

  def test_eps_must_be_positive(self) -> None:
    with pytest.raises(ValueError, match="eps"):
      GTKBicop(eps=0.0)

  def test_sigma_min_must_be_positive(self) -> None:
    with pytest.raises(ValueError, match="sigma_min"):
      GTKBicop(sigma_min=0.0)

  def test_sigma_u_init_must_exceed_sigma_min(self) -> None:
    with pytest.raises(ValueError, match="sigma_u_init"):
      GTKBicop(sigma_u_init=0.005, sigma_min=0.01)

  def test_sigma_v_init_must_exceed_sigma_min(self) -> None:
    with pytest.raises(ValueError, match="sigma_v_init"):
      GTKBicop(sigma_v_init=0.005, sigma_min=0.01)

  def test_logits_shape(self) -> None:
    m = make_gtk(m_u=5, m_v=7)
    assert m.logits.shape == (5, 7)

  def test_weights_sum_to_one(self) -> None:
    assert make_gtk().weights.sum().item() == pytest.approx(1.0, abs=1e-5)

  def test_weights_nonneg(self) -> None:
    assert (make_gtk().weights >= 0).all()

  def test_sigma_shapes(self) -> None:
    m = make_gtk(m_u=6, m_v=8)
    assert m.sigma_u.shape == (6,)
    assert m.sigma_v.shape == (8,)

  def test_sigma_above_sigma_min(self) -> None:
    m = make_gtk(sigma_min=0.05)
    assert (m.sigma_u > 0.05).all()
    assert (m.sigma_v > 0.05).all()

  def test_learn_sigma_false_registers_buffer(self) -> None:
    m = GTKBicop(learn_sigma=False)
    bufs = dict(m.named_buffers())
    assert "raw_sigma_u" in bufs
    assert "raw_sigma_v" in bufs

  def test_default_sigma_equals_grid_spacing(self) -> None:
    # With m=5: vinecopulib spacing = 6.5 / (5-1) = 1.625
    m = GTKBicop(m_u=5, m_v=5)
    sigma_init = 6.5 / 4  # default = vinecopulib grid spacing
    assert sigma_init == pytest.approx(m.sigma_u[0].item(), abs=0.01)


# ===========================================================================
# Input validation
# ===========================================================================


class TestGTKInputValidation:
  def test_pdf_rejects_1d(self) -> None:
    m = make_gtk()
    with pytest.raises(ValueError, match="shape"):
      m.pdf(torch.rand(10))

  def test_pdf_rejects_wrong_columns(self) -> None:
    m = make_gtk()
    with pytest.raises(ValueError, match="shape"):
      m.pdf(torch.rand(10, 3))

  def test_margin_u_rejects_2d(self) -> None:
    m = make_gtk()
    with pytest.raises(ValueError, match="1-D"):
      m.margin_u(torch.rand(5, 1))

  def test_basis_pdf_u_rejects_2d(self) -> None:
    m = make_gtk()
    with pytest.raises(ValueError, match="1-D"):
      m.basis_pdf_u(torch.rand(5, 1))


# ===========================================================================
# PDF
# ===========================================================================


class TestGTKPDF:
  def test_pdf_nonneg(self) -> None:
    m = make_gtk()
    assert (m.pdf(random_uv()) >= 0).all()

  def test_pdf_integrates_to_one(self) -> None:
    # Integrate in Z-space via midpoint quadrature on [-6, 6].
    # f_Z(z1, z2) = c(Phi(z1), Phi(z2)) * phi(z1) * phi(z2)
    # integrates to 1 over R^2 by construction.  The wide z-range is
    # necessary because GTK centers sit at z = ±3.25 (the vinecopulib
    # grid boundary), so the copula density has large values very close
    # to u = 0 and u = 1 that a u-space uniform grid would miss.
    import math

    m = make_gtk()
    n, z_lo, z_hi = 100, -6.0, 6.0
    z = torch.linspace(z_lo, z_hi, n) + (z_hi - z_lo) / (2 * n)
    z1, z2 = torch.meshgrid(z, z, indexing="ij")
    half = math.sqrt(2.0 * math.pi)
    phi1 = torch.exp(-0.5 * z1.square()) / half
    phi2 = torch.exp(-0.5 * z2.square()) / half
    u1 = 0.5 * (1.0 + torch.erf(z1 / math.sqrt(2.0)))
    u2 = 0.5 * (1.0 + torch.erf(z2 / math.sqrt(2.0)))
    UV = torch.stack([u1.reshape(-1), u2.reshape(-1)], dim=1)
    f_Z = m.pdf(UV) * phi1.reshape(-1) * phi2.reshape(-1)
    integral = f_Z.mean().item() * (z_hi - z_lo) ** 2
    assert integral == pytest.approx(1.0, abs=0.05)

  def test_log_pdf_is_finite(self) -> None:
    m = make_gtk()
    assert torch.isfinite(m.log_pdf(random_uv())).all()


# ===========================================================================
# Margins
# ===========================================================================


class TestGTKMargins:
  def test_margin_u_formula_consistent(self) -> None:
    m = make_gtk()
    u = unit_grid()
    W = m.weights
    fu = m.basis_pdf_u(u)
    expected = fu @ W.sum(dim=1)
    assert torch.allclose(m.margin_u(u), expected, atol=1e-5)

  def test_margin_v_formula_consistent(self) -> None:
    m = make_gtk()
    v = unit_grid()
    W = m.weights
    fv = m.basis_pdf_v(v)
    expected = fv @ W.sum(dim=0)
    assert torch.allclose(m.margin_v(v), expected, atol=1e-5)

  def test_margin_u_nonneg(self) -> None:
    assert (make_gtk().margin_u(unit_grid()) >= 0).all()

  def test_margin_v_nonneg(self) -> None:
    assert (make_gtk().margin_v(unit_grid()) >= 0).all()


# ===========================================================================
# H-functions
# ===========================================================================


class TestGTKHfunc:
  def test_hfunc1_at_u1_equals_margin_v(self) -> None:
    # With z_max=3.25 the rightmost kernel center is at Phi(3.25)≈0.9994,
    # so Fv_j(1) ≈ 0.74 for the outermost center (m=10, sigma≈0.72).
    # The identity hfunc1(1,v) = margin_v(v) holds exactly only at u→∞;
    # we therefore use a relaxed tolerance.
    m = make_gtk()
    v = unit_grid()
    UV = torch.stack([torch.ones_like(v), v], dim=1)
    assert torch.allclose(
      m.hfunc1(UV, normalized=False), m.margin_v(v), atol=0.01
    )

  def test_hfunc2_at_v1_equals_margin_u(self) -> None:
    m = make_gtk()
    u = unit_grid()
    UV = torch.stack([u, torch.ones_like(u)], dim=1)
    assert torch.allclose(
      m.hfunc2(UV, normalized=False), m.margin_u(u), atol=0.01
    )

  def test_hfunc1_normalized_in_unit_interval(self) -> None:
    m = make_gtk()
    h1n = m.hfunc1(random_uv(), normalized=True)
    assert (h1n >= -1e-6).all() and (h1n <= 1.0 + 1e-6).all()

  def test_hfunc2_normalized_in_unit_interval(self) -> None:
    m = make_gtk()
    h2n = m.hfunc2(random_uv(), normalized=True)
    assert (h2n >= -1e-6).all() and (h2n <= 1.0 + 1e-6).all()

  def test_hfunc1_raw_nonneg(self) -> None:
    assert (make_gtk().hfunc1(random_uv()) >= 0).all()

  def test_hfunc2_raw_nonneg(self) -> None:
    assert (make_gtk().hfunc2(random_uv()) >= 0).all()


# ===========================================================================
# CDF
# ===========================================================================


class TestGTKCDF:
  def test_cdf_nonneg(self) -> None:
    assert (make_gtk().cdf(random_uv()) >= 0).all()

  def test_cdf_at_one_one(self) -> None:
    # With m=10 and z_max=3.25 the outermost center sits at Phi(3.25)≈0.9994;
    # Fv_j(0.9999) ≈ 0.74 for that center, so the CDF is noticeably < 1 at
    # (0.9999, 0.9999).  Use a tolerance that reflects this boundary
    # approximation error for small m with the vinecopulib grid.
    m = make_gtk()
    assert m.cdf(torch.tensor([[0.9999, 0.9999]])).item() == pytest.approx(
      1.0, abs=0.07
    )


# ===========================================================================
# Gradients
# ===========================================================================


class TestGTKGradients:
  def test_nll_gradient_flows_through_logits(self) -> None:
    m = make_gtk(m_u=8, m_v=8)
    m.nll(random_uv(100)).backward()
    assert m.logits.grad is not None
    assert torch.isfinite(m.logits.grad).all()

  def test_nll_gradient_flows_through_sigma_u(self) -> None:
    m = make_gtk(m_u=8, m_v=8)
    m.nll(random_uv(100)).backward()
    assert m.raw_sigma_u.grad is not None
    assert torch.isfinite(m.raw_sigma_u.grad).all()

  def test_nll_gradient_flows_through_sigma_v(self) -> None:
    m = make_gtk(m_u=8, m_v=8)
    m.nll(random_uv(100)).backward()
    assert m.raw_sigma_v.grad is not None
    assert torch.isfinite(m.raw_sigma_v.grad).all()

  def test_no_nan_gradients(self) -> None:
    m = make_gtk(m_u=8, m_v=8)
    m.nll(random_uv(100)).backward()
    for name, p in m.named_parameters():
      assert p.grad is not None, f"No grad for {name}"
      assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"


# ===========================================================================
# Marginal penalty
# ===========================================================================


class TestGTKMarginalPenalty:
  def test_penalty_nonneg(self) -> None:
    assert make_gtk().marginal_penalty().item() >= 0.0

  def test_penalty_is_scalar(self) -> None:
    assert make_gtk().marginal_penalty().shape == ()

  def test_penalty_gradient_flows(self) -> None:
    m = make_gtk()
    m.marginal_penalty().backward()
    assert m.logits.grad is not None
    assert torch.isfinite(m.logits.grad).all()


# ===========================================================================
# Smoothness penalties
# ===========================================================================


class TestGTKSmoothnessPenalties:
  def test_logit_penalty_nonneg(self) -> None:
    assert make_gtk().logit_smoothness_penalty().item() >= 0.0

  def test_logit_penalty_scalar(self) -> None:
    assert make_gtk().logit_smoothness_penalty().shape == ()

  def test_logit_penalty_zero_for_constant_logits(self) -> None:
    m = make_gtk()
    with torch.no_grad():
      m.logits.fill_(2.0)
    assert m.logit_smoothness_penalty().item() == pytest.approx(0.0, abs=1e-6)

  def test_logit_penalty_gradient_flows(self) -> None:
    m = make_gtk()
    m.logit_smoothness_penalty().backward()
    assert m.logits.grad is not None
    assert torch.isfinite(m.logits.grad).all()

  def test_scale_penalty_nonneg(self) -> None:
    assert make_gtk().scale_smoothness_penalty().item() >= 0.0

  def test_scale_penalty_scalar(self) -> None:
    assert make_gtk().scale_smoothness_penalty().shape == ()

  def test_scale_penalty_zero_for_constant_sigma(self) -> None:
    # Default init sets all raw_sigma to the same value.
    assert make_gtk().scale_smoothness_penalty().item() == pytest.approx(
      0.0, abs=1e-6
    )

  def test_scale_penalty_gradient_flows(self) -> None:
    m = make_gtk(learn_sigma=True)
    m.scale_smoothness_penalty().backward()
    assert m.raw_sigma_u.grad is not None
    assert m.raw_sigma_v.grad is not None
    assert torch.isfinite(m.raw_sigma_u.grad).all()
    assert torch.isfinite(m.raw_sigma_v.grad).all()

  def test_scale_penalty_m1_returns_zero(self) -> None:
    assert make_gtk(
      m_u=1, m_v=1
    ).scale_smoothness_penalty().item() == pytest.approx(0.0, abs=1e-6)
