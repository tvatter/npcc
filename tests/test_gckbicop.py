"""Tests for GCKBicop."""

import pytest
import torch

from npcc.gckbicop import GCKBicop

from .conftest import random_uv, unit_grid


def make_gck(m_u: int = 10, m_v: int = 10, **kwargs: object) -> GCKBicop:
  return GCKBicop(m_u=m_u, m_v=m_v, **kwargs)  # ty: ignore[invalid-argument-type]


# ===========================================================================
# Initialisation
# ===========================================================================


class TestGCKInit:
  def test_m_u_must_be_at_least_1(self) -> None:
    with pytest.raises(ValueError, match="m_u"):
      GCKBicop(m_u=0)

  def test_m_v_must_be_at_least_1(self) -> None:
    with pytest.raises(ValueError, match="m_v"):
      GCKBicop(m_v=0)

  def test_eps_must_be_positive(self) -> None:
    with pytest.raises(ValueError, match="eps"):
      GCKBicop(eps=0.0)

  def test_eps_must_be_below_half(self) -> None:
    with pytest.raises(ValueError, match="eps"):
      GCKBicop(eps=0.5)

  def test_max_abs_rho_must_be_in_open_unit_interval(self) -> None:
    with pytest.raises(ValueError, match="max_abs_rho"):
      GCKBicop(max_abs_rho=1.0)

  def test_rho_u_init_must_be_within_max_abs_rho(self) -> None:
    with pytest.raises(ValueError, match="rho_u_init"):
      GCKBicop(rho_u_init=0.9, max_abs_rho=0.8)

  def test_rho_v_init_must_be_within_max_abs_rho(self) -> None:
    with pytest.raises(ValueError, match="rho_v_init"):
      GCKBicop(rho_v_init=0.9, max_abs_rho=0.8)

  def test_logits_shape(self) -> None:
    m = make_gck(m_u=5, m_v=7)
    assert m.logits.shape == (5, 7)

  def test_weights_sum_to_one(self) -> None:
    m = make_gck()
    assert m.weights.sum().item() == pytest.approx(1.0, abs=1e-5)

  def test_weights_nonneg(self) -> None:
    m = make_gck()
    assert (m.weights >= 0).all()

  def test_rho_shapes(self) -> None:
    m = make_gck(m_u=6, m_v=8)
    assert m.rho_u.shape == (6,)
    assert m.rho_v.shape == (8,)

  def test_rho_within_bounds(self) -> None:
    m = make_gck(max_abs_rho=0.9)
    assert (m.rho_u.abs() < 0.9).all()
    assert (m.rho_v.abs() < 0.9).all()

  def test_learn_rho_false_registers_buffer(self) -> None:
    m = GCKBicop(learn_rho=False)
    assert "raw_rho_u" in dict(m.named_buffers())
    assert "raw_rho_v" in dict(m.named_buffers())


# ===========================================================================
# Input validation
# ===========================================================================


class TestGCKInputValidation:
  def test_pdf_rejects_1d(self) -> None:
    m = make_gck()
    with pytest.raises(ValueError, match="shape"):
      m.pdf(torch.rand(10))

  def test_pdf_rejects_wrong_columns(self) -> None:
    m = make_gck()
    with pytest.raises(ValueError, match="shape"):
      m.pdf(torch.rand(10, 3))

  def test_cdf_rejects_bad_shape(self) -> None:
    m = make_gck()
    with pytest.raises(ValueError, match="shape"):
      m.cdf(torch.rand(10, 3))

  def test_hfunc1_rejects_bad_shape(self) -> None:
    m = make_gck()
    with pytest.raises(ValueError, match="shape"):
      m.hfunc1(torch.rand(5))

  def test_hfunc2_rejects_bad_shape(self) -> None:
    m = make_gck()
    with pytest.raises(ValueError, match="shape"):
      m.hfunc2(torch.rand(5, 3))

  def test_margin_u_rejects_2d(self) -> None:
    m = make_gck()
    with pytest.raises(ValueError, match="1-D"):
      m.margin_u(torch.rand(5, 1))

  def test_margin_v_rejects_2d(self) -> None:
    m = make_gck()
    with pytest.raises(ValueError, match="1-D"):
      m.margin_v(torch.rand(5, 1))

  def test_basis_pdf_u_rejects_2d(self) -> None:
    m = make_gck()
    with pytest.raises(ValueError, match="1-D"):
      m.basis_pdf_u(torch.rand(5, 1))

  def test_basis_cdf_v_rejects_2d(self) -> None:
    m = make_gck()
    with pytest.raises(ValueError, match="1-D"):
      m.basis_cdf_v(torch.rand(5, 1))


# ===========================================================================
# PDF
# ===========================================================================


class TestGCKPDF:
  def test_pdf_nonneg(self) -> None:
    m = make_gck()
    UV = random_uv()
    assert (m.pdf(UV) >= 0).all()

  def test_pdf_integrates_to_one(self) -> None:
    # Midpoint quadrature: t_k = (k + 0.5) / n covers [0, 1] without
    # endpoints; mean(c) approximates int_0^1 int_0^1 c(u,v) du dv = 1.
    m = make_gck()
    n = 60
    t = (torch.arange(n, dtype=torch.float32) + 0.5) / n
    uu, vv = torch.meshgrid(t, t, indexing="ij")
    UV = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
    assert m.pdf(UV).mean().item() == pytest.approx(1.0, abs=0.15)

  def test_log_pdf_is_finite(self) -> None:
    m = make_gck()
    UV = random_uv()
    assert torch.isfinite(m.log_pdf(UV)).all()


# ===========================================================================
# Margins
# ===========================================================================


class TestGCKMargins:
  def test_margin_u_formula_consistent(self) -> None:
    m = make_gck()
    u = unit_grid()
    W = m.weights
    fu = m.basis_pdf_u(u)
    expected = fu @ W.sum(dim=1)
    assert torch.allclose(m.margin_u(u), expected, atol=1e-5)

  def test_margin_v_formula_consistent(self) -> None:
    m = make_gck()
    v = unit_grid()
    W = m.weights
    fv = m.basis_pdf_v(v)
    expected = fv @ W.sum(dim=0)
    assert torch.allclose(m.margin_v(v), expected, atol=1e-5)

  def test_margin_u_nonneg(self) -> None:
    m = make_gck()
    assert (m.margin_u(unit_grid()) >= 0).all()

  def test_margin_v_nonneg(self) -> None:
    m = make_gck()
    assert (m.margin_v(unit_grid()) >= 0).all()


# ===========================================================================
# H-functions
# ===========================================================================


class TestGCKHfunc:
  def test_hfunc1_at_u1_equals_margin_v(self) -> None:
    m = make_gck()
    v = unit_grid()
    ones = torch.ones_like(v)
    UV = torch.stack([ones, v], dim=1)
    h1 = m.hfunc1(UV, normalized=False)
    mv = m.margin_v(v)
    assert torch.allclose(h1, mv, atol=1e-4)

  def test_hfunc2_at_v1_equals_margin_u(self) -> None:
    m = make_gck()
    u = unit_grid()
    ones = torch.ones_like(u)
    UV = torch.stack([u, ones], dim=1)
    h2 = m.hfunc2(UV, normalized=False)
    mu = m.margin_u(u)
    assert torch.allclose(h2, mu, atol=1e-4)

  def test_hfunc1_normalized_in_unit_interval(self) -> None:
    m = make_gck()
    UV = random_uv()
    h1n = m.hfunc1(UV, normalized=True)
    assert (h1n >= -1e-6).all() and (h1n <= 1.0 + 1e-6).all()

  def test_hfunc2_normalized_in_unit_interval(self) -> None:
    m = make_gck()
    UV = random_uv()
    h2n = m.hfunc2(UV, normalized=True)
    assert (h2n >= -1e-6).all() and (h2n <= 1.0 + 1e-6).all()

  def test_hfunc1_raw_nonneg(self) -> None:
    m = make_gck()
    assert (m.hfunc1(random_uv()) >= 0).all()

  def test_hfunc2_raw_nonneg(self) -> None:
    m = make_gck()
    assert (m.hfunc2(random_uv()) >= 0).all()


# ===========================================================================
# CDF
# ===========================================================================


class TestGCKCDF:
  def test_cdf_nonneg(self) -> None:
    m = make_gck()
    assert (m.cdf(random_uv()) >= 0).all()

  def test_cdf_at_one_one(self) -> None:
    m = make_gck()
    UV = torch.tensor([[0.9999, 0.9999]])
    assert m.cdf(UV).item() == pytest.approx(1.0, abs=0.01)


# ===========================================================================
# Gradients
# ===========================================================================


class TestGCKGradients:
  def test_nll_gradient_flows_through_logits(self) -> None:
    m = make_gck(m_u=8, m_v=8)
    m.nll(random_uv(100)).backward()
    assert m.logits.grad is not None
    assert torch.isfinite(m.logits.grad).all()

  def test_nll_gradient_flows_through_rho_u(self) -> None:
    m = make_gck(m_u=8, m_v=8)
    m.nll(random_uv(100)).backward()
    assert m.raw_rho_u.grad is not None
    assert torch.isfinite(m.raw_rho_u.grad).all()

  def test_nll_gradient_flows_through_rho_v(self) -> None:
    m = make_gck(m_u=8, m_v=8)
    m.nll(random_uv(100)).backward()
    assert m.raw_rho_v.grad is not None
    assert torch.isfinite(m.raw_rho_v.grad).all()

  def test_no_nan_gradients(self) -> None:
    m = make_gck(m_u=8, m_v=8)
    m.nll(random_uv(100)).backward()
    for name, p in m.named_parameters():
      assert p.grad is not None, f"No grad for {name}"
      assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"


# ===========================================================================
# Marginal penalty
# ===========================================================================


class TestGCKMarginalPenalty:
  def test_penalty_nonneg(self) -> None:
    assert make_gck().marginal_penalty().item() >= 0.0

  def test_penalty_is_scalar(self) -> None:
    assert make_gck().marginal_penalty().shape == ()

  def test_penalty_gradient_flows(self) -> None:
    m = make_gck()
    m.marginal_penalty().backward()
    assert m.logits.grad is not None
    assert torch.isfinite(m.logits.grad).all()


# ===========================================================================
# Smoothness penalties
# ===========================================================================


class TestGCKSmoothnessPenalties:
  def test_logit_penalty_nonneg(self) -> None:
    assert make_gck().logit_smoothness_penalty().item() >= 0.0

  def test_logit_penalty_scalar(self) -> None:
    assert make_gck().logit_smoothness_penalty().shape == ()

  def test_logit_penalty_zero_for_constant_logits(self) -> None:
    m = make_gck()
    with torch.no_grad():
      m.logits.fill_(1.23)
    assert m.logit_smoothness_penalty().item() == pytest.approx(0.0, abs=1e-6)

  def test_logit_penalty_gradient_flows(self) -> None:
    m = make_gck()
    m.logit_smoothness_penalty().backward()
    assert m.logits.grad is not None
    assert torch.isfinite(m.logits.grad).all()

  def test_logit_penalty_known_value(self) -> None:
    # 3x3 logit matrix with hand-computed penalty = 3.5 (see comments below).
    #
    #   L = [[0, 1, 3],
    #        [2, 2, 2],
    #        [1, 0, 0]]
    #
    # Row diffs (6 values): squared [4,1,1,1,4,4] -> mean = 15/6 = 2.5
    # Col diffs (6 values): squared [1,0,1,4,0,0] -> mean =  6/6 = 1.0
    # Total = 3.5
    m = make_gck(m_u=3, m_v=3)
    L = torch.tensor([[0.0, 1.0, 3.0], [2.0, 2.0, 2.0], [1.0, 0.0, 0.0]])
    with torch.no_grad():
      m.logits.copy_(L)
    assert m.logit_smoothness_penalty().item() == pytest.approx(3.5, abs=1e-5)

  def test_rho_penalty_nonneg(self) -> None:
    assert make_gck().rho_smoothness_penalty().item() >= 0.0

  def test_rho_penalty_scalar(self) -> None:
    assert make_gck().rho_smoothness_penalty().shape == ()

  def test_rho_penalty_zero_for_constant_rho(self) -> None:
    # Default init sets all raw_rho to the same value.
    assert make_gck().rho_smoothness_penalty().item() == pytest.approx(
      0.0, abs=1e-6
    )

  def test_rho_penalty_gradient_flows(self) -> None:
    m = make_gck(learn_rho=True)
    m.rho_smoothness_penalty().backward()
    assert m.raw_rho_u.grad is not None
    assert m.raw_rho_v.grad is not None
    assert torch.isfinite(m.raw_rho_u.grad).all()
    assert torch.isfinite(m.raw_rho_v.grad).all()

  def test_rho_penalty_m1_returns_zero(self) -> None:
    assert make_gck(
      m_u=1, m_v=1
    ).rho_smoothness_penalty().item() == pytest.approx(0.0, abs=1e-6)
