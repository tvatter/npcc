import pytest
import torch

from npcc.gcbicop import GCBicop

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_model(m_u: int = 10, m_v: int = 10, **kwargs: object) -> GCBicop:
  return GCBicop(m_u=m_u, m_v=m_v, **kwargs)  # type: ignore[arg-type]


def random_uv(n: int = 100, seed: int = 0) -> torch.Tensor:
  g = torch.Generator()
  g.manual_seed(seed)
  return torch.rand(n, 2, generator=g) * 0.9 + 0.05


# ---------------------------------------------------------------------------
# Initialization validation
# ---------------------------------------------------------------------------


class TestInit:
  def test_m_u_must_be_at_least_1(self) -> None:
    with pytest.raises(ValueError, match="m_u"):
      GCBicop(m_u=0)

  def test_m_v_must_be_at_least_1(self) -> None:
    with pytest.raises(ValueError, match="m_v"):
      GCBicop(m_v=0)

  def test_eps_must_be_in_open_interval(self) -> None:
    with pytest.raises(ValueError, match="eps"):
      GCBicop(eps=0.0)
    with pytest.raises(ValueError, match="eps"):
      GCBicop(eps=0.5)
    with pytest.raises(ValueError, match="eps"):
      GCBicop(eps=-1e-7)

  def test_z_max_must_be_positive(self) -> None:
    with pytest.raises(ValueError, match="z_max"):
      GCBicop(z_max=0.0)
    with pytest.raises(ValueError, match="z_max"):
      GCBicop(z_max=-1.0)

  def test_max_abs_rho_must_be_in_open_01(self) -> None:
    with pytest.raises(ValueError, match="max_abs_rho"):
      GCBicop(max_abs_rho=1.0)
    with pytest.raises(ValueError, match="max_abs_rho"):
      GCBicop(max_abs_rho=0.0)
    with pytest.raises(ValueError, match="max_abs_rho"):
      GCBicop(max_abs_rho=-0.5)

  def test_rho_u_init_must_be_strictly_inside_range(self) -> None:
    with pytest.raises(ValueError, match="rho_u_init"):
      GCBicop(rho_u_init=0.995, max_abs_rho=0.995)

  def test_rho_v_init_must_be_strictly_inside_range(self) -> None:
    with pytest.raises(ValueError, match="rho_v_init"):
      GCBicop(rho_v_init=-0.995, max_abs_rho=0.995)

  def test_raw_rho_shapes(self) -> None:
    model = make_model(m_u=8, m_v=12)
    assert model.raw_rho_u.shape == (8,)
    assert model.raw_rho_v.shape == (12,)

  def test_rho_shapes(self) -> None:
    model = make_model(m_u=8, m_v=12)
    assert model.rho_u.shape == (8,)
    assert model.rho_v.shape == (12,)

  def test_rho_values_in_range(self) -> None:
    model = make_model(
      m_u=5, m_v=5, rho_u_init=0.5, rho_v_init=-0.3, max_abs_rho=0.995
    )
    assert (model.rho_u.abs() < 0.995).all()
    assert (model.rho_v.abs() < 0.995).all()

  def test_weights_nonneg_sum_to_one(self) -> None:
    model = make_model()
    W = model.weights
    assert (W >= 0).all()
    assert W.sum().item() == pytest.approx(1.0, abs=1e-5)

  def test_learn_rho_false_registers_buffers(self) -> None:
    model = make_model(learn_rho=False)
    assert "raw_rho_u" not in dict(model.named_parameters())
    assert "raw_rho_v" not in dict(model.named_parameters())


# ---------------------------------------------------------------------------
# PDF: nonnegativity and integral
# ---------------------------------------------------------------------------


class TestInputValidation:
  def test_pdf_rejects_wrong_shape(self) -> None:
    model = make_model()
    with pytest.raises(ValueError, match="UV must have shape"):
      model.pdf(torch.rand(10, 3))
    with pytest.raises(ValueError, match="UV must have shape"):
      model.pdf(torch.rand(10))

  def test_cdf_rejects_wrong_shape(self) -> None:
    model = make_model()
    with pytest.raises(ValueError, match="UV must have shape"):
      model.cdf(torch.rand(10, 3))

  def test_hfunc1_rejects_wrong_shape(self) -> None:
    model = make_model()
    with pytest.raises(ValueError, match="UV must have shape"):
      model.hfunc1(torch.rand(5, 1))

  def test_hfunc2_rejects_wrong_shape(self) -> None:
    model = make_model()
    with pytest.raises(ValueError, match="UV must have shape"):
      model.hfunc2(torch.rand(5, 1))

  def test_margin_u_rejects_2d(self) -> None:
    model = make_model()
    with pytest.raises(ValueError, match="u must be 1-D"):
      model.margin_u(torch.rand(5, 2))

  def test_margin_v_rejects_2d(self) -> None:
    model = make_model()
    with pytest.raises(ValueError, match="v must be 1-D"):
      model.margin_v(torch.rand(5, 2))

  def test_basis_pdf_u_rejects_2d(self) -> None:
    model = make_model()
    with pytest.raises(ValueError, match="u must be 1-D"):
      model.basis_pdf_u(torch.rand(5, 2))

  def test_basis_cdf_v_rejects_2d(self) -> None:
    model = make_model()
    with pytest.raises(ValueError, match="v must be 1-D"):
      model.basis_cdf_v(torch.rand(5, 2))


class TestPDF:
  def test_pdf_nonnegative(self) -> None:
    model = make_model()
    UV = random_uv(200)
    vals = model.pdf(UV)
    assert (vals >= 0).all()

  def test_pdf_integral_approx_one(self) -> None:
    """
    Since each 1D kernel integrates to exactly 1 over (0, 1), the full
    mixture integrates to 1 over (0,1)^2 for any weights.  The numerical
    integral over [0.001, 0.999]^2 (trapezoid rule, 150 pts per axis)
    should be within 3 % of 1.
    """
    model = make_model(m_u=15, m_v=15)
    n = 150
    u_grid = torch.linspace(0.001, 0.999, n)
    v_grid = torch.linspace(0.001, 0.999, n)
    uu, vv = torch.meshgrid(u_grid, v_grid, indexing="ij")
    UV = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
    with torch.no_grad():
      pdf_vals = model.pdf(UV).reshape(n, n)
    integral = torch.trapezoid(
      torch.trapezoid(pdf_vals, u_grid, dim=0), v_grid
    ).item()
    assert integral == pytest.approx(1.0, abs=0.03)

  def test_log_pdf_finite(self) -> None:
    model = make_model()
    UV = random_uv(100)
    vals = model.log_pdf(UV)
    assert torch.isfinite(vals).all()


# ---------------------------------------------------------------------------
# Margins: formula consistency
# ---------------------------------------------------------------------------


class TestMargins:
  def test_margin_u_equals_basis_times_alpha(self) -> None:
    """m_u(u) = basis_pdf_u(u) @ alpha, alpha = W.sum(1)."""
    model = make_model(m_u=8, m_v=8)
    u = torch.linspace(0.05, 0.95, 30)
    alpha = model.weights.sum(dim=1)
    expected = model.basis_pdf_u(u) @ alpha
    actual = model.margin_u(u)
    assert torch.allclose(actual, expected, rtol=1e-5)

  def test_margin_v_equals_basis_times_beta(self) -> None:
    """m_v(v) = basis_pdf_v(v) @ beta, beta = W.sum(0)."""
    model = make_model(m_u=8, m_v=8)
    v = torch.linspace(0.05, 0.95, 30)
    beta = model.weights.sum(dim=0)
    expected = model.basis_pdf_v(v) @ beta
    actual = model.margin_v(v)
    assert torch.allclose(actual, expected, rtol=1e-5)

  def test_margin_u_nonneg(self) -> None:
    model = make_model()
    u = torch.linspace(0.01, 0.99, 50)
    assert (model.margin_u(u) >= 0).all()

  def test_margin_v_nonneg(self) -> None:
    model = make_model()
    v = torch.linspace(0.01, 0.99, 50)
    assert (model.margin_v(v) >= 0).all()


# ---------------------------------------------------------------------------
# H-functions: boundary conditions
# ---------------------------------------------------------------------------


class TestHfunc:
  def test_hfunc1_at_u1_equals_margin_v(self) -> None:
    """hfunc1((1, v), normalized=False) → margin_v(v) as u → 1."""
    model = make_model(m_u=12, m_v=12)
    v = torch.linspace(0.05, 0.95, 40)
    # Use u = 1 - eps so stdnorm_icdf clamps to z_max ≈ +4.75
    u_top = torch.full_like(v, 1.0 - model.eps)
    UV = torch.stack([u_top, v], dim=1)
    with torch.no_grad():
      h1 = model.hfunc1(UV, normalized=False)
      mv = model.margin_v(v)
    assert torch.allclose(h1, mv, rtol=1e-2, atol=1e-3)

  def test_hfunc2_at_v1_equals_margin_u(self) -> None:
    """hfunc2((u, 1), normalized=False) → margin_u(u) as v → 1."""
    model = make_model(m_u=12, m_v=12)
    u = torch.linspace(0.05, 0.95, 40)
    v_top = torch.full_like(u, 1.0 - model.eps)
    UV = torch.stack([u, v_top], dim=1)
    with torch.no_grad():
      h2 = model.hfunc2(UV, normalized=False)
      mu = model.margin_u(u)
    assert torch.allclose(h2, mu, rtol=1e-2, atol=1e-3)

  def test_hfunc1_normalized_in_unit_interval(self) -> None:
    """Normalized hfunc1 ∈ [0, 1] because K_u ∈ [0, 1]."""
    model = make_model()
    UV = random_uv(200)
    with torch.no_grad():
      vals = model.hfunc1(UV, normalized=True)
    assert (vals >= -1e-5).all()
    assert (vals <= 1.0 + 1e-5).all()

  def test_hfunc2_normalized_in_unit_interval(self) -> None:
    """Normalized hfunc2 ∈ [0, 1] because K_v ∈ [0, 1]."""
    model = make_model()
    UV = random_uv(200)
    with torch.no_grad():
      vals = model.hfunc2(UV, normalized=True)
    assert (vals >= -1e-5).all()
    assert (vals <= 1.0 + 1e-5).all()

  def test_hfunc_raw_nonneg(self) -> None:
    model = make_model()
    UV = random_uv(100)
    with torch.no_grad():
      assert (model.hfunc1(UV) >= 0).all()
      assert (model.hfunc2(UV) >= 0).all()


# ---------------------------------------------------------------------------
# CDF: boundary values
# ---------------------------------------------------------------------------


class TestCDF:
  def test_cdf_nonneg(self) -> None:
    model = make_model()
    UV = random_uv(100)
    with torch.no_grad():
      assert (model.cdf(UV) >= 0).all()

  def test_cdf_at_one_approx_one(self) -> None:
    """C(1, 1) should be 1 since all K_u and K_v → 1."""
    model = make_model()
    eps = model.eps
    UV = torch.tensor([[1.0 - eps, 1.0 - eps]])
    with torch.no_grad():
      val = model.cdf(UV).item()
    assert val == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradients:
  def test_gradient_through_logits(self) -> None:
    model = make_model(m_u=8, m_v=8)
    UV = random_uv(50)
    model.nll(UV).backward()
    assert model.logits.grad is not None
    assert torch.isfinite(model.logits.grad).all()

  def test_gradient_through_raw_rho_u(self) -> None:
    model = make_model(m_u=8, m_v=8)
    UV = random_uv(50)
    model.nll(UV).backward()
    assert model.raw_rho_u.grad is not None
    assert torch.isfinite(model.raw_rho_u.grad).all()

  def test_gradient_through_raw_rho_v(self) -> None:
    model = make_model(m_u=8, m_v=8)
    UV = random_uv(50)
    model.nll(UV).backward()
    assert model.raw_rho_v.grad is not None
    assert torch.isfinite(model.raw_rho_v.grad).all()

  def test_no_nan_gradients(self) -> None:
    model = make_model(m_u=8, m_v=8)
    UV = random_uv(100)
    model.nll(UV).backward()
    for name, p in model.named_parameters():
      assert p.grad is not None, f"No grad for {name}"
      assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"


# ---------------------------------------------------------------------------
# Marginal penalty
# ---------------------------------------------------------------------------


class TestMarginalPenalty:
  def test_penalty_nonneg(self) -> None:
    model = make_model()
    assert model.marginal_penalty().item() >= 0.0

  def test_penalty_is_scalar(self) -> None:
    model = make_model()
    p = model.marginal_penalty()
    assert p.shape == ()

  def test_penalty_gradient_flows(self) -> None:
    model = make_model()
    model.marginal_penalty().backward()
    assert model.logits.grad is not None
    assert torch.isfinite(model.logits.grad).all()
