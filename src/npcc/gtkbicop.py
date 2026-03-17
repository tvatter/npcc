"""
gtkbicop.py — Gaussian-Transformation-Kernel bivariate pair-copula model.

The construction works in Gaussianised Z-space.  A density there is defined
as a softmax-weighted mixture of separable Gaussian kernels, and the copula
density is recovered via the standard-normal Jacobian.  All quantities (CDF,
h-functions, margins) are available in closed form via the kernel primitive
B = Phi(a_i).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from npcc.bicopbase import (
  _BicopBase,
  _eval_basis_1d,
  _softplus_inv,
  stdnorm_icdf,
)


class GTKBicop(_BicopBase):
  """
  Gaussian-Transformation-Kernel bivariate pair-copula (GTKBicop).

  The construction works in Gaussianised Z-space.  Let z1 = Phi^{-1}(u)
  and z2 = Phi^{-1}(v).  A density in Z-space is defined as::

      f_Z(z1, z2) = sum_{i=1}^{m_u} sum_{j=1}^{m_v}
                        W_{ij} b_u(z1; mu_i, sigma_{u,i})
                                b_v(z2; nu_j, sigma_{v,j}),

  where W_{ij} >= 0, sum W_{ij} = 1, and b is a Gaussian kernel::

      b(z; mu, sigma) = (1 / sigma) * phi((z - mu) / sigma).

  The corresponding copula density is obtained via the Jacobian formula::

      c(u, v) = f_Z(Phi^{-1}(u), Phi^{-1}(v))
                / (phi(Phi^{-1}(u)) * phi(Phi^{-1}(v))).

  Because the smoothing is applied in the unbounded Z-space rather than on
  the compact copula scale, this construction is often better behaved in
  the tails and is closer in spirit to TLL-type nonparametric estimators.

  Closed-form kernel primitive
  ----------------------------
  ::

      B(z; mu, sigma) = int_{-inf}^z b(t; mu, sigma) dt
                      = Phi((z - mu) / sigma).

  Effective kernel and primitive (used by the base class)
  --------------------------------------------------------
  ::

      fu_i(u) = b(z1; mu_i, sigma_{u,i}) / phi(z1)
              = exp(-0.5 * (a_i^2 - z1^2) - log(sigma_{u,i})),
      Fu_i(u) = B(z1; mu_i, sigma_{u,i}) = Phi(a_i),

  where a_i = (z1 - mu_i) / sigma_{u,i} and z1 = Phi^{-1}(u).

  Copula CDF and h-functions (closed form)
  -----------------------------------------
  ::

      C(u, v)     = sum_{i,j} W_{ij} Fu_i(u) Fv_j(v)
      H1^raw(u,v) = sum_{i,j} W_{ij} Fu_i(u) fv_j(v)
      H2^raw(u,v) = sum_{i,j} W_{ij} fu_i(u) Fv_j(v)

  Induced margins
  ---------------
  ::

      m_u(u) = (1/phi(z1)) sum_i alpha_i b_u(z1; mu_i, sigma_{u,i})
             = sum_i alpha_i fu_i(u),   alpha_i = sum_j W_{ij},
      m_v(v) = (1/phi(z2)) sum_j beta_j  b_v(z2; nu_j, sigma_{v,j})
             = sum_j beta_j  fv_j(v),   beta_j  = sum_i W_{ij}.

  Weight parameterisation
  -----------------------
  ::

      W_{ij} = exp(L_{ij}) / sum_{k,l} exp(L_{kl}).

  Scale parameterisation
  ----------------------
  Per-center positive scales via softplus + sigma_min::

      sigma_{u,i} = sigma_min + softplus(eta_{u,i}),
      sigma_{v,j} = sigma_min + softplus(eta_{v,j}).

  Parameters
  ----------
  m_u, m_v : int
      Number of Z-space grid centers.  Must be >= 1.
  sigma_u_init, sigma_v_init : float or None
      Shared initial scale.  If None, defaults to the vinecopulib grid
      spacing 6.5 / max(m - 1, 1).  Must satisfy init > sigma_min.
  learn_sigma : bool
      Whether the scale parameters are learnable.
  eps : float
      Clamp value for u away from {0, 1}.  Must be in (0, 0.5).
  sigma_min : float
      Minimum scale value (lower bound).  Must be > 0.
  """

  mu_u_centers: torch.Tensor
  mu_v_centers: torch.Tensor
  raw_sigma_u: torch.Tensor
  raw_sigma_v: torch.Tensor

  def __init__(
    self,
    m_u: int = 25,
    m_v: int = 25,
    sigma_u_init: float | None = None,
    sigma_v_init: float | None = None,
    learn_sigma: bool = True,
    eps: float = 1e-6,
    sigma_min: float = 0.01,
  ) -> None:
    if m_u < 1:
      raise ValueError(f"m_u must be >= 1, got {m_u}")
    if m_v < 1:
      raise ValueError(f"m_v must be >= 1, got {m_v}")
    if not (0.0 < eps < 0.5):
      raise ValueError(f"eps must be in (0, 0.5), got {eps}")
    if sigma_min <= 0.0:
      raise ValueError(f"sigma_min must be > 0, got {sigma_min}")

    # Default sigma = one vinecopulib grid spacing (6.5 / (m - 1))
    _spacing_u = 6.5 / max(m_u - 1, 1)
    _spacing_v = 6.5 / max(m_v - 1, 1)
    if sigma_u_init is None:
      sigma_u_init = _spacing_u
    if sigma_v_init is None:
      sigma_v_init = _spacing_v

    if sigma_u_init <= sigma_min:
      raise ValueError(
        f"sigma_u_init={sigma_u_init:.6g} must be > sigma_min={sigma_min}"
      )
    if sigma_v_init <= sigma_min:
      raise ValueError(
        f"sigma_v_init={sigma_v_init:.6g} must be > sigma_min={sigma_min}"
      )

    super().__init__(m_u, m_v, eps)
    self.sigma_min = sigma_min

    self.register_buffer(
      "mu_u_centers", self._make_normal_grid(m_u, z_scale=True)
    )
    self.register_buffer(
      "mu_v_centers", self._make_normal_grid(m_v, z_scale=True)
    )

    raw_u_val = _softplus_inv(sigma_u_init - sigma_min)
    raw_v_val = _softplus_inv(sigma_v_init - sigma_min)
    raw_u = torch.full((m_u,), raw_u_val, dtype=torch.float32)
    raw_v = torch.full((m_v,), raw_v_val, dtype=torch.float32)

    if learn_sigma:
      self.raw_sigma_u = nn.Parameter(raw_u)
      self.raw_sigma_v = nn.Parameter(raw_v)
    else:
      self.register_buffer("raw_sigma_u", raw_u)
      self.register_buffer("raw_sigma_v", raw_v)

  @property
  def sigma_u(self) -> torch.Tensor:
    """Per-center u-scales [m_u], in (sigma_min, inf)."""
    return self.sigma_min + F.softplus(self.raw_sigma_u)

  @property
  def sigma_v(self) -> torch.Tensor:
    """Per-center v-scales [m_v], in (sigma_min, inf)."""
    return self.sigma_min + F.softplus(self.raw_sigma_v)

  def _basis_u(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    u-basis via the GTK kernel in Z-space.

    For each center i::

        a_i  = (z1 - mu_i) / sigma_{u,i},
        fu_i = b(z1; mu_i, sigma_{u,i}) / phi(z1)
             = exp(-0.5 * (a_i^2 - z1^2) - log(sigma_{u,i})),
        Fu_i = Phi(a_i).
    """
    z = stdnorm_icdf(u, eps=self.eps).unsqueeze(-1)  # [B, 1]
    z_shift = self.mu_u_centers.unsqueeze(0)  # [1, m_u]
    scale = self.sigma_u.unsqueeze(0)  # [1, m_u]
    return _eval_basis_1d(z, z_shift, scale)

  def _basis_v(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    v-basis via the GTK kernel in Z-space.

    For each center j::

        a_j  = (z2 - nu_j) / sigma_{v,j},
        fv_j = b(z2; nu_j, sigma_{v,j}) / phi(z2)
             = exp(-0.5 * (a_j^2 - z2^2) - log(sigma_{v,j})),
        Fv_j = Phi(a_j).
    """
    z = stdnorm_icdf(v, eps=self.eps).unsqueeze(-1)  # [B, 1]
    z_shift = self.mu_v_centers.unsqueeze(0)  # [1, m_v]
    scale = self.sigma_v.unsqueeze(0)  # [1, m_v]
    return _eval_basis_1d(z, z_shift, scale)

  def scale_smoothness_penalty(self) -> torch.Tensor:
    """
    First-order smoothness penalty on the per-center scale vectors.

    Penalises large steps between consecutive transformed scales,
    encouraging smooth sigma profiles across the Z-space grid.

    Formula
    -------
    ::

        P_sigma = mean_i (sigma_{u,i+1} - sigma_{u,i})^2
                + mean_j (sigma_{v,j+1} - sigma_{v,j})^2

    Returns
    -------
    Tensor scalar.
    """
    sigma_u = self.sigma_u
    sigma_v = self.sigma_v
    pu = (
      (sigma_u[1:] - sigma_u[:-1]).square().mean()
      if self.m_u > 1
      else sigma_u.new_zeros(())
    )
    pv = (
      (sigma_v[1:] - sigma_v[:-1]).square().mean()
      if self.m_v > 1
      else sigma_v.new_zeros(())
    )
    return pu + pv
