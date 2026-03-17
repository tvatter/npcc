"""
gckbicop.py — Gaussian-Copula-Kernel bivariate pair-copula model.

The copula density is a positive softmax-weighted mixture of separable
Gaussian-copula kernel basis functions defined directly on the copula
scale (0, 1).  All quantities (CDF, h-functions, margins) are available
in closed form via the kernel primitive K = Phi(a_i).
"""

import torch
import torch.nn as nn

from npcc.bicopbase import (
  _BicopBase,
  _eval_basis_1d,
  atanh,
  stdnorm_icdf,
)


class GCKBicop(_BicopBase):
  """
  Gaussian-Copula-Kernel bivariate pair-copula (GCKBicop).

  The copula density is a positive mixture of separable Gaussian-copula
  kernel basis functions defined directly on the copula scale (0, 1)::

      c(u, v) = sum_{i=1}^{m_u} sum_{j=1}^{m_v}
                    W_{ij} k_u(u; U_i, rho_{u,i}) k_v(v; V_j, rho_{v,j}),

  where W_{ij} >= 0 and sum_{i,j} W_{ij} = 1.

  1-D kernel (Gaussian-copula conditional density)
  -------------------------------------------------
  ::

      k(u; U_i, rho_i) = c_GC(u, U_i; rho_i) = phi(a_i) / (s_i * phi(z)),

  with::

      z   = Phi^{-1}(u),   z_i = Phi^{-1}(U_i),
      s_i = sqrt(1 - rho_i^2),
      a_i = (z - rho_i * z_i) / s_i.

  On log scale::

      log k(u; U_i, rho_i) = -0.5 * (a_i^2 - z^2) - log(s_i).

  Kernel primitive (closed form)
  --------------------------------
  ::

      K(u; U_i, rho_i) = int_0^u k(t; U_i, rho_i) dt = Phi(a_i).

  Induced margins
  ---------------
  Each 1-D kernel integrates to 1 over (0, 1), so the mixture integrates
  to 1 over (0, 1)^2 for any weight matrix.  The induced margins are::

      m_u(u) = sum_i alpha_i k_u(u; U_i, rho_{u,i}),  alpha_i = sum_j W_{ij},
      m_v(v) = sum_j beta_j  k_v(v; V_j, rho_{v,j}),  beta_j  = sum_i W_{ij}.

  They are in general not uniform; marginal_penalty() can be used to
  encourage approximate uniformity during training.

  Weight parameterisation
  -----------------------
  ::

      W_{ij} = exp(L_{ij}) / sum_{k,l} exp(L_{kl}).

  Correlation parameterisation
  ----------------------------
  Per-center unconstrained parameters eta mapped to (-rho_max, rho_max)::

      rho_{u,i} = rho_max * tanh(eta_{u,i}),
      rho_{v,j} = rho_max * tanh(eta_{v,j}).

  Parameters
  ----------
  m_u, m_v : int
      Number of grid centers.  Must be >= 1.
  rho_u_init, rho_v_init : float
      Shared initial correlation value.
      Must satisfy abs(init) < max_abs_rho.
  learn_rho : bool
      Whether the correlation parameters are learnable.
  eps : float
      Clamp value for u away from {0, 1}.  Must be in (0, 0.5).
  max_abs_rho : float
      Upper bound on |rho|.  Must be in (0, 1).
  """

  U_centers: torch.Tensor
  V_centers: torch.Tensor
  Z_u_centers: torch.Tensor
  Z_v_centers: torch.Tensor
  raw_rho_u: torch.Tensor
  raw_rho_v: torch.Tensor

  def __init__(
    self,
    m_u: int = 25,
    m_v: int = 25,
    rho_u_init: float = 0.5,
    rho_v_init: float = 0.5,
    learn_rho: bool = True,
    eps: float = 1e-6,
    max_abs_rho: float = 0.995,
  ) -> None:
    if m_u < 1:
      raise ValueError(f"m_u must be >= 1, got {m_u}")
    if m_v < 1:
      raise ValueError(f"m_v must be >= 1, got {m_v}")
    if not (0.0 < eps < 0.5):
      raise ValueError(f"eps must be in (0, 0.5), got {eps}")
    if not (0.0 < max_abs_rho < 1.0):
      raise ValueError(f"max_abs_rho must be in (0, 1), got {max_abs_rho}")
    if abs(rho_u_init) >= max_abs_rho:
      raise ValueError(
        f"abs(rho_u_init)={abs(rho_u_init):.6g} must be"
        f" < max_abs_rho={max_abs_rho}"
      )
    if abs(rho_v_init) >= max_abs_rho:
      raise ValueError(
        f"abs(rho_v_init)={abs(rho_v_init):.6g} must be"
        f" < max_abs_rho={max_abs_rho}"
      )

    super().__init__(m_u, m_v, eps)
    self.max_abs_rho = max_abs_rho

    U = self._make_normal_grid(m_u)
    V = self._make_normal_grid(m_v)
    self.register_buffer("U_centers", U)
    self.register_buffer("V_centers", V)
    self.register_buffer("Z_u_centers", stdnorm_icdf(U, eps=eps))
    self.register_buffer("Z_v_centers", stdnorm_icdf(V, eps=eps))

    raw_u_val = atanh(rho_u_init / max_abs_rho)
    raw_v_val = atanh(rho_v_init / max_abs_rho)
    raw_u = torch.full((m_u,), raw_u_val, dtype=torch.float32)
    raw_v = torch.full((m_v,), raw_v_val, dtype=torch.float32)

    if learn_rho:
      self.raw_rho_u = nn.Parameter(raw_u)
      self.raw_rho_v = nn.Parameter(raw_v)
    else:
      self.register_buffer("raw_rho_u", raw_u)
      self.register_buffer("raw_rho_v", raw_v)

  @property
  def rho_u(self) -> torch.Tensor:
    """Per-center u-correlations [m_u], in (-max_abs_rho, max_abs_rho)."""
    return self.max_abs_rho * torch.tanh(self.raw_rho_u)

  @property
  def rho_v(self) -> torch.Tensor:
    """Per-center v-correlations [m_v], in (-max_abs_rho, max_abs_rho)."""
    return self.max_abs_rho * torch.tanh(self.raw_rho_v)

  def _basis_u(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    u-basis via the GCK kernel.

    For each center i::

        z_shift_i = rho_{u,i} * Phi^{-1}(U_i),
        scale_i   = sqrt(1 - rho_{u,i}^2),
        a_i       = (z - z_shift_i) / scale_i,
        fu_i      = exp(-0.5 * (a_i^2 - z^2) - log(scale_i)),
        Fu_i      = Phi(a_i).
    """
    z = stdnorm_icdf(u, eps=self.eps).unsqueeze(-1)  # [B, 1]
    rho = self.rho_u.unsqueeze(0)  # [1, m_u]
    z_shift = rho * self.Z_u_centers.unsqueeze(0)  # [1, m_u]
    scale = torch.sqrt(1.0 - rho.square())  # [1, m_u]
    return _eval_basis_1d(z, z_shift, scale)

  def _basis_v(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    v-basis via the GCK kernel.

    For each center j::

        z_shift_j = rho_{v,j} * Phi^{-1}(V_j),
        scale_j   = sqrt(1 - rho_{v,j}^2),
        a_j       = (z - z_shift_j) / scale_j,
        fv_j      = exp(-0.5 * (a_j^2 - z^2) - log(scale_j)),
        Fv_j      = Phi(a_j).
    """
    z = stdnorm_icdf(v, eps=self.eps).unsqueeze(-1)  # [B, 1]
    rho = self.rho_v.unsqueeze(0)  # [1, m_v]
    z_shift = rho * self.Z_v_centers.unsqueeze(0)  # [1, m_v]
    scale = torch.sqrt(1.0 - rho.square())  # [1, m_v]
    return _eval_basis_1d(z, z_shift, scale)

  def rho_smoothness_penalty(self) -> torch.Tensor:
    """
    First-order smoothness penalty on the per-center correlation vectors.

    Penalises large steps between consecutive transformed correlations,
    encouraging smooth rho profiles across the grid.

    Formula
    -------
    ::

        P_rho = mean_i (rho_{u,i+1} - rho_{u,i})^2
              + mean_j (rho_{v,j+1} - rho_{v,j})^2

    Returns
    -------
    Tensor scalar.
    """
    rho_u = self.rho_u
    rho_v = self.rho_v
    pu = (
      (rho_u[1:] - rho_u[:-1]).square().mean()
      if self.m_u > 1
      else rho_u.new_zeros(())
    )
    pv = (
      (rho_v[1:] - rho_v[:-1]).square().mean()
      if self.m_v > 1
      else rho_v.new_zeros(())
    )
    return pu + pv
