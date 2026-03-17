import math
from typing import Literal

import torch
import torch.nn as nn


def stdnorm_pdf(z: torch.Tensor) -> torch.Tensor:
  return torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)


def stdnorm_cdf(z: torch.Tensor) -> torch.Tensor:
  return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def stdnorm_icdf(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  u = u.clamp(eps, 1.0 - eps)
  return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


def atanh(x: float) -> float:
  return 0.5 * math.log((1.0 + x) / (1.0 - x))


class GCBicop(nn.Module):
  """
  Bivariate copula based on Gaussian-copula kernel basis functions.

  Model
  -----
  The copula density is represented as a positive mixture of separable
  basis terms::

      c(u, v) = sum_{i=1}^{m_u} sum_{j=1}^{m_v}
                    W_{ij} k_u(u; U_i, rho_{u,i}) k_v(v; V_j, rho_{v,j})

  with weights W_{ij} >= 0 and sum_{i,j} W_{ij} = 1.

  Each one-dimensional basis function is a Gaussian-copula density
  evaluated along one coordinate with the other held fixed at a grid
  center::

      k(u; U_i, rho_i) = c_GC(u, U_i; rho_i) = phi(a_i) / (s_i * phi(z)),

  where::

      z   = Phi^{-1}(u),    z_i = Phi^{-1}(U_i),
      s_i = sqrt(1 - rho_i^2),
      a_i = (z - rho_i * z_i) / s_i.

  Its primitive is available in closed form::

      K(u; U_i, rho_i) = int_0^u k(t; U_i, rho_i) dt = Phi(a_i).

  These closed-form expressions yield the density, CDF, and h-functions
  entirely in terms of normal CDFs and PDFs.

  Note
  ----
  Since each 1D kernel integrates to exactly 1 over (0, 1)::

      int_0^1 k(u; U_i, rho_i) du = Phi(+inf) - Phi(-inf) = 1,

  the mixture density integrates to 1 over (0, 1)^2 for any weights.
  The induced margins are in general not uniform, however.  A marginal
  penalty is provided to encourage approximate uniformity.

  Weight parameterization
  -----------------------
  Weights are obtained via softmax over the logit matrix L::

      W_{ij} = exp(L_{ij}) / sum_{k,l} exp(L_{kl}).

  Correlation parameterization
  ----------------------------
  Per-center correlation parameters are stored as unconstrained scalars
  ``raw_rho_u`` (shape [m_u]) and ``raw_rho_v`` (shape [m_v]) and
  mapped to (-max_abs_rho, max_abs_rho) via::

      rho_{u,i} = rho_max * tanh(eta_{u,i}),
      rho_{v,j} = rho_max * tanh(eta_{v,j}).

  Parameters
  ----------
  m_u, m_v:
      Number of grid centers along each margin.  Must be >= 1.
  rho_u_init, rho_v_init:
      Initial correlation value shared by all u / v centers.
      Must satisfy ``abs(rho_{u,v}_init) < max_abs_rho``.
  learn_rho:
      Whether to treat the correlation parameters as learnable.
  grid:
      Grid type for placing centers: ``"uniform"`` or ``"probit"``.
  z_max:
      Half-range of the z-grid used when ``grid="probit"``.  Must be > 0.
  eps:
      Small clamp value to keep u away from 0 and 1.  Must be in (0, 0.5).
  max_abs_rho:
      Upper bound on abs(rho).  Must be in (0, 1).
  """

  # Class-level buffer annotations so static type checkers know these
  # are Tensors, not the ambiguous Tensor | Module union that nn.Module's
  # __getattr__ would otherwise imply.
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
    grid: Literal["uniform", "probit"] = "probit",
    z_max: float = 2.5,
    eps: float = 1e-6,
    max_abs_rho: float = 0.995,
  ) -> None:
    super().__init__()

    if m_u < 1:
      raise ValueError(f"m_u must be >= 1, got {m_u}")
    if m_v < 1:
      raise ValueError(f"m_v must be >= 1, got {m_v}")
    if not (0.0 < eps < 0.5):
      raise ValueError(f"eps must be in (0, 0.5), got {eps}")
    if z_max <= 0.0:
      raise ValueError(f"z_max must be > 0, got {z_max}")
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

    self.m_u = m_u
    self.m_v = m_v
    self.eps = eps
    self.max_abs_rho = max_abs_rho

    self.logits = nn.Parameter(torch.zeros(m_u, m_v))

    U = self._make_grid(m_u, grid=grid, z_max=z_max)
    V = self._make_grid(m_v, grid=grid, z_max=z_max)
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

  @staticmethod
  def _make_grid(
    m: int,
    grid: Literal["uniform", "probit"],
    z_max: float,
  ) -> torch.Tensor:
    if grid == "uniform":
      return torch.linspace(0.0, 1.0, m + 2, dtype=torch.float32)[1:-1]
    if grid == "probit":
      z = torch.linspace(-z_max, z_max, m, dtype=torch.float32)
      return stdnorm_cdf(z)
    raise ValueError(f"Unknown grid={grid!r}")

  @staticmethod
  def _check_UV(UV: torch.Tensor) -> None:
    """Raise ValueError if UV does not have shape [B, 2]."""
    if UV.ndim != 2 or UV.shape[1] != 2:
      raise ValueError(f"UV must have shape [B, 2], got {tuple(UV.shape)}")

  @staticmethod
  def _check_1d(t: torch.Tensor, name: str = "input") -> None:
    """Raise ValueError if t is not 1-D."""
    if t.ndim != 1:
      raise ValueError(f"{name} must be 1-D, got shape {tuple(t.shape)}")

  @property
  def rho_u(self) -> torch.Tensor:
    """Per-center u-correlations, shape [m_u], in (-max_abs_rho, max_abs_rho)."""
    return self.max_abs_rho * torch.tanh(self.raw_rho_u)

  @property
  def rho_v(self) -> torch.Tensor:
    """Per-center v-correlations, shape [m_v], in (-max_abs_rho, max_abs_rho)."""
    return self.max_abs_rho * torch.tanh(self.raw_rho_v)

  @property
  def weights(self) -> torch.Tensor:
    """Weight matrix W of shape [m_u, m_v]; W_{ij} >= 0, sum = 1."""
    w = torch.softmax(self.logits.reshape(-1), dim=0)
    return w.reshape(self.m_u, self.m_v)

  def _eval_basis_1d(
    self,
    z: torch.Tensor,
    z_centers: torch.Tensor,
    rho: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Kernel pdf and cdf for one variable, sharing intermediate computations.

    Parameters
    ----------
    z : Tensor of shape [B, 1]
        Phi^{-1}(u) for the batch.
    z_centers : Tensor of shape [1, m]
        Phi^{-1}(U_i) for the grid centers.
    rho : Tensor of shape [1, m]
        Per-center correlations rho_i.

    Returns
    -------
    f : Tensor of shape [B, m]
        Kernel pdf k(u; U_i, rho_i).
    F : Tensor of shape [B, m]
        Kernel cdf K(u; U_i, rho_i) = int_0^u k(t; U_i, rho_i) dt.

    Formula
    -------
    With::

        s_i = sqrt(1 - rho_i^2),
        a_i = (z - rho_i * z_i) / s_i,

    the kernel and its primitive are::

        k(u; U_i, rho_i) = exp(-0.5 * (a_i^2 - z^2) - log(s_i)),
        K(u; U_i, rho_i) = Phi(a_i).

    Both quantities share the s_i and a_i intermediate results.
    """
    s = torch.sqrt(1.0 - rho.square())  # [1, m]
    a = (z - rho * z_centers) / s  # [B, m]
    f = (-0.5 * (a.square() - z.square()) - s.log()).exp()
    return f, stdnorm_cdf(a)

  def basis_pdf_u(self, u: torch.Tensor) -> torch.Tensor:
    """Evaluate the u-basis kernel PDFs; returns shape [B, m_u]."""
    self._check_1d(u, "u")
    z = stdnorm_icdf(u, eps=self.eps).unsqueeze(-1)  # [B, 1]
    f, _ = self._eval_basis_1d(
      z, self.Z_u_centers.unsqueeze(0), self.rho_u.unsqueeze(0)
    )
    return f

  def basis_pdf_v(self, v: torch.Tensor) -> torch.Tensor:
    """Evaluate the v-basis kernel PDFs; returns shape [B, m_v]."""
    self._check_1d(v, "v")
    z = stdnorm_icdf(v, eps=self.eps).unsqueeze(-1)  # [B, 1]
    f, _ = self._eval_basis_1d(
      z, self.Z_v_centers.unsqueeze(0), self.rho_v.unsqueeze(0)
    )
    return f

  def basis_cdf_u(self, u: torch.Tensor) -> torch.Tensor:
    """Evaluate the u-basis kernel CDFs (primitives); returns shape [B, m_u]."""
    self._check_1d(u, "u")
    z = stdnorm_icdf(u, eps=self.eps).unsqueeze(-1)  # [B, 1]
    _, F = self._eval_basis_1d(
      z, self.Z_u_centers.unsqueeze(0), self.rho_u.unsqueeze(0)
    )
    return F

  def basis_cdf_v(self, v: torch.Tensor) -> torch.Tensor:
    """Evaluate the v-basis kernel CDFs (primitives); returns shape [B, m_v]."""
    self._check_1d(v, "v")
    z = stdnorm_icdf(v, eps=self.eps).unsqueeze(-1)  # [B, 1]
    _, F = self._eval_basis_1d(
      z, self.Z_v_centers.unsqueeze(0), self.rho_v.unsqueeze(0)
    )
    return F

  def margin_u(self, u: torch.Tensor) -> torch.Tensor:
    """
    Induced u-margin (marginal density of the first coordinate).

    Parameters
    ----------
    u : Tensor of shape [B].

    Returns
    -------
    Tensor of shape [B].

    Formula
    -------
    ::

        m_u(u) = int_0^1 c(u, v) dv
               = sum_i alpha_i k_u(u; U_i, rho_{u,i}),

    where ``alpha_i = sum_j W_{ij}`` are the row marginal weights.
    """
    self._check_1d(u, "u")
    W = self.weights
    z_u = stdnorm_icdf(u, eps=self.eps).unsqueeze(-1)  # [B, 1]
    fu, _ = self._eval_basis_1d(
      z_u, self.Z_u_centers.unsqueeze(0), self.rho_u.unsqueeze(0)
    )  # [B, m_u]
    return fu @ W.sum(dim=1)  # [B]

  def margin_v(self, v: torch.Tensor) -> torch.Tensor:
    """
    Induced v-margin (marginal density of the second coordinate).

    Parameters
    ----------
    v : Tensor of shape [B].

    Returns
    -------
    Tensor of shape [B].

    Formula
    -------
    ::

        m_v(v) = int_0^1 c(u, v) du
               = sum_j beta_j k_v(v; V_j, rho_{v,j}),

    where ``beta_j = sum_i W_{ij}`` are the column marginal weights.
    """
    self._check_1d(v, "v")
    W = self.weights
    z_v = stdnorm_icdf(v, eps=self.eps).unsqueeze(-1)  # [B, 1]
    fv, _ = self._eval_basis_1d(
      z_v, self.Z_v_centers.unsqueeze(0), self.rho_v.unsqueeze(0)
    )  # [B, m_v]
    return fv @ W.sum(dim=0)  # [B]

  def pdf(self, UV: torch.Tensor) -> torch.Tensor:
    """
    Copula density.

    Parameters
    ----------
    UV : Tensor of shape [B, 2], values in (0, 1)^2.

    Returns
    -------
    Tensor of shape [B].

    Formula
    -------
    ::

        c(u, v) = sum_{i,j} W_{ij}
                      k_u(u; U_i, rho_{u,i}) k_v(v; V_j, rho_{v,j}).
    """
    self._check_UV(UV)
    u, v = UV[:, 0], UV[:, 1]
    z_u = stdnorm_icdf(u, eps=self.eps).unsqueeze(-1)  # [B, 1]
    z_v = stdnorm_icdf(v, eps=self.eps).unsqueeze(-1)  # [B, 1]
    fu, _ = self._eval_basis_1d(
      z_u, self.Z_u_centers.unsqueeze(0), self.rho_u.unsqueeze(0)
    )  # [B, m_u]
    fv, _ = self._eval_basis_1d(
      z_v, self.Z_v_centers.unsqueeze(0), self.rho_v.unsqueeze(0)
    )  # [B, m_v]
    return torch.einsum("bi,bj,ij->b", fu, fv, self.weights)

  def cdf(self, UV: torch.Tensor) -> torch.Tensor:
    """
    Copula CDF.

    Parameters
    ----------
    UV : Tensor of shape [B, 2], values in (0, 1)^2.

    Returns
    -------
    Tensor of shape [B].

    Formula
    -------
    ::

        C(u, v) = sum_{i,j} W_{ij}
                      K_u(u; U_i, rho_{u,i}) K_v(v; V_j, rho_{v,j}).
    """
    self._check_UV(UV)
    u, v = UV[:, 0], UV[:, 1]
    z_u = stdnorm_icdf(u, eps=self.eps).unsqueeze(-1)  # [B, 1]
    z_v = stdnorm_icdf(v, eps=self.eps).unsqueeze(-1)  # [B, 1]
    _, Fu = self._eval_basis_1d(
      z_u, self.Z_u_centers.unsqueeze(0), self.rho_u.unsqueeze(0)
    )  # [B, m_u]
    _, Fv = self._eval_basis_1d(
      z_v, self.Z_v_centers.unsqueeze(0), self.rho_v.unsqueeze(0)
    )  # [B, m_v]
    return torch.einsum("bi,bj,ij->b", Fu, Fv, self.weights)

  def hfunc1(self, UV: torch.Tensor, normalized: bool = False) -> torch.Tensor:
    """
    First h-function: partial integral over u.

    Parameters
    ----------
    UV : Tensor of shape [B, 2], values in (0, 1)^2.
    normalized : bool
        If ``False`` (default), return the raw partial integral.
        If ``True``, divide by the v-margin to obtain H1(u | v).

    Returns
    -------
    Tensor of shape [B].

    Formula
    -------
    Raw form::

        H1^raw(u, v) = int_0^u c(s, v) ds
                     = sum_{i,j} W_{ij}
                           K_u(u; U_i, rho_{u,i}) k_v(v; V_j, rho_{v,j}).

    Normalized form::

        H1(u | v) = H1^raw(u, v) / m_v(v),

    where ``m_v(v) = sum_j beta_j k_v(v; V_j, rho_{v,j})``.

    Since K_u in [0, 1], the raw form satisfies
    ``H1^raw(u, v) <= m_v(v)`` for all u, v, so the normalized form
    lies in [0, 1].
    """
    self._check_UV(UV)
    u, v = UV[:, 0], UV[:, 1]
    z_u = stdnorm_icdf(u, eps=self.eps).unsqueeze(-1)  # [B, 1]
    z_v = stdnorm_icdf(v, eps=self.eps).unsqueeze(-1)  # [B, 1]
    _, Fu = self._eval_basis_1d(
      z_u, self.Z_u_centers.unsqueeze(0), self.rho_u.unsqueeze(0)
    )  # [B, m_u]
    fv, _ = self._eval_basis_1d(
      z_v, self.Z_v_centers.unsqueeze(0), self.rho_v.unsqueeze(0)
    )  # [B, m_v]
    W = self.weights
    h1_raw = torch.einsum("bi,bj,ij->b", Fu, fv, W)
    if not normalized:
      return h1_raw
    mv = fv @ W.sum(dim=0)  # reuse fv and W; avoids a redundant basis eval
    return h1_raw / mv.clamp_min(self.eps)

  def hfunc2(self, UV: torch.Tensor, normalized: bool = False) -> torch.Tensor:
    """
    Second h-function: partial integral over v.

    Parameters
    ----------
    UV : Tensor of shape [B, 2], values in (0, 1)^2.
    normalized : bool
        If ``False`` (default), return the raw partial integral.
        If ``True``, divide by the u-margin to obtain H2(v | u).

    Returns
    -------
    Tensor of shape [B].

    Formula
    -------
    Raw form::

        H2^raw(u, v) = int_0^v c(u, t) dt
                     = sum_{i,j} W_{ij}
                           k_u(u; U_i, rho_{u,i}) K_v(v; V_j, rho_{v,j}).

    Normalized form::

        H2(v | u) = H2^raw(u, v) / m_u(u),

    where ``m_u(u) = sum_i alpha_i k_u(u; U_i, rho_{u,i})``.

    Since K_v in [0, 1], the raw form satisfies
    ``H2^raw(u, v) <= m_u(u)`` for all u, v, so the normalized form
    lies in [0, 1].
    """
    self._check_UV(UV)
    u, v = UV[:, 0], UV[:, 1]
    z_u = stdnorm_icdf(u, eps=self.eps).unsqueeze(-1)  # [B, 1]
    z_v = stdnorm_icdf(v, eps=self.eps).unsqueeze(-1)  # [B, 1]
    fu, _ = self._eval_basis_1d(
      z_u, self.Z_u_centers.unsqueeze(0), self.rho_u.unsqueeze(0)
    )  # [B, m_u]
    _, Fv = self._eval_basis_1d(
      z_v, self.Z_v_centers.unsqueeze(0), self.rho_v.unsqueeze(0)
    )  # [B, m_v]
    W = self.weights
    h2_raw = torch.einsum("bi,bj,ij->b", fu, Fv, W)
    if not normalized:
      return h2_raw
    mu = fu @ W.sum(dim=1)  # reuse fu and W; avoids a redundant basis eval
    return h2_raw / mu.clamp_min(self.eps)

  def log_pdf(self, UV: torch.Tensor) -> torch.Tensor:
    return torch.log(self.pdf(UV).clamp_min(1e-16))

  def nll(self, UV: torch.Tensor) -> torch.Tensor:
    return -self.log_pdf(UV).sum()

  def marginal_penalty(self, n_grid: int = 257) -> torch.Tensor:
    """
    Soft copula marginal penalty.

    Approximates::

        int_0^1 (m_u(u) - 1)^2 du + int_0^1 (m_v(v) - 1)^2 dv

    on a uniform grid, where the induced margins are::

        m_u(u) = sum_i alpha_i k_u(u; U_i, rho_{u,i}),
        m_v(v) = sum_j beta_j  k_v(v; V_j, rho_{v,j}),

    with ``alpha_i = sum_j W_{ij}`` and ``beta_j = sum_i W_{ij}``.

    Parameters
    ----------
    n_grid : int
        Number of integration points.

    Returns
    -------
    Tensor scalar.
    """
    grid = torch.linspace(
      self.eps,
      1.0 - self.eps,
      n_grid,
      dtype=self.logits.dtype,
      device=self.logits.device,
    )
    mu = self.margin_u(grid)
    mv = self.margin_v(grid)
    return (mu - 1.0).square().mean() + (mv - 1.0).square().mean()

  def forward(self, UV: torch.Tensor) -> torch.Tensor:
    return self.pdf(UV)
