"""
bicopbase.py — Shared infrastructure for differentiable bivariate pair-copula models.

Provides:
  - Module-level numeric helpers (stdnorm_pdf, stdnorm_cdf, stdnorm_icdf, …).
  - The unified kernel evaluator _eval_basis_1d used by both GCKBicop and
    GTKBicop.
  - The abstract base class _BicopBase, which implements the full public API
    (pdf, cdf, hfunc1, hfunc2, margin_u, margin_v, penalties, training helpers)
    generically in terms of the abstract _basis_u / _basis_v methods.
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def stdnorm_pdf(z: torch.Tensor) -> torch.Tensor:
  """Standard normal PDF phi(z)."""
  return torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)


def stdnorm_cdf(z: torch.Tensor) -> torch.Tensor:
  """Standard normal CDF Phi(z)."""
  return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def stdnorm_icdf(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  """Standard normal quantile Phi^{-1}(u), clamped to avoid infinities."""
  u = u.clamp(eps, 1.0 - eps)
  return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


def atanh(x: float) -> float:
  """Scalar inverse hyperbolic tangent."""
  return 0.5 * math.log((1.0 + x) / (1.0 - x))


def _softplus_inv(y: float) -> float:
  """Scalar inverse of softplus: softplus_inv(y) = log(exp(y) - 1)."""
  return math.log(math.expm1(y))


def _eval_basis_1d(
  z: torch.Tensor,
  z_shift: torch.Tensor,
  scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """
  Shared kernel evaluation used by both GCKBicop and GTKBicop.

  Computes::

      a = (z - z_shift) / scale                             [B, m]
      f = exp(-0.5 * (a^2 - z^2) - log(scale))             [B, m]
      F = Phi(a)                                             [B, m]

  Parameters
  ----------
  z       : Tensor of shape [B, 1]  — transformed batch values.
  z_shift : Tensor of shape [1, m]  — per-center shift.
  scale   : Tensor of shape [1, m]  — per-center positive scale.

  Returns
  -------
  f : Tensor [B, m]
      Effective kernel divided by phi(z):
        GCKBicop: f = k(u; U_i, rho_i) = phi(a_i) / (s_i * phi(z)),
        GTKBicop: f = b(z; mu_i, sigma_i) / phi(z).
  F : Tensor [B, m]
      Kernel primitive:
        GCKBicop: F = K(u; U_i, rho_i) = Phi(a_i),
        GTKBicop: F = B(z; mu_i, sigma_i) = Phi(a_i).

  Parameterisation mapping
  -------------------------
  GCKBicop:  z_shift = rho_i * Phi^{-1}(U_i),  scale = sqrt(1 - rho_i^2).
  GTKBicop:  z_shift = mu_i,                    scale = sigma_i.
  """
  a = (z - z_shift) / scale  # [B, m]
  f = (-0.5 * (a.square() - z.square()) - scale.log()).exp()
  return f, stdnorm_cdf(a)


# ---------------------------------------------------------------------------
# Internal base class
# ---------------------------------------------------------------------------


class _BicopBase(nn.Module):
  """
  Shared infrastructure for GCKBicop and GTKBicop.

  Provides:
  - Softmax weight parameterisation (logit matrix of shape [m_u, m_v]).
  - Input-shape validators (_check_UV, _check_1d).
  - Full public API implemented generically via _basis_u / _basis_v:
      pdf, cdf, hfunc1, hfunc2, margin_u, margin_v,
      basis_pdf_u, basis_pdf_v, basis_cdf_u, basis_cdf_v.
  - Logit-smoothness penalty, marginal-uniformity penalty.
  - Training helpers: log_pdf, nll, forward.

  Subclasses must implement ``_basis_u`` and ``_basis_v``.

  Generic formulas (basis-agnostic)
  ----------------------------------
  Let fu[b, i] and Fu[b, i] be the effective kernel pdf and primitive for
  the u-direction at batch index b and center i (and similarly fv, Fv for
  v).  Then all public quantities reduce to::

      pdf(u, v)   = sum_{i,j} W_{ij} fu_i(u) fv_j(v)
      cdf(u, v)   = sum_{i,j} W_{ij} Fu_i(u) Fv_j(v)
      hfunc1_raw  = sum_{i,j} W_{ij} Fu_i(u) fv_j(v)
      hfunc2_raw  = sum_{i,j} W_{ij} fu_i(u) Fv_j(v)
      margin_u(u) = sum_i alpha_i fu_i(u),  alpha_i = sum_j W_{ij}
      margin_v(v) = sum_j beta_j  fv_j(v),  beta_j  = sum_i W_{ij}

  This structure is identical for both model families.
  """

  logits: nn.Parameter

  @staticmethod
  def _make_normal_grid(m: int, z_scale: bool = False) -> torch.Tensor:
    """Vinecopulib-style normal grid over (-3.25, 3.25).

    Replicates ``KernelBicop::make_normal_grid`` from vinecopulib::

        z_i = -3.25 + i * (6.5 / (m - 1)),  i = 0, ..., m-1.

    Parameters
    ----------
    m : int
        Number of grid points.
    z_scale : bool
        If True, return the raw z-values (suitable as GTKBicop centers in
        Z-space).  If False (default), return Phi(z) in (0, 1) (suitable
        as GCKBicop centers on the copula scale).

    Returns
    -------
    Tensor of shape [m].
    """
    z = torch.linspace(-3.25, 3.25, m, dtype=torch.float32)
    return z if z_scale else stdnorm_cdf(z)

  def __init__(self, m_u: int, m_v: int, eps: float) -> None:
    super().__init__()
    self.m_u = m_u
    self.m_v = m_v
    self.eps = eps
    self.logits = nn.Parameter(torch.zeros(m_u, m_v))

  # --- Validators ---

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

  # --- Weights ---

  @property
  def weights(self) -> torch.Tensor:
    """Weight matrix W of shape [m_u, m_v]; W_{ij} >= 0, sum = 1.

    ::

        W_{ij} = exp(L_{ij}) / sum_{k,l} exp(L_{kl}).
    """
    w = torch.softmax(self.logits.reshape(-1), dim=0)
    return w.reshape(self.m_u, self.m_v)

  # --- Abstract basis helpers (must be overridden) ---

  def _basis_u(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    u-basis: effective kernel pdf fu and primitive Fu, both [B, m_u].

    Parameters
    ----------
    u : Tensor of shape [B].
    """
    raise NotImplementedError

  def _basis_v(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    v-basis: effective kernel pdf fv and primitive Fv, both [B, m_v].

    Parameters
    ----------
    v : Tensor of shape [B].
    """
    raise NotImplementedError

  # --- Public basis accessors ---

  def basis_pdf_u(self, u: torch.Tensor) -> torch.Tensor:
    """u-basis kernel pdfs; shape [B, m_u]."""
    self._check_1d(u, "u")
    f, _ = self._basis_u(u)
    return f

  def basis_pdf_v(self, v: torch.Tensor) -> torch.Tensor:
    """v-basis kernel pdfs; shape [B, m_v]."""
    self._check_1d(v, "v")
    f, _ = self._basis_v(v)
    return f

  def basis_cdf_u(self, u: torch.Tensor) -> torch.Tensor:
    """u-basis kernel CDFs (primitives); shape [B, m_u]."""
    self._check_1d(u, "u")
    _, F = self._basis_u(u)
    return F

  def basis_cdf_v(self, v: torch.Tensor) -> torch.Tensor:
    """v-basis kernel CDFs (primitives); shape [B, m_v]."""
    self._check_1d(v, "v")
    _, F = self._basis_v(v)
    return F

  # --- Public API ---

  def margin_u(self, u: torch.Tensor) -> torch.Tensor:
    """
    Induced u-marginal density.

    Parameters
    ----------
    u : Tensor of shape [B].

    Returns
    -------
    Tensor of shape [B].

    Formula
    -------
    ::

        m_u(u) = sum_i alpha_i fu_i(u),  alpha_i = sum_j W_{ij}.

    See the class docstring for the model-specific form of fu_i.
    """
    self._check_1d(u, "u")
    W = self.weights
    fu, _ = self._basis_u(u)
    return fu @ W.sum(dim=1)

  def margin_v(self, v: torch.Tensor) -> torch.Tensor:
    """
    Induced v-marginal density.

    Parameters
    ----------
    v : Tensor of shape [B].

    Returns
    -------
    Tensor of shape [B].

    Formula
    -------
    ::

        m_v(v) = sum_j beta_j fv_j(v),  beta_j = sum_i W_{ij}.
    """
    self._check_1d(v, "v")
    W = self.weights
    fv, _ = self._basis_v(v)
    return fv @ W.sum(dim=0)

  def pdf(self, UV: torch.Tensor) -> torch.Tensor:
    """
    Copula density c(u, v).

    Parameters
    ----------
    UV : Tensor of shape [B, 2], values in (0, 1)^2.

    Returns
    -------
    Tensor of shape [B].

    Formula
    -------
    ::

        c(u, v) = sum_{i,j} W_{ij} fu_i(u) fv_j(v).
    """
    self._check_UV(UV)
    fu, _ = self._basis_u(UV[:, 0])
    fv, _ = self._basis_v(UV[:, 1])
    return torch.einsum("bi,bj,ij->b", fu, fv, self.weights)

  def cdf(self, UV: torch.Tensor) -> torch.Tensor:
    """
    Copula CDF C(u, v).

    Parameters
    ----------
    UV : Tensor of shape [B, 2], values in (0, 1)^2.

    Returns
    -------
    Tensor of shape [B].

    Formula
    -------
    ::

        C(u, v) = sum_{i,j} W_{ij} Fu_i(u) Fv_j(v).
    """
    self._check_UV(UV)
    _, Fu = self._basis_u(UV[:, 0])
    _, Fv = self._basis_v(UV[:, 1])
    return torch.einsum("bi,bj,ij->b", Fu, Fv, self.weights)

  def hfunc1(self, UV: torch.Tensor, normalized: bool = False) -> torch.Tensor:
    """
    First h-function: partial integral int_0^u c(s, v) ds.

    Parameters
    ----------
    UV : Tensor of shape [B, 2].
    normalized : bool
        If False (default), return the raw partial integral.
        If True, divide by m_v(v) to obtain H1(u | v).

    Returns
    -------
    Tensor of shape [B].

    Formula
    -------
    Raw form::

        H1^raw(u, v) = sum_{i,j} W_{ij} Fu_i(u) fv_j(v).

    Normalized form::

        H1(u | v) = H1^raw(u, v) / m_v(v),

    where m_v(v) = sum_j beta_j fv_j(v).  Since Fu_i in [0, 1],
    H1^raw <= m_v(v), so the normalized form lies in [0, 1].
    """
    self._check_UV(UV)
    _, Fu = self._basis_u(UV[:, 0])
    fv, _ = self._basis_v(UV[:, 1])
    W = self.weights
    h1_raw = torch.einsum("bi,bj,ij->b", Fu, fv, W)
    if not normalized:
      return h1_raw
    mv = fv @ W.sum(dim=0)  # reuse fv; avoids redundant basis eval
    return h1_raw / mv.clamp_min(self.eps)

  def hfunc2(self, UV: torch.Tensor, normalized: bool = False) -> torch.Tensor:
    """
    Second h-function: partial integral int_0^v c(u, t) dt.

    Parameters
    ----------
    UV : Tensor of shape [B, 2].
    normalized : bool
        If False (default), return the raw partial integral.
        If True, divide by m_u(u) to obtain H2(v | u).

    Returns
    -------
    Tensor of shape [B].

    Formula
    -------
    Raw form::

        H2^raw(u, v) = sum_{i,j} W_{ij} fu_i(u) Fv_j(v).

    Normalized form::

        H2(v | u) = H2^raw(u, v) / m_u(u),

    where m_u(u) = sum_i alpha_i fu_i(u).  Since Fv_j in [0, 1],
    H2^raw <= m_u(u), so the normalized form lies in [0, 1].
    """
    self._check_UV(UV)
    fu, _ = self._basis_u(UV[:, 0])
    _, Fv = self._basis_v(UV[:, 1])
    W = self.weights
    h2_raw = torch.einsum("bi,bj,ij->b", fu, Fv, W)
    if not normalized:
      return h2_raw
    mu = fu @ W.sum(dim=1)  # reuse fu; avoids redundant basis eval
    return h2_raw / mu.clamp_min(self.eps)

  # --- Penalties ---

  def logit_smoothness_penalty(self) -> torch.Tensor:
    """
    First-order smoothness penalty on the pre-softmax logit matrix.

    Penalises large differences between neighbouring logit values on the
    2-D grid, encouraging a smooth weight surface.

    Formula
    -------
    ::

        P_logits = mean_{i,j} (L_{i+1,j} - L_{i,j})^2   (row diffs)
                 + mean_{i,j} (L_{i,j+1} - L_{i,j})^2   (col diffs)

    Returns
    -------
    Tensor scalar.
    """
    L = self.logits
    row_diff = (L[1:, :] - L[:-1, :]).square().mean()
    col_diff = (L[:, 1:] - L[:, :-1]).square().mean()
    return row_diff + col_diff

  def marginal_penalty(self, n_grid: int = 257) -> torch.Tensor:
    """
    Soft copula marginal uniformity penalty.

    Approximates::

        int_0^1 (m_u(u) - 1)^2 du + int_0^1 (m_v(v) - 1)^2 dv

    on a uniform grid, where the induced margins are::

        m_u(u) = sum_i alpha_i fu_i(u),  alpha_i = sum_j W_{ij},
        m_v(v) = sum_j beta_j  fv_j(v),  beta_j  = sum_i W_{ij}.

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

  # --- Training helpers ---

  def log_pdf(self, UV: torch.Tensor) -> torch.Tensor:
    """Log copula density, clamped to avoid -inf."""
    return torch.log(self.pdf(UV).clamp_min(1e-16))

  def nll(self, UV: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood."""
    return -self.log_pdf(UV).mean()

  def forward(self, UV: torch.Tensor) -> torch.Tensor:
    """Alias for pdf(UV)."""
    return self.pdf(UV)
