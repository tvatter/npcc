"""
conditional_distribution1d.py — backend-neutral abstract base for
univariate conditional predictive distributions.

A concrete instance, once :py:meth:`fit` has been called, represents the
conditional distribution of ``Y`` given ``W`` learned by *some*
distributional regressor.  The base class is deliberately free of any
model dependency: it owns only

- the optional support transforms (and their Jacobians / inverses),
- the ``transform`` / ``eps`` / ``device`` / ``batch_size`` fields set up
  at construction time,
- the ``fit`` skeleton, which transforms ``Y -> Z = T(Y)`` and delegates
  the actual model training to the abstract :py:meth:`_fit_model`,
- the public ``pdf`` / ``cdf`` / ``icdf`` / ``pdf_grid`` / ``cdf_grid``
  interface (all abstract),

and accepts ``np.ndarray`` or ``torch.Tensor`` inputs, returning the same
type the caller passed in.

Concrete backends live in :mod:`npcc.core.backends`.  The two TabPFN
read-outs (``TabPFNCriterionBackend`` — native binned head, the default;
``TabPFNQuantileBackend`` — numerical inversion of the quantile table)
implement this interface, as do the optional non-TabPFN backends.

The grid methods (:py:meth:`pdf_grid` / :py:meth:`cdf_grid`) are
**abstract on purpose**.  There is no slow tiling fallback: every backend
must evaluate a Cartesian ``w x y_grid`` by predicting **at most once per
conditioning row** and reusing that prediction across all ``y`` values.
This "predict-once-per-row" contract is the core speed invariant of the
whole library (see :mod:`npcc.core.quantile_table_distribution1d`, which
satisfies it for every quantile-based backend, and the native backends,
which override the grids directly).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Literal, Self

import torch

from npcc.core._common import (
  TensorLike,
  _as_2d,
  _logit,
  _resolve_device,
  _to_tensor,
)


class ConditionalDistribution1D(ABC):
  """Abstract base class for univariate conditional predictive distributions.

  Parameters
  ----------
  transform
      ``"identity"`` fits directly on ``Y``.  ``"logit"`` fits on
      ``Z = logit(Y)`` and applies the inverse Jacobian when evaluating
      densities; this is the only sensible choice when ``Y`` is bounded
      in ``(0, 1)``, which is always the case for copula scores.
      ``"probit"`` uses ``Z = Phi^{-1}(Y)`` with the standard-normal
      Jacobian.
  eps
      Clip distance from the boundary of ``(0, 1)`` used by the support
      transforms.
  device
      Device for internal tensors and inference.  ``None`` auto-selects
      ``cuda`` if available, else ``cpu``.
  batch_size
      Default chunk size for batched inference.  ``None`` (default) uses
      400 on CPU and 2000 on CUDA.  Backends that read the predictive
      distribution in one forward pass may ignore it.
  """

  transform: Literal["identity", "logit", "probit"]
  eps: float
  batch_size: int
  _device: torch.device
  _fitted: bool

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
  ) -> None:
    self.transform = transform
    self.eps = eps
    self._device = _resolve_device(device)
    if batch_size is None:
      self.batch_size = 2000 if self._device.type == "cuda" else 400
    else:
      if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
      self.batch_size = batch_size
    self._fitted = False

  def _resolve_batch_size(self, batch_size: int | None) -> int:
    """Per-call override of :attr:`batch_size`, validated positive."""
    effective = self.batch_size if batch_size is None else batch_size
    if effective <= 0:
      raise ValueError("batch_size must be positive.")
    return effective

  def _check_fitted(self) -> None:
    """Raise if :py:meth:`fit` has not been called yet."""
    if not self._fitted:
      raise RuntimeError("The model is not fitted.")

  # ------------------------------------------------------------------
  # Shared concrete helpers (support-transform machinery).
  # ------------------------------------------------------------------

  def _transform_y(self, y: torch.Tensor) -> torch.Tensor:
    if self.transform == "identity":
      return y
    if self.transform == "logit":
      y_clip = torch.clamp(y, self.eps, 1.0 - self.eps)
      return _logit(y_clip)
    if self.transform == "probit":
      y_clip = torch.clamp(y, self.eps, 1.0 - self.eps)
      return math.sqrt(2.0) * torch.erfinv(2.0 * y_clip - 1.0)
    raise ValueError(f"Unknown transform: {self.transform}")

  def _jacobian_inverse(self, y: torch.Tensor) -> torch.Tensor:
    if self.transform == "identity":
      return torch.ones_like(y)
    if self.transform == "logit":
      y_clip = torch.clamp(y, self.eps, 1.0 - self.eps)
      return 1.0 / (y_clip * (1.0 - y_clip))
    if self.transform == "probit":
      y_clip = torch.clamp(y, self.eps, 1.0 - self.eps)
      z = math.sqrt(2.0) * torch.erfinv(2.0 * y_clip - 1.0)
      phi_z = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
      return 1.0 / phi_z
    raise ValueError(f"Unknown transform: {self.transform}")

  def _inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
    """Map z-space samples back to the y-scale (inverse of ``_transform_y``)."""
    if self.transform == "identity":
      return z
    if self.transform == "logit":
      return torch.sigmoid(z)
    if self.transform == "probit":
      return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    raise ValueError(f"Unknown transform: {self.transform}")

  # ------------------------------------------------------------------
  # Shared fit skeleton (Y -> Z = T(Y), then delegate to the backend).
  # ------------------------------------------------------------------

  def fit(self, w: TensorLike, y: TensorLike) -> Self:
    """Fit the backend regressor on ``(w, Z = transform(y))``.

    The fit-time tensors live on CPU: several backends (e.g. TabPFN) do
    their own device placement internally and reject non-CPU tensors at
    the input boundary.  Subclasses receive already-transformed,
    CPU-resident, float64 ``(w, z)`` via :py:meth:`_fit_model`.
    """
    cpu = torch.device("cpu")
    w_t = _as_2d(w, device=cpu)
    y_t = _to_tensor(y, device=cpu).reshape(-1)
    if y_t.shape[0] != w_t.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_t)
    self._fit_model(w_t, z)
    self._fitted = True
    return self

  @abstractmethod
  def _fit_model(self, w: torch.Tensor, z: torch.Tensor) -> None:
    """Train the backend regressor on ``(w, z)`` (CPU float64 tensors)."""

  # ------------------------------------------------------------------
  # Abstract public API.
  # ------------------------------------------------------------------

  @abstractmethod
  def pdf(
    self, w: TensorLike, y: TensorLike, *, batch_size: int | None = None
  ) -> TensorLike:
    """Conditional density ``f(y_i | w_i)`` per row."""

  @abstractmethod
  def cdf(
    self, w: TensorLike, y: TensorLike, *, batch_size: int | None = None
  ) -> TensorLike:
    """Conditional CDF ``F(y_i | w_i) = P(Y <= y_i | W = w_i)`` per row."""

  @abstractmethod
  def icdf(
    self, w: TensorLike, alphas: TensorLike, *, batch_size: int | None = None
  ) -> TensorLike:
    """Conditional quantile ``F^{-1}(alphas_i | w_i)`` per row, on the y-scale."""

  @abstractmethod
  def pdf_grid(
    self, w: TensorLike, y_grid: TensorLike, *, batch_size: int | None = None
  ) -> TensorLike:
    """Density on the Cartesian product of ``w`` rows and ``y_grid``.

    Returns shape ``(n_w, n_y)`` with ``out[i, j] = f(y_grid[j] | w[i])``.
    Implementations MUST predict at most once per ``w`` row (never tile
    over ``y_grid``) — this is the library's core speed invariant.
    """

  @abstractmethod
  def cdf_grid(
    self, w: TensorLike, y_grid: TensorLike, *, batch_size: int | None = None
  ) -> TensorLike:
    """CDF on the Cartesian product of ``w`` rows and ``y_grid``.

    Returns shape ``(n_w, n_y)`` with ``out[i, j] = F(y_grid[j] | w[i])``.
    Same predict-once-per-row contract as :py:meth:`pdf_grid`.
    """
