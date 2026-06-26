"""
tabpfn_distribution1d.py — abstract base for univariate conditional
predictive distributions backed by TabPFN.

Concrete implementations live in
:mod:`npcc.tabpfn_criterion_distribution1d` (uses TabPFN's binned
distribution head directly) and
:mod:`npcc.tabpfn_quantile_distribution1d` (numerical inversion of the
predicted quantile table).  Both share:

- the optional support transforms (and their Jacobians / inverses),
- the ``transform`` / ``eps`` / ``device`` / ``model_kwargs`` / ``model_``
  fields set up at construction time,
- the public ``fit`` / ``pdf`` / ``cdf`` / ``icdf`` interface,
- accept ``np.ndarray`` or ``torch.Tensor`` inputs and return the same
  type the caller passed in.

The base class is :class:`abc.ABC` so the abstract methods are enforced
at instantiation time; the per-class fast paths (``pdf_grid`` /
``cdf_grid`` on the criterion subclass) stay subclass-specific.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any, Literal, Self

import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from npcc.core._common import (
  TensorLike,
  _as_2d,
  _logit,
  _resolve_device,
  _to_tensor,
)

_DEFAULT_MODEL_VERSION = ModelVersion.V3
"""Default TabPFN model version shared by every npcc distribution/copula class.

Centralized here so a version bump is a one-line change rather than four
scattered defaults (the two inner distributions and
:class:`~npcc.pfnr_bicop.PFNRBicop` all reference this single value).
"""


class TabPFNDistribution1D(ABC):
  """Abstract base class for univariate conditional predictive distributions.

  A concrete instance, once :py:meth:`fit` has been called, represents
  the conditional distribution of ``Y`` given ``W`` learned by a
  TabPFN regressor.  Subclasses differ only in *how* that distribution
  is read off the regressor's output.

  Parameters
  ----------
  transform
      ``"identity"`` fits TabPFN directly on ``Y``.  ``"logit"`` fits
      on ``Z = logit(Y)`` and applies the inverse Jacobian when
      evaluating densities; this is the only sensible choice when
      ``Y`` is bounded in ``(0, 1)``, which is always the case for
      copula scores. ``"probit"`` uses ``Z = Phi^{-1}(Y)`` with the
      standard-normal Jacobian.
  eps
      Clip distance from the boundary of ``(0, 1)`` used by support
      transforms.
  device
      Device for internal tensors and TabPFN inference.  ``None``
      auto-selects ``cuda`` if available, else ``cpu``.  Forwarded into
      ``model_kwargs["device"]`` (without overriding an explicit user
      setting).
  batch_size
      Default chunk size for batched inference.  ``None`` (default) uses
      400 on CPU and 2000 on CUDA.  Subclasses that read the predictive
      distribution in one forward pass (e.g. the quantile method) ignore
      it; it is honoured by the criterion method's chunked ``pdf`` /
      ``cdf``.
  model_kwargs
      Forwarded to
      :py:meth:`TabPFNRegressor.create_default_for_version` by
      subclasses.
  """

  transform: Literal["identity", "logit", "probit"]
  eps: float
  batch_size: int
  model_kwargs: dict[str, Any]
  model_version: ModelVersion | None
  model_: TabPFNRegressor | None
  _device: torch.device

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
    model_version: ModelVersion | None = _DEFAULT_MODEL_VERSION,
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
    self.model_kwargs = dict(model_kwargs or {})
    self.model_kwargs.setdefault("device", str(self._device))
    self.model_version = model_version
    self.model_ = None

  def _resolve_batch_size(self, batch_size: int | None) -> int:
    """Per-call override of :attr:`batch_size`, validated positive."""
    effective = self.batch_size if batch_size is None else batch_size
    if effective <= 0:
      raise ValueError("batch_size must be positive.")
    return effective

  def _make_model(self) -> TabPFNRegressor:
    if self.model_version is None:
      return TabPFNRegressor(**self.model_kwargs)

    return TabPFNRegressor.create_default_for_version(
      self.model_version,
      **self.model_kwargs,
    )

  # ------------------------------------------------------------------
  # Shared concrete helpers (support transform machinery).
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
  # Shared fit (TabPFN-v3 regressor on (w, transform(y))).
  # ------------------------------------------------------------------

  def fit(self, w: TensorLike, y: TensorLike) -> Self:
    """Fit a TabPFN regressor on ``(w, transform(y))``.

    Defaults to TabPFN-v3 (configurable via ``model_version`` /
    ``model_kwargs``).  The fit-time tensors live on CPU: TabPFN does its
    own GPU placement internally and would reject CUDA tensors at the
    input boundary.
    """
    cpu = torch.device("cpu")
    w_t = _as_2d(w, device=cpu)
    y_t = _to_tensor(y, device=cpu).reshape(-1)
    if y_t.shape[0] != w_t.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_t)

    self.model_ = self._make_model()
    self.model_.fit(w_t, z)
    return self

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
  def icdf(self, w: TensorLike, alphas: TensorLike) -> TensorLike:
    """Conditional quantile ``F^{-1}(alphas_i | w_i)`` per row, on the y-scale."""
