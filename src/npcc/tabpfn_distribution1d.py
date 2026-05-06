"""
tabpfn_distribution1d.py — abstract base for univariate conditional
predictive distributions backed by TabPFN.

Concrete implementations live in
:mod:`npcc.tabpfn_criterion_distribution1d` (uses TabPFN's binned
distribution head directly) and
:mod:`npcc.tabpfn_quantile_distribution1d` (numerical inversion of the
predicted quantile table).  Both share:

- the optional logit support transform (and its Jacobian / inverse),
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
from typing import Any, Literal, Self

import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from npcc._common import (
  TensorLike,
  _as_2d,
  _logit,
  _resolve_device,
  _to_tensor,
)


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
      copula scores.
  eps
      Clip distance from the boundary of ``(0, 1)`` used by the logit
      transform.
  device
      Device for internal tensors and TabPFN inference.  ``None``
      auto-selects ``cuda`` if available, else ``cpu``.  Forwarded into
      ``model_kwargs["device"]`` (without overriding an explicit user
      setting).
  model_kwargs
      Forwarded to
      :py:meth:`TabPFNRegressor.create_default_for_version` by
      subclasses.
  """

  transform: Literal["identity", "logit"]
  eps: float
  model_kwargs: dict[str, Any]
  model_: TabPFNRegressor | None
  _device: torch.device

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit"] = "logit",
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    model_kwargs: dict[str, Any] | None = None,
  ) -> None:
    self.transform = transform
    self.eps = eps
    self._device = _resolve_device(device)
    self.model_kwargs = dict(model_kwargs or {})
    self.model_kwargs.setdefault("device", str(self._device))
    self.model_ = None

  # ------------------------------------------------------------------
  # Shared concrete helpers (logit support transform machinery).
  # ------------------------------------------------------------------

  def _transform_y(self, y: torch.Tensor) -> torch.Tensor:
    if self.transform == "identity":
      return y
    if self.transform == "logit":
      y_clip = torch.clamp(y, self.eps, 1.0 - self.eps)
      return _logit(y_clip)
    raise ValueError(f"Unknown transform: {self.transform}")

  def _jacobian_inverse(self, y: torch.Tensor) -> torch.Tensor:
    if self.transform == "identity":
      return torch.ones_like(y)
    if self.transform == "logit":
      y_clip = torch.clamp(y, self.eps, 1.0 - self.eps)
      return 1.0 / (y_clip * (1.0 - y_clip))
    raise ValueError(f"Unknown transform: {self.transform}")

  def _inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
    """Map z-space samples back to the y-scale (inverse of ``_transform_y``)."""
    if self.transform == "identity":
      return z
    if self.transform == "logit":
      return torch.sigmoid(z)
    raise ValueError(f"Unknown transform: {self.transform}")

  # ------------------------------------------------------------------
  # Shared fit (TabPFN-v2.5 regressor on (w, transform(y))).
  # ------------------------------------------------------------------

  def fit(self, w: TensorLike, y: TensorLike) -> Self:
    """Fit a TabPFN-v2.5 regressor on ``(w, transform(y))``.

    Locked to TabPFN-v2.5 because v2.6 has reported regressions on
    tabular regression tasks.  Override via ``model_kwargs`` if needed.

    The fit-time tensors live on CPU: TabPFN does its own GPU
    placement internally and would reject CUDA tensors at the input
    boundary.
    """
    cpu = torch.device("cpu")
    w_t = _as_2d(w, device=cpu)
    y_t = _to_tensor(y, device=cpu).reshape(-1)
    if y_t.shape[0] != w_t.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_t)

    self.model_ = TabPFNRegressor.create_default_for_version(
      ModelVersion.V2_5, **self.model_kwargs
    )
    self.model_.fit(w_t, z)
    return self

  # ------------------------------------------------------------------
  # Abstract public API.
  # ------------------------------------------------------------------

  @abstractmethod
  def pdf(self, w: TensorLike, y: TensorLike) -> TensorLike:
    """Conditional density ``f(y_i | w_i)`` per row."""

  @abstractmethod
  def cdf(self, w: TensorLike, y: TensorLike) -> TensorLike:
    """Conditional CDF ``F(y_i | w_i) = P(Y <= y_i | W = w_i)`` per row."""

  @abstractmethod
  def icdf(self, w: TensorLike, alphas: TensorLike) -> TensorLike:
    """Conditional quantile ``F^{-1}(alphas_i | w_i)`` per row, on the y-scale."""
