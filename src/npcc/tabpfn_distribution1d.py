"""
tabpfn_distribution1d.py — abstract base for univariate conditional
predictive distributions backed by TabPFN.

Concrete implementations live in
:mod:`npcc.tabpfn_criterion_distribution1d` (uses TabPFN's binned
distribution head directly) and
:mod:`npcc.tabpfn_quantile_distribution1d` (numerical inversion of the
predicted quantile table).  Both share:

- the optional logit support transform (and its Jacobian / inverse),
- the ``transform`` / ``eps`` / ``model_kwargs`` / ``model_`` fields
  set up at construction time,
- the public ``fit`` / ``pdf`` / ``cdf`` / ``icdf`` interface.

The base class is :class:`abc.ABC` so the abstract methods are enforced
at instantiation time; the per-class fast paths (``pdf_grid`` /
``cdf_grid`` on the criterion subclass) stay subclass-specific.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike
from tabpfn import TabPFNRegressor

from npcc._common import _logit


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
  model_kwargs
      Forwarded to
      :py:meth:`TabPFNRegressor.create_default_for_version` by
      subclasses.
  """

  transform: Literal["identity", "logit"]
  eps: float
  model_kwargs: dict[str, Any]
  model_: TabPFNRegressor | None

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit"] = "logit",
    eps: float = 1e-6,
    model_kwargs: dict[str, Any] | None = None,
  ) -> None:
    self.transform = transform
    self.eps = eps
    self.model_kwargs = model_kwargs or {}
    self.model_ = None

  # ------------------------------------------------------------------
  # Shared concrete helpers (logit support transform machinery).
  # ------------------------------------------------------------------

  def _transform_y(self, y: np.ndarray) -> np.ndarray:
    if self.transform == "identity":
      return y
    if self.transform == "logit":
      y_clip = np.clip(y, self.eps, 1.0 - self.eps)
      return _logit(y_clip)
    raise ValueError(f"Unknown transform: {self.transform}")

  def _jacobian_inverse(self, y: np.ndarray) -> np.ndarray:
    if self.transform == "identity":
      return np.ones_like(y)
    if self.transform == "logit":
      y_clip = np.clip(y, self.eps, 1.0 - self.eps)
      return 1.0 / (y_clip * (1.0 - y_clip))
    raise ValueError(f"Unknown transform: {self.transform}")

  def _inverse_transform(self, z: np.ndarray) -> np.ndarray:
    """Map z-space samples back to the y-scale (inverse of ``_transform_y``)."""
    if self.transform == "identity":
      return z
    if self.transform == "logit":
      return 1.0 / (1.0 + np.exp(-z))  # sigmoid
    raise ValueError(f"Unknown transform: {self.transform}")

  # ------------------------------------------------------------------
  # Abstract public API.
  # ------------------------------------------------------------------

  @abstractmethod
  def fit(self, w: ArrayLike, y: ArrayLike) -> Self:
    """Fit the underlying regressor on ``(w, transform(y))``."""

  @abstractmethod
  def pdf(self, w: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Conditional density ``f(y_i | w_i)`` per row."""

  @abstractmethod
  def cdf(self, w: ArrayLike, y: ArrayLike) -> np.ndarray:
    """Conditional CDF ``F(y_i | w_i) = P(Y <= y_i | W = w_i)`` per row."""

  @abstractmethod
  def icdf(self, w: ArrayLike, alphas: ArrayLike) -> np.ndarray:
    """Conditional quantile ``F^{-1}(alphas_i | w_i)`` per row, on the y-scale."""
