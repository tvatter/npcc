"""
Backend-neutral conditional margins.

A concrete conditional margin represents the distribution of a univariate
response ``Y`` given an optional feature matrix ``X``. It implements
pyvinecopulib's :class:`MarginBase` contract and adds efficient Cartesian-grid
evaluation methods used by NPCC pair copulas.

Concrete backends implement:

- ``_fit_model`` to train on transformed targets;
- ``pdf`` and ``cdf`` as required by ``MarginBase``;
- ``icdf`` when a native or table-based inverse is available;
- ``pdf_grid`` and ``cdf_grid`` using at most one prediction per conditioning
  row.

All public numerical inputs and outputs are torch tensors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Literal, Self

import torch
from pyvinecopulib.core import MarginBase

from npcc.core._common import _logit, _resolve_device


class ConditionalDistribution1D(MarginBase[torch.Tensor], ABC):
  """Abstract backend-powered conditional continuous margin.

  Parameters
  ----------
  transform
    Transformation applied to the response before fitting. ``"identity"``
    is appropriate for original-scale margins. ``"logit"`` and ``"probit"``
    map values from ``(0, 1)`` to the real line and are appropriate for
    copula-scale data.
  eps
    Distance used when clipping values away from the boundaries of ``(0, 1)``
    before applying the logit or probit transform.
  device
    Device used for fitting, inference and sampling. If omitted, CUDA is
    selected when available and CPU otherwise.
  batch_size
      Default inference chunk size. If omitted, 400 is used on CPU and 2000 on
      CUDA.
  """

  supports_covariates: bool = True
  supports_weights: bool = False
  supported_var_types: tuple[str, ...] = ("c",)

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

  @property
  def is_fitted(self) -> bool:
    """Whether this margin has been fitted."""
    return self._fitted

  @property
  def family_name(self) -> str:
    """Name shown in margin summaries."""
    return type(self).__name__

  def _resolve_batch_size(self, batch_size: int | None) -> int:
    """Resolve and validate a method-level batch-size override."""
    effective = self.batch_size if batch_size is None else batch_size

    if effective <= 0:
      raise ValueError("batch_size must be positive.")

    return effective

  def _check_fitted(self) -> None:
    """Raise if the margin has not been fitted."""
    if not self._fitted:
      raise RuntimeError("The model is not fitted.")

  def _conditioning(
    self,
    values: torch.Tensor,
    x: torch.Tensor | None,
  ) -> torch.Tensor:
    """Prepare conditioning features for fitting or evaluation.

    When no covariates are supplied, a constant feature is created because
    distributional-regression backends generally require at least one feature.

    Parameters
    ----------
    values
      Response values or probabilities. Only the leading dimension is needed
      to construct unconditional features.
    x
      Optional conditioning features.

    Returns
    -------
    torch.Tensor
      Two-dimensional float64 features on ``self._device``.
    """
    n = values.shape[0]

    if x is None:
      return torch.zeros(
        (n, 1),
        dtype=torch.float64,
        device=self._device,
      )

    if x.ndim == 1:
      x = x.reshape(-1, 1)

    return x.to(
      device=self._device,
      dtype=torch.float64,
    )

  def _transform_y(self, y: torch.Tensor) -> torch.Tensor:
    """Transform responses to the backend's modeled scale."""
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
    """Return the inverse-transform Jacobian evaluation on the response scale."""
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
    """Map transformed values back to the response scale."""
    if self.transform == "identity":
      return z

    if self.transform == "logit":
      return torch.sigmoid(z)

    if self.transform == "probit":
      return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

    raise ValueError(f"Unknown transform: {self.transform}")

  def fit(
    self,
    y: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
  ) -> Self:
    """Fit the backend to responses and optional conditioning features.

    Inputs are normalized to float64 tensors on ``self._device``. A backend
    requiring CPU or NumPy data must perform that conversion privately inside
    :meth:`_fit_model`.
    """
    y_t = y.to(
      device=self._device,
      dtype=torch.float64,
    ).reshape(-1)

    x_t = self._conditioning(y_t, x=x)
    z_t = self._transform_y(y_t)

    self._fitted = False
    self._fit_model(x_t, z_t)
    self._fitted = True

    return self

  @abstractmethod
  def _fit_model(self, w: torch.Tensor, z: torch.Tensor) -> None:
    """Train the backend using tensors on the configured device.

    A third-party backend that requires host-side inputs is responsible for
    detaching and moving these tensors to CPU within its own adapter.
    """

  @abstractmethod
  def pdf(
    self,
    y: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate the conditional density ``f(y_i | x_i)``."""

  @abstractmethod
  def cdf(
    self,
    y: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate the conditional CDF ``F(y_i | x_i)``."""

  @abstractmethod
  def icdf(
    self,
    p: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate the conditional quantile ``F^-1(y_i | x_i)``."""

  @abstractmethod
  def pdf_grid(
    self,
    y_grid: torch.Tensor,
    /,
    *,
    x: torch.Tensor,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate the conditional density on the Cartesian product of ``x`` and ``y_grid``.

    Returns
    -------
    torch.Tensor
      Matrix with shape ``(n_x, n_y)`` where element ``(i, j)`` is
      ``f(y_grid[j] | x[i])``.

    Notes
    -----
    Implementations must predict at most once per conditioning row. They must
    not tile each row of ``x`` across the response grid before prediction.
    """

  @abstractmethod
  def cdf_grid(
    self,
    y_grid: torch.Tensor,
    /,
    *,
    x: torch.Tensor,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    """Evaluate the conditional CDF on the Cartesian product of ``x`` and ``y_grid``.

    Returns
    -------
    torch.Tensor
      Matrix with shape ``(n_x, n_y)`` where element ``(i, j)`` is
      ``F(y_grid[j] | x[i])``.

    Notes
    -----
    Implementations follow the same predict-once-per-conditioning-row
    requirement as :meth:`pdf_grid`.
    """

  def _sample_uniform(
    self,
    n: int,
    seeds: list[int],
  ) -> torch.Tensor:
    """Draw base uniforms on the configured device."""
    generator = torch.Generator(device=self._device)

    if seeds:
      generator.manual_seed(int(seeds[0]))
    else:
      generator.seed()

    return torch.rand(
      n,
      generator=generator,
      dtype=torch.float64,
      device=self._device,
    )
