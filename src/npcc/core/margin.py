"""Backend-backed conditional margins for vine distributions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self

import numpy as np
import torch
from array_api_compat import is_torch_array
from pyvinecopulib.core import MarginBase

from npcc.core._common import TensorLike, _resolve_device
from npcc.core.conditional_distribution1d import ConditionalDistribution1D
from npcc.core.quantile_table_distribution1d import QuantileGridConfig
from npcc.core.registry import create_backend


class BackendMargin(MarginBase[TensorLike]):
  """Continuous real-valued margin modeled by a registered NPCC backend.

  The backend model ``Y | X`` when covariates are supplied. Without
  covariates, a constant zero-valued feature is used so backends that require
  at least one feature can still estimate an unconditional margin.
  """

  supports_covariates: bool = True
  supports_weights: bool = False
  supported_var_types: tuple[str, ...] = ("c",)

  def __init__(
    self,
    *,
    backend: str = "tabpfn-criterion",
    quantile_config: QuantileGridConfig | None = None,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    backend_kwargs: Mapping[str, object] | None = None,
  ) -> None:
    self.backend = backend
    self.quantile_config = quantile_config or QuantileGridConfig()
    self._device = _resolve_device(device)
    self.batch_size = batch_size
    self.backend_kwargs = dict(backend_kwargs or {})
    self._is_fitted = False

    self._distribution: ConditionalDistribution1D = create_backend(
      backend,
      transform="identity",
      config=self.quantile_config,
      device=self._device,
      batch_size=self.batch_size,
      backend_kwargs=self.backend_kwargs,
    )

  @property
  def is_fitted(self) -> bool:
    """Whether the underlying backend has been fitted."""
    return self._is_fitted

  @property
  def family_name(self) -> str:
    """Backend identifier used in margin summaries."""
    return self.backend

  def _check_fitted(self) -> None:
    if not self._is_fitted:
      raise RuntimeError("The margin is not fitted.")

  @staticmethod
  def _validate_vector(values: TensorLike, *, name: str) -> int:
    if values.ndim != 1:
      raise ValueError(f"{name} must have shape (n,).")
    return int(values.shape[0])

  @classmethod
  def _validate_probabilities(cls, p: TensorLike) -> None:
    cls._validate_vector(p, name="p")

    if isinstance(p, torch.Tensor):
      invalid = ~torch.isfinite(p) | (p < 0.0) | (p > 1.0)
      has_invalid = bool(torch.any(invalid))
    else:
      invalid = ~np.isfinite(p) | (p < 0.0) | (p > 1.0)
      has_invalid = bool(np.any(invalid))

    if has_invalid:
      raise ValueError("p must contain finite values in [0, 1].")

  def _conditioning(
    self,
    values: TensorLike,
    x: TensorLike | None,
  ) -> TensorLike:
    n = self._validate_vector(values, name="values")

    if x is None:
      if isinstance(values, torch.Tensor):
        return torch.zeros(
          (n, 1),
          dtype=torch.float64,
          device=values.device,
        )
      return np.zeros((n, 1), dtype=np.float64)

    if is_torch_array(values) != is_torch_array(x):
      raise TypeError("values and x must use the same array namespace.")

    if x.ndim == 1:
      x = x.reshape(-1, 1)
    elif x.ndim != 2:
      raise ValueError("x must have shape (n,) or (n, p).")

    if x.shape[0] != n:
      raise ValueError("values and x must have the same number of rows.")

    return x

  def fit(
    self,
    y: TensorLike,
    /,
    *,
    x: TensorLike | None = None,
    weights: TensorLike | None = None,
  ) -> Self:
    """Fit the backend to a continuous response and optional covariates."""
    if weights is not None:
      raise TypeError("BackendMargin does not support observation weights.")

    features = self._conditioning(y, x)
    self._is_fitted = False
    self._distribution.fit(features, y)
    self._is_fitted = True
    return self

  def pdf(
    self,
    y: TensorLike,
    /,
    *,
    x: TensorLike | None = None,
  ) -> TensorLike:
    """Evaluate the conditional marginal density."""
    self._check_fitted()
    features = self._conditioning(y, x)
    return self._distribution.pdf(features, y)

  def cdf(
    self,
    y: TensorLike,
    /,
    *,
    x: TensorLike | None = None,
  ) -> TensorLike:
    """Evaluate the conditional marginal distribution function."""
    self._check_fitted()
    features = self._conditioning(y, x)
    return self._distribution.cdf(features, y)

  def icdf(
    self,
    p: TensorLike,
    /,
    *,
    x: TensorLike | None = None,
  ) -> TensorLike:
    """Evaluate conditional quantiles using the backend's native inverse."""
    self._check_fitted()
    self._validate_probabilities(p)
    features = self._conditioning(p, x)
    return self._distribution.icdf(features, p)

  def sample(
    self, n: int, *, x: TensorLike | None = None, seeds: list[int] | None = None
  ) -> TensorLike:
    """Draw samples, preserving the covariate namespace when supplied."""
    base_t = self._sample_uniform(n, list(seeds or []))

    if x is not None and not is_torch_array(x):
      base: TensorLike = base_t.detach().cpu().numpy()
    else:
      base = base_t

    return self.icdf(base, x=x)

  def _sample_uniform(self, n: int, seeds: list[int]) -> torch.Tensor:
    """Draw uniforms on the backend device for inherited sampling."""
    if n < 0:
      raise ValueError("n must be non-negative.")

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
