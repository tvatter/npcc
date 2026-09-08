"""Fit controls for Rosenblatt pair copulas and vines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import torch

from npcc.core.quantile_table_distribution1d import QuantileGridConfig
from npcc.core.registry import validate_backend_kwargs

Transform = Literal["identity", "logit", "probit"]


@dataclass
class FitControlsRosenblattBicop:
  """Configuration for fitting a Rosenblatt pair copula.

  These controls configure the conditional-distribution backend used for both
  Rosenblatt directions. They satisfy pyvinecopulib's ``ControlsLike`` contract
  through :meth:`to_dict``.

  Attributes
  ----------
  backend
    Registered conditional-margin backend.
  quantile_config
    Quantile-grid and boundary configuration.
  transform
    Transformation applied to copula-scale responses before backend fitting.
  device
    Torch device used for fitting and evaluation. ``None`` leaves placement to
    the input data.
  batch_size
    Default backend inference batch size.
  backend_kwargs
    Backend-specific constructor arguments.
  sinkhorn_iters
    Number of Sinkhorn projection iterations. ``None`` disables projection.
  projection_grid_size
    Number of points per axis in the projection grid.
  """

  backend: str = "tabpfn-criterion"
  quantile_config: QuantileGridConfig = field(
    default_factory=QuantileGridConfig
  )
  transform: Transform = "logit"
  device: str | torch.device | None = None
  batch_size: int | None = None
  backend_kwargs: Mapping[str, object] = field(default_factory=dict)
  sinkhorn_iters: int | None = None
  projection_grid_size: int = 101

  def __post_init__(self) -> None:
    """Validate and normalize the controls."""
    if self.transform not in ("identity", "logit", "probit"):
      raise ValueError("transform must be 'identity', 'logit', or 'probit'.")

    if self.device is not None:
      self.device = torch.device(self.device)

    if self.batch_size is not None and self.batch_size <= 0:
      raise ValueError("batch_size must be positive.")

    if self.sinkhorn_iters is not None and self.sinkhorn_iters <= 0:
      raise ValueError("sinkhorn_iters must be None or a positive integer.")

    if self.projection_grid_size < 2:
      raise ValueError("projection_grid_size must be at least 2.")

    self.backend_kwargs = dict(self.backend_kwargs)
    validate_backend_kwargs(self.backend, self.backend_kwargs)

  def to_dict(self) -> dict[str, object]:
    """Return the settings as a plain dictionary."""
    return {
      "backend": self.backend,
      "quantile_config": self.quantile_config,
      "transform": self.transform,
      "device": self.device,
      "batch_size": self.batch_size,
      "backend_kwargs": dict(self.backend_kwargs),
      "sinkhorn_iters": self.sinkhorn_iters,
      "projection_grid_size": self.projection_grid_size,
    }


@dataclass
class FitControlsRosenblattVinecop(FitControlsRosenblattBicop):
  """Configuration for fitting a fixed-structure Rosenblatt vine.

  A vine's controls are also valid pair-copula controls. Consequently, the
  backend configuration is passed unchanged to every pair copula in the vine.

  The class currently adds no vine-specific fields because NPCC requires a
  fixed structure and does not perform selection.
  """
