"""Original-scale vine distributions with backend-backed margins."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Self, cast

import numpy as np
import torch
from array_api_compat import is_torch_array
from pyvinecopulib import RVineStructure
from pyvinecopulib.core import Vinedist

from npcc.core._common import TensorLike
from npcc.core.margin import BackendMargin
from npcc.core.quantile_table_distribution1d import QuantileGridConfig
from npcc.core.vinecop import RosenblattVinecop


class RosenblattVinedist(Vinedist[TensorLike]):
  """Original-scale distribution with backend margins and a Rosenblatt vine.

  One independent :class:`BackendMargin` is fitted to each response column.
  Their probability integral transforms are then used to fit a fixed-structure
  :class:`RosenblattVinecop`.
  """

  @classmethod
  def from_data(
    cls,
    y: TensorLike,
    *,
    x: TensorLike | None = None,
    margins: object = None,
    controls: object | None = None,
    structure: object | None = None,
    weights: object | None = None,
    names: Sequence[str] | None = None,
    margin_backend: str = "tabpfn-criterion",
    margin_quantile_config: QuantileGridConfig | None = None,
    margin_backend_kwargs: Mapping[str, object] | None = None,
    pair_backend: str = "tabpfn-criterion",
    pair_quantile_config: QuantileGridConfig | None = None,
    pair_backend_kwargs: Mapping[str, object] | None = None,
    pair_transform: Literal["identity", "logit", "probit"] = "logit",
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    sinkhorn_iters: int | None = None,
    projection_grid_size: int = 101,
  ) -> Self:
    """Fit backend margins and a fixed-structure Rosenblatt vine.

    Parameters inherited from :class:`Vinedist` are retained for API
    compatibility. Options unsupported by the current Rosenblatt estimator
    must remain ``None``.
    """
    if margins is not None:
      raise NotImplementedError(
        "Custom margins are not supported; RosenblattVinedist fits one "
        "BackendMargin per response column."
      )
    if controls is not None:
      raise NotImplementedError(
        "controls are not supported by RosenblattVinecop fitting."
      )
    if weights is not None:
      raise NotImplementedError(
        "Observation weights are not supported by backend margins or "
        "RosenblattVinecop fitting."
      )
    if names is not None:
      raise NotImplementedError(
        "Variable names are not supported because custom margin resolution "
        "is currently disabled."
      )

    if structure is None:
      raise ValueError(
        "structure is required because RosenblattVinecop does not select "
        "vine structures."
      )
    if not isinstance(structure, RVineStructure):
      raise TypeError("structure must be an RVineStructure.")

    if not isinstance(y, (np.ndarray, torch.Tensor)):
      raise TypeError("y must be a NumPy array or torch tensor.")

    if y.ndim != 2:
      raise ValueError("y must have shape (n, d).")

    n, d = int(y.shape[0]), int(y.shape[1])
    if d != structure.dim:
      raise ValueError(
        f"y has {d} columns, but the structure has dimension {structure.dim}."
      )

    if x is not None:
      if not isinstance(x, (np.ndarray, torch.Tensor)):
        raise TypeError("x must be a NumPy array or torch tensor.")
      if is_torch_array(y) != is_torch_array(x):
        raise TypeError("y and x must use the same array namespace.")
      if x.ndim not in (1, 2):
        raise ValueError("x must have shape (n,) or (n, p).")
      if x.shape[0] != n:
        raise ValueError("y and x must have the same number of rows.")

    fitted_margins = [
      BackendMargin(
        backend=margin_backend,
        quantile_config=margin_quantile_config,
        device=device,
        batch_size=batch_size,
        backend_kwargs=margin_backend_kwargs,
      ).fit(y[:, j], x=x)
      for j in range(d)
    ]

    u = cast(
      TensorLike,
      Vinedist.copula_data(
        fitted_margins,
        y,
        x=x,
      ),
    )

    copula = RosenblattVinecop.from_data(
      u,
      structure,
      x=x,
      backend=pair_backend,
      quantile_config=pair_quantile_config,
      transform=pair_transform,
      device=device,
      batch_size=batch_size,
      backend_kwargs=dict(pair_backend_kwargs or {}),
      sinkhorn_iters=sinkhorn_iters,
      projection_grid_size=projection_grid_size,
    )

    return cls(copula, fitted_margins)
