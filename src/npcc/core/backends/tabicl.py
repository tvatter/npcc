"""
TabICL quantile conditional margin.

TabICL is an in-context tabular foundation model whose regressor exposes
conditional quantiles directly. It predicts every requested probability level
in a single call.

PDF, CDF, inverse CDF, and grid evaluation are inherited from
:class:`QuantileTableDistribution1D`.

TabICL uses a NumPy-facing estimator interface. Input conversion is isolated
inside this adapter, and predictions are immediately converted back to torch
tensors on the input device and with the input dtype.

This backend requires the ``tabicl`` optional dependency.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from pyvinecopulib.core.extend import to_numpy
from tabicl import TabICLRegressor

from npcc.core.margin_quantile_table import (
  QuantileTableConfig,
  QuantileTableDistribution1D,
)


class TabICLBackend(QuantileTableDistribution1D):
  """Conditional margin using TabICL's quantile output.

  Parameters
  ----------
  transform
    Response transformation inherited from
    :class:`QuantileTableDistribution1D`.
  quantile_table_config
    Quantile-table reconstruction configuration.
  eps
    Distance used when clipping values away from the boundaries of
    ``(0, 1)`` before the logit or probit transform.
  device
    Device used for model fitting, prediction, and returned tensors.
  batch_size
    Maximum number of conditioning rows evaluated in one prediction call.
  model_kwargs
    Additional arguments passed to ``TabICLRegressor``.
  """

  model_: TabICLRegressor | None

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    quantile_table_config: QuantileTableConfig | None = None,
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
  ) -> None:
    super().__init__(
      transform=transform,
      quantile_table_config=quantile_table_config,
      eps=eps,
      device=device,
      batch_size=batch_size,
    )
    self.model_kwargs = dict(model_kwargs or {})
    self.model_kwargs.setdefault("device", str(self._device))
    self.model_ = None

  def _fit_model(
    self,
    x: torch.Tensor,
    z: torch.Tensor,
  ) -> None:
    """Prepare TabICL for conditional prediction."""
    self.model_ = TabICLRegressor(**self.model_kwargs)

    self.model_.fit(
      to_numpy(x),
      to_numpy(z),
    )

  def _predict_quantiles(
    self,
    x: torch.Tensor,
    alphas: torch.Tensor,
  ) -> torch.Tensor:
    """Predict one quantile table for a chunk of conditioning rows."""
    assert self.model_ is not None

    predicted = self.model_.predict(
      to_numpy(x),
      output_type="quantiles",
      alphas=alphas.tolist(),
    )

    quantiles = torch.as_tensor(
      predicted,
      dtype=x.dtype,
      device=x.device,
    )

    n_chunk = x.shape[0]
    n_alphas = alphas.shape[0]
    expected_shape = (n_chunk, n_alphas)
    transposed_shape = (n_alphas, n_chunk)

    if quantiles.shape == expected_shape:
      return quantiles

    if quantiles.shape == transposed_shape:
      return quantiles.T

    raise RuntimeError(
      "Unexpected TabICL quantile output shape. "
      f"Got {tuple(quantiles.shape)}, expected {expected_shape} "
      f"or {transposed_shape}."
    )
