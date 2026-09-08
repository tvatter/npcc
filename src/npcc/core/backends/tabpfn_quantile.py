"""
Quantile-based TabPFN conditional margin.

This backend reads TabPFN's ``output_type="quantiles"`` result into the
quantile-table reconstruction machinery. It implements model fitting and
quantile prediction; PDF, CDF, inverse CDF, and grid evaluation are inherited
from :class:`QuantileTableDistribution1D`.

TabPFN receives host-side input tensors. These conversions are isolated in this
adapter, while all outputs are returned as torch tensors on the configured
model device.
"""

from __future__ import annotations

from typing import Any, Literal

import torch

from npcc.core.backends.tabpfn_common import (
  _DEFAULT_MODEL_VERSION,
  ModelVersion,
  make_tabpfn_regressor,
)
from npcc.core.quantile_table_distribution1d import (
  QuantileTableConfig,
  QuantileTableDistribution1D,
)


class TabPFNQuantileBackend(QuantileTableDistribution1D):
  """Conditional margin using TabPFN's predicted quantiles.

  Parameters
  ----------
  transform
    Response transformation inherited from
    :class:`QuantileTableDistribution1D`.
  config
    Quantile-grid configuration.
  device
    Device used by the TabPFN model and returned tensors.
  batch_size
    Maximum number of conditioning rows evaluated in one prediction call.
  model_kwargs
    Additional arguments passed to ``TabPFNRegressor``.
  model_version
    TabPFN model version. The default is the shared current version.
  """

  model_: Any | None

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    quantile_table_config: QuantileTableConfig | None = None,
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
    model_version: ModelVersion | None = _DEFAULT_MODEL_VERSION,
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
    self.model_version = model_version
    self.model_ = None

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    """Fit TabPFN."""
    self.model_ = make_tabpfn_regressor(
      self.model_version,
      self.model_kwargs,
    )

    # TabPFN converts inputs to NumPy internally
    # CUDA inputs cannot be converted
    self.model_.fit(
      x.detach().cpu(),
      z.detach().cpu(),
    )

  def _predict_quantiles(
    self,
    x: torch.Tensor,
    alphas: torch.Tensor,
  ) -> torch.Tensor:
    """Predict one quantile table for a chunk of conditioning rows.

    TabPFN receives host-side conditioning rows and a Python list of
    probability levels. Its output is converted back to a torch tensor on the
    configured device.

    Depending on the TabPFN version, the output may have shape
    ``(n_chunk, n_alphas)`` or ``(n_alphas, n_chunk)``.
    """
    assert self.model_ is not None

    predicted = self.model_.predict(
      x.detach().cpu(),
      output_type="quantiles",
      quantiles=alphas.detach().cpu().tolist(),
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
      "Unexpected quantile output shape. "
      f"Got {tuple(quantiles.shape)}, expected {expected_shape} "
      f"or {transposed_shape}."
    )
