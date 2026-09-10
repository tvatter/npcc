"""
XGBoost multi-quantile conditional margin.

XGBoost's ``reg:quantileerror`` objective fits a single multi-output model
predicting every probability level in ``config.alphas()``.

PDF, CDF, inverse CDF, and grid evaluation are inherited from
:class:`QuantileTableDistribution1D`.

XGBoost supports torch tensors directly. Model predictions are converted
immediately to torch tensors on the input device and with the input dtype.

This backend requires XGBoost 2.0 or newer.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from xgboost import XGBRegressor

from npcc.core.quantile_table_distribution1d import (
  QuantileTableConfig,
  QuantileTableDistribution1D,
)


class XGBQuantileBackend(QuantileTableDistribution1D):
  """Conditional margin using XGBoost multi-quantile regression.

  Parameters
  ----------
  transform
    Response transformation inherited from
    :class:`QuantileTableDistribution1D`.
  config
    Quantile-grid configuration.
  device:
    Device used for model fitting, prediction, and returned tensors.
  batch_size
    Maximum number of conditioning rows evaluated in one prediction call.
  n_estimators
    Number of XGBoost boosting rounds.
  tree_method
    XGBoost tree construction algorithm.
  **xgb_kwargs
      Additional arguments passed to ``XGBRegressor``.
  """

  model_: XGBRegressor | None

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    quantile_table_config: QuantileTableConfig | None = None,
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    n_estimators: int = 200,
    tree_method: str = "hist",
    **xgb_kwargs: Any,  # noqa: ANN401 - forwarded to XGBRegressor
  ) -> None:
    super().__init__(
      transform=transform,
      quantile_table_config=quantile_table_config,
      eps=eps,
      device=device,
      batch_size=batch_size,
    )
    self.n_estimators = n_estimators
    self.tree_method = tree_method
    self.xgb_kwargs = dict(xgb_kwargs)
    self.xgb_kwargs.setdefault(
      "device",
      str(self._device),
    )
    self.model_ = None

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    """Fit the multi-quantile XGBoost model."""
    alphas = self.quantile_table_config.alphas(
      device="cpu",
      dtype=torch.float64,
    )

    model = XGBRegressor(
      objective="reg:quantileerror",
      quantile_alpha=alphas.tolist(),
      tree_method=self.tree_method,
      multi_strategy="multi_output_tree",
      n_estimators=self.n_estimators,
      **self.xgb_kwargs,
    )

    model.fit(
      x.detach(),
      z.detach(),
    )
    self.model_ = model

  def _predict_quantiles(
    self,
    x: torch.Tensor,
    alphas: torch.Tensor,
  ) -> torch.Tensor:
    """Predict one quantile table for a chunk of conditioning rows."""
    assert self.model_ is not None

    del alphas

    predicted = self.model_.predict(x.detach())

    quantiles = torch.as_tensor(
      predicted,
      dtype=x.dtype,
      device=x.device,
    )

    if quantiles.ndim == 1:
      quantiles = quantiles.unsqueeze(1)

    return quantiles
