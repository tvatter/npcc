"""
Gradient-boosted quantile conditional margin.

This backend fits one scikit-learn ``GradientBoostingRegressor`` with
``loss="quantile"`` for every probability level in ``config.alphas()``.

Scikit-learn receives detached host-side tensors. Its predictors are converted
immediately to torch tensors on the input device and with the input dtype.

This backend requires the ``bgm`` optional dependency.
"""

from __future__ import annotations

from typing import Literal

import torch
from sklearn.ensemble import GradientBoostingRegressor

from npcc.core.margin_quantile_table import (
  QuantileTableConfig,
  QuantileTableDistribution1D,
)


class QuantileGBMBackend(QuantileTableDistribution1D):
  """Conditional margin using per-quantile gradient-boosting models.

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
    Device used for returned tensors.
  batch_size
    Maximum number of conditioning rows evaluated in one prediction call.
  **gbm_kwargs
    Additional arguments passed to every
    ``GradientBoostingRegressor``. The ``loss`` and ``alpha`` arguments are
    controlled by this backend.
  """

  models_: list[GradientBoostingRegressor] | None

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    quantile_table_config: QuantileTableConfig | None = None,
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    **gbm_kwargs: object,
  ) -> None:
    super().__init__(
      transform=transform,
      quantile_table_config=quantile_table_config,
      eps=eps,
      device=device,
      batch_size=batch_size,
    )
    self.gbm_kwargs = dict(gbm_kwargs)
    self.models_ = None

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    """Fit one gradient-boosting model per probability level."""
    x_host = x.detach().cpu()
    z_host = z.detach().cpu()

    alphas = self.quantile_table_config.alphas(
      device="cpu",
      dtype=torch.float64,
    )

    self.models_ = []

    for alpha in alphas.tolist():
      model = GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        **self.gbm_kwargs,
      )
      model.fit(
        x_host,
        z_host,
      )
      self.models_.append(model)

  def _predict_quantiles(
    self,
    x: torch.Tensor,
    alphas: torch.Tensor,
  ) -> torch.Tensor:
    """Predict one quantile table for a chunk of conditioning rows."""
    assert self.models_ is not None

    del alphas

    x_host = x.detach().cpu()
    columns = [
      torch.as_tensor(
        model.predict(x_host),
        dtype=x.dtype,
        device=x.device,
      )
      for model in self.models_
    ]

    return torch.stack(columns, dim=1)
