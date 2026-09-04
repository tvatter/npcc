"""
CatBoost MultiQuantile conditional margin.

CatBoost's ``MultiQuantile`` loss fits one model that predicts every
probability level in ``config.alphas()``.

PDF, CDF, inverse CDF, and grid evaluation are inherited from
:class:`QuantileTableDistribution1D``.

CatBoost receives NumPy-backed host arrays because it does not support torch
tensors directly. This conversion is isolated inside the adapter. Predictions
are converted immediately to torch tensors on the input device and with the
input dtype.

This backend requires the ``catboost`` optional dependency.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from catboost import CatBoostRegressor

from npcc.core.quantile_table_distribution1d import (
  QuantileGridConfig,
  QuantileTableDistribution1D,
)


class CatBoostBackend(QuantileTableDistribution1D):
  """Conditional margin using CatBoost MultiQuantile regression.

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
  iterations
    Number of CatBoost boosting iterations.
  **catboost_kwargs
    Additional arguments passed to ``CatBoostRegressor``. The
    ``loss_function`` and ``logging_level`` arguments are controlled by this
    backend.
  """

  model_: CatBoostRegressor | None

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    config: QuantileGridConfig | None = None,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    iterations: int = 1000,
    **catboost_kwargs: Any,  # noqa: ANN401 - forwarded to CatBoostRegressor
  ) -> None:
    super().__init__(
      transform=transform,
      config=config,
      device=device,
      batch_size=batch_size,
    )
    self.iterations = iterations
    self.catboost_kwargs = dict(catboost_kwargs)
    self.model_ = None

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    """Fit the CatBoost multi-quantile model."""
    alphas = self.config.alphas(
      device="cpu",
      dtype=torch.float64,
    )

    alpha_string = ",".join(f"{alpha:.6f}" for alpha in alphas)

    model = CatBoostRegressor(
      iterations=self.iterations,
      loss_function=f"MultiQuantile:alpha={alpha_string}",
      logging_level="Silent",
      **self.catboost_kwargs,
    )

    x_host = (
      x.detach()
      .to(
        device="cpu",
        dtype=torch.float32,
      )
      .numpy()
    )
    z_host = z.detach().cpu().numpy()

    model.fit(x_host, z_host)

    self.model_ = model

  def _predict_quantiles(
    self,
    x: torch.Tensor,
    alphas: torch.Tensor,
  ) -> torch.Tensor:
    """Predict one quantile table for a chunk of conditioning rows."""
    assert self.model_ is not None

    del alphas

    x_host = (
      x.detach()
      .to(
        device="cpu",
        dtype=torch.float32,
      )
      .numpy()
    )

    predicted = self.model_.predict(x_host)

    quantiles = torch.as_tensor(
      predicted,
      dtype=x.dtype,
      device=x.device,
    )

    if quantiles.ndim == 1:
      quantiles = quantiles.unsqueeze(1)

    return quantiles
