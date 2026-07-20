"""
catboost.py — CatBoost MultiQuantile backend (extra).

CatBoost with the ``MultiQuantile`` loss fits a **single** model that predicts
all requested quantile levels at once (unlike the per-level ``gbm`` backend,
which fits one sklearn model per level).  It maps onto
:class:`~npcc.core.quantile_table_distribution1d.QuantileTableDistribution1D`
via the single ``_predict_quantiles`` hook.

Requires the ``catboost`` extra (``pip install npcc[catboost]``).

Reference: Prokhorenkova, Gusev, Vorobev, Dorogush, Gulin, "CatBoost: unbiased
boosting with categorical features", NeurIPS 2018.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from catboost import CatBoostRegressor

from npcc.core.quantile_table_distribution1d import (
  QuantileGridConfig,
  QuantileTableDistribution1D,
)


class CatBoostBackend(QuantileTableDistribution1D):
  """Conditional predictive distribution via CatBoost MultiQuantile.

  Parameters
  ----------
  transform, config, device, batch_size
      Forwarded to :class:`QuantileTableDistribution1D`.
  iterations
      Number of boosting iterations.
  **catboost_kwargs
      Extra keyword arguments for ``CatBoostRegressor`` (e.g. ``depth``,
      ``learning_rate``, ``task_type="GPU"``).  ``loss_function`` and
      ``logging_level`` are managed internally.
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
    **catboost_kwargs: Any,  # noqa: ANN401 - passthrough to CatBoostRegressor
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

  def _fit_model(self, w: torch.Tensor, z: torch.Tensor) -> None:
    alphas = self.config.alphas()
    alpha_str = ",".join(f"{a:.6f}" for a in alphas)
    model = CatBoostRegressor(
      iterations=self.iterations,
      loss_function=f"MultiQuantile:alpha={alpha_str}",
      logging_level="Silent",
      **self.catboost_kwargs,
    )
    model.fit(w.detach().cpu().numpy(), z.detach().cpu().numpy())
    self.model_ = model

  def _predict_quantiles(
    self, w: torch.Tensor, alphas: np.ndarray
  ) -> torch.Tensor:
    assert self.model_ is not None
    q = np.asarray(self.model_.predict(w.detach().cpu().numpy()), dtype=float)
    if q.ndim == 1:
      q = q[:, None]
    return torch.as_tensor(q, dtype=torch.float64, device=self._device)
