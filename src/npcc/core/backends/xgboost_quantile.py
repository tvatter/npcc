"""
xgboost_quantile.py — XGBoost multi-quantile backend (extra).

XGBoost's ``reg:quantileerror`` objective with a vector ``quantile_alpha``
fits a single multi-output model predicting all quantile levels at once,
mapping onto
:class:`~npcc.core.quantile_table_distribution1d.QuantileTableDistribution1D`.

Requires the ``xgboost`` extra (``pip install npcc[xgboost]``; XGBoost >= 2.0
for vector-valued ``quantile_alpha``).

Reference: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System",
KDD 2016.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from xgboost import XGBRegressor

from npcc.core.quantile_table_distribution1d import (
  QuantileGridConfig,
  QuantileTableDistribution1D,
)


class XGBQuantileBackend(QuantileTableDistribution1D):
  """Conditional predictive distribution via XGBoost multi-quantile regression.

  Parameters
  ----------
  transform, config, device, batch_size
      Forwarded to :class:`QuantileTableDistribution1D`.
  n_estimators, tree_method
      XGBoost boosting rounds / tree method.
  **xgb_kwargs
      Extra ``XGBRegressor`` kwargs (e.g. ``max_depth``, ``learning_rate``,
      ``device="cuda"``).
  """

  model_: XGBRegressor | None

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    config: QuantileGridConfig | None = None,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    n_estimators: int = 200,
    tree_method: str = "hist",
    **xgb_kwargs: Any,  # noqa: ANN401 - passthrough to XGBRegressor
  ) -> None:
    super().__init__(
      transform=transform,
      config=config,
      device=device,
      batch_size=batch_size,
    )
    self.n_estimators = n_estimators
    self.tree_method = tree_method
    self.xgb_kwargs = dict(xgb_kwargs)
    self.model_ = None

  def _fit_model(self, w: torch.Tensor, z: torch.Tensor) -> None:
    model = XGBRegressor(
      objective="reg:quantileerror",
      quantile_alpha=np.asarray(self.config.alphas(), dtype=float),
      tree_method=self.tree_method,
      multi_strategy="multi_output_tree",
      n_estimators=self.n_estimators,
      **self.xgb_kwargs,
    )
    model.fit(
      w.detach().cpu().numpy().astype(np.float32),
      z.detach().cpu().numpy(),
    )
    self.model_ = model

  def _predict_quantiles(
    self, w: torch.Tensor, alphas: np.ndarray
  ) -> torch.Tensor:
    assert self.model_ is not None
    q = np.asarray(
      self.model_.predict(w.detach().cpu().numpy().astype(np.float32)),
      dtype=float,
    )
    if q.ndim == 1:
      q = q[:, None]
    return torch.as_tensor(q, dtype=torch.float64, device=self._device)
