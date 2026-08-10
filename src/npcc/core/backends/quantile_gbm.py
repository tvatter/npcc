"""
quantile_gbm.py — gradient-boosted quantile-regression backend (extra).

Fits one scikit-learn ``GradientBoostingRegressor(loss="quantile")`` per
alpha level of ``config.alphas()`` (frozen at fit time), then predicts the
conditional quantile table.  All pdf/cdf/icdf/grid inversion is inherited
from :class:`~npcc.core.quantile_table_distribution1d.QuantileTableDistribution1D`.

Requires the ``gbm`` extra (``pip install npcc[gbm]`` / scikit-learn).
Fitting is ``O(K)`` models (one per quantile) — a one-time cost; the
protected conditional-regression *inference* path is plain tree prediction.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor

from npcc.core.quantile_table_distribution1d import (
  QuantileGridConfig,
  QuantileTableDistribution1D,
)


class QuantileGBMBackend(QuantileTableDistribution1D):
  """Conditional predictive distribution via per-quantile GBM regression.

  Parameters
  ----------
  transform, config, device, batch_size
      Forwarded to :class:`QuantileTableDistribution1D`.
  **gbm_kwargs
      Forwarded to every ``GradientBoostingRegressor`` (e.g.
      ``n_estimators``, ``max_depth``, ``learning_rate``).  ``loss`` and
      ``alpha`` are set internally and must not be overridden.
  """

  models_: list[GradientBoostingRegressor] | None

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    config: QuantileGridConfig | None = None,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    **gbm_kwargs: object,
  ) -> None:
    super().__init__(
      transform=transform,
      config=config,
      device=device,
      batch_size=batch_size,
    )
    self.gbm_kwargs = dict(gbm_kwargs)
    self.models_ = None

  def _fit_model(self, w: torch.Tensor, z: torch.Tensor) -> None:
    w_np = w.detach().cpu().numpy()
    z_np = z.detach().cpu().numpy()
    self.models_ = []
    for alpha in self.config.alphas():
      model = GradientBoostingRegressor(
        loss="quantile", alpha=float(alpha), **self.gbm_kwargs
      )
      model.fit(w_np, z_np)
      self.models_.append(model)

  def _predict_quantiles(
    self, w: torch.Tensor, alphas: np.ndarray
  ) -> torch.Tensor:
    assert self.models_ is not None
    w_np = w.detach().cpu().numpy()
    cols = [model.predict(w_np) for model in self.models_]
    q = np.stack(cols, axis=1)
    return torch.as_tensor(q, dtype=torch.float64, device=self._device)
