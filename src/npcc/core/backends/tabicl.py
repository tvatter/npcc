"""
tabicl.py — TabICL foundation-model backend (extra).

TabICL is an in-context tabular foundation model whose regressor exposes
conditional quantiles directly:
``predict(X, output_type="quantiles", alphas=[...])`` returns an
``(n_samples, n_quantiles)`` array, order-matched to ``alphas``.  That is
exactly the :class:`~npcc.core.quantile_table_distribution1d.QuantileTableDistribution1D`
hook, so this backend implements only :py:meth:`_predict_quantiles`; the
chunked table prediction, monotone sort, and pdf/cdf/icdf/grid inversion
are inherited.

Requires the ``tabicl`` extra (``pip install npcc[tabicl]``).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from tabicl import TabICLRegressor

from npcc.core.quantile_table_distribution1d import (
  QuantileGridConfig,
  QuantileTableDistribution1D,
)


class TabICLBackend(QuantileTableDistribution1D):
  """Conditional predictive distribution via TabICL's quantile output.

  Parameters
  ----------
  transform, config, device, batch_size
      Forwarded to :class:`QuantileTableDistribution1D`.
  model_kwargs
      Forwarded to the ``TabICLRegressor`` constructor (e.g.
      ``n_estimators``, ``random_state``).
  """

  model_: TabICLRegressor | None

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    config: QuantileGridConfig | None = None,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
  ) -> None:
    super().__init__(
      transform=transform,
      config=config,
      device=device,
      batch_size=batch_size,
    )
    self.model_kwargs = dict(model_kwargs or {})
    self.model_kwargs.setdefault("device", str(self._device))
    self.model_ = None

  def _fit_model(self, w: torch.Tensor, z: torch.Tensor) -> None:
    self.model_ = TabICLRegressor(**self.model_kwargs)
    self.model_.fit(w.detach().cpu().numpy(), z.detach().cpu().numpy())

  def _predict_quantiles(
    self, w: torch.Tensor, alphas: np.ndarray
  ) -> torch.Tensor:
    assert self.model_ is not None
    q = self.model_.predict(
      w.detach().cpu().numpy(),
      output_type="quantiles",
      alphas=alphas.tolist(),
    )
    q_arr = np.asarray(q, dtype=float)
    n_chunk = w.shape[0]
    n_alphas = len(alphas)
    if q_arr.shape == (n_alphas, n_chunk):
      q_arr = q_arr.T
    if q_arr.shape != (n_chunk, n_alphas):
      raise RuntimeError(
        "Unexpected TabICL quantile output shape. "
        f"Got {q_arr.shape}, expected {(n_chunk, n_alphas)}."
      )
    return torch.as_tensor(q_arr, dtype=torch.float64, device=self._device)
