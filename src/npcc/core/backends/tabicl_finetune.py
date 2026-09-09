"""Fine-tuned TabICL backend, read through its quantile head.

Fine-tunes TabICL (`FinetunedTabICLRegressor`) and reads the predictive
distribution off the same quantile head as
:class:`~npcc.core.backends.tabicl.TabICLBackend` — so it subclasses that
backend and overrides only ``_fit_model``.

Note: TabICL fine-tunes on a **pinball (quantile) loss** over its raw
quantile outputs (not configurable to a density/NLL loss), so — unlike the
TabPFN fine-tune backend — its objective is calibration-oriented rather than
the density-optimal bar-NLL.

Requires the ``tabicl`` extra (``pip install npcc[tabicl]``; GPU-recommended).
Reference: ScoringBench, arXiv:2603.08206 (fine-tuning tabular foundation
models with proper scoring rules); TabICL, arXiv:2502.05564.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from pyvinecopulib.core import to_numpy

from npcc.core.backends.tabicl import TabICLBackend
from npcc.core.margin_quantile_table import QuantileTableConfig


class FinetunedTabICLBackend(TabICLBackend):
  """TabICL fine-tuned on pinball loss; quantile read-out.

  Parameters
  ----------
  transform, quantile_table_config, eps, device, batch_size
      Forwarded to :class:`TabICLBackend`.
  epochs, learning_rate
      Fine-tuning schedule for ``FinetunedTabICLRegressor``.
  **finetune_kwargs
      Extra keyword arguments for ``FinetunedTabICLRegressor`` (e.g.
      ``early_stopping``, ``patience``).
  """

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    quantile_table_config: QuantileTableConfig | None = None,
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    epochs: int = 30,
    learning_rate: float = 1e-5,
    **finetune_kwargs: Any,  # noqa: ANN401 - passthrough to the finetuner
  ) -> None:
    super().__init__(
      transform=transform,
      quantile_table_config=quantile_table_config,
      eps=eps,
      device=device,
      batch_size=batch_size,
    )
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.finetune_kwargs = dict(finetune_kwargs)

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    from tabicl import FinetunedTabICLRegressor

    model = FinetunedTabICLRegressor(
      device=str(self._device),
      epochs=self.epochs,
      learning_rate=self.learning_rate,
      **self.finetune_kwargs,
    )
    model.fit(to_numpy(x), to_numpy(z))
    self.model_ = model
