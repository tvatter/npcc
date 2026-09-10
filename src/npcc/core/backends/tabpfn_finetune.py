"""Fine-tuned TabPFN backend, read through its native head.

Fine-tunes TabPFN (`FinetunedTabPFNRegressor`, which fine-tunes the v2.5
checkpoint) with a proper-scoring-rule loss, then reads the predictive
distribution off the same native bar-distribution head as
:class:`~npcc.core.backends.tabpfn_criterion.TabPFNCriterionBackend` — so it
subclasses that backend and overrides only ``_fit_model``.

**Loss for our problem.** The downstream metric is the copula KL / log
predictive density, and by the Rosenblatt factorization the inner conditional
NLL equals the copula log-density (per direction).  So the density-optimal
fine-tuning objective is the bar-distribution **negative log-likelihood**
(`ce_loss_weight`), i.e. maximum likelihood = minimize KL.  We default to
``ce_loss_weight=1`` with the other terms off (TabPFN's own default is
``crps + mse``, which optimizes calibration/point, not density).  All loss
weights stay tunable.

Requires the core ``tabpfn`` package (fine-tuning is native, GPU-recommended).
Reference: ScoringBench, "Distributional Regression with Tabular Foundation
Models ... Proper Scoring Rules", arXiv:2603.08206.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from pyvinecopulib.core.extend import to_numpy

from npcc.core.backends.tabpfn_criterion import TabPFNCriterionBackend


class FinetunedTabPFNCriterionBackend(TabPFNCriterionBackend):
  """TabPFN fine-tuned with a scoring-rule loss; criterion (bar) read-out.

  Parameters
  ----------
  transform, eps, device, batch_size
      Forwarded to :class:`TabPFNCriterionBackend`.
  epochs, learning_rate
      Fine-tuning schedule for ``FinetunedTabPFNRegressor``.
  ce_loss_weight, crps_loss_weight, crls_loss_weight, mse_loss_weight, mae_loss_weight
      Proper-scoring-rule loss weights.  Default = pure bar-distribution NLL
      (`ce_loss_weight=1`, rest 0) — the density-optimal objective for copula
      KL (see module docstring).
  **finetune_kwargs
      Extra keyword arguments for ``FinetunedTabPFNRegressor`` (e.g.
      ``early_stopping``, ``n_estimators_final_inference``).
  """

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    epochs: int = 30,
    learning_rate: float = 1e-5,
    ce_loss_weight: float = 1.0,
    crps_loss_weight: float = 0.0,
    crls_loss_weight: float = 0.0,
    mse_loss_weight: float = 0.0,
    mae_loss_weight: float = 0.0,
    **finetune_kwargs: Any,  # noqa: ANN401 - passthrough to the finetuner
  ) -> None:
    super().__init__(
      transform=transform,
      eps=eps,
      device=device,
      batch_size=batch_size,
    )
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.ce_loss_weight = ce_loss_weight
    self.crps_loss_weight = crps_loss_weight
    self.crls_loss_weight = crls_loss_weight
    self.mse_loss_weight = mse_loss_weight
    self.mae_loss_weight = mae_loss_weight
    self.finetune_kwargs = dict(finetune_kwargs)

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor

    model = FinetunedTabPFNRegressor(
      device=str(self._device),
      epochs=self.epochs,
      learning_rate=self.learning_rate,
      ce_loss_weight=self.ce_loss_weight,
      crps_loss_weight=self.crps_loss_weight,
      crls_loss_weight=self.crls_loss_weight,
      mse_loss_weight=self.mse_loss_weight,
      mae_loss_weight=self.mae_loss_weight,
      **self.finetune_kwargs,
    )
    model.fit(to_numpy(x), to_numpy(z))
    self.model_ = model
