"""
pytabkit.py — PyTabKit RealMLP / TabM backends (extra).

PyTabKit's regressors expose ``predict_quantiles(X, quantiles=…)``, mapping
directly onto
:class:`~npcc.core.quantile_table_distribution1d.QuantileTableDistribution1D`.
Two model families are provided, each with a tuned-defaults (TD) estimator
and an optional HPO estimator (selected via ``hpo=True``):

- RealMLP — a strong pre-tuned MLP (Holzmüller, Grinsztajn, Steinwart,
  "Better by Default", NeurIPS 2024, arXiv:2407.04491).
- TabM — parameter-efficient MLP ensembling (Gorishniy, Kotelnikov, Babenko,
  ICLR 2025, arXiv:2410.24210).

Requires the ``pytabkit`` extra (``pip install npcc[pytabkit]``).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal

import numpy as np
import torch

from npcc.core.quantile_table_distribution1d import (
  QuantileGridConfig,
  QuantileTableDistribution1D,
)


class _PyTabKitBackend(QuantileTableDistribution1D):
  """Shared PyTabKit plumbing: fit a regressor, read ``predict_quantiles``."""

  model_: Any | None

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    config: QuantileGridConfig | None = None,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    hpo: bool = False,
    **model_kwargs: Any,  # noqa: ANN401 - passthrough to the pytabkit estimator
  ) -> None:
    super().__init__(
      transform=transform,
      config=config,
      device=device,
      batch_size=batch_size,
    )
    self.hpo = hpo
    self.model_kwargs = dict(model_kwargs)
    self.model_ = None

  @abstractmethod
  def _estimator_cls(self, *, hpo: bool) -> type:
    """Return the pytabkit estimator class (TD or HPO variant)."""

  def _fit_model(self, w: torch.Tensor, z: torch.Tensor) -> None:
    # PyTabKit estimators have no ``predict_quantiles``; instead they are
    # trained with a ``multi_pinball(<alphas>)`` metric so that ``predict``
    # returns the quantile matrix at exactly those levels.
    metric = (
      "multi_pinball("
      + ",".join(f"{a:.6f}" for a in self.config.alphas())
      + ")"
    )
    kwargs = dict(self.model_kwargs)
    kwargs.setdefault("train_metric_name", metric)
    kwargs.setdefault("val_metric_name", metric)
    cls = self._estimator_cls(hpo=self.hpo)
    model = cls(**kwargs)
    model.fit(w.detach().cpu().numpy(), z.detach().cpu().numpy())
    self.model_ = model

  def _predict_quantiles(
    self, w: torch.Tensor, alphas: np.ndarray
  ) -> torch.Tensor:
    assert self.model_ is not None
    q = np.asarray(self.model_.predict(w.detach().cpu().numpy()), dtype=float)
    n_chunk, n_alphas = w.shape[0], len(alphas)
    if q.ndim == 1:
      q = q.reshape(n_chunk, n_alphas)
    elif q.shape == (n_alphas, n_chunk):
      q = q.T
    if q.shape != (n_chunk, n_alphas):
      raise RuntimeError(
        "Unexpected pytabkit quantile shape. "
        f"Got {q.shape}, expected {(n_chunk, n_alphas)}."
      )
    return torch.as_tensor(q, dtype=torch.float64, device=self._device)


class PyTabKitRealMLPBackend(_PyTabKitBackend):
  """RealMLP (`RealMLP_TD_Regressor` / `RealMLP_HPO_Regressor`)."""

  def _estimator_cls(self, *, hpo: bool) -> type:
    if hpo:
      from pytabkit import RealMLP_HPO_Regressor

      return RealMLP_HPO_Regressor
    from pytabkit import RealMLP_TD_Regressor

    return RealMLP_TD_Regressor


class PyTabKitTabMBackend(_PyTabKitBackend):
  """TabM (`TabM_D_Regressor` / `TabM_HPO_Regressor`)."""

  def _estimator_cls(self, *, hpo: bool) -> type:
    if hpo:
      from pytabkit import TabM_HPO_Regressor

      return TabM_HPO_Regressor
    from pytabkit import TabM_D_Regressor

    return TabM_D_Regressor
