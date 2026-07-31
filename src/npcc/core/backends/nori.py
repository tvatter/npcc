"""
nori.py — Synthefy Nori foundation-model backend (extra).

Nori is a tabular foundation model for regression via in-context learning; its
regressor exposes conditional quantiles directly
(``predict(X, output_type="quantiles", quantiles=[...])``), mapping onto
:class:`~npcc.core.quantile_table_distribution1d.QuantileTableDistribution1D`
via the single ``_predict_quantiles`` hook.

Install (until the torch-uncapping fix reaches PyPI, use the fork; ``uv``
cannot lock it because its packaging pins the torch cu128 index — see the note
in ``pyproject.toml``)::

    uv sync --extra cu128 --extra backends
    uv pip install "synthefy-nori @ \
git+https://github.com/tvatter/synthefy-nori.git@allow-newer-torch-cuda"

Package: ``synthefy-nori``.  Reference: Synthefy Nori model card
(https://huggingface.co/Synthefy/Nori); no peer-reviewed paper at time of
writing.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from synthefy_nori import NoriRegressor

from npcc.core.quantile_table_distribution1d import (
  QuantileGridConfig,
  QuantileTableDistribution1D,
)


class NoriBackend(QuantileTableDistribution1D):
  """Conditional predictive distribution via Synthefy Nori's quantile head.

  Parameters
  ----------
  transform, config, device, batch_size
      Forwarded to :class:`QuantileTableDistribution1D`.
  **nori_kwargs
      Extra keyword arguments for ``NoriRegressor`` (e.g. ``model_path``,
      ``augmentations``).  ``device`` defaults to the resolved backend device.
  """

  model_: NoriRegressor | None

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    config: QuantileGridConfig | None = None,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    **nori_kwargs: Any,  # noqa: ANN401 - passthrough to NoriRegressor
  ) -> None:
    super().__init__(
      transform=transform,
      config=config,
      device=device,
      batch_size=batch_size,
    )
    self.nori_kwargs = dict(nori_kwargs)
    self.nori_kwargs.setdefault("device", str(self._device))
    self.model_ = None

  def _fit_model(self, w: torch.Tensor, z: torch.Tensor) -> None:
    model = NoriRegressor(**self.nori_kwargs)
    model.fit(
      w.detach().cpu().numpy().astype(np.float32),
      z.detach().cpu().numpy().astype(np.float64),
    )
    self.model_ = model

  def _predict_quantiles(
    self, w: torch.Tensor, alphas: np.ndarray
  ) -> torch.Tensor:
    assert self.model_ is not None
    q = np.asarray(
      self.model_.predict(
        w.detach().cpu().numpy().astype(np.float32),
        output_type="quantiles",
        quantiles=alphas.tolist(),
      ),
      dtype=float,
    )
    n_chunk, n_alphas = w.shape[0], len(alphas)
    if q.ndim == 1:
      q = q.reshape(n_chunk, n_alphas)
    elif q.shape == (n_alphas, n_chunk):
      q = q.T
    if q.shape != (n_chunk, n_alphas):
      raise RuntimeError(
        "Unexpected Nori quantile output shape. "
        f"Got {q.shape}, expected {(n_chunk, n_alphas)}."
      )
    return torch.as_tensor(q, dtype=torch.float64, device=self._device)
