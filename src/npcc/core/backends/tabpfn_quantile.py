"""
tabpfn_quantile.py — quantile-based TabPFN backend.

Reads TabPFN's ``output_type="quantiles"`` API into the universal
:class:`~npcc.core.quantile_table_distribution1d.QuantileTableDistribution1D`
machinery: it only implements :py:meth:`_predict_quantiles`; the chunked
table prediction, monotone sort, and pdf/cdf/icdf/grid inversion are
inherited.

The quantile read-out is model-agnostic but, for TabPFN, is slower than
the native ``criterion`` head (see
:class:`~npcc.core.backends.tabpfn_criterion.TabPFNCriterionBackend`,
the default).  It exists mainly as the shared path that the non-TabPFN
quantile backends reuse.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

from npcc.core.backends.tabpfn_common import (
  _DEFAULT_MODEL_VERSION,
  ModelVersion,
  make_tabpfn_regressor,
)
from npcc.core.quantile_table_distribution1d import (
  QuantileGridConfig,
  QuantileTableDistribution1D,
)


class TabPFNQuantileBackend(QuantileTableDistribution1D):
  """Conditional predictive distribution via TabPFN's quantile output.

  Parameters
  ----------
  transform, config, device, batch_size
      Forwarded to :class:`QuantileTableDistribution1D`.
  model_kwargs
      Forwarded to the ``TabPFNRegressor`` constructor.
  model_version
      TabPFN model version (default: v3).
  """

  model_: Any | None

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    config: QuantileGridConfig | None = None,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
    model_version: ModelVersion | None = _DEFAULT_MODEL_VERSION,
  ) -> None:
    super().__init__(
      transform=transform,
      config=config,
      device=device,
      batch_size=batch_size,
    )
    self.model_kwargs = dict(model_kwargs or {})
    self.model_kwargs.setdefault("device", str(self._device))
    self.model_version = model_version
    self.model_ = None

  def _fit_model(self, w: torch.Tensor, z: torch.Tensor) -> None:
    self.model_ = make_tabpfn_regressor(self.model_version, self.model_kwargs)
    self.model_.fit(w, z)

  def _predict_quantiles(
    self, w: torch.Tensor, alphas: np.ndarray
  ) -> torch.Tensor:
    """One ``output_type="quantiles"`` forward pass for a chunk of rows.

    TabPFN's predict input must be on CPU.  Its quantile output may be
    ``(n_chunk, n_alphas)`` or its transpose depending on the version;
    prefer the documented ``(n_chunk, n_alphas)`` layout so a square
    chunk is never spuriously transposed.
    """
    assert self.model_ is not None
    q_pred = self.model_.predict(
      w.detach().cpu(),
      output_type="quantiles",
      quantiles=alphas.tolist(),
    )
    q = np.asarray(q_pred, dtype=float)
    n_chunk = w.shape[0]
    n_alphas = len(alphas)

    if q.shape == (n_chunk, n_alphas):
      pass
    elif q.shape == (n_alphas, n_chunk):
      q = q.T
    else:
      raise RuntimeError(
        "Unexpected quantile output shape. "
        f"Got {q.shape}, expected {(n_chunk, n_alphas)} "
        f"or {(n_alphas, n_chunk)}."
      )
    return torch.as_tensor(q, dtype=torch.float64, device=self._device)
