"""
Synthefy Nori quantile conditional margin.

Nori is an in-context tabular foundation model whose regressor exposes
conditional quantiles directly. It predicts every requested probability level
in a single call.

PDF, CDF, inverse CDF, and grid evaluation are inherited from
:class:`QuantileTableDistribution1D`.

Nori uses a NumPy-facing estimator interface. Input conversion is isolated
inside this adapter, and predictions are immediately converted back to torch
tensors on the input device and with the input dtype.

This backend requires the ``synthefy-nori`` optional dependency.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from synthefy_nori import NoriRegressor

from npcc.core.quantile_table_distribution1d import (
  QuantileGridConfig,
  QuantileTableDistribution1D,
)


class NoriBackend(QuantileTableDistribution1D):
  """Conditional margin using Synthefy Nori's quantile output.

  Parameters
  ----------
  transform
    Response transformation inherited from
    :class:`QuantileTableDistribution1D`.
  config
    Quantile-grid configuration.
  device
    Device used by Nori and returned tensors.
  batch_size
    Maximum number of conditioning rows evaluated in one prediction call.
  **nori_kwargs
    Additional arguments passed to ``NoriRegressor``.
  """

  model_: NoriRegressor | None

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    config: QuantileGridConfig | None = None,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    **nori_kwargs: Any,  # noqa: ANN401 - forwarded to NoriRegressor
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

  def _fit_model(
    self,
    x: torch.Tensor,
    z: torch.Tensor,
  ) -> None:
    """Prepare Nori for conditional prediction."""
    model = NoriRegressor(**self.nori_kwargs)

    x_host = (
      x.detach()
      .to(
        device="cpu",
        dtype=torch.float64,
      )
      .numpy()
    )
    z_host = (
      z.detach()
      .to(
        device="cpu",
        dtype=torch.float64,
      )
      .numpy()
    )

    model.fit(x_host, z_host)

    self.model_ = model

  def _predict_quantiles(
    self,
    x: torch.Tensor,
    alphas: torch.Tensor,
  ) -> torch.Tensor:
    """Predict one quantile table for a chunk of conditioning rows."""
    assert self.model_ is not None

    x_host = (
      x.detach()
      .to(
        device="cpu",
        dtype=torch.float32,
      )
      .numpy()
    )

    predicted = self.model_.predict(
      x_host,
      output_type="quantiles",
      quantiles=alphas.detach().cpu().tolist(),
    )

    quantiles = torch.as_tensor(
      predicted,
      dtype=x.dtype,
      device=x.device,
    )

    n_chunk = x.shape[0]
    n_alphas = alphas.shape[0]
    expected_shape = (n_chunk, n_alphas)
    nori_shape = (n_alphas, n_chunk)

    if quantiles.ndim == 1:
      quantiles = quantiles.reshape(nori_shape)

    if quantiles.shape == nori_shape:
      return quantiles.T

    if quantiles.shape == expected_shape:
      return quantiles

    raise RuntimeError(
      "Unexpected Nori quantile output shape. "
      f"Got {quantiles.shape}, expected {nori_shape} "
      f"or {expected_shape}."
    )
