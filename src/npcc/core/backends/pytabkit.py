"""
PyTabKit RealMLP and TabM quantile conditional margins.

PyTabKit estimators can be trained using a ``multi_pinball`` metric containing
the requested probability levels. Their ordinary ``predict`` method then
returns the corresponding conditional quantile table.

PDF, CDF, inverse CDF, and grid evaluation are inherited from
:class:`QuantileTableDistribution1D`.

PyTabKit uses a NumPy-facing estimator interface. Input conversion is isolated
inside this adapter, and predictions are immediately converted back to torch
tensors on the input device and with the input dtype.

This backend requires the ``pytabkit`` optional dependency.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal, cast

import torch
from pyvinecopulib.core import to_numpy

from npcc.core.margin_quantile_table import (
  QuantileTableConfig,
  QuantileTableDistribution1D,
)


class _PyTabKitBackend(QuantileTableDistribution1D):
  """Shared PyTabKit quantile-model implementation."""

  model_: Any | None  # PyTabKit has multiple unrelated estimator classes.

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    quantile_table_config: QuantileTableConfig | None = None,
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    hpo: bool = False,
    **model_kwargs: Any,  # noqa: ANN401 - forwarded to PyTabKit
  ) -> None:
    super().__init__(
      transform=transform,
      quantile_table_config=quantile_table_config,
      eps=eps,
      device=device,
      batch_size=batch_size,
    )
    self.hpo = hpo
    self.model_kwargs = dict(model_kwargs)
    self.model_ = None

  @abstractmethod
  def _estimator_cls(self, *, hpo: bool) -> type:
    """Return the selected PyTabKit estimator class."""

  def _fit_model(
    self,
    x: torch.Tensor,
    z: torch.Tensor,
  ) -> None:
    """Fit a PyTabKit model using the fixed probability grid."""
    alphas = self.quantile_table_config.alphas(
      device="cpu",
      dtype=torch.float64,
    )

    alpha_string = ",".join(f"{alpha:.6f}" for alpha in alphas.tolist())

    metric = f"multi_pinball({alpha_string})"

    model_kwargs = dict(self.model_kwargs)
    model_kwargs.setdefault("train_metric_name", metric)
    model_kwargs.setdefault("val_metric_name", metric)

    estimator_class = self._estimator_cls(hpo=self.hpo)
    model = estimator_class(**model_kwargs)

    model.fit(
      to_numpy(x),
      to_numpy(z),
    )

    self.model_ = model

  def _predict_quantiles(
    self,
    x: torch.Tensor,
    alphas: torch.Tensor,
  ) -> torch.Tensor:
    """Predict one quantile table for a chunk of conditioning rows."""
    assert self.model_ is not None

    predicted = self.model_.predict(to_numpy(x))

    quantiles = torch.as_tensor(
      predicted,
      dtype=x.dtype,
      device=x.device,
    )

    n_chunk = x.shape[0]
    n_alphas = alphas.shape[0]
    expected_shape = (n_chunk, n_alphas)
    transposed_shape = (n_alphas, n_chunk)

    if quantiles.ndim == 1:
      quantiles = quantiles.reshape(expected_shape)

    if quantiles.shape == expected_shape:
      return quantiles

    if quantiles.shape == transposed_shape:
      return quantiles.T

    raise RuntimeError(
      "Unexpected pytabkit quantile shape. "
      f"Got {quantiles.shape}, expected {expected_shape} "
      f"or {transposed_shape}."
    )


class PyTabKitRealMLPBackend(_PyTabKitBackend):
  """Conditional quantiles from a PyTabKit RealMLP estimator."""

  def _estimator_cls(self, *, hpo: bool) -> type:
    """Return the tuned-default or HPO RealMLP estimator."""
    if hpo:
      from pytabkit import RealMLP_HPO_Regressor

      return cast("type", RealMLP_HPO_Regressor)

    from pytabkit import RealMLP_TD_Regressor

    return cast("type", RealMLP_TD_Regressor)


class PyTabKitTabMBackend(_PyTabKitBackend):
  """Conditional quantiles from a PyTabKit TabM estimator."""

  def _estimator_cls(self, *, hpo: bool) -> type:
    """Return the default or HPO TabM estimator."""
    if hpo:
      from pytabkit import TabM_HPO_Regressor

      return cast("type", TabM_HPO_Regressor)

    from pytabkit import TabM_D_Regressor

    return cast("type", TabM_D_Regressor)
