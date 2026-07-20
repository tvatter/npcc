"""Quantile conditional distribution backed by TabICLv2."""

from __future__ import annotations

import importlib
from typing import Literal, Protocol, cast

import numpy as np
import torch

from npcc.core._common import TensorLike
from npcc.core.errors import MissingProviderDependencyError
from npcc.core.providers import TabICLConfig
from npcc.core.quantile_inversion import (
  QuantileInversionConfig,
  _QuantileInversionDistribution,
)


class _TabICLRegressorLike(Protocol):
  def fit(self, x: np.ndarray, y: np.ndarray) -> object: ...
  def predict(
    self, x: np.ndarray, *, output_type: str, alphas: list[float]
  ) -> object: ...


class _TabICLRegressorFactory(Protocol):
  def __call__(self, **kwargs: object) -> _TabICLRegressorLike: ...


class _TabICLQuantileAdapter:
  """Translate TabICL's quantile API to the adapter used by NPCC."""

  def __init__(
    self, config: TabICLConfig, device: torch.device, random_state: int
  ) -> None:
    try:
      module = importlib.import_module("tabicl")
    except ImportError as exc:
      raise MissingProviderDependencyError(
        "TabICL is unavailable. Install npcc[tabicl], then make checkpoint "
        f"{config.checkpoint!r} available in the local model cache."
      ) from exc
    kwargs = dict(config.extra_regressor_kwargs)
    kwargs.update(
      checkpoint_version=config.checkpoint,
      n_estimators=config.n_estimators,
      batch_size=config.ensemble_batch_size,
      kv_cache=False if config.cache_mode == "none" else config.cache_mode,
      device=str(device),
      random_state=random_state,
    )
    factory = cast(_TabICLRegressorFactory, module.TabICLRegressor)
    self._model = factory(**kwargs)

  def fit(self, x: TensorLike, y: TensorLike) -> _TabICLQuantileAdapter:
    self._model.fit(np.asarray(x), np.asarray(y))
    return self

  def predict(
    self,
    x: TensorLike,
    *,
    output_type: str = "mean",
    quantiles: list[float] | None = None,
  ) -> object:
    if output_type != "quantiles" or quantiles is None:
      raise ValueError("TabICL adapter supports quantile predictions only.")
    return self._model.predict(
      np.asarray(x), output_type="quantiles", alphas=quantiles
    )


class _TabICLQuantileDistribution(_QuantileInversionDistribution):
  """TabICL implementation of NPCC's numerical quantile distribution."""

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    config: QuantileInversionConfig | None = None,
    support_epsilon: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    provider_config: TabICLConfig,
    random_state: int = 317,
  ) -> None:
    super().__init__(
      transform=transform,
      config=config,
      support_epsilon=support_epsilon,
      device=device,
      batch_size=batch_size,
      model_version=None,
    )
    self.provider_config = provider_config
    self.random_state = random_state

  def _make_model(self) -> _TabICLQuantileAdapter:
    return _TabICLQuantileAdapter(
      self.provider_config, self._device, self.random_state
    )
