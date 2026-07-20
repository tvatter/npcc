"""Foundation-model provider contracts and built-in provider resolution."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, cast
import warnings

import torch

from npcc.core.conditional_distribution import (
  ConditionalDistribution,
  SupportTransform,
)
from npcc.core.errors import (
  FoundationModelRangeWarning,
  MissingProviderDependencyError,
  ProviderConfigurationError,
  UnsupportedRecoveryError,
)
from npcc.core.quantile_inversion import QuantileInversionConfig

JSONValue = (
  None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)


class Recovery(StrEnum):
  """How a provider prediction is recovered as a distribution."""

  NATIVE_DISTRIBUTION = "native_distribution"
  QUANTILE_INVERSION = "quantile_inversion"


def _validate_json(value: object, path: str = "extra_regressor_kwargs") -> None:
  if value is None or isinstance(value, bool | int | float | str):
    return
  if isinstance(value, list):
    for index, item in enumerate(value):
      _validate_json(item, f"{path}[{index}]")
    return
  if isinstance(value, dict) and all(isinstance(key, str) for key in value):
    for key, item in value.items():
      _validate_json(item, f"{path}.{key}")
    return
  raise ProviderConfigurationError(f"{path} must contain only JSON values.")


def _freeze_kwargs(
  values: Mapping[str, JSONValue], explicit: frozenset[str]
) -> Mapping[str, JSONValue]:
  copied = deepcopy(dict(values))
  _validate_json(copied)
  duplicates = explicit.intersection(copied)
  if duplicates:
    names = ", ".join(sorted(duplicates))
    raise ProviderConfigurationError(
      f"Explicit provider fields cannot be repeated in "
      f"extra_regressor_kwargs: {names}."
    )
  return MappingProxyType(copied)


@dataclass(frozen=True)
class TabPFNConfig:
  """Stable, serializable configuration for TabPFN."""

  model_version: str = "v3"
  n_estimators: int | None = None
  ignore_pretraining_limits: bool = False
  extra_regressor_kwargs: Mapping[str, JSONValue] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.model_version:
      raise ProviderConfigurationError("model_version must be non-empty.")
    if self.n_estimators is not None and self.n_estimators <= 0:
      raise ProviderConfigurationError("n_estimators must be positive or None.")
    frozen = _freeze_kwargs(
      self.extra_regressor_kwargs,
      frozenset({"model_version", "n_estimators", "ignore_pretraining_limits"}),
    )
    object.__setattr__(self, "extra_regressor_kwargs", frozen)


@dataclass(frozen=True)
class TabICLConfig:
  """Stable, serializable configuration for TabICLv2."""

  checkpoint: str = "tabicl-regressor-v2-20260212.ckpt"
  n_estimators: int = 8
  ensemble_batch_size: int = 8
  cache_mode: str = "repr"
  extra_regressor_kwargs: Mapping[str, JSONValue] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not self.checkpoint:
      raise ProviderConfigurationError("checkpoint must be non-empty.")
    if self.n_estimators <= 0 or self.ensemble_batch_size <= 0:
      raise ProviderConfigurationError(
        "n_estimators and ensemble_batch_size must be positive."
      )
    if self.cache_mode not in {"none", "kv", "repr"}:
      raise ProviderConfigurationError(
        "cache_mode must be one of 'none', 'kv', or 'repr'."
      )
    frozen = _freeze_kwargs(
      self.extra_regressor_kwargs,
      frozenset(
        {"checkpoint", "n_estimators", "ensemble_batch_size", "cache_mode"}
      ),
    )
    object.__setattr__(self, "extra_regressor_kwargs", frozen)


class FoundationModelProvider(Protocol):
  """Python extension point for foundation-model distributions."""

  @property
  def name(self) -> str: ...

  @property
  def model_id(self) -> str: ...

  @property
  def supported_recoveries(self) -> frozenset[Recovery]: ...

  def create_distribution(
    self,
    *,
    recovery: Recovery,
    transform: SupportTransform,
    quantile_inversion: QuantileInversionConfig | None,
    support_epsilon: float,
    device: torch.device,
    inference_chunk_size: int,
    random_state: int,
  ) -> ConditionalDistribution: ...

  def warn_if_outside_documented_range(self, n_samples: int) -> None: ...


@dataclass(frozen=True)
class _TabPFNProvider:
  config: TabPFNConfig
  name: str = "tabpfn"
  supported_recoveries: frozenset[Recovery] = frozenset(
    {Recovery.NATIVE_DISTRIBUTION, Recovery.QUANTILE_INVERSION}
  )

  @property
  def model_id(self) -> str:
    return self.config.model_version

  def warn_if_outside_documented_range(self, n_samples: int) -> None:
    maximum = 50_000 if self.config.model_version == "v2.5" else 100_000
    if n_samples > maximum:
      warnings.warn(
        f"provider=tabpfn model_id={self.model_id}: observed n={n_samples}; "
        f"documented range is n<={maximum}.",
        FoundationModelRangeWarning,
        stacklevel=3,
      )

  def create_distribution(
    self,
    *,
    recovery: Recovery,
    transform: SupportTransform,
    quantile_inversion: QuantileInversionConfig | None,
    support_epsilon: float,
    device: torch.device,
    inference_chunk_size: int,
    random_state: int,
  ) -> ConditionalDistribution:
    try:
      from tabpfn.constants import ModelVersion
    except ImportError as exc:
      raise MissingProviderDependencyError(
        "TabPFN is unavailable. Install npcc[tabpfn], then make the model "
        "available in the local cache or configure TabPFN authentication."
      ) from exc
    from npcc.core.tabpfn_backend import (
      _NativeTabPFNDistribution,
      _TabPFNQuantileDistribution,
    )

    try:
      version = ModelVersion(self.config.model_version)
    except ValueError as exc:
      raise ProviderConfigurationError(
        f"Unknown TabPFN model version: {self.config.model_version!r}."
      ) from exc
    kwargs = dict(self.config.extra_regressor_kwargs)
    kwargs["random_state"] = random_state
    kwargs["ignore_pretraining_limits"] = self.config.ignore_pretraining_limits
    if self.config.n_estimators is not None:
      kwargs["n_estimators"] = self.config.n_estimators
    common = {
      "transform": transform.value,
      "device": device,
      "batch_size": inference_chunk_size,
      "model_kwargs": kwargs,
      "model_version": version,
    }
    if recovery is Recovery.NATIVE_DISTRIBUTION:
      return cast(
        ConditionalDistribution,
        _NativeTabPFNDistribution(eps=support_epsilon, **common),
      )
    assert quantile_inversion is not None
    return cast(
      ConditionalDistribution,
      _TabPFNQuantileDistribution(
        config=quantile_inversion,
        support_epsilon=support_epsilon,
        **common,
      ),
    )


@dataclass(frozen=True)
class _TabICLProvider:
  config: TabICLConfig
  name: str = "tabicl"
  supported_recoveries: frozenset[Recovery] = frozenset(
    {Recovery.QUANTILE_INVERSION}
  )

  @property
  def model_id(self) -> str:
    return self.config.checkpoint

  def warn_if_outside_documented_range(self, n_samples: int) -> None:
    if n_samples < 300 or n_samples > 48_000:
      warnings.warn(
        f"provider=tabicl model_id={self.model_id}: observed n={n_samples}; "
        "documented range is 300<=n<=48000.",
        FoundationModelRangeWarning,
        stacklevel=3,
      )

  def create_distribution(
    self,
    *,
    recovery: Recovery,
    transform: SupportTransform,
    quantile_inversion: QuantileInversionConfig | None,
    support_epsilon: float,
    device: torch.device,
    inference_chunk_size: int,
    random_state: int,
  ) -> ConditionalDistribution:
    if recovery is not Recovery.QUANTILE_INVERSION:
      raise UnsupportedRecoveryError(
        "TabICL supports only recovery='quantile_inversion'."
      )
    from npcc.core.tabicl_backend import _TabICLQuantileDistribution

    assert quantile_inversion is not None
    return cast(
      ConditionalDistribution,
      _TabICLQuantileDistribution(
        transform=transform.value,
        config=quantile_inversion,
        support_epsilon=support_epsilon,
        device=device,
        batch_size=inference_chunk_size,
        provider_config=self.config,
        random_state=random_state,
      ),
    )


def resolve_provider(
  provider: str | FoundationModelProvider,
  config: TabPFNConfig | TabICLConfig | None,
) -> FoundationModelProvider:
  """Resolve a built-in provider or validate a custom provider object."""
  if not isinstance(provider, str):
    if config is not None:
      raise ProviderConfigurationError(
        "provider_config must be omitted for a custom provider."
      )
    return provider
  if provider == "tabpfn":
    if not isinstance(config, TabPFNConfig):
      raise ProviderConfigurationError(
        "provider='tabpfn' requires an explicit TabPFNConfig."
      )
    return _TabPFNProvider(config)
  if provider == "tabicl":
    if not isinstance(config, TabICLConfig):
      raise ProviderConfigurationError(
        "provider='tabicl' requires an explicit TabICLConfig."
      )
    return _TabICLProvider(config)
  raise ProviderConfigurationError(f"Unknown provider: {provider!r}.")
