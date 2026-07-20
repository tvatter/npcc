"""Tests for provider configuration and recovery contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from npcc import (
  ProviderConfigurationError,
  Recovery,
  TabICLConfig,
  TabPFNConfig,
)
from npcc.core.providers import resolve_provider


def test_provider_capabilities_and_model_ids() -> None:
  tabpfn = resolve_provider("tabpfn", TabPFNConfig(model_version="v3"))
  tabicl = resolve_provider("tabicl", TabICLConfig(checkpoint="checkpoint"))
  assert tabpfn.supported_recoveries == frozenset(
    {Recovery.NATIVE_DISTRIBUTION, Recovery.QUANTILE_INVERSION}
  )
  assert tabicl.supported_recoveries == frozenset({Recovery.QUANTILE_INVERSION})
  assert tabpfn.model_id == "v3"
  assert tabicl.model_id == "checkpoint"


def test_wrong_provider_config_rejected() -> None:
  with pytest.raises(ProviderConfigurationError):
    resolve_provider("tabpfn", TabICLConfig())


def test_builtin_provider_config_is_required() -> None:
  with pytest.raises(ProviderConfigurationError):
    resolve_provider("tabpfn", None)


def test_extra_kwargs_reject_explicit_duplicate() -> None:
  with pytest.raises(ProviderConfigurationError, match="repeated"):
    TabPFNConfig(extra_regressor_kwargs={"n_estimators": 4})


def test_extra_kwargs_must_be_json_compatible() -> None:
  with pytest.raises(ProviderConfigurationError, match="JSON"):
    invalid = cast(dict[str, int], {"callback": object()})
    TabPFNConfig(extra_regressor_kwargs=invalid)


def test_configs_are_immutable_and_defensively_copied() -> None:
  source = {"foo": [1, 2]}
  config = TabPFNConfig(extra_regressor_kwargs=source)
  source["bar"] = 3
  assert "bar" not in config.extra_regressor_kwargs
  with pytest.raises(FrozenInstanceError):
    setattr(config, "model_version", "v2.5")


@pytest.mark.parametrize("n", [299, 48_001])
def test_tabicl_range_warning(n: int) -> None:
  provider = resolve_provider("tabicl", TabICLConfig())
  with pytest.warns(UserWarning, match="provider=tabicl"):
    provider.warn_if_outside_documented_range(n)
