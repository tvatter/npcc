"""Opt-in, cache-only smoke tests for real foundation-model providers."""

from __future__ import annotations

import os

import numpy as np
import pytest

from npcc import (
  FoundationModelBicop,
  Recovery,
  TabICLConfig,
  TabPFNConfig,
)

pytestmark = pytest.mark.integration


def _require_integration() -> None:
  if os.environ.get("NPCC_RUN_MODEL_INTEGRATION") != "1":
    pytest.skip("set NPCC_RUN_MODEL_INTEGRATION=1 for cached-model tests")


def _training_data(n: int) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(317)
  return rng.uniform(0.05, 0.95, n), rng.uniform(0.05, 0.95, n)


def test_tabpfn_cached_model_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
  _require_integration()
  monkeypatch.setenv("HF_HUB_OFFLINE", "1")
  u, v = _training_data(300)
  model = FoundationModelBicop(
    provider="tabpfn",
    recovery=Recovery.NATIVE_DISTRIBUTION,
    provider_config=TabPFNConfig(),
  ).fit(u, v)
  assert np.isfinite(model.pdf(u[:2], v[:2])).all()


def test_tabicl_cached_model_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
  _require_integration()
  monkeypatch.setenv("HF_HUB_OFFLINE", "1")
  u, v = _training_data(300)
  model = FoundationModelBicop(
    provider="tabicl",
    recovery=Recovery.QUANTILE_INVERSION,
    provider_config=TabICLConfig(),
  ).fit(u, v)
  assert np.isfinite(model.pdf(u[:2], v[:2])).all()
