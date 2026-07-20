"""Tests for the provider-neutral foundation-model bicop."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from npcc import (
  FoundationModelBicop,
  NotFittedError,
  ProviderConfigurationError,
  QuantileInversionConfig,
  Recovery,
  SupportTransform,
  TabICLConfig,
  TabPFNConfig,
  UnsupportedRecoveryError,
)
from npcc.core.foundation_model_bicop import _sinkhorn_project


def _model(
  recovery: Recovery = Recovery.QUANTILE_INVERSION,
  *,
  transform: SupportTransform = SupportTransform.LOGIT,
  quantile_inversion: QuantileInversionConfig | None = None,
) -> FoundationModelBicop:
  return FoundationModelBicop(
    provider="tabpfn",
    recovery=recovery,
    provider_config=TabPFNConfig(),
    transform=transform,
    quantile_inversion=quantile_inversion,
  )


def test_selection_is_explicit() -> None:
  with pytest.raises(TypeError):
    FoundationModelBicop()  # ty: ignore[missing-argument]


def test_tabicl_rejects_native_recovery() -> None:
  with pytest.raises(UnsupportedRecoveryError):
    FoundationModelBicop(
      provider="tabicl",
      recovery=Recovery.NATIVE_DISTRIBUTION,
      provider_config=TabICLConfig(),
    )


def test_native_rejects_quantile_configuration() -> None:
  with pytest.raises(ProviderConfigurationError):
    _model(
      Recovery.NATIVE_DISTRIBUTION,
      quantile_inversion=QuantileInversionConfig(),
    )


def test_defaults_are_precise() -> None:
  model = _model()
  assert model.random_state == 317
  assert model.transform is SupportTransform.LOGIT
  assert model.support_epsilon == pytest.approx(1e-6)
  assert model.quantile_inversion == QuantileInversionConfig()


def test_unfitted_access_raises_typed_error() -> None:
  with pytest.raises(NotFittedError):
    _model().pdf(np.array([0.5]), np.array([0.5]))


@pytest.mark.parametrize(
  "recovery",
  [Recovery.NATIVE_DISTRIBUTION, Recovery.QUANTILE_INVERSION],
)
def test_fit_and_density_paths_match_uniform_fake(
  patch_uniform: None, recovery: Recovery
) -> None:
  rng = np.random.default_rng(1)
  u = rng.uniform(0.15, 0.85, 40)
  v = rng.uniform(0.15, 0.85, 40)
  model = _model(recovery, transform=SupportTransform.IDENTITY).fit(u, v)
  assert model.is_fitted
  density = model.pdf(u[:5], v[:5])
  assert isinstance(density, np.ndarray)
  np.testing.assert_allclose(density, 0.25, atol=1e-10)


def test_torch_output_is_float64(patch_uniform: None) -> None:
  values = torch.linspace(0.2, 0.8, 20, dtype=torch.float32)
  model = _model(transform=SupportTransform.IDENTITY).fit(values, values)
  result = model.pdf(values[:4], values[:4])
  assert isinstance(result, torch.Tensor)
  assert result.dtype == torch.float64
  assert result.device == values.device


def test_mixed_inputs_rejected(patch_uniform: None) -> None:
  values = np.linspace(0.2, 0.8, 20)
  model = _model().fit(values, values)
  with pytest.raises(TypeError):
    model.pdf(torch.tensor(values[:3]), values[:3])


def test_transactional_failed_refit_preserves_model(
  patch_uniform: None, monkeypatch: pytest.MonkeyPatch
) -> None:
  values = np.linspace(0.2, 0.8, 20)
  model = _model().fit(values, values)
  previous = model._v_given_ux

  def fail(_: object, __: object) -> None:
    raise RuntimeError("fit failed")

  candidate = model._make_distribution(model.random_state)
  monkeypatch.setattr(candidate, "fit", fail)
  monkeypatch.setattr(model, "_make_distribution", lambda _: candidate)
  with pytest.raises(RuntimeError, match="fit failed"):
    model.fit(values, values)
  assert model._v_given_ux is previous
  assert model.is_fitted


def test_sinkhorn_scalings_normalize_grid() -> None:
  density = torch.tensor([[2.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
  weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
  r, s = _sinkhorn_project(density, weights, weights, 50)
  projected = r[:, None] * density * s[None, :]
  expected = torch.ones(2, dtype=torch.float64)
  torch.testing.assert_close(projected @ weights, expected)
  torch.testing.assert_close(projected.T @ weights, expected)
