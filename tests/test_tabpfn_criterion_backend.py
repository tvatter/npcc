"""Tests for the criterion-based TabPFN conditional margin."""

from __future__ import annotations

import math
from typing import Literal, cast

import pytest
import torch

from npcc.core.backends.tabpfn_criterion import TabPFNCriterionBackend


def make_fitted_backend(
  patch_uniform: None,
  *,
  transform: Literal["identity", "logit", "probit"] = "logit",
  batch_size: int | None = None,
) -> TabPFNCriterionBackend:
  del patch_uniform
  backend = TabPFNCriterionBackend(
    transform=transform,
    device="cpu",
    batch_size=batch_size,
  )
  backend.fit(
    torch.full((10,), 0.5, dtype=torch.float64),
    x=torch.zeros((10, 1), dtype=torch.float64),
  )
  return backend


def uniform_density(y: torch.Tensor) -> torch.Tensor:
  return 0.25 / (y * (1.0 - y))


def standard_normal_cdf(z: torch.Tensor) -> torch.Tensor:
  return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def standard_normal_icdf(p: torch.Tensor) -> torch.Tensor:
  return math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)


class TestTabPFNCriterionBackend:
  def test_default_batch_size_on_cpu_is_400(self) -> None:
    backend = TabPFNCriterionBackend(device="cpu")

    assert backend.batch_size == 400

  def test_default_batch_size_on_cuda_is_2000(
    self,
    monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    backend = TabPFNCriterionBackend(device="cuda")

    assert backend.batch_size == 2000

  def test_custom_batch_size_overrides_device_default(self) -> None:
    backend = TabPFNCriterionBackend(device="cpu", batch_size=123)

    assert backend.batch_size == 123

  def test_nonpositive_constructor_batch_size_rejected(self) -> None:
    with pytest.raises(ValueError, match="batch_size"):
      TabPFNCriterionBackend(batch_size=0)

  def test_pdf_before_fit_raises(self, patch_uniform: None) -> None:
    backend = TabPFNCriterionBackend(device="cpu")

    with pytest.raises(RuntimeError, match="not fitted"):
      backend.pdf(
        torch.tensor([0.5], dtype=torch.float64),
        x=torch.zeros((1, 1), dtype=torch.float64),
      )

  @pytest.mark.parametrize("feature_dim", [1, 3])
  def test_fit_accepts_feature_matrix(
    self,
    patch_uniform: None,
    feature_dim: int,
  ) -> None:
    backend = TabPFNCriterionBackend(transform="logit", device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(0)
    x = torch.rand(
      (20, feature_dim),
      generator=generator,
      dtype=torch.float64,
    )
    y = 0.1 + 0.8 * torch.rand(
      20,
      generator=generator,
      dtype=torch.float64,
    )

    backend.fit(y, x=x)

    assert backend.model_ is not None

  def test_fit_rejects_one_dimensional_features(
    self,
    patch_uniform: None,
  ) -> None:
    """``(n,)`` says nothing about which axis is which, so it is refused.

    Matching on the shape clause rather than on "one row per observation":
    the row-count refusal contains that phrase too, so the looser regex
    would pass on the wrong branch.
    """
    backend = TabPFNCriterionBackend(transform="logit", device="cpu")
    values = torch.linspace(0.1, 0.9, 20, dtype=torch.float64)

    with pytest.raises(ValueError, match=r"must have shape \(n, p\)"):
      backend.fit(values, x=values)

  def test_fit_rejects_length_mismatch(self, patch_uniform: None) -> None:
    backend = TabPFNCriterionBackend(transform="logit", device="cpu")

    with pytest.raises(ValueError, match="one row per observation"):
      backend.fit(
        torch.zeros(6, dtype=torch.float64),
        x=torch.zeros((5, 1), dtype=torch.float64),
      )

  def test_pdf_rejects_length_mismatch(self, patch_uniform: None) -> None:
    backend = make_fitted_backend(patch_uniform)

    with pytest.raises(ValueError, match="one row per observation"):
      backend.pdf(
        torch.full((6,), 0.5, dtype=torch.float64),
        x=torch.zeros((5, 1), dtype=torch.float64),
      )

  def test_pdf_rejects_nonpositive_batch_size(
    self,
    patch_uniform: None,
  ) -> None:
    backend = make_fitted_backend(patch_uniform)

    with pytest.raises(ValueError, match="batch_size"):
      backend.pdf(
        torch.tensor([0.3, 0.5], dtype=torch.float64),
        x=torch.zeros((2, 1), dtype=torch.float64),
        batch_size=0,
      )

  def test_pdf_logit_jacobian_matches_analytic(
    self,
    patch_uniform: None,
  ) -> None:
    backend = make_fitted_backend(patch_uniform)
    y = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)
    x = torch.zeros((len(y), 1), dtype=torch.float64)

    result = backend.pdf(y, x=x)

    torch.testing.assert_close(
      result,
      uniform_density(y),
      atol=1e-6,
      rtol=1e-6,
    )

  def test_pdf_zero_outside_support(self, patch_uniform: None) -> None:
    backend = make_fitted_backend(patch_uniform)
    y = torch.tensor([0.02, 0.98], dtype=torch.float64)

    result = backend.pdf(
      y,
      x=torch.zeros((len(y), 1), dtype=torch.float64),
    )

    torch.testing.assert_close(result, torch.zeros_like(result))

  def test_pdf_respects_batch_size(self, patch_uniform: None) -> None:
    backend = make_fitted_backend(patch_uniform)
    generator = torch.Generator(device="cpu").manual_seed(0)
    y = 0.15 + 0.7 * torch.rand(
      17,
      generator=generator,
      dtype=torch.float64,
    )
    x = torch.zeros((len(y), 1), dtype=torch.float64)

    full = backend.pdf(y, x=x)
    chunked = backend.pdf(y, x=x, batch_size=4)

    torch.testing.assert_close(full, chunked)

  @pytest.mark.parametrize(
    ("method", "batch_size", "expected_calls"),
    [("pdf", None, 5), ("cdf", 10, 2)],
  )
  def test_inference_uses_resolved_batch_size(
    self,
    patch_uniform: None,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    batch_size: int | None,
    expected_calls: int,
  ) -> None:
    backend = make_fitted_backend(patch_uniform, batch_size=4)
    y = torch.linspace(0.15, 0.85, 17, dtype=torch.float64)
    x = torch.zeros((len(y), 1), dtype=torch.float64)
    calls = 0
    original = backend._predict_full

    def spy(features: torch.Tensor) -> object:
      nonlocal calls
      calls += 1
      return original(features)

    monkeypatch.setattr(backend, "_predict_full", spy)

    if method == "pdf":
      backend.pdf(y, x=x, batch_size=batch_size)
    else:
      backend.cdf(y, x=x, batch_size=batch_size)

    assert calls == expected_calls

  def test_pdf_grid_matches_pointwise_pdf(
    self,
    patch_uniform: None,
  ) -> None:
    backend = make_fitted_backend(patch_uniform)
    x = torch.zeros((3, 1), dtype=torch.float64)
    y_grid = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)

    grid = backend.pdf_grid(y_grid, x=x)
    tiled = backend.pdf(
      y_grid.repeat(x.shape[0]),
      x=x.repeat_interleave(len(y_grid), dim=0),
    ).reshape(x.shape[0], len(y_grid))

    assert grid.shape == (3, 3)
    torch.testing.assert_close(grid, tiled)

  def test_pdf_grid_before_fit_raises(self, patch_uniform: None) -> None:
    backend = TabPFNCriterionBackend(device="cpu")

    with pytest.raises(RuntimeError, match="not fitted"):
      backend.pdf_grid(
        torch.tensor([0.5], dtype=torch.float64),
        x=torch.zeros((1, 1), dtype=torch.float64),
      )

  def test_cdf_before_fit_raises(self, patch_uniform: None) -> None:
    backend = TabPFNCriterionBackend(device="cpu")

    with pytest.raises(RuntimeError, match="not fitted"):
      backend.cdf(
        torch.tensor([0.5], dtype=torch.float64),
        x=torch.zeros((1, 1), dtype=torch.float64),
      )

  def test_cdf_rejects_length_mismatch(self, patch_uniform: None) -> None:
    backend = make_fitted_backend(patch_uniform)

    with pytest.raises(ValueError, match="one row per observation"):
      backend.cdf(
        torch.full((6,), 0.5, dtype=torch.float64),
        x=torch.zeros((5, 1), dtype=torch.float64),
      )

  def test_cdf_logit_matches_analytic(self, patch_uniform: None) -> None:
    backend = make_fitted_backend(patch_uniform)
    y = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)

    result = backend.cdf(
      y,
      x=torch.zeros((len(y), 1), dtype=torch.float64),
    )
    expected = torch.clamp((torch.logit(y) + 2.0) / 4.0, 0.0, 1.0)

    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

  def test_cdf_clips_outside_support(self, patch_uniform: None) -> None:
    backend = make_fitted_backend(patch_uniform)
    y = torch.tensor([0.02, 0.98], dtype=torch.float64)

    result = backend.cdf(
      y,
      x=torch.zeros((len(y), 1), dtype=torch.float64),
    )

    torch.testing.assert_close(
      result,
      torch.tensor([0.0, 1.0], dtype=torch.float64),
    )

  def test_cdf_respects_batch_size(self, patch_uniform: None) -> None:
    backend = make_fitted_backend(patch_uniform)
    y = torch.linspace(0.15, 0.85, 17, dtype=torch.float64)
    x = torch.zeros((len(y), 1), dtype=torch.float64)

    full = backend.cdf(y, x=x)
    chunked = backend.cdf(y, x=x, batch_size=4)

    torch.testing.assert_close(full, chunked)

  def test_cdf_grid_matches_pointwise_cdf(
    self,
    patch_uniform: None,
  ) -> None:
    backend = make_fitted_backend(patch_uniform)
    x = torch.zeros((3, 1), dtype=torch.float64)
    y_grid = torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64)

    grid = backend.cdf_grid(y_grid, x=x)
    tiled = backend.cdf(
      y_grid.repeat(x.shape[0]),
      x=x.repeat_interleave(len(y_grid), dim=0),
    ).reshape(x.shape[0], len(y_grid))

    assert grid.shape == (3, 3)
    torch.testing.assert_close(grid, tiled)

  def test_cdf_grid_before_fit_raises(self, patch_uniform: None) -> None:
    backend = TabPFNCriterionBackend(device="cpu")

    with pytest.raises(RuntimeError, match="not fitted"):
      backend.cdf_grid(
        torch.tensor([0.5], dtype=torch.float64),
        x=torch.zeros((1, 1), dtype=torch.float64),
      )

  def test_cdf_monotone_in_y(self, patch_uniform: None) -> None:
    backend = make_fitted_backend(patch_uniform)
    y = torch.linspace(0.05, 0.95, 20, dtype=torch.float64)

    result = backend.cdf(
      y,
      x=torch.zeros((len(y), 1), dtype=torch.float64),
    )

    assert torch.all(torch.diff(result) >= -1e-9)

  def test_icdf_inverts_uniform_fake(self, patch_uniform: None) -> None:
    backend = make_fitted_backend(patch_uniform)
    probabilities = torch.tensor(
      [0.1, 0.3, 0.5, 0.7, 0.9],
      dtype=torch.float64,
    )

    result = backend.icdf(
      probabilities,
      x=torch.zeros((len(probabilities), 1), dtype=torch.float64),
    )
    expected = torch.sigmoid(-2.0 + 4.0 * probabilities)

    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

  def test_icdf_rejects_probability_outside_unit(
    self,
    patch_uniform: None,
  ) -> None:
    backend = make_fitted_backend(patch_uniform)

    with pytest.raises(ValueError, match="strictly inside"):
      backend.icdf(
        torch.tensor([0.5, 1.0], dtype=torch.float64),
        x=torch.zeros((2, 1), dtype=torch.float64),
      )

  def test_icdf_rejects_length_mismatch(self, patch_uniform: None) -> None:
    backend = make_fitted_backend(patch_uniform)

    with pytest.raises(ValueError, match="one row per observation"):
      backend.icdf(
        torch.tensor([0.5], dtype=torch.float64),
        x=torch.zeros((5, 1), dtype=torch.float64),
      )

  def test_identity_transform_returns_transformed_density(
    self,
    patch_uniform: None,
  ) -> None:
    backend = make_fitted_backend(patch_uniform, transform="identity")
    z = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)

    result = backend.pdf(
      z,
      x=torch.zeros((len(z), 1), dtype=torch.float64),
    )

    torch.testing.assert_close(result, torch.full_like(result, 0.25))

  def test_probit_transform_matches_analytic(
    self,
    patch_uniform: None,
  ) -> None:
    backend = make_fitted_backend(patch_uniform, transform="probit")
    y = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    x = torch.zeros((len(y), 1), dtype=torch.float64)
    z = standard_normal_icdf(y)
    phi_z = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)

    torch.testing.assert_close(
      backend.pdf(y, x=x),
      0.25 / phi_z,
      atol=1e-6,
      rtol=1e-6,
    )
    torch.testing.assert_close(
      backend.cdf(y, x=x),
      torch.clamp((z + 2.0) / 4.0, 0.0, 1.0),
      atol=1e-6,
      rtol=1e-6,
    )

    probabilities = torch.tensor(
      [0.1, 0.3, 0.5, 0.7, 0.9],
      dtype=torch.float64,
    )
    torch.testing.assert_close(
      backend.icdf(
        probabilities,
        x=torch.zeros(
          (len(probabilities), 1),
          dtype=torch.float64,
        ),
      ),
      standard_normal_cdf(-2.0 + 4.0 * probabilities),
      atol=1e-6,
      rtol=1e-6,
    )

  def test_unknown_transform_raises(self) -> None:
    invalid = cast("Literal['identity', 'logit', 'probit']", "exp")
    backend = TabPFNCriterionBackend(transform=invalid)

    with pytest.raises(ValueError, match="Unknown transform"):
      backend._transform_y(torch.tensor([0.5]))
