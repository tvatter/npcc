"""Tests for original-scale Rosenblatt vine distributions."""

from __future__ import annotations

import pytest
import torch
from pyvinecopulib import RVineStructure
from pyvinecopulib.core import VinedistBase

from npcc.core.controls import FitControlsRosenblattVinecop
from npcc.core.margin import ConditionalMargin
from npcc.core.registry import create_backend
from npcc.core.vinecop import RosenblattVinecop
from npcc.core.vinedist import RosenblattVinedist


def make_structure() -> RVineStructure:
  return RVineStructure.from_order([1, 2, 3])


def make_data() -> torch.Tensor:
  generator = torch.Generator(device="cpu")
  generator.manual_seed(42)
  return (
    2.0
    * torch.rand(
      (40, 3),
      generator=generator,
      dtype=torch.float64,
    )
    - 1.0
  )


def make_controls(
  backend: str = "uniform-native",
) -> FitControlsRosenblattVinecop:
  return FitControlsRosenblattVinecop(
    backend=backend,
    device="cpu",
  )


def make_distribution(
  controls: FitControlsRosenblattVinecop,
) -> RosenblattVinedist:
  structure = make_structure()
  margins = [
    create_backend(
      controls.backend,
      transform="identity",
      quantile_table_config=controls.quantile_table_config,
      eps=controls.eps,
      device=controls.device,
      batch_size=controls.batch_size,
      backend_kwargs=controls.backend_kwargs,
    )
    for _ in range(structure.dim)
  ]
  vinecop = RosenblattVinecop(
    None,
    structure,
    device=controls.device,
  )
  return RosenblattVinedist(vinecop, margins)


def fit_distribution(
  register_uniform_backends: None,
  *,
  backend: str = "uniform-native",
  x: torch.Tensor | None = None,
) -> RosenblattVinedist:
  controls = make_controls(backend)
  distribution = make_distribution(controls)
  return distribution.fit(make_data(), controls, x=x)


def test_rosenblatt_vinedist_subclasses_vinedist_base() -> None:
  assert issubclass(RosenblattVinedist, VinedistBase)


def test_constructor_binds_fixed_parts(
  register_uniform_backends: None,
) -> None:
  controls = make_controls()
  distribution = make_distribution(controls)

  assert distribution.dim == 3
  assert len(distribution.margins) == 3
  assert isinstance(distribution.vinecop, RosenblattVinecop)
  assert distribution.vinecop.structure == make_structure()


def test_fit_fits_one_independent_margin_per_column(
  register_uniform_backends: None,
) -> None:
  distribution = fit_distribution(register_uniform_backends)

  assert all(
    isinstance(margin, ConditionalMargin) and margin.is_fitted
    for margin in distribution.margins
  )
  assert len({id(margin) for margin in distribution.margins}) == 3


def test_fit_uses_one_backend_for_margins_and_pairs(
  register_uniform_backends: None,
) -> None:
  distribution = fit_distribution(
    register_uniform_backends,
    backend="uniform-quantile",
  )

  for margin in distribution.margins:
    assert isinstance(margin, ConditionalMargin)
    assert margin.family_name == "_UniformQuantileBackend"

  vinecop = distribution.vinecop
  assert isinstance(vinecop, RosenblattVinecop)
  assert all(
    pair.backend == "uniform-quantile"
    for row in vinecop.pair_copulas
    for pair in row
  )


def test_marginal_cdf_is_probability_integral_transform(
  register_uniform_backends: None,
) -> None:
  distribution = fit_distribution(register_uniform_backends)
  y = torch.tensor(
    [
      [-1.0, 0.0, 1.0],
      [0.5, -0.5, 0.0],
    ],
    dtype=torch.float64,
  )

  result = distribution.marginal_cdf(y)
  expected = (y + 2.0) / 4.0

  torch.testing.assert_close(result, expected)


def test_joint_pdf_matches_sklar_factorization(
  register_uniform_backends: None,
) -> None:
  distribution = fit_distribution(register_uniform_backends)
  y = torch.tensor(
    [
      [-0.5, 0.0, 0.5],
      [0.5, -0.5, 0.0],
    ],
    dtype=torch.float64,
  )

  u = distribution.marginal_cdf(y)
  copula_density = distribution.vinecop.pdf(u)
  expected = copula_density * 0.25**3

  torch.testing.assert_close(distribution.pdf(y), expected)


def test_conditional_fit_and_pdf_return_tensors(
  register_uniform_backends: None,
) -> None:
  generator = torch.Generator(device="cpu")
  generator.manual_seed(43)
  x = torch.randn(
    (40, 2),
    generator=generator,
    dtype=torch.float64,
  )
  distribution = fit_distribution(register_uniform_backends, x=x)

  result = distribution.pdf(make_data()[:5], x=x[:5])

  assert result.shape == (5,)
  assert result.dtype == torch.float64
  assert result.device.type == "cpu"


def test_rosenblatt_round_trip(
  register_uniform_backends: None,
) -> None:
  distribution = fit_distribution(register_uniform_backends)
  y = make_data()[:5]

  transformed = distribution.rosenblatt(y)
  recovered = distribution.inverse_rosenblatt(transformed)

  torch.testing.assert_close(recovered, y, atol=1e-8, rtol=1e-8)


def test_unconditional_sample_returns_original_scale_tensor(
  register_uniform_backends: None,
) -> None:
  distribution = fit_distribution(register_uniform_backends)

  first = distribution.sample(8, seeds=[42])
  second = distribution.sample(8, seeds=[42])

  assert first.shape == (8, 3)
  assert first.dtype == torch.float64
  assert first.device.type == "cpu"
  torch.testing.assert_close(first, second)
  assert torch.all((first >= -2.0) & (first <= 2.0))


def test_fit_rejects_non_tensor_data(
  register_uniform_backends: None,
) -> None:
  controls = make_controls()
  distribution = make_distribution(controls)

  with pytest.raises(TypeError, match="y must be a torch tensor"):
    distribution.fit([[0.0, 0.0, 0.0]], controls)


def test_fit_rejects_dimension_mismatch(
  register_uniform_backends: None,
) -> None:
  controls = make_controls()
  distribution = make_distribution(controls)
  y = torch.ones((10, 2), dtype=torch.float64)

  with pytest.raises(ValueError, match="has 3 variables"):
    distribution.fit(y, controls)


def test_fit_rejects_covariate_row_mismatch(
  register_uniform_backends: None,
) -> None:
  controls = make_controls()
  distribution = make_distribution(controls)
  y = torch.ones((10, 3), dtype=torch.float64)
  x = torch.ones((9, 2), dtype=torch.float64)

  with pytest.raises(ValueError, match="x must have one row per observation"):
    distribution.fit(y, controls, x=x)


def test_fit_rejects_weights(
  register_uniform_backends: None,
) -> None:
  controls = make_controls()
  distribution = make_distribution(controls)
  y = make_data()
  weights = torch.ones(y.shape[0], dtype=torch.float64)

  with pytest.raises(ValueError, match="cannot weight the copula half"):
    distribution.fit(y, controls, weights=weights)


def test_conditional_sample_returns_tensor(
  register_uniform_backends: None,
) -> None:
  generator = torch.Generator(device="cpu")
  generator.manual_seed(43)
  x = torch.randn(
    (40, 2),
    generator=generator,
    dtype=torch.float64,
  )
  distribution = fit_distribution(register_uniform_backends, x=x)

  result = distribution.sample(5, x=x[:5], seeds=[42])

  assert result.shape == (5, 3)
  assert result.dtype == torch.float64
  assert result.device.type == "cpu"


def test_vinedist_types_are_publicly_exported() -> None:
  import npcc

  assert npcc.ConditionalMargin is ConditionalMargin
  assert npcc.RosenblattVinedist is RosenblattVinedist
