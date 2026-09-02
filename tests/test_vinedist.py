"""Tests for original-scale Rosenblatt vine distributions."""

from __future__ import annotations

import numpy as np
import torch
import pytest
from pyvinecopulib import RVineStructure
from pyvinecopulib.core import Vinedist

from npcc.core._common import TensorLike
from npcc.core.margin import BackendMargin
from npcc.core.vinedist import RosenblattVinedist


def make_structure() -> RVineStructure:
  return RVineStructure.from_order([1, 2, 3])


def make_data() -> np.ndarray:
  rng = np.random.default_rng(42)
  return rng.uniform(-1.0, 1.0, size=(40, 3))


def fit_distribution(
  register_uniform_backends: None,
  *,
  pair_backend: str = "uniform-native",
) -> RosenblattVinedist:
  return RosenblattVinedist.from_data(
    make_data(),
    structure=make_structure(),
    margin_backend="uniform-native",
    pair_backend=pair_backend,
    device="cpu",
  )


def test_rosenblatt_vine_dist_subclasses_vinedist() -> None:
  assert issubclass(RosenblattVinedist, Vinedist)


def test_from_data_fits_one_independent_backend_margin_per_column(
  register_uniform_backends: None,
) -> None:
  dist = fit_distribution(register_uniform_backends)

  assert dist.dim == 3
  assert len(dist.margins) == 3
  assert all(
    isinstance(margin, BackendMargin) and margin.is_fitted
    for margin in dist.margins
  )

  assert len({id(margin) for margin in dist.margins}) == 3
  assert (
    len(
      {
        id(margin._distribution)
        for margin in dist.margins
        if isinstance(margin, BackendMargin)
      }
    )
    == 3
  )


def test_marginal_cdf_is_probability_integral_transform(
  register_uniform_backends: None,
) -> None:
  dist = fit_distribution(register_uniform_backends)
  y = np.array(
    [
      [-1.0, 0.0, 1.0],
      [0.5, -0.5, 0.0],
    ]
  )

  result = dist.marginal_cdf(y)
  expected = (y + 2.0) / 4.0

  assert isinstance(result, np.ndarray)
  np.testing.assert_allclose(result, expected)


def test_margin_and_pair_backends_are_independent_settings(
  register_uniform_backends: None,
) -> None:
  dist = fit_distribution(
    register_uniform_backends,
    pair_backend="uniform-quantile",
  )

  assert all(
    isinstance(margin, BackendMargin) and margin.backend == "uniform-native"
    for margin in dist.margins
  )
  assert all(
    pair.backend == "uniform-quantile"
    for row in dist.copula.pair_copulas
    for pair in row
  )


def test_from_data_requires_structure(
  register_uniform_backends: None,
) -> None:
  with pytest.raises(ValueError, match="structure is required"):
    RosenblattVinedist.from_data(
      make_data(),
      margin_backend="uniform-native",
      pair_backend="uniform-native",
      device="cpu",
    )


def test_from_data_rejects_custom_margins(
  register_uniform_backends: None,
) -> None:
  with pytest.raises(NotImplementedError, match="Custom margins"):
    RosenblattVinedist.from_data(
      make_data(),
      structure=make_structure(),
      margins=BackendMargin(
        backend="uniform-native",
        device="cpu",
      ),
      margin_backend="uniform-native",
      pair_backend="uniform-native",
      device="cpu",
    )


def test_from_data_rejects_controls(
  register_uniform_backends: None,
) -> None:
  with pytest.raises(NotImplementedError, match="controls"):
    RosenblattVinedist.from_data(
      make_data(),
      structure=make_structure(),
      controls=object(),
      margin_backend="uniform-native",
      pair_backend="uniform-native",
      device="cpu",
    )


def test_from_data_rejects_weights(
  register_uniform_backends: None,
) -> None:
  with pytest.raises(NotImplementedError, match="weights"):
    RosenblattVinedist.from_data(
      make_data(),
      structure=make_structure(),
      weights=np.ones(40),
      margin_backend="uniform-native",
      pair_backend="uniform-native",
      device="cpu",
    )


def test_from_data_rejects_names(
  register_uniform_backends: None,
) -> None:
  with pytest.raises(NotImplementedError, match="Variable names"):
    RosenblattVinedist.from_data(
      make_data(),
      structure=make_structure(),
      names=["a", "b", "c"],
      margin_backend="uniform-native",
      pair_backend="uniform-native",
      device="cpu",
    )


def test_joint_pdf_matches_sklar_factorization(
  register_uniform_backends: None,
) -> None:
  dist = fit_distribution(register_uniform_backends)
  y = np.array(
    [
      [-0.5, 0.0, 0.5],
      [0.5, -0.5, 0.0],
    ]
  )

  u = dist.marginal_cdf(y)
  copula_density = dist.copula.pdf(u)
  expected = copula_density * 0.25**3

  np.testing.assert_allclose(dist.pdf(y), expected)


@pytest.mark.parametrize("array_type", ["numpy", "torch"])
def test_conditional_fit_and_pdf_preserve_array_namespace(
  register_uniform_backends: None,
  array_type: str,
) -> None:
  y_np = make_data()
  x_np = np.random.default_rng(43).normal(size=(40, 2))

  if array_type == "torch":
    y: TensorLike = torch.as_tensor(y_np)
    x: TensorLike = torch.as_tensor(x_np)
  else:
    y = y_np
    x = x_np

  dist = RosenblattVinedist.from_data(
    y,
    structure=make_structure(),
    x=x,
    margin_backend="uniform-native",
    pair_backend="uniform-native",
    device="cpu",
  )
  result = dist.pdf(y[:5], x=x[:5])

  if array_type == "torch":
    assert isinstance(result, torch.Tensor)
  else:
    assert isinstance(result, np.ndarray)

  assert result.shape == (5,)


def test_rosenblatt_round_trip(
  register_uniform_backends: None,
) -> None:
  dist = fit_distribution(register_uniform_backends)
  y = make_data()[:5]

  transformed = dist.rosenblatt(y)
  recovered = dist.inverse_rosenblatt(transformed)

  np.testing.assert_allclose(recovered, y, atol=1e-8, rtol=1e-8)


def test_unconditional_sample_returns_original_scale_torch_data(
  register_uniform_backends: None,
) -> None:
  dist = fit_distribution(register_uniform_backends)

  first = dist.sample(8, seeds=[42])
  second = dist.sample(8, seeds=[42])

  assert isinstance(first, torch.Tensor)
  assert first.shape == (8, 3)
  torch.testing.assert_close(first, second)
  assert torch.all((first >= -2.0) & (first <= 2.0))


def test_from_data_rejects_structure_dimension_mismatch(
  register_uniform_backends: None,
) -> None:
  with pytest.raises(ValueError, match="structure has dimension 3"):
    RosenblattVinedist.from_data(
      np.ones((10, 2)),
      structure=make_structure(),
      margin_backend="uniform-native",
      pair_backend="uniform-native",
      device="cpu",
    )


def test_from_data_rejects_covariate_row_mismatch(
  register_uniform_backends: None,
) -> None:
  with pytest.raises(ValueError, match="same number of rows"):
    RosenblattVinedist.from_data(
      np.ones((10, 3)),
      structure=make_structure(),
      x=np.ones((9, 2)),
      margin_backend="uniform-native",
      pair_backend="uniform-native",
      device="cpu",
    )


def test_from_data_rejects_mixed_array_namespaces(
  register_uniform_backends: None,
) -> None:
  with pytest.raises(TypeError, match="same array namespace"):
    RosenblattVinedist.from_data(
      torch.ones((10, 3)),
      structure=make_structure(),
      x=np.ones((10, 2)),
      margin_backend="uniform-native",
      pair_backend="uniform-native",
      device="cpu",
    )
