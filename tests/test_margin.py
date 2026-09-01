"""Tests for the backend-backed pyvinecopulib margin."""

from __future__ import annotations

import copy
import numpy as np
import pytest
import torch
from pyvinecopulib.core import MarginBase

from npcc.core.margin import BackendMargin


def make_margin(
  register_uniform_backends: None,
) -> BackendMargin:
  return BackendMargin(
    backend="uniform-native",
    device="cpu",
  )


def test_backend_margin_subclasses_margin_base(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)

  assert isinstance(margin, MarginBase)
  assert margin.supports_covariates is True
  assert margin.supported_var_types == ("c",)
  assert margin.var_type == "c"
  assert margin.support == (float("-inf"), float("inf"))
  assert margin.is_fitted is False
  assert margin.family_name == "uniform-native"


@pytest.mark.parametrize(
  ("x", "expected_width"),
  [
    (None, 1),
    (np.ones(8), 1),
    (np.ones((8, 3)), 3),
  ],
)
def test_fit_constructs_expected_feature_matrix(
  register_uniform_backends: None,
  monkeypatch: pytest.MonkeyPatch,
  x: np.ndarray | None,
  expected_width: int,
) -> None:
  margin = make_margin(register_uniform_backends)
  observed_shapes: list[tuple[int, ...]] = []
  original = margin._distribution._fit_model

  def record_fit(w: torch.Tensor, z: torch.Tensor) -> None:
    observed_shapes.append(tuple(w.shape))
    original(w, z)

  monkeypatch.setattr(margin._distribution, "_fit_model", record_fit)

  y = np.linspace(-1.0, 1.0, 8)
  result = margin.fit(y, x=x)

  assert result is margin
  assert margin.is_fitted is True
  assert observed_shapes == [(8, expected_width)]


def test_identity_scale_distribution_operations(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)
  margin.fit(np.linspace(-1.0, 1.0, 20))

  y = np.array([-1.0, 0.0, 1.0])
  p = np.array([0.25, 0.5, 0.75])

  np.testing.assert_allclose(margin.pdf(y), 0.25)
  np.testing.assert_allclose(margin.cdf(y), p)
  np.testing.assert_allclose(margin.icdf(p), y)


def test_torch_input_returns_torch(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)
  y = torch.linspace(-1.0, 1.0, 20, dtype=torch.float64)
  margin.fit(y)

  evaluation = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
  out = margin.cdf(evaluation)

  assert isinstance(out, torch.Tensor)
  torch.testing.assert_close(
    out,
    torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64),
  )


def test_evaluation_before_fit_raises(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)

  with pytest.raises(RuntimeError, match="margin is not fitted"):
    margin.pdf(np.array([0.0]))


def test_rejects_mismatched_rows(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)

  with pytest.raises(ValueError, match="same number of rows"):
    margin.fit(
      np.linspace(-1.0, 1.0, 8),
      x=np.ones((7, 2)),
    )


def test_rejects_mixed_array_namespaces(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)

  with pytest.raises(TypeError, match="same array namespace"):
    margin.fit(
      torch.linspace(-1.0, 1.0, 8),
      x=np.ones((8, 2)),
    )


def test_rejects_weights(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)

  with pytest.raises(TypeError, match="does not support observation weights"):
    margin.fit(
      np.linspace(-1.0, 1.0, 8),
      weights=np.ones(8),
    )


def test_inherited_margin_operations(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)
  margin.fit(np.linspace(-1.0, 1.0, 20))

  y = np.array([-1.0, 0.0, 1.0])
  expected_logpdf = np.full(3, np.log(0.25))

  np.testing.assert_allclose(margin.logpdf(y), expected_logpdf)
  np.testing.assert_allclose(margin.cdf_left(y), margin.cdf(y))
  assert margin.loglik(y) == pytest.approx(expected_logpdf.sum())


def test_cdf_icdf_round_trip_with_covariates(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)
  train_y = np.linspace(-1.0, 1.0, 20)
  train_x = np.column_stack([train_y, train_y**2])
  margin.fit(train_y, x=train_x)

  p = np.array([0.2, 0.5, 0.8])
  x = np.array(
    [
      [-1.0, 1.0],
      [0.0, 0.0],
      [1.0, 1.0],
    ]
  )

  quantiles = margin.icdf(p, x=x)

  np.testing.assert_allclose(margin.cdf(quantiles, x=x), p)


@pytest.mark.parametrize(
  "invalid",
  [
    np.array([-0.1, 0.5]),
    np.array([0.5, 1.1]),
    np.array([0.5, np.nan]),
    np.array([0.5, np.inf]),
  ],
)
def test_icdf_rejects_invalid_probabilities(
  register_uniform_backends: None,
  invalid: np.ndarray,
) -> None:
  margin = make_margin(register_uniform_backends)
  margin.fit(np.linspace(-1.0, 1.0, 20))

  with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
    margin.icdf(invalid)


def test_sample_without_covariates_returns_torch(
  register_uniform_backends: None,
) -> None:
  margin = make_margin(register_uniform_backends)
  margin.fit(np.linspace(-1.0, 1.0, 20))

  first = margin.sample(10, seeds=[42])
  second = margin.sample(10, seeds=[42])

  assert isinstance(first, torch.Tensor)
  torch.testing.assert_close(first, second)
  assert torch.all((first >= -2.0) & (first <= 2.0))


@pytest.mark.parametrize("array_type", ["numpy", "torch"])
def test_conditional_sample_preserves_covariate_type(
  register_uniform_backends: None,
  array_type: str,
) -> None:
  margin = make_margin(register_uniform_backends)
  train_y = np.linspace(-1.0, 1.0, 20)
  train_x = np.ones((20, 2))
  margin.fit(train_y, x=train_x)

  x_np = np.ones((5, 2))
  x = torch.as_tensor(x_np) if array_type == "torch" else x_np
  result = margin.sample(5, x=x, seeds=[42])

  if array_type == "torch":
    assert isinstance(result, torch.Tensor)
  else:
    assert isinstance(result, np.ndarray)

  assert result.shape == (5,)


def test_copied_prototypes_have_independent_backends(
  register_uniform_backends: None,
) -> None:
  first = make_margin(register_uniform_backends)
  second = copy.deepcopy(first)

  assert first is not second
  assert first._distribution is not second._distribution

  first.fit(np.linspace(-1.0, 1.0, 20))

  assert first.is_fitted is True
  assert second.is_fitted is False
