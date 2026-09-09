"""Tests for the Torch helpers in ``npcc.core._placement`` / ``_interp`` / ``_trim``."""

from __future__ import annotations

import pytest
import torch

from npcc.core._interp import (
  gradient_1d,
  interp,
  interp_batched_fp,
  interp_batched_xp,
)
from npcc.core._placement import resolve_device
from npcc.core._trim import check_uv, logit


def test_resolve_explicit_cpu_device() -> None:
  assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_object() -> None:
  device = torch.device("cpu")

  assert resolve_device(device) == device


class TestCheckUv:
  def test_rejects_lower_boundary(self) -> None:
    u = torch.tensor([0.5, 0.0], dtype=torch.float64)
    v = torch.tensor([0.5, 0.5], dtype=torch.float64)

    with pytest.raises(ValueError, match="strictly inside"):
      check_uv(u, v, 1e-6)

  def test_rejects_upper_boundary(self) -> None:
    u = torch.tensor([0.5, 1.0], dtype=torch.float64)
    v = torch.tensor([0.5, 0.5], dtype=torch.float64)

    with pytest.raises(ValueError, match="strictly inside"):
      check_uv(u, v, 1e-6)

  def test_rejects_shape_mismatch(self) -> None:
    u = torch.tensor([0.5], dtype=torch.float64)
    v = torch.tensor([0.5, 0.5], dtype=torch.float64)

    with pytest.raises(ValueError, match="same shape"):
      check_uv(u, v, 1e-6)

  def test_clips_into_eps_band(self) -> None:
    u = torch.tensor([1e-9, 0.5], dtype=torch.float64)
    v = torch.tensor(
      [0.5, 1.0 - 1e-9],
      dtype=torch.float64,
    )

    actual_u, actual_v = check_uv(u, v, eps=1e-6)

    expected_u = torch.tensor(
      [1e-6, 0.5],
      dtype=torch.float64,
    )
    expected_v = torch.tensor(
      [0.5, 1.0 - 1e-6],
      dtype=torch.float64,
    )

    torch.testing.assert_close(actual_u, expected_u)
    torch.testing.assert_close(actual_v, expected_v)

  def test_flattens_coordinates(self) -> None:
    u = torch.tensor([[0.2], [0.4]], dtype=torch.float64)
    v = torch.tensor([[0.6], [0.8]], dtype=torch.float64)

    actual_u, actual_v = check_uv(u, v, eps=1e-6)

    assert actual_u.shape == (2,)
    assert actual_v.shape == (2,)


class TestLogit:
  def test_logit_at_half_is_zero(self) -> None:
    p = torch.tensor([0.5], dtype=torch.float64)

    result = logit(p)

    torch.testing.assert_close(result, torch.zeros_like(p))

  def test_logit_is_antisymmetric(self) -> None:
    p = torch.tensor([0.1, 0.4], dtype=torch.float64)

    torch.testing.assert_close(
      logit(p),
      -logit(1.0 - p),
    )


class TestTorchInterp:
  def test_interpolates_and_clamps_boundaries(self) -> None:
    x = torch.tensor(
      [-0.5, 0.5, 2.5],
      dtype=torch.float64,
    )
    xp = torch.tensor(
      [0.0, 1.0, 2.0],
      dtype=torch.float64,
    )
    fp = torch.tensor(
      [0.0, 10.0, 20.0],
      dtype=torch.float64,
    )

    result = interp(x, xp, fp)

    expected = torch.tensor(
      [0.0, 5.0, 20.0],
      dtype=torch.float64,
    )
    torch.testing.assert_close(result, expected)


class TestTorchInterpBatchedXp:
  def test_interpolates_row_specific_coordinates(self) -> None:
    x = torch.tensor(
      [0.5, 3.0],
      dtype=torch.float64,
    )
    xp = torch.tensor(
      [
        [0.0, 1.0, 2.0],
        [0.0, 2.0, 4.0],
      ],
      dtype=torch.float64,
    )
    fp = torch.tensor(
      [
        [0.0, 10.0, 20.0],
        [0.0, 20.0, 40.0],
      ],
      dtype=torch.float64,
    )

    result = interp_batched_xp(x, xp, fp)

    expected = torch.tensor(
      [5.0, 30.0],
      dtype=torch.float64,
    )
    torch.testing.assert_close(result, expected)


class TestTorchInterpBatchedFp:
  def test_interpolates_row_specific_values(self) -> None:
    x = torch.tensor(
      [0.5, 1.5],
      dtype=torch.float64,
    )
    xp = torch.tensor(
      [0.0, 1.0, 2.0],
      dtype=torch.float64,
    )
    fp = torch.tensor(
      [
        [0.0, 10.0, 20.0],
        [0.0, 100.0, 200.0],
      ],
      dtype=torch.float64,
    )

    result = interp_batched_fp(x, xp, fp)

    expected = torch.tensor(
      [5.0, 150.0],
      dtype=torch.float64,
    )
    torch.testing.assert_close(result, expected)


class TestTorchGradient1d:
  def test_uses_central_and_one_sided_differences(self) -> None:
    x = torch.tensor(
      [0.0, 1.0, 2.0],
      dtype=torch.float64,
    )
    y = x.square()

    result = gradient_1d(y, x)

    expected = torch.tensor(
      [1.0, 2.0, 3.0],
      dtype=torch.float64,
    )
    torch.testing.assert_close(result, expected)

  def test_supports_leading_dimensions(self) -> None:
    x = torch.tensor(
      [0.0, 1.0, 2.0],
      dtype=torch.float64,
    )
    y = torch.stack(
      [
        x,
        2.0 * x,
      ]
    )

    result = gradient_1d(y, x)

    expected = torch.tensor(
      [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
      ],
      dtype=torch.float64,
    )
    torch.testing.assert_close(result, expected)

  def test_rejects_single_coordinate(self) -> None:
    x = torch.tensor([0.0], dtype=torch.float64)
    y = torch.tensor([1.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="at least 2 points"):
      gradient_1d(y, x)
