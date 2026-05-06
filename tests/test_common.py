"""Tests for the private helpers in ``npcc._common``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from npcc._common import _as_2d, _check_uv, _logit


class TestAs2d:
  def test_reshapes_1d(self) -> None:
    out = _as_2d(np.arange(5, dtype=float))
    assert out.shape == (5, 1)

  def test_keeps_2d(self) -> None:
    out = _as_2d(np.zeros((3, 4)))
    assert out.shape == (3, 4)

  def test_rejects_3d(self) -> None:
    with pytest.raises(ValueError, match="1D or 2D"):
      _as_2d(np.zeros((2, 3, 4)))

  def test_accepts_torch_tensor(self) -> None:
    out = _as_2d(torch.zeros(5))
    assert out.shape == (5, 1)
    assert isinstance(out, torch.Tensor)


class TestCheckUv:
  def test_rejects_outside_unit(self) -> None:
    with pytest.raises(ValueError, match="strictly inside"):
      _check_uv(np.array([0.5, 0.0]), np.array([0.5, 0.5]), 1e-6)
    with pytest.raises(ValueError, match="strictly inside"):
      _check_uv(np.array([0.5, 1.0]), np.array([0.5, 0.5]), 1e-6)

  def test_rejects_shape_mismatch(self) -> None:
    with pytest.raises(ValueError, match="same shape"):
      _check_uv(np.array([0.5]), np.array([0.5, 0.5]), 1e-6)

  def test_clips_into_eps_band(self) -> None:
    u, v = _check_uv(
      np.array([1e-9, 0.5]), np.array([0.5, 1.0 - 1e-9]), eps=1e-6
    )
    assert float(u[0].item()) == pytest.approx(1e-6)
    assert float(v[1].item()) == pytest.approx(1.0 - 1e-6)


class TestLogit:
  def test_logit_at_half_is_zero(self) -> None:
    out = _logit(np.array([0.5]))
    assert float(out[0].item()) == pytest.approx(0.0)

  def test_logit_is_antisymmetric(self) -> None:
    p = np.array([0.1, 0.4])
    np.testing.assert_allclose(
      _logit(p).cpu().numpy(),
      -_logit(1.0 - p).cpu().numpy(),
      atol=1e-7,
    )
