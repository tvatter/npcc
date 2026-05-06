"""
Shared test fixtures and fakes for npcc tests.

Skips the entire suite when ``tabpfn`` cannot be imported (the package
is a hard runtime dependency, but importing it is gated on a working
auth setup, so we cushion the test layer).

Provides a ``_UniformQuantileRegressor`` fake that mimics
``TabPFNRegressor`` and supports both density-recovery paths:

- ``output_type="quantiles"`` returns the analytic quantile function of
  ``Z ~ Uniform(-2, 2)``: ``Q(alpha) = -2 + 4 * alpha``.
- ``output_type="full"`` returns ``{"logits", "criterion"}`` where
  ``criterion.pdf`` is the matching uniform density (``0.25`` on the
  support, ``0`` outside).

Both paths give the same density on the ``Y`` scale (with logit
transform) — namely ``f_Y(y) = 0.25 / (y * (1 - y))`` — so the two
recovery methods can be tested against the same analytic ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("tabpfn")

# Modules whose ``TabPFNRegressor`` symbol we patch in the fixtures.
# Both density classes import ``TabPFNRegressor`` directly, so each
# module attribute needs its own monkeypatch.
_TABPFN_REGRESSOR_TARGETS = (
  "npcc.tabpfn_density1d.TabPFNRegressor",
  "npcc.tabpfn_quantile_density1d.TabPFNRegressor",
)


class _UniformCriterion:
  """Fake ``criterion`` for ``Z ~ Uniform(-2, 2)``: linear CDF and constant PDF."""

  Q_LO: float = -2.0
  Q_HI: float = 2.0

  def pdf(self, logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    z_flat = z.reshape(-1)
    in_support = (z_flat > self.Q_LO) & (z_flat < self.Q_HI)
    return torch.where(
      in_support,
      torch.full_like(z_flat, 1.0 / (self.Q_HI - self.Q_LO)),
      torch.zeros_like(z_flat),
    )

  def cdf(self, logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    z_flat = z.reshape(-1)
    return torch.clamp((z_flat - self.Q_LO) / (self.Q_HI - self.Q_LO), 0.0, 1.0)

  def icdf(self, logits: torch.Tensor, left_prob: float) -> torch.Tensor:
    """Inverse CDF: ``Q(alpha) = -2 + 4 * alpha``.  Returns shape ``(n,)``."""
    n = logits.shape[0]
    return torch.full(
      (n,),
      self.Q_LO + (self.Q_HI - self.Q_LO) * float(left_prob),
      dtype=torch.float32,
      device=logits.device,
    )


class _UniformQuantileRegressor:
  """Mimics ``TabPFNRegressor`` with predictable Z ~ Uniform(-2, 2) outputs.

  Supports both ``output_type="quantiles"`` and ``output_type="full"``.
  """

  Q_LO: float = -2.0
  Q_HI: float = 2.0

  def __init__(self, **_: object) -> None:
    self.fitted_: bool = False

  @classmethod
  def create_default_for_version(
    cls, version: object, **overrides: object
  ) -> _UniformQuantileRegressor:
    return cls(**overrides)

  def fit(self, X: np.ndarray, y: np.ndarray) -> _UniformQuantileRegressor:
    self.fitted_ = True
    return self

  def predict(
    self,
    X: np.ndarray,
    *,
    output_type: str = "mean",
    quantiles: list[float] | None = None,
  ) -> object:
    n = X.shape[0]
    if output_type == "quantiles":
      if quantiles is None:
        raise ValueError("quantiles is required for output_type='quantiles'.")
      alphas = np.asarray(quantiles, dtype=float)
      qrow = self.Q_LO + (self.Q_HI - self.Q_LO) * alphas
      return np.broadcast_to(qrow[None, :], (n, len(alphas))).copy()
    if output_type == "full":
      return {
        "logits": torch.zeros((n, 8), dtype=torch.float32),
        "criterion": _UniformCriterion(),
      }
    raise ValueError(f"Unsupported output_type: {output_type}")


class _UniformQuantileRegressorTransposed(_UniformQuantileRegressor):
  """Same distribution, but quantile output has shape ``(n_q, n_obs)``."""

  def predict(
    self,
    X: np.ndarray,
    *,
    output_type: str = "mean",
    quantiles: list[float] | None = None,
  ) -> object:
    out = super().predict(X, output_type=output_type, quantiles=quantiles)
    if output_type == "quantiles":
      assert isinstance(out, np.ndarray)
      return out.T
    return out


class _BadShapeQuantileRegressor(_UniformQuantileRegressor):
  """Returns a 3-D quantile array to trigger the unexpected-shape error."""

  def predict(
    self,
    X: np.ndarray,
    *,
    output_type: str = "mean",
    quantiles: list[float] | None = None,
  ) -> object:
    if output_type == "quantiles":
      return np.zeros((2, 3, 4))
    return super().predict(X, output_type=output_type, quantiles=quantiles)


def _patch_all(monkeypatch: pytest.MonkeyPatch, fake: type) -> None:
  for target in _TABPFN_REGRESSOR_TARGETS:
    monkeypatch.setattr(target, fake)


@pytest.fixture
def patch_uniform(monkeypatch: pytest.MonkeyPatch) -> None:
  _patch_all(monkeypatch, _UniformQuantileRegressor)


@pytest.fixture
def patch_transposed(monkeypatch: pytest.MonkeyPatch) -> None:
  _patch_all(monkeypatch, _UniformQuantileRegressorTransposed)


@pytest.fixture
def patch_bad_shape(monkeypatch: pytest.MonkeyPatch) -> None:
  _patch_all(monkeypatch, _BadShapeQuantileRegressor)


def uniform_density_y(y: np.ndarray) -> np.ndarray:
  """Analytic ``f_Y(y)`` under logit transform with Z ~ Uniform(-2, 2)."""
  return 0.25 / (y * (1.0 - y))
