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

from collections.abc import Iterator

import numpy as np
import pytest
import torch

pytest.importorskip("tabpfn")

from npcc.core.margin import (  # noqa: E402
  ConditionalMargin,
)
from npcc.core.margin_quantile_table import (  # noqa: E402
  QuantileTableDistribution1D,
)

# Both TabPFN backends build their regressor through
# ``npcc.core.backends.tabpfn_common.make_tabpfn_regressor``, so that is the
# single symbol we patch.
_TABPFN_REGRESSOR_TARGETS = (
  "npcc.core.backends.tabpfn_common.TabPFNRegressor",
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


# ---------------------------------------------------------------------------
# Hermetic, TabPFN-free backends (no monkeypatch): same Z ~ Uniform(-2, 2)
# ground truth as the fakes above, so ``uniform_density_y`` still applies.
# Used to prove the registry and pluggable-backend extension point without
# touching TabPFN.
# ---------------------------------------------------------------------------


class _UniformQuantileBackend(QuantileTableDistribution1D):
  """Closed-form quantile backend for Z ~ Uniform(-2, 2); no external model."""

  Q_LO: float = -2.0
  Q_HI: float = 2.0

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    return None

  def _predict_quantiles(
    self, x: torch.Tensor, alphas: torch.Tensor
  ) -> torch.Tensor:
    row = self.Q_LO + (self.Q_HI - self.Q_LO) * alphas
    return row.unsqueeze(0).expand(x.shape[0], -1).clone()


class _UniformNativeBackend(ConditionalMargin):
  """Closed-form native backend for Z ~ Uniform(-2, 2) (non-criterion path)."""

  Q_LO: float = -2.0
  Q_HI: float = 2.0

  def _fit_model(self, x: torch.Tensor, z: torch.Tensor) -> None:
    return None

  def _pdf_z(self, z: torch.Tensor) -> torch.Tensor:
    inside = (z > self.Q_LO) & (z < self.Q_HI)
    return torch.where(
      inside,
      torch.full_like(z, 1.0 / (self.Q_HI - self.Q_LO)),
      torch.zeros_like(z),
    )

  def _cdf_z(self, z: torch.Tensor) -> torch.Tensor:
    return torch.clamp((z - self.Q_LO) / (self.Q_HI - self.Q_LO), 0.0, 1.0)

  def pdf(
    self,
    y: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    del x, batch_size
    self._check_fitted()
    y_t = y.reshape(-1)
    out = self._pdf_z(self._transform_y(y_t)) * self._jacobian_inverse(y_t)
    return out

  def cdf(
    self,
    y: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    del x, batch_size
    self._check_fitted()
    return self._cdf_z(self._transform_y(y.reshape(-1)))

  def icdf(
    self,
    p: torch.Tensor,
    /,
    *,
    x: torch.Tensor | None = None,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    del x, batch_size
    self._check_fitted()
    z = self.Q_LO + (self.Q_HI - self.Q_LO) * p.reshape(-1)
    return self._inverse_transform(z)

  def pdf_grid(
    self,
    y_grid: torch.Tensor,
    /,
    *,
    x: torch.Tensor,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    del batch_size
    self._check_fitted()
    y_t = y_grid.reshape(-1)
    row = self._pdf_z(self._transform_y(y_t)) * self._jacobian_inverse(y_t)
    return row.unsqueeze(0).expand(x.shape[0], -1).clone()

  def cdf_grid(
    self,
    y_grid: torch.Tensor,
    /,
    *,
    x: torch.Tensor,
    batch_size: int | None = None,
  ) -> torch.Tensor:
    del batch_size
    self._check_fitted()
    row = self._cdf_z(self._transform_y(y_grid.reshape(-1)))
    return row.unsqueeze(0).expand(x.shape[0], -1).clone()


@pytest.fixture
def register_uniform_backends() -> Iterator[None]:
  """Register the hermetic ``uniform-quantile`` / ``uniform-native`` backends."""
  from npcc.core import registry

  registry.register_backend(
    "uniform-quantile",
    lambda *, transform, quantile_table_config, eps, device, batch_size, **kw: (
      _UniformQuantileBackend(
        transform=transform,
        quantile_table_config=quantile_table_config,
        eps=eps,
        device=device,
        batch_size=batch_size,
      )
    ),
  )
  registry.register_backend(
    "uniform-native",
    lambda *, transform, quantile_table_config, eps, device, batch_size, **kw: (
      _UniformNativeBackend(
        transform=transform,
        eps=eps,
        device=device,
        batch_size=batch_size,
      )
    ),
  )
  yield
  registry._REGISTRY.pop("uniform-quantile", None)
  registry._REGISTRY.pop("uniform-native", None)
