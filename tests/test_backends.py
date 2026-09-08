"""Registry behaviour and TabPFN-free pluggable-backend tests.

These prove the backend seam end-to-end without touching TabPFN: a
hermetic quantile backend and a hermetic native backend (both
``Z ~ Uniform(-2, 2)``) are registered and driven through
``RosenblattBicop``.  They also lock the core speed invariant — grid
methods predict at most once per conditioning row, never per grid cell.
"""

from __future__ import annotations

import math

import pytest
import torch

from npcc.core.bicop import RosenblattBicop
from npcc.core.controls import (
  FitControlsRosenblattBicop,
  FitControlsRosenblattVinecop,
)
from npcc.core.errors import (
  InvalidBackendKwargsError,
  MissingBackendDependencyError,
  UnknownBackendError,
)
from npcc.core.quantile_table_distribution1d import QuantileGridConfig
from npcc.core.registry import (
  available_backends,
  create_backend,
  documented_n_range,
  validate_backend_kwargs,
)
from tests.conftest import (
  _UniformNativeBackend,
  _UniformQuantileBackend,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
  def test_builtins_registered(self) -> None:
    names = available_backends()
    for expected in (
      "tabpfn-criterion",
      "tabpfn-quantiles",
      "ngboost",
      "gbm",
      "tabicl",
    ):
      assert expected in names

  def test_unknown_backend_raises(self) -> None:
    with pytest.raises(UnknownBackendError, match="Unknown backend"):
      create_backend(
        "does-not-exist",
        transform="logit",
        config=QuantileGridConfig(),
        device="cpu",
        batch_size=None,
      )

  def test_missing_extra_gives_helpful_error(
    self, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    # Simulate the optional dependency being absent by blocking the
    # backend module import; the factory must re-raise with a pip hint.
    import sys

    monkeypatch.setitem(sys.modules, "npcc.core.backends.ngboost", None)
    with pytest.raises(
      MissingBackendDependencyError, match=r"pip install npcc\[ngboost\]"
    ):
      create_backend(
        "ngboost",
        transform="logit",
        config=QuantileGridConfig(),
        device="cpu",
        batch_size=None,
      )

  def test_unknown_backend_kwarg_is_rejected(self) -> None:
    # An unknown backend kwarg fails fast (typed error), not deep in the
    # backend constructor.
    with pytest.raises(InvalidBackendKwargsError, match="unknown backend"):
      create_backend(
        "tabpfn-criterion",
        transform="logit",
        config=QuantileGridConfig(),
        device="cpu",
        batch_size=None,
        backend_kwargs={"not_a_real_kwarg": 123},
      )


class TestBackendSpec:
  def test_curated_kwargs_accepted(self) -> None:
    validate_backend_kwargs("gbm", {"max_depth": 4, "subsample": 0.8})

  def test_unknown_kwarg_rejected(self) -> None:
    with pytest.raises(InvalidBackendKwargsError, match="unknown backend"):
      validate_backend_kwargs("gbm", {"max_dpeth": 4})

  def test_mistyped_kwarg_rejected(self) -> None:
    with pytest.raises(InvalidBackendKwargsError, match="max_depth"):
      validate_backend_kwargs("gbm", {"max_depth": 3.5})

  def test_nested_kwarg_must_be_table(self) -> None:
    with pytest.raises(InvalidBackendKwargsError, match="model_kwargs"):
      validate_backend_kwargs("tabpfn-criterion", {"model_kwargs": 5})

  def test_unrestricted_backend_only_checks_json(self) -> None:
    # `nori` has no allow-list; any JSON-shaped kwarg is accepted.
    validate_backend_kwargs("nori", {"anything": [1, 2, 3]})

  def test_non_json_value_rejected(self) -> None:
    with pytest.raises(InvalidBackendKwargsError, match="JSON"):
      validate_backend_kwargs("nori", {"bad": object()})

  def test_documented_n_range(self) -> None:
    assert documented_n_range("tabicl") == (300, 48000)
    assert documented_n_range("gbm") is None


# ---------------------------------------------------------------------------
# Hermetic end-to-end through RosenblattBicop (no TabPFN, no monkeypatch)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["uniform-quantile", "uniform-native"])
class TestHermeticBackendEndToEnd:
  def test_pdf_matches_analytic(
    self, register_uniform_backends: None, backend: str
  ) -> None:
    generator = torch.Generator().manual_seed(0)
    u = 0.2 + 0.6 * torch.rand(40, generator=generator)
    v = 0.2 + 0.6 * torch.rand(40, generator=generator)
    controls = FitControlsRosenblattBicop(
      backend=backend,
      device="cpu",
    )
    m = RosenblattBicop.from_data(
      torch.column_stack((u, v)),
      controls,
    )

    # Symmetric average of two identical Uniform(-2,2)-logit densities.
    y = torch.tensor([0.3, 0.5, 0.7])
    out = m.pdf(torch.column_stack((y, y)))
    expected = 0.25 / (y * (1.0 - y))
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

  def test_hfunc_in_unit_interval(
    self, register_uniform_backends: None, backend: str
  ) -> None:
    generator = torch.Generator().manual_seed(1)
    u = 0.2 + 0.6 * torch.rand(30, generator=generator)
    v = 0.2 + 0.6 * torch.rand(30, generator=generator)
    controls = FitControlsRosenblattBicop(
      backend=backend,
      device="cpu",
    )
    m = RosenblattBicop.from_data(
      torch.column_stack((u, v)),
      controls,
    )
    h = m.hfunc1(torch.tensor([[0.3, 0.4], [0.5, 0.5], [0.7, 0.6]]))
    assert ((h >= 0.0) & (h <= 1.0)).all()

  def test_pdf_grid_matches_pointwise(
    self, register_uniform_backends: None, backend: str
  ) -> None:
    generator = torch.Generator().manual_seed(2)
    u = 0.2 + 0.6 * torch.rand(30, generator=generator)
    v = 0.2 + 0.6 * torch.rand(30, generator=generator)
    controls = FitControlsRosenblattBicop(
      backend=backend,
      device="cpu",
    )
    m = RosenblattBicop.from_data(
      torch.column_stack((u, v)),
      controls,
    )
    u_g = torch.linspace(0.25, 0.75, 4)
    v_g = torch.linspace(0.3, 0.7, 5)
    grid = m.pdf_grid(u_g, v_g)
    u_tile = u_g.repeat_interleave(len(v_g))
    v_tile = v_g.repeat(len(u_g))
    expected = m.pdf(torch.column_stack((u_tile, v_tile))).reshape(
      len(u_g), len(v_g)
    )
    torch.testing.assert_close(grid, expected)

  def test_cdf_grid_available(
    self, register_uniform_backends: None, backend: str
  ) -> None:
    generator = torch.Generator().manual_seed(3)
    u = 0.2 + 0.6 * torch.rand(30, generator=generator)
    v = 0.2 + 0.6 * torch.rand(30, generator=generator)
    controls = FitControlsRosenblattBicop(
      backend=backend,
      device="cpu",
    )
    m = RosenblattBicop.from_data(
      torch.column_stack((u, v)),
      controls,
    )
    grid = torch.linspace(0.2, 0.8, 4)
    out = m.cdf_grid(grid, grid)
    assert out.shape == (4, 4)
    assert ((out >= 0.0) & (out <= 1.0)).all()

  def test_sinkhorn_runs(
    self, register_uniform_backends: None, backend: str
  ) -> None:
    generator = torch.Generator().manual_seed(4)
    u = 0.2 + 0.6 * torch.rand(30, generator=generator)
    v = 0.2 + 0.6 * torch.rand(30, generator=generator)
    controls = FitControlsRosenblattBicop(
      backend=backend,
      device="cpu",
      sinkhorn_iters=3,
      projection_grid_size=20,
    )
    m = RosenblattBicop.from_data(
      torch.column_stack((u, v)),
      controls,
    )
    out = m.pdf(
      torch.tensor(
        [[0.4, 0.4], [0.6, 0.6]],
        dtype=torch.float64,
      )
    )
    assert (out > 0).all()


# ---------------------------------------------------------------------------
# FAST invariant: grids predict once per conditioning row, never per cell.
# ---------------------------------------------------------------------------


class TestGridPredictsOncePerRow:
  def test_quantile_pdf_grid_predicts_per_row_chunk_not_per_cell(
    self, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    backend = _UniformQuantileBackend(transform="logit", batch_size=1000)
    backend.fit(
      torch.full((10,), 0.5, dtype=torch.float64),
      x=torch.zeros((10, 1), dtype=torch.float64),
    )

    calls = 0
    original = backend._predict_quantiles

    def _spy(w: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
      nonlocal calls
      calls += 1
      return original(w, alphas)

    monkeypatch.setattr(backend, "_predict_quantiles", _spy)

    n_w, n_y = 6, 25
    w = torch.zeros((n_w, 1), dtype=torch.float64)
    y_grid = torch.linspace(0.2, 0.8, n_y, dtype=torch.float64)
    backend.pdf_grid(y_grid, x=w)

    # One chunk (batch_size >= n_w) => exactly one prediction, regardless
    # of n_y.  A per-cell path would call n_w * n_y = 150 times.
    assert calls == math.ceil(n_w / backend.batch_size) == 1

  def test_quantile_cdf_grid_predicts_per_row_chunk(
    self, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    backend = _UniformQuantileBackend(transform="logit", batch_size=2)
    backend.fit(
      torch.full((10,), 0.5, dtype=torch.float64),
      x=torch.zeros((10, 1), dtype=torch.float64),
    )

    calls = 0
    original = backend._predict_quantiles

    def _spy(w: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
      nonlocal calls
      calls += 1
      return original(w, alphas)

    monkeypatch.setattr(backend, "_predict_quantiles", _spy)

    n_w, n_y = 5, 30
    w = torch.zeros((n_w, 1), dtype=torch.float64)
    y_grid = torch.linspace(0.2, 0.8, n_y, dtype=torch.float64)
    backend.cdf_grid(y_grid, x=w)
    # ceil(5 / 2) = 3 chunks, independent of n_y (not 5 * 30 = 150).
    assert calls == math.ceil(n_w / backend.batch_size) == 3


def test_native_backend_is_conditional_distribution() -> None:
  from npcc.core.margin import ConditionalMargin

  assert issubclass(_UniformNativeBackend, ConditionalMargin)


def test_bicop_from_data_accepts_fit_controls(
  register_uniform_backends: None,
) -> None:
  controls = FitControlsRosenblattBicop(
    backend="uniform-native",
    transform="logit",
    device="cpu",
    batch_size=17,
  )
  generator = torch.Generator().manual_seed(42)
  u = 0.2 + 0.6 * torch.rand(
    (30, 2),
    generator=generator,
    dtype=torch.float64,
  )

  pair = RosenblattBicop.from_data(u, controls)

  assert pair.backend == "uniform-native"
  assert pair.transform == "logit"
  assert pair._device == torch.device("cpu")
  assert pair.batch_size == 17
  assert pair.v_given_ux_.is_fitted
  assert pair.u_given_vx_.is_fitted


def test_bicop_accepts_vinecop_controls(
  register_uniform_backends: None,
) -> None:
  controls = FitControlsRosenblattVinecop(
    backend="uniform-native",
    device="cpu",
  )
  generator = torch.Generator().manual_seed(43)
  u = 0.2 + 0.6 * torch.rand(
    (30, 2),
    generator=generator,
    dtype=torch.float64,
  )

  pair = RosenblattBicop.from_data(u, controls)

  assert pair.backend == controls.backend
  assert pair.v_given_ux_.is_fitted
  assert pair.u_given_vx_.is_fitted


def test_fit_controls_replace_conditional_estimators(
  register_uniform_backends: None,
) -> None:
  pair = RosenblattBicop(
    backend="uniform-quantile",
    device="cpu",
  )
  old_forward = pair.v_given_ux_
  old_reverse = pair.u_given_vx_

  controls = FitControlsRosenblattBicop(
    backend="uniform-native",
    device="cpu",
  )
  generator = torch.Generator().manual_seed(44)
  u = 0.2 + 0.6 * torch.rand(
    (30, 2),
    generator=generator,
    dtype=torch.float64,
  )

  result = pair.fit(u, controls)

  assert result is pair
  assert pair.backend == "uniform-native"
  assert pair.v_given_ux_ is not old_forward
  assert pair.u_given_vx_ is not old_reverse
  assert pair.v_given_ux_.is_fitted
  assert pair.u_given_vx_.is_fitted


def test_fit_without_controls_retains_configuration(
  register_uniform_backends: None,
) -> None:
  pair = RosenblattBicop(
    backend="uniform-native",
    device="cpu",
    batch_size=23,
  )
  forward = pair.v_given_ux_
  reverse = pair.u_given_vx_

  generator = torch.Generator().manual_seed(45)
  u = 0.2 + 0.6 * torch.rand(
    (30, 2),
    generator=generator,
    dtype=torch.float64,
  )

  pair.fit(u)

  assert pair.backend == "uniform-native"
  assert pair.batch_size == 23
  assert pair.v_given_ux_ is forward
  assert pair.u_given_vx_ is reverse
