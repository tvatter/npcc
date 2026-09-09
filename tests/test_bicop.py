"""Tests for the Torch-native Rosenblatt pair-copula estimator."""

from __future__ import annotations

import numpy
import pytest
import torch
from pyvinecopulib.core import BicopBase, BicopLike

from npcc.core.backends.tabpfn_criterion import TabPFNCriterionBackend
from npcc.core.backends.tabpfn_quantile import TabPFNQuantileBackend
from npcc.core.bicop import RosenblattBicop, _sinkhorn_project
from npcc.core.controls import (
  FitControlsRosenblattBicop,
  FitControlsRosenblattVinecop,
)


def random_uv(n: int = 30, *, seed: int = 0) -> torch.Tensor:
  generator = torch.Generator(device="cpu").manual_seed(seed)
  return 0.15 + 0.7 * torch.rand(
    (n, 2),
    generator=generator,
    dtype=torch.float64,
  )


def make_controls(
  method: str = "criterion",
  *,
  transform: str = "logit",
  batch_size: int | None = None,
  sinkhorn_iters: int | None = None,
) -> FitControlsRosenblattBicop:
  return FitControlsRosenblattBicop(
    backend=f"tabpfn-{method}",
    transform=transform,  # ty: ignore[invalid-argument-type]
    device="cpu",
    batch_size=batch_size,
    sinkhorn_iters=sinkhorn_iters,
    projection_grid_size=21,
  )


def fit_bicop(
  patch_uniform: None,
  method: str = "criterion",
  *,
  x: torch.Tensor | None = None,
  transform: str = "logit",
  sinkhorn_iters: int | None = None,
) -> RosenblattBicop:
  del patch_uniform
  model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))
  return model.fit(
    random_uv(),
    make_controls(
      method,
      transform=transform,
      sinkhorn_iters=sinkhorn_iters,
    ),
    x=x,
  )


class TestRosenblattBicopConstruction:
  def test_implements_bicop_contract(self) -> None:
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))

    assert isinstance(model, BicopBase)
    assert isinstance(model, BicopLike)

  def test_default_backend_is_criterion(self) -> None:
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))

    assert model.backend == "tabpfn-criterion"
    assert model.transform == "logit"
    assert isinstance(model.v_given_ux_, TabPFNCriterionBackend)
    assert isinstance(model.u_given_vx_, TabPFNCriterionBackend)

  def test_fit_controls_replace_backend_configuration(
    self,
    patch_uniform: None,
  ) -> None:
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))
    controls = make_controls(
      "quantiles",
      transform="identity",
      batch_size=17,
    )

    model.fit(random_uv(), controls)

    assert model.backend == "tabpfn-quantiles"
    assert model.transform == "identity"
    assert model.batch_size == 17
    assert isinstance(model.v_given_ux_, TabPFNQuantileBackend)
    assert isinstance(model.u_given_vx_, TabPFNQuantileBackend)

  def test_default_batch_size_on_cpu_is_400(self) -> None:
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))

    assert model.batch_size == 400
    assert model.v_given_ux_.batch_size == 400


@pytest.mark.parametrize("method", ["criterion", "quantiles"])
class TestRosenblattBicopValidation:
  @pytest.mark.parametrize(
    "uv",
    [
      torch.tensor([0.2, 0.3], dtype=torch.float64),
      torch.ones((2, 1), dtype=torch.float64),
      torch.ones((2, 3), dtype=torch.float64),
    ],
  )
  def test_fit_rejects_invalid_shape(
    self,
    patch_uniform: None,
    method: str,
    uv: torch.Tensor,
  ) -> None:
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))

    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
      model.fit(uv, make_controls(method))

  def test_fit_rejects_covariate_length_mismatch(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))

    with pytest.raises(ValueError, match="one row per observation"):
      model.fit(
        random_uv(10),
        make_controls(method),
        x=torch.zeros((5, 2), dtype=torch.float64),
      )

  def test_pdf_rejects_covariate_length_mismatch(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)

    with pytest.raises(ValueError, match="one row per observation"):
      model.pdf(
        random_uv(10),
        x=torch.zeros((5, 2), dtype=torch.float64),
      )

  def test_fit_rejects_boundary_values(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    uv = torch.tensor(
      [[0.5, 0.3], [0.0, 0.4]],
      dtype=torch.float64,
    )
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))

    with pytest.raises(ValueError, match="strictly inside"):
      model.fit(uv, make_controls(method))


def test_sinkhorn_project_rejects_nonpositive_iterations() -> None:
  density = torch.ones((3, 3), dtype=torch.float64)
  weights = torch.ones(3, dtype=torch.float64)

  with pytest.raises(ValueError, match="n_iters must be positive"):
    _sinkhorn_project(density, weights, weights, 0)


def test_trapezoidal_weights_singleton_grid() -> None:
  grid = torch.tensor([0.25], dtype=torch.float64)

  result = RosenblattBicop._trapezoidal_weights(grid)

  torch.testing.assert_close(
    result,
    torch.tensor([1.0], dtype=torch.float64),
  )


@pytest.mark.parametrize("method", ["criterion", "quantiles"])
class TestRosenblattBicopEvaluation:
  def test_pdf_returns_positive_tensor(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)
    uv = random_uv(12, seed=1)

    result = model.pdf(uv)

    assert result.shape == (12,)
    assert result.dtype == torch.float64
    assert result.device.type == "cpu"
    assert torch.isfinite(result).all()
    assert torch.all(result > 0.0)

  def test_pdf_accepts_covariates(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    train_x = torch.zeros((30, 2), dtype=torch.float64)
    model = fit_bicop(patch_uniform, method, x=train_x)

    result = model.pdf(
      random_uv(8, seed=2),
      x=torch.zeros((8, 2), dtype=torch.float64),
    )

    assert result.shape == (8,)

  def test_log_pdf_matches_pdf(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)
    uv = random_uv(8, seed=3)

    torch.testing.assert_close(model.log_pdf(uv), torch.log(model.pdf(uv)))

  def test_loglik_matches_summed_log_pdf(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)
    uv = random_uv(8, seed=4)

    torch.testing.assert_close(model.loglik(uv), model.log_pdf(uv).sum())

  def test_pdf_grid_matches_pointwise_pdf(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)
    u_grid = torch.linspace(0.25, 0.75, 4, dtype=torch.float64)
    v_grid = torch.linspace(0.3, 0.7, 5, dtype=torch.float64)

    result = model.pdf_grid(u_grid, v_grid)
    uv = torch.column_stack(
      (
        u_grid.repeat_interleave(len(v_grid)),
        v_grid.repeat(len(u_grid)),
      )
    )
    expected = model.pdf(uv).reshape(len(u_grid), len(v_grid))

    assert result.shape == (4, 5)
    torch.testing.assert_close(result, expected)

  def test_pdf_grid_rejects_multiple_covariate_rows(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)
    grid = torch.linspace(0.2, 0.8, 4, dtype=torch.float64)

    with pytest.raises(ValueError, match=r"shape \(p,\) or \(1, p\)"):
      model.pdf_grid(
        grid,
        grid,
        x_row=torch.zeros((2, 1), dtype=torch.float64),
      )

  def test_hfuncs_return_probabilities(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)
    uv = random_uv(10, seed=5)

    first = model.hfunc1(uv)
    second = model.hfunc2(uv)

    assert first.shape == (10,)
    assert second.shape == (10,)
    assert torch.all((first > 0.0) & (first < 1.0))
    assert torch.all((second > 0.0) & (second < 1.0))

  def test_inverse_hfuncs_invert_forward_hfuncs(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)
    uv = random_uv(10, seed=6)

    first_input = torch.column_stack((uv[:, 0], model.hfunc1(uv)))
    second_input = torch.column_stack((model.hfunc2(uv), uv[:, 1]))

    torch.testing.assert_close(
      model.hinv1(first_input),
      uv[:, 1],
      atol=2e-2,
      rtol=2e-2,
    )
    torch.testing.assert_close(
      model.hinv2(second_input),
      uv[:, 0],
      atol=2e-2,
      rtol=2e-2,
    )

  def test_cdf_returns_probabilities(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)

    result = model.cdf(random_uv(6, seed=7), n_int=16)

    assert result.shape == (6,)
    assert torch.all((result >= 0.0) & (result <= 1.0))

  def test_cdf_rejects_small_integration_grid(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)

    with pytest.raises(ValueError, match="at least 2"):
      model.cdf(random_uv(2), n_int=1)

  def test_cdf_grid_matches_pointwise_cdf(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)
    grid = torch.linspace(0.25, 0.75, 4, dtype=torch.float64)

    result = model.cdf_grid(grid, grid, n_int=32)
    uv = torch.column_stack(
      (
        grid.repeat_interleave(len(grid)),
        grid.repeat(len(grid)),
      )
    )
    expected = model.cdf(uv, n_int=32).reshape(len(grid), len(grid))

    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("method", ["criterion", "quantiles"])
class TestRosenblattBicopSampling:
  def test_sample_is_seeded(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)

    first = model.sample(10, seeds=[42])
    second = model.sample(10, seeds=[42])

    assert first.shape == (10, 2)
    assert first.dtype == torch.float64
    assert first.device.type == "cpu"
    torch.testing.assert_close(first, second)

  def test_qrng_sample_is_seeded(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)

    first = model.sample(10, qrng=True, seeds=[42])
    second = model.sample(10, qrng=True, seeds=[42])

    torch.testing.assert_close(first, second)

  def test_sample_accepts_covariates(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    train_x = torch.zeros((30, 2), dtype=torch.float64)
    model = fit_bicop(patch_uniform, method, x=train_x)

    result = model.sample(
      8,
      x=torch.zeros((8, 2), dtype=torch.float64),
      seeds=[42],
    )

    assert result.shape == (8, 2)

  def test_flip_remains_unsupported(
    self,
    patch_uniform: None,
    method: str,
  ) -> None:
    model = fit_bicop(patch_uniform, method)

    with pytest.raises(NotImplementedError):
      model.flip()


class TestRosenblattBicopTau:
  def test_tau_is_deterministic_and_bounded(
    self,
    patch_uniform: None,
  ) -> None:
    model = fit_bicop(patch_uniform)

    first = model.tau(n=100, seeds=[42])
    second = model.tau(n=100, seeds=[42])

    assert -1.0 <= first <= 1.0
    assert first == pytest.approx(second)

  def test_tau_rejects_small_sample(self, patch_uniform: None) -> None:
    model = fit_bicop(patch_uniform)

    with pytest.raises(ValueError, match="at least 10"):
      model.tau(n=9)

  def test_tau_rejects_multiple_covariate_rows(
    self,
    patch_uniform: None,
  ) -> None:
    model = fit_bicop(patch_uniform)

    with pytest.raises(ValueError, match=r"shape \(p,\) or \(1, p\)"):
      model.tau(torch.zeros((2, 1), dtype=torch.float64))


class TestSinkhornProjection:
  def test_projection_preserves_shape_and_nonnegativity(
    self,
    patch_uniform: None,
  ) -> None:
    model = fit_bicop(patch_uniform, sinkhorn_iters=3)
    uv = random_uv(10, seed=8)

    result = model.pdf(uv)

    assert result.shape == (10,)
    assert torch.all(result >= 0.0)

  def test_projection_grid_is_cached(self, patch_uniform: None) -> None:
    model = fit_bicop(patch_uniform, sinkhorn_iters=3)

    assert model._u_grid_borders_ is not None
    assert model._v_grid_borders_ is not None

    first_u = model._u_grid_borders_
    first_v = model._v_grid_borders_
    model._get_grid_borders()

    assert model._u_grid_borders_ is not first_u
    assert model._v_grid_borders_ is not first_v

  def test_projection_grid_pdf_is_finite(self, patch_uniform: None) -> None:
    model = fit_bicop(patch_uniform, sinkhorn_iters=3)
    grid = torch.linspace(0.2, 0.8, 8, dtype=torch.float64)

    result = model.pdf_grid(grid, grid)

    assert result.shape == (8, 8)
    assert torch.isfinite(result).all()
    assert torch.all(result >= 0.0)


class TestPlacement:
  """Inputs are brought onto the estimator's dtype and device.

  ``PlacementMixin._prep`` infers a placement from arrays the object holds,
  and finds none on these estimators -- so it returned its argument untouched,
  and the half-placement written in its stead placed the copula arguments and
  left the covariates alone. On CUDA that met a host ``x`` with a device ``u``
  inside a ``column_stack``; on every device it let a float32 input stay
  float32 under a float64 contract.
  """

  def test_prep_normalizes_dtype_and_device(self, patch_uniform: None) -> None:
    model = fit_bicop(patch_uniform)

    placed = model._prep(numpy.asarray([[0.3, 0.4]], dtype=numpy.float32))

    assert placed.dtype is torch.float64
    assert placed.device == model._device

  @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
  def test_pdf_returns_float64_whatever_it_is_given(
    self, patch_uniform: None, dtype: torch.dtype
  ) -> None:
    model = fit_bicop(patch_uniform)
    uv = torch.tensor([[0.3, 0.4], [0.6, 0.7]], dtype=dtype)

    assert model.pdf(uv).dtype is torch.float64

  def test_covariates_are_placed_alongside_the_observations(
    self, patch_uniform: None
  ) -> None:
    """A float32 covariate matrix must not poison the feature matrix.

    `u` and `x` are concatenated, so placing only one of them is what a
    device mismatch -- and, on one device, a silent downcast -- comes from.
    """
    x = torch.zeros((30, 2), dtype=torch.float64)
    model = fit_bicop(patch_uniform, x=x)

    out = model.pdf(
      torch.tensor([[0.3, 0.4]], dtype=torch.float32),
      x=torch.zeros((1, 2), dtype=torch.float32),
    )

    assert out.dtype is torch.float64

  def test_the_inherited_plot_runs(self, patch_uniform: None) -> None:
    """``BicopBase.plot`` manufactures a NumPy grid and places it via ``_prep``.

    The one path that reaches these estimators with a foreign array type, and
    the reason ``_prep`` returning its argument untouched was survivable
    before: the density evaluation re-placed it. Nothing re-places the grid.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    model = fit_bicop(patch_uniform)
    model.plot(grid_size=8)

    matplotlib.pyplot.close("all")

  def test_cdf_grid_accumulates_in_float64(self, patch_uniform: None) -> None:
    """The cumulative integral used to land in a dtype-less ``torch.zeros``."""
    model = fit_bicop(patch_uniform)
    grid = torch.linspace(0.1, 0.9, 5, dtype=torch.float64)

    assert model.cdf_grid(grid, grid, n_int=8).dtype is torch.float64


class TestControlsContract:
  def test_tau_accepts_a_one_dimensional_covariate_row(
    self, patch_uniform: None
  ) -> None:
    """``(p,)`` is documented, and was rejected.

    The reshape tested ``x_row.dim == 1``, comparing a bound method to an
    int, so a 1-D row was never reshaped and then tripped the shape check.
    """
    x = torch.zeros((30, 1), dtype=torch.float64)
    model = fit_bicop(patch_uniform, x=x)

    flat = model.tau(torch.zeros(1, dtype=torch.float64), n=50)
    column = model.tau(torch.zeros((1, 1), dtype=torch.float64), n=50)

    assert flat == pytest.approx(column)

  def test_an_array_in_the_controls_slot_names_the_argument_order(
    self, patch_uniform: None
  ) -> None:
    """``fit(uv, x)`` binds covariates to ``controls``; say so."""
    del patch_uniform
    model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))

    with pytest.raises(TypeError, match="array where `controls` goes"):
      model.fit(
        random_uv(),
        # A tensor where controls go, on purpose: the guard under test.
        torch.zeros((30, 2), dtype=torch.float64),  # ty: ignore[invalid-argument-type]
      )

  def test_a_setting_that_cannot_be_honored_is_refused(self) -> None:
    """``ControlsLike`` asks a consumer to refuse, not to drop silently."""

    class ForeignControls:
      def to_dict(self) -> dict[str, object]:
        return {"backend": "tabpfn-criterion", "tree_criterion": "tau"}

    with pytest.raises(ValueError, match="cannot honor tree_criterion"):
      RosenblattBicop(ForeignControls())

  def test_a_foreign_controls_object_carrying_only_known_settings_works(
    self,
  ) -> None:
    class ForeignControls:
      def to_dict(self) -> dict[str, object]:
        return {"backend": "tabpfn-criterion", "device": "cpu", "eps": 1e-5}

    model = RosenblattBicop(ForeignControls())

    assert model.eps == pytest.approx(1e-5)
    assert model._device == torch.device("cpu")

  def test_vine_controls_are_valid_pair_controls(self) -> None:
    model = RosenblattBicop(
      FitControlsRosenblattVinecop(device="cpu", eps=1e-4)
    )

    assert model.eps == pytest.approx(1e-4)
