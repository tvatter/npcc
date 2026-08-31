import numpy as np
import pytest
import torch

from pyvinecopulib import RVineStructure
from pyvinecopulib.core import VinecopBase

from npcc import RosenblattBicop, RosenblattVinecop
from npcc.core._common import TensorLike


@pytest.fixture
def fitted_vine(
  register_uniform_backends: None,
) -> RosenblattVinecop:
  rng = np.random.default_rng(42)
  u = rng.uniform(0.1, 0.9, size=(40, 3))
  return RosenblattVinecop.from_data(
    u,
    make_structure(),
    backend="uniform-native",
    device="cpu",
  )


def make_structure() -> RVineStructure:
  return RVineStructure.from_order([1, 2, 3])


def make_pairs() -> list[list[RosenblattBicop]]:
  return [
    [RosenblattBicop(device="cpu"), RosenblattBicop(device="cpu")],
    [RosenblattBicop(device="cpu")],
  ]


def test_rosenblatt_vinecop_subclass_vinecop_base() -> None:
  assert issubclass(RosenblattVinecop, VinecopBase)


def test_constructor_binds_structure_and_pairs() -> None:
  structure = make_structure()
  pairs = make_pairs()

  vine = RosenblattVinecop(pairs, structure)

  assert vine.dim == 3
  assert vine.trunc_lvl == 2
  assert vine.order == (1, 2, 3)
  assert vine.supports_covariates is True
  assert vine.get_pair_copula(0, 1) is pairs[0][1]
  assert vine.get_pair_copula(1, 0) is pairs[1][0]


def test_constructor_rejects_wrong_tree_count() -> None:
  with pytest.raises(ValueError, match="has 1 trees, expected 2"):
    RosenblattVinecop([[RosenblattBicop()]], make_structure())


def test_constructor_rejects_wrong_edge_count() -> None:
  pairs = [[RosenblattBicop()], [RosenblattBicop()]]

  with pytest.raises(ValueError, match="tree 0 has 1 edges, expected 2"):
    RosenblattVinecop(pairs, make_structure())


def test_constructor_copies_pair_rows() -> None:
  pairs = make_pairs()
  vine = RosenblattVinecop(pairs, make_structure())

  pairs[0].clear()

  assert len(vine.pair_copulas[0]) == 2


def test_from_data_fits_fixed_structure(
  register_uniform_backends: None,
) -> None:
  rng = np.random.default_rng(42)
  u = rng.uniform(0.1, 0.9, size=(40, 3))

  vine = RosenblattVinecop.from_data(
    u,
    make_structure(),
    backend="uniform-native",
    device="cpu",
  )

  assert [len(row) for row in vine.pair_copulas] == [2, 1]
  for row in vine.pair_copulas:
    for pair in row:
      assert pair.v_given_ux_._fitted is True
      assert pair.u_given_vx_._fitted is True


def test_from_data_assembles_non_simplified_context(
  register_uniform_backends: None,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  rng = np.random.default_rng(42)
  u = rng.uniform(0.1, 0.9, size=(40, 3))
  x = rng.normal(size=(40, 2))
  context_widths: list[int | None] = []

  original_fit = RosenblattBicop.fit

  def record_fit(
    self: RosenblattBicop,
    uv: TensorLike,
    x_edge: TensorLike | None = None,
  ) -> RosenblattBicop:
    width = None if x_edge is None else int(x_edge.shape[1])
    context_widths.append(width)
    return original_fit(self, uv, x_edge)

  monkeypatch.setattr(RosenblattBicop, "fit", record_fit)

  RosenblattVinecop.from_data(
    u, make_structure(), x=x, backend="uniform-native", device="cpu"
  )

  assert context_widths == [2, 2, 3]


def test_constructor_infers_pair_device() -> None:
  vine = RosenblattVinecop(make_pairs(), make_structure())

  assert vine._device == torch.device("cpu")


def test_constructor_rejects_mixed_pair_devices() -> None:
  pairs = make_pairs()
  pairs[0][1]._device = torch.device("cuda:0")

  with pytest.raises(ValueError, match="must use the same device"):
    RosenblattVinecop(pairs, make_structure())


def test_constructor_rejects_explicit_device_mismatch() -> None:
  with pytest.raises(ValueError, match="does not match"):
    RosenblattVinecop(
      make_pairs(),
      make_structure(),
      device="cuda:0",
    )


def test_sample_is_seeded_and_on_configured_device(
  register_uniform_backends: None,
) -> None:
  rng = np.random.default_rng(42)
  u = rng.uniform(0.1, 0.9, size=(40, 3))
  vine = RosenblattVinecop.from_data(
    u, make_structure(), backend="uniform-native", device="cpu"
  )

  first = vine.sample(8, seeds=[42])
  second = vine.sample(8, seeds=[42])

  assert isinstance(first, torch.Tensor)
  assert isinstance(second, torch.Tensor)
  assert first.shape == (8, 3)
  assert first.dtype == torch.float64
  assert first.device.type == "cpu"
  assert torch.equal(first, second)
  assert torch.all((first > 0.0) & (first < 1.0))


def test_sample_qrng_is_reproducible(
  register_uniform_backends: None,
) -> None:
  rng = np.random.default_rng(42)
  u = rng.uniform(0.1, 0.9, size=(40, 3))
  vine = RosenblattVinecop.from_data(
    u,
    make_structure(),
    backend="uniform-native",
    device="cpu",
  )

  first = vine.sample(8, qrng=True, seeds=[42])
  second = vine.sample(8, qrng=True, seeds=[42])

  assert isinstance(first, torch.Tensor)
  assert isinstance(second, torch.Tensor)
  assert torch.equal(first, second)


def test_pdf_preserves_numpy(
  fitted_vine: RosenblattVinecop,
) -> None:
  rng = np.random.default_rng(42)
  u = rng.uniform(0.2, 0.8, size=(5, 3))

  result = fitted_vine.pdf(u)

  assert isinstance(result, np.ndarray)
  assert result.shape == (5,)
  assert np.isfinite(result).all()
  assert (result > 0.0).all()


def test_pdf_preserves_torch(
  fitted_vine: RosenblattVinecop,
) -> None:
  rng = np.random.default_rng(42)
  u = torch.as_tensor(rng.uniform(0.2, 0.8, size=(5, 3)))

  result = fitted_vine.pdf(u)

  assert isinstance(result, torch.Tensor)
  assert result.shape == (5,)
  assert result.dtype == torch.float64
  assert result.device.type == "cpu"
  assert torch.isfinite(result).all()
  assert torch.all(result > 0.0)


@pytest.mark.parametrize("array_type", ["numpy", "torch"])
def test_rosenblatt_inverse_round_trip(
  fitted_vine: RosenblattVinecop,
  array_type: str,
) -> None:
  rng = np.random.default_rng(42)
  u_np = rng.uniform(0.2, 0.8, size=(5, 3))
  u: TensorLike = (
    torch.as_tensor(u_np.copy()) if array_type == "torch" else u_np.copy()
  )

  transformed = fitted_vine.rosenblatt(u)
  recovered = fitted_vine.inverse_rosenblatt(transformed)

  if array_type == "torch":
    assert isinstance(recovered, torch.Tensor)
    torch.testing.assert_close(
      recovered,
      torch.as_tensor(u_np),
      atol=1e-8,
      rtol=1e-8,
    )
  else:
    assert isinstance(recovered, np.ndarray)
    np.testing.assert_allclose(recovered, u_np, atol=1e-8, rtol=1e-8)


def test_loglik_matches_summed_log_pdf(
  fitted_vine: RosenblattVinecop,
) -> None:
  rng = np.random.default_rng(42)
  u = rng.uniform(0.2, 0.8, size=(5, 3))

  density = fitted_vine.pdf(u)
  result = fitted_vine.loglik(u)

  assert isinstance(density, np.ndarray)
  assert isinstance(result, np.floating)
  assert result == pytest.approx(np.log(np.clip(density, 1e-20, None)).sum())


@pytest.mark.parametrize("array_type", ["numpy", "torch"])
def test_pdf_with_same_namespace_covariates(
  register_uniform_backends: None,
  array_type: str,
) -> None:
  rng = np.random.default_rng(42)
  train_u_np = rng.uniform(0.1, 0.9, size=(40, 3))
  train_x_np = rng.normal(size=(40, 2))
  query_u_np = rng.uniform(0.2, 0.8, size=(5, 3))
  query_x_np = rng.normal(size=(5, 2))

  if array_type == "torch":
    train_u: TensorLike = torch.as_tensor(train_u_np)
    train_x: TensorLike = torch.as_tensor(train_x_np)
    query_u: TensorLike = torch.as_tensor(query_u_np)
    query_x: TensorLike = torch.as_tensor(query_x_np)
  else:
    train_u = train_u_np
    train_x = train_x_np
    query_u = query_u_np
    query_x = query_x_np

  vine = RosenblattVinecop.from_data(
    train_u,
    make_structure(),
    x=train_x,
    backend="uniform-native",
    device="cpu",
  )
  result = vine.pdf(query_u, x=query_x)

  if array_type == "torch":
    assert isinstance(result, torch.Tensor)
  else:
    assert isinstance(result, np.ndarray)

  assert result.shape == (5,)


@pytest.mark.parametrize("array_type", ["numpy", "torch"])
def test_cdf_preserves_input_type(
  fitted_vine: RosenblattVinecop,
  array_type: str,
) -> None:
  rng = np.random.default_rng(42)
  query_np = rng.uniform(0.2, 0.8, size=(3, 3))
  query: TensorLike = (
    torch.as_tensor(query_np) if array_type == "torch" else query_np
  )

  result = fitted_vine.cdf(query, N=64, seeds=[42])

  if array_type == "torch":
    assert isinstance(result, torch.Tensor)
    assert torch.all((result >= 0.0) & (result <= 1.0))
  else:
    assert isinstance(result, np.ndarray)
    assert ((result >= 0.0) & (result <= 1.0)).all()

  assert result.shape == (3,)


@pytest.mark.parametrize("array_type", ["numpy", "torch"])
def test_sample_conditional_preserves_input_type(
  fitted_vine: RosenblattVinecop,
  array_type: str,
) -> None:
  u_cond_np = np.array([[0.3], [0.5], [0.7]])
  u_cond: TensorLike = (
    torch.as_tensor(u_cond_np) if array_type == "torch" else u_cond_np
  )

  result = fitted_vine.sample_conditional(
    u_cond,
    seeds=[42],
  )

  if array_type == "torch":
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result[:, 2], torch.as_tensor(u_cond_np[:, 0]))
  else:
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result[:, 2], u_cond_np[:, 0])

  assert result.shape == (3, 3)
