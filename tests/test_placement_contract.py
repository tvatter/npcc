"""Upstream's ``place`` must be able to find every estimator's placement.

``pyvinecopulib.core.prepare_covariates`` reaches placement through the
module-level ``place``, never through the object's ``_prep`` hook, and ``place``
answers by looking for a float array the object *holds*. So an estimator that
keeps its device as a ``torch.device`` handle and nothing else is skipped
silently: the covariates come back untouched and meet placed copula arguments
inside a concatenation, several frames later.

That is reachable from `BicopBase.loglik`, `sample`, `hinv1`, `hinv2` and every
edge of the vine cascade, and it is invisible on CPU -- a host array and a host
tensor concatenate without complaint. `_set_placement` plants an empty float64
tensor so `place` resolves; these tests are what keep it planted.
"""

from __future__ import annotations

import numpy
import pytest
import torch
from pyvinecopulib import RVineStructure
from pyvinecopulib.core import place, prepare_covariates, reference_array

from npcc.core._placement import TensorPlacement, resolve_device
from npcc.core.bicop import RosenblattBicop
from npcc.core.controls import FitControlsRosenblattBicop
from npcc.core.margin_quantile_table import QuantileTableConfig
from npcc.core.registry import create_backend
from npcc.core.vinecop import RosenblattVinecop
from npcc.core.vinedist import RosenblattVinedist

# Parameterized over both devices, with the mark on the *param*: `-m "not
# cuda"` then deselects the GPU half, and a CPU-only runner never opens a
# context. A mark applied from inside the test body lands after collection and
# would do neither.
_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]


def _estimators(device: str) -> dict[str, TensorPlacement]:
  """One of each estimator, on ``device``.

  Parameters
  ----------
  device : str
      The device to construct them on.

  Returns
  -------
  dict
      The four estimators, keyed by a short name. Typed by the mixin they
      share, which is what carries ``_prep`` and ``_placement_ref``.
  """
  controls = FitControlsRosenblattBicop(backend="uniform-native", device=device)
  margin = create_backend(
    "uniform-native",
    transform="identity",
    quantile_table_config=controls.quantile_table_config
    or QuantileTableConfig(),
    eps=controls.eps,
    device=device,
    batch_size=None,
  )
  structure = RVineStructure.from_order([1, 2, 3])
  vinecop = RosenblattVinecop(None, structure, device=device)
  margins = [
    create_backend(
      "uniform-native",
      transform="identity",
      quantile_table_config=controls.quantile_table_config
      or QuantileTableConfig(),
      eps=controls.eps,
      device=device,
      batch_size=None,
    )
    for _ in range(3)
  ]
  return {
    "bicop": RosenblattBicop(controls),
    "margin": margin,
    "vinecop": vinecop,
    "vinedist": RosenblattVinedist(vinecop, margins),
  }


@pytest.mark.parametrize("name", ["bicop", "margin", "vinecop", "vinedist"])
@pytest.mark.parametrize("device", _DEVICES)
def test_place_finds_every_estimators_placement(
  register_uniform_backends: None,
  device: str,
  name: str,
) -> None:
  """Every estimator holds a float array naming where its numerics run."""
  del register_uniform_backends

  expected = resolve_device(device)
  estimator = _estimators(device)[name]
  host = numpy.arange(6, dtype=float).reshape(3, 2)

  reference = reference_array(estimator)

  assert isinstance(reference, torch.Tensor)
  assert reference.dtype is torch.float64
  assert reference.device == expected

  placed = place(estimator, host)
  covariates = prepare_covariates(estimator, host, 3)

  assert isinstance(placed, torch.Tensor)
  assert isinstance(covariates, torch.Tensor)
  assert placed.dtype is torch.float64
  assert placed.device == expected
  assert covariates.device == expected
  torch.testing.assert_close(placed, estimator._prep(host))


@pytest.mark.parametrize("name", ["bicop", "margin", "vinecop", "vinedist"])
def test_the_reference_is_the_planted_one(
  register_uniform_backends: None,
  name: str,
) -> None:
  """``_placement_ref`` is the only float array on the object, by construction.

  ``reference_array`` prefers a float array and falls back to an integer one,
  and for the vine the planted tensor is *last* in ``vars()`` -- the accessors
  ahead of it (``order``, ``inverse_order``) are integer. A subclass that
  stored a float32 tensor ahead of it would make ``place`` adopt float32 on
  the covariate path, with nothing raising anywhere.
  """
  del register_uniform_backends

  estimator = _estimators("cpu")[name]

  assert reference_array(estimator) is estimator._placement_ref


def test_a_gradient_survives_prep_but_not_place() -> None:
  """The reason ``_prep`` cannot simply delegate to ``place``.

  ``torch.as_tensor`` carries a gradient across a dtype change; ``place``
  reaches placement through ``array_api_compat``'s ``xp.asarray``, which
  severs it. This is the durable justification for the override, so it is
  worth failing here if either side ever changes.
  """
  model = RosenblattBicop(FitControlsRosenblattBicop(device="cpu"))
  tracked = torch.tensor([[0.3, 0.4]], dtype=torch.float32, requires_grad=True)

  assert model._prep(tracked).requires_grad
  assert not place(model, tracked).requires_grad
