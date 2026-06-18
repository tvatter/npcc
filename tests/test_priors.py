"""Tests for the conditional-copula prior (``npcc.priors``).

Hermetic and TabPFN-free: only numpy + pyvinecopulib. Covers registry validity,
simulator/density correctness (uniform margins, including *per-x* conditional
uniformity), the mixture identity, determinism, the TabPFN dataset layout, and
config validation.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
import pytest

from npcc.priors import (
  ConditionalCopulaPrior,
  DEFAULT_REGISTRY,
  PriorDraw,
  XSampler,
  direction_datasets,
  sample_pool,
  sample_spec,
)
from npcc.priors.components import (
  CopulaComponent,
  CopulaMixture,
  ParameterProcess,
)
from npcc.priors.families import FamilySpec, build_bicop, params_from_tau
from npcc.priors.links import CovariateLink, affine_to_interval

INTERIOR = np.array(
  [[u, v] for u in (0.1, 0.3, 0.5, 0.7, 0.9) for v in (0.1, 0.3, 0.5, 0.7, 0.9)]
)


def _const_link(intercept: float = 0.0, p: int = 0) -> CovariateLink:
  return CovariateLink(intercept, np.zeros(p), None, None, None)


def _const_process(
  spec: FamilySpec, intercept: float = 0.0
) -> ParameterProcess:
  """Unconditional (p=0) process for ``spec`` at a fixed link intercept."""
  if spec.is_independence:
    return ParameterProcess(spec, "none", ())
  if spec.tau_capable:
    return ParameterProcess(spec, "tau", (_const_link(intercept),))
  return ParameterProcess(
    spec, "native", tuple(_const_link(intercept) for _ in range(spec.n_params))
  )


def _const_draw(spec: FamilySpec, intercept: float = 0.0) -> PriorDraw:
  mixture = CopulaMixture(
    (CopulaComponent(_const_process(spec, intercept)),), np.ones(1)
  )
  return PriorDraw(mixture, 0, 1, XSampler((), np.zeros(0), np.zeros(0)))


def _corner_params(spec: FamilySpec) -> list[np.ndarray]:
  """Parameter vectors at the corners of the family's valid region."""
  if spec.is_independence:
    return [np.zeros((0, 1))]
  if spec.tau_capable:
    return [
      params_from_tau(spec, spec.tau_lo),
      params_from_tau(spec, spec.tau_hi),
    ]
  return [
    np.asarray(combo, float).reshape(-1, 1)
    for combo in itertools.product(*spec.native_bounds)
  ]


@pytest.mark.parametrize("spec", DEFAULT_REGISTRY, ids=lambda s: s.name)
def test_registry_corners_construct(spec: FamilySpec) -> None:
  """Every family constructs at its range corners with finite pdf / hinv1."""
  for params in _corner_params(spec):
    bicop = build_bicop(spec, params)
    pdf = np.asarray(bicop.pdf(INTERIOR))
    hinv = np.asarray(bicop.hinv1(INTERIOR))
    assert np.all(np.isfinite(pdf)) and np.all(pdf >= 0.0)
    assert np.all(np.isfinite(hinv)) and np.all((hinv >= 0.0) & (hinv <= 1.0))


@pytest.mark.parametrize("spec", DEFAULT_REGISTRY, ids=lambda s: s.name)
def test_simulated_margins_uniform(spec: FamilySpec) -> None:
  """Simulated (u, v) have uniform margins for every family."""
  draw = _const_draw(spec)
  _, u, v = draw.simulate(np.random.default_rng(0), n=4000)
  for col in (u, v):
    assert col.mean() == pytest.approx(0.5, abs=0.03)
    counts = np.histogram(col, bins=10, range=(0.0, 1.0))[0]
    # Expected 400/bin; allow a loose band (no estimator, just sanity).
    assert counts.min() > 250 and counts.max() < 560


def test_per_x_conditional_margins_uniform() -> None:
  """A genuinely conditional copula has uniform margins *at each fixed x*."""
  spec = next(s for s in DEFAULT_REGISTRY if s.name == "clayton_0")
  link = CovariateLink(
    0.0, np.array([2.0]), None, None, None
  )  # tau varies with x
  mixture = CopulaMixture(
    (CopulaComponent(ParameterProcess(spec, "tau", (link,))),), np.ones(1)
  )
  rng = np.random.default_rng(1)
  for x0 in (-1.5, 0.0, 1.5):
    x = np.full((5000, 1), x0)
    u = rng.uniform(0, 1, 5000)
    w = rng.uniform(0, 1, 5000)
    v = mixture.simulate(u, w, x, rng)
    assert u.mean() == pytest.approx(0.5, abs=0.03)
    assert v.mean() == pytest.approx(0.5, abs=0.03)


@pytest.mark.parametrize("spec", DEFAULT_REGISTRY, ids=lambda s: s.name)
def test_true_density_finite_positive(spec: FamilySpec) -> None:
  """True density is finite and non-negative on the interior grid."""
  draw = _const_draw(spec)
  u, v = INTERIOR[:, 0], INTERIOR[:, 1]
  x = np.zeros((len(u), 0))
  c = draw.true_density(u, v, x)
  assert np.all(np.isfinite(c)) and np.all(c >= 0.0)


@pytest.mark.parametrize("family", ["gaussian", "frank"])
def test_smooth_density_integrates_to_one(family: str) -> None:
  """Smooth tau-capable families integrate to ~1 (pins density + quadrature)."""
  spec = next(s for s in DEFAULT_REGISTRY if s.name == family)
  level = (0.5 - spec.tau_lo) / (spec.tau_hi - spec.tau_lo)  # target tau = 0.5
  draw = _const_draw(spec, intercept=float(np.log(level / (1.0 - level))))
  m = 200
  g = (np.arange(m) + 0.5) / m
  uu, vv = (a.ravel() for a in np.meshgrid(g, g, indexing="ij"))
  c = draw.true_density(uu, vv, np.zeros((len(uu), 0)))
  assert c.mean() == pytest.approx(1.0, abs=0.02)


def test_mixture_density_is_weighted_sum() -> None:
  draw = sample_spec(
    ConditionalCopulaPrior(max_components=3), np.random.default_rng(5)
  )
  x, u, v = draw.simulate(np.random.default_rng(6), n=300)
  manual = sum(
    w * comp.pdf(u, v, x)
    for w, comp in zip(draw.mixture.weights, draw.mixture.components)
  )
  np.testing.assert_allclose(manual, draw.true_density(u, v, x), rtol=1e-12)


def test_determinism() -> None:
  prior = ConditionalCopulaPrior()
  a = sample_pool(prior, 4, 128, np.random.default_rng(42))
  b = sample_pool(prior, 4, 128, np.random.default_rng(42))
  assert len(a) == len(b) == 8  # 4 draws * both directions
  for (wa, ya), (wb, yb) in zip(a, b):
    np.testing.assert_array_equal(wa, wb)
    np.testing.assert_array_equal(ya, yb)


def test_both_directions_layout_and_identity() -> None:
  x = np.arange(6.0).reshape(3, 2)
  u = np.array([0.1, 0.2, 0.3])
  v = np.array([0.4, 0.5, 0.6])
  ds = direction_datasets(x, u, v, both_directions=True)
  assert len(ds) == 2
  (w1, y1), (w2, y2) = ds
  np.testing.assert_array_equal(w1[:, 0], u)
  np.testing.assert_array_equal(w1[:, 1:], x)
  np.testing.assert_array_equal(y1, v)
  np.testing.assert_array_equal(w2[:, 0], v)
  np.testing.assert_array_equal(y2, u)
  # No covariates -> single conditioner column.
  ((w0, _),) = direction_datasets(np.zeros((3, 0)), u, v, both_directions=False)
  assert w0.shape == (3, 1)


def test_independence_pdf_and_hinv1() -> None:
  spec = next(s for s in DEFAULT_REGISTRY if s.is_independence)
  comp = CopulaComponent(ParameterProcess(spec, "none", ()))
  u = np.array([0.2, 0.5, 0.8])
  w = np.array([0.3, 0.6, 0.9])
  x = np.zeros((3, 0))
  np.testing.assert_allclose(comp.pdf(u, w, x), 1.0)
  np.testing.assert_allclose(
    comp.hinv1(u, w, x), w
  )  # C(v|u)=v under independence


def test_unconditional_params_constant_across_rows() -> None:
  # p == 0: parameters cannot depend on covariates.
  draw = sample_spec(
    ConditionalCopulaPrior(p_min=0, p_max=0), np.random.default_rng(3)
  )
  comp = draw.mixture.components[0]
  if comp.process.mode != "none":
    params = comp.process.params(np.zeros((50, 0)))
    assert np.allclose(params, params[0])
  # unconditional_prob == 1 with covariates present: still constant.
  draw2 = sample_spec(
    ConditionalCopulaPrior(p_min=2, p_max=2, unconditional_prob=1.0),
    np.random.default_rng(4),
  )
  x = np.random.default_rng(0).standard_normal((50, 2))
  for comp in draw2.mixture.components:
    if comp.process.mode != "none":
      params = comp.process.params(x)
      assert np.allclose(params, params[0])


@pytest.mark.parametrize(
  "make",
  [
    lambda: ConditionalCopulaPrior(p_min=-1),
    lambda: ConditionalCopulaPrior(p_min=3, p_max=2),
    lambda: ConditionalCopulaPrior(n_min=1),
    lambda: ConditionalCopulaPrior(n_min=500, n_max=200),
    lambda: ConditionalCopulaPrior(max_components=0),
    lambda: ConditionalCopulaPrior(unconditional_prob=1.5),
    lambda: ConditionalCopulaPrior(families=()),
  ],
)
def test_config_validation_raises(make: Callable[[], object]) -> None:
  with pytest.raises(ValueError):
    make()


def test_affine_to_interval_bounds() -> None:
  z = np.array([-50.0, 0.0, 50.0])
  out = affine_to_interval(z, -0.9, 0.9)
  assert out[0] == pytest.approx(-0.9, abs=1e-3)
  assert out[1] == pytest.approx(0.0, abs=1e-9)
  assert out[2] == pytest.approx(0.9, abs=1e-3)


def test_covariate_link_constant_when_no_covariates() -> None:
  link = CovariateLink(0.7, np.zeros(0), None, None, None)
  out = link(np.zeros((4, 0)))
  np.testing.assert_allclose(out, 0.7)


def test_sample_eval_dataset_exposes_true_density() -> None:
  from npcc.priors import sample_eval_dataset

  draw, x, u, v = sample_eval_dataset(
    ConditionalCopulaPrior(), np.random.default_rng(0), rows=120
  )
  assert len(u) == 120 and len(v) == 120 and len(x) == 120
  assert np.all(np.isfinite(draw.true_density(u, v, x)))


def test_sample_pool_validation_raises() -> None:
  prior = ConditionalCopulaPrior()
  with pytest.raises(ValueError):
    sample_pool(prior, 0, 64, np.random.default_rng(0))
  with pytest.raises(ValueError):
    sample_pool(prior, 1, 1, np.random.default_rng(0))


def test_independence_only_prior_uses_none_mode() -> None:
  indep_spec = next(s for s in DEFAULT_REGISTRY if s.is_independence)
  draw = sample_spec(
    ConditionalCopulaPrior(families=(indep_spec,)), np.random.default_rng(0)
  )
  assert all(c.process.mode == "none" for c in draw.mixture.components)
  x, u, v = draw.simulate(np.random.default_rng(1), n=200)
  np.testing.assert_allclose(draw.true_density(u, v, x), 1.0)


def test_non_mixed_x_dist_uses_single_kind() -> None:
  draw = sample_spec(
    ConditionalCopulaPrior(x_dist="normal", p_min=2, p_max=2),
    np.random.default_rng(0),
  )
  assert set(draw.x_sampler.kinds) == {"normal"}
