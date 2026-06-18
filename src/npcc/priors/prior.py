"""
prior.py — the rich prior over conditional bivariate copula densities.

:class:`ConditionalCopulaPrior` is the hyperprior; :func:`sample_spec` draws one
:class:`PriorDraw` (a covariate sampler + a :class:`CopulaMixture`), from which
``simulate`` produces a synthetic dataset ``(X, u, v)`` and ``true_density``
returns the exact ``c(u, v | x)`` for evaluation — both through the *same*
mixture, so the simulator and the density target can never silently diverge.

The prior is rich by construction:

- mixtures of 1..``max_components`` components, each a uniformly chosen
  ``(family, rotation)`` from the registry, with Dirichlet weights;
- per-component tau-mode (gampcc Kendall's-tau link) or native-parameter mode,
  chosen at random for tau-capable families;
- covariate dimension ``p`` from ``p_min..p_max`` (``p == 0`` is a pure
  unconditional draw) with normal / uniform / mixed marginals;
- an explicit unconditional probability that forces *constant* links even when
  covariates are present (constant Kendall's tau, covariates irrelevant);
- smooth non-linear covariate effects via a random-Fourier basis.

Everything is driven by one ``numpy`` ``Generator``; a fixed seed gives a
byte-identical draw and dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from npcc.priors.components import (
  CopulaComponent,
  CopulaMixture,
  ParameterProcess,
)
from npcc.priors.families import DEFAULT_REGISTRY, FamilySpec
from npcc.priors.links import CovariateLink


@dataclass(frozen=True)
class XSampler:
  """Per-column covariate sampler fixed at draw time (so draws are reproducible)."""

  kinds: tuple[str, ...]
  locs: np.ndarray
  scales: np.ndarray

  def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw covariate rows, shape ``(n, p)`` (``(n, 0)`` when ``p == 0``)."""
    if len(self.kinds) == 0:
      return np.zeros((n, 0))
    cols = []
    for kind, loc, scale in zip(self.kinds, self.locs, self.scales):
      if kind == "normal":
        cols.append(loc + scale * rng.standard_normal(n))
      else:
        cols.append(rng.uniform(loc - scale, loc + scale, n))
    return np.column_stack(cols)


@dataclass(frozen=True)
class PriorDraw:
  """One realised conditional copula: a covariate sampler + a copula mixture."""

  mixture: CopulaMixture
  p: int
  n: int
  x_sampler: XSampler
  eps: float = 1e-6

  def simulate(
    self, rng: np.random.Generator, n: int | None = None
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample ``(X, u, v)``; ``u`` is uniform and ``v`` is drawn from the mixture."""
    rows = self.n if n is None else n
    x = self.x_sampler.sample(rows, rng)
    u = rng.uniform(self.eps, 1.0 - self.eps, rows)
    w = rng.uniform(self.eps, 1.0 - self.eps, rows)
    v = self.mixture.simulate(u, w, x, rng)
    v = np.clip(v, self.eps, 1.0 - self.eps)
    return x, u, v

  def true_density(
    self, u: np.ndarray, v: np.ndarray, x: np.ndarray
  ) -> np.ndarray:
    """Exact conditional copula density ``c(u_i, v_i | x_i)`` (for metrics)."""
    return self.mixture.pdf(u, v, x)


@dataclass
class ConditionalCopulaPrior:
  """Hyperprior over conditional bivariate copula densities.

  Attributes
  ----------
  families
      Registry of ``(family, rotation)`` specs to sample components from.
  p_min, p_max
      Inclusive range for the covariate dimension (``p_min == 0`` allows pure
      unconditional draws with no covariates).
  n_min, n_max
      Inclusive range for the per-dataset sample size.
  max_components
      Maximum number of mixture components (1..``max_components``).
  unconditional_prob
      Probability that a draw uses constant links (covariate-independent copula,
      i.e. constant Kendall's tau) even when ``p > 0``.
  tau_mode_prob
      Probability of the Kendall's-tau parameterisation when a family supports
      it (otherwise the native-parameter parameterisation is used).
  x_dist
      Covariate marginal family: ``"normal"``, ``"uniform"``, or ``"mixed"``.
  intercept_scale, linear_scale, basis_scale
      Standard deviations of the link's intercept / linear / basis coefficients.
  n_basis_max
      Maximum number of random-Fourier basis functions per link (0 disables).
  basis_lengthscale
      Length-scale of the random-Fourier frequencies (larger = smoother).
  eps
      Boundary clip for simulated ``(u, v)``.
  """

  families: tuple[FamilySpec, ...] = DEFAULT_REGISTRY
  p_min: int = 0
  p_max: int = 4
  n_min: int = 200
  n_max: int = 2000
  max_components: int = 3
  unconditional_prob: float = 0.25
  tau_mode_prob: float = 0.5
  x_dist: str = "mixed"
  intercept_scale: float = 1.5
  linear_scale: float = 1.0
  basis_scale: float = 1.0
  n_basis_max: int = 3
  basis_lengthscale: float = 1.0
  eps: float = 1e-6

  def __post_init__(self) -> None:
    if not 0 <= self.p_min <= self.p_max:
      raise ValueError("require 0 <= p_min <= p_max.")
    if not 2 <= self.n_min <= self.n_max:
      raise ValueError("require 2 <= n_min <= n_max.")
    if self.max_components < 1:
      raise ValueError("max_components must be >= 1.")
    if not 0.0 <= self.unconditional_prob <= 1.0:
      raise ValueError("unconditional_prob must be in [0, 1].")
    if len(self.families) == 0:
      raise ValueError("families registry must be non-empty.")


def _sample_link(
  prior: ConditionalCopulaPrior,
  p: int,
  constant: bool,
  rng: np.random.Generator,
) -> CovariateLink:
  intercept = prior.intercept_scale * float(rng.standard_normal())
  if constant or p == 0:
    return CovariateLink(intercept, np.zeros(p), None, None, None)
  linear = (prior.linear_scale / np.sqrt(p)) * rng.standard_normal(p)
  n_basis = int(rng.integers(0, prior.n_basis_max + 1))
  if n_basis == 0:
    return CovariateLink(intercept, linear, None, None, None)
  freqs = rng.standard_normal((n_basis, p)) / prior.basis_lengthscale
  phases = rng.uniform(0.0, 2.0 * np.pi, n_basis)
  basis_coef = (prior.basis_scale / np.sqrt(n_basis)) * rng.standard_normal(
    n_basis
  )
  return CovariateLink(intercept, linear, freqs, phases, basis_coef)


def _sample_process(
  prior: ConditionalCopulaPrior,
  spec: FamilySpec,
  p: int,
  constant: bool,
  rng: np.random.Generator,
) -> ParameterProcess:
  if spec.is_independence:
    return ParameterProcess(spec, "none", ())
  use_tau = spec.tau_capable and (
    not spec.native_capable or rng.random() < prior.tau_mode_prob
  )
  if use_tau:
    return ParameterProcess(
      spec, "tau", (_sample_link(prior, p, constant, rng),)
    )
  links = tuple(
    _sample_link(prior, p, constant, rng) for _ in range(spec.n_params)
  )
  return ParameterProcess(spec, "native", links)


def _sample_x_sampler(
  prior: ConditionalCopulaPrior, p: int, rng: np.random.Generator
) -> XSampler:
  if p == 0:
    return XSampler((), np.zeros(0), np.zeros(0))
  if prior.x_dist == "mixed":
    kinds = tuple(rng.choice(("normal", "uniform")) for _ in range(p))
  else:
    kinds = (prior.x_dist,) * p
  locs = rng.uniform(-1.0, 1.0, p)
  scales = rng.uniform(0.5, 2.0, p)
  return XSampler(kinds, locs, scales)


def sample_spec(
  prior: ConditionalCopulaPrior, rng: np.random.Generator
) -> PriorDraw:
  """Draw one :class:`PriorDraw` (covariate sampler + copula mixture)."""
  p = int(rng.integers(prior.p_min, prior.p_max + 1))
  n = int(rng.integers(prior.n_min, prior.n_max + 1))
  constant = (p == 0) or (rng.random() < prior.unconditional_prob)
  x_sampler = _sample_x_sampler(prior, p, rng)

  k = int(rng.integers(1, prior.max_components + 1))
  indices = rng.integers(0, len(prior.families), size=k)
  components = tuple(
    CopulaComponent(_sample_process(prior, prior.families[i], p, constant, rng))
    for i in indices
  )
  weights = rng.dirichlet(np.ones(k)) if k > 1 else np.ones(1)
  return PriorDraw(
    CopulaMixture(components, weights), p, n, x_sampler, prior.eps
  )
