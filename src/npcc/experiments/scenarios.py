"""Ground-truth copula scenarios for the simulation study.

A *scenario* pairs a copula family with a Kendall's-tau regime and defines:

- :func:`sample` — draw a training set ``(u, v[, x])`` from the known copula;
- :func:`eval_grid` — the fixed points at which estimates are scored;
- :func:`ground_truth` — the exact ``pdf`` / ``cdf`` / ``hfunc1`` / ``hfunc2``
  on that evaluation grid.

Everything is backed by :mod:`pyvinecopulib`: ``Bicop.tau_to_parameters`` maps a
target Kendall's tau to the family's parameter, and ``Bicop.hinv1`` provides a
generic inverse-Rosenblatt sampler, so the study generalises to any
one-parameter family without per-family closed forms.

Conditional scenarios vary ``tau`` with a scalar covariate ``x`` (``x`` is a
deterministic ``linspace`` over the unit interval, matching the original
notebook study); unconditional scenarios use a single fixed ``tau`` and no
covariate (``x is None``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pyvinecopulib as pv

# Families restricted to the single-parameter set, for which
# ``tau_to_parameters`` is an unambiguous scalar map. Resolved via getattr
# because pyvinecopulib's type stub does not expose the enum members.
_FAMILY_NAMES: tuple[str, ...] = (
  "clayton",
  "gumbel",
  "frank",
  "gaussian",
  "joe",
)
FAMILIES: dict[str, pv.BicopFamily] = {
  name: getattr(pv.BicopFamily, name) for name in _FAMILY_NAMES
}

QUANTITIES: tuple[str, ...] = ("pdf", "cdf", "hfunc1", "hfunc2")

# Evaluation grid shared with the original notebook study.
U_LEVELS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
V_LEVELS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
UV_PAIRS: tuple[tuple[float, float], ...] = tuple(
  (u, v) for u in U_LEVELS for v in V_LEVELS
)
N_X_EVAL: int = 50
X_MIN: float = 0.01
X_MAX: float = 0.99

# Unconditional evaluation: an interior uv grid (boundaries excluded).
UV_GRID_N: int = 20

# Keep tau inside a band where every family's parameter stays well-conditioned.
TAU_LO: float = 0.05
TAU_HI: float = 0.90
_EPS: float = 1e-9


@dataclass(frozen=True)
class ScenarioSpec:
  """A Kendall's-tau regime: conditional ``tau(x)`` or a fixed ``tau``."""

  name: str
  conditional: bool
  tau_of_x: Callable[[np.ndarray], np.ndarray] | None = None
  tau: float | None = None


def _clip_tau(tau: np.ndarray) -> np.ndarray:
  return np.clip(tau, TAU_LO, TAU_HI)


TAU_SCENARIOS: dict[str, ScenarioSpec] = {
  "linear": ScenarioSpec(
    "linear", True, lambda x: _clip_tau(TAU_LO + (TAU_HI - TAU_LO) * x)
  ),
  "constant": ScenarioSpec("constant", True, lambda x: np.full_like(x, 0.5)),
  "sin": ScenarioSpec(
    "sin", True, lambda x: _clip_tau(0.5 + 0.4 * np.sin(2.0 * np.pi * x))
  ),
  "quadratic": ScenarioSpec(
    "quadratic",
    True,
    lambda x: _clip_tau(TAU_LO + (TAU_HI - TAU_LO) * (2.0 * x - 1.0) ** 2),
  ),
  "uncond25": ScenarioSpec("uncond25", False, tau=0.25),
  "uncond50": ScenarioSpec("uncond50", False, tau=0.50),
  "uncond75": ScenarioSpec("uncond75", False, tau=0.75),
}


@dataclass(frozen=True)
class EvalGrid:
  """Flattened evaluation points and the shape to reshape predictions to.

  ``u_flat`` / ``v_flat`` are length ``n_points``; ``x_flat`` is the same
  length for conditional scenarios and ``None`` for unconditional ones.
  ``shape`` is ``(n_pairs, n_x)`` (conditional) or ``(n_points,)``
  (unconditional) so the runner can reshape per-quantity ground truth and
  predictions consistently.
  """

  u_flat: np.ndarray
  v_flat: np.ndarray
  x_flat: np.ndarray | None
  shape: tuple[int, ...]
  conditional: bool
  x_axis: np.ndarray | None


def _bicop(family: pv.BicopFamily, tau: float) -> pv.Bicop:
  """Bicop of ``family`` whose Kendall's tau equals ``tau``."""
  proto = pv.Bicop(family=family)
  params = np.asarray(proto.tau_to_parameters(float(tau)), dtype=np.float64)
  return pv.Bicop(family=family, parameters=params)


def is_conditional(scenario: str) -> bool:
  return TAU_SCENARIOS[scenario].conditional


def eval_grid(scenario: str) -> EvalGrid:
  """Fixed evaluation grid for ``scenario`` (conditional or unconditional)."""
  spec = TAU_SCENARIOS[scenario]
  if spec.conditional:
    u_pairs = np.array([p[0] for p in UV_PAIRS], dtype=np.float64)
    v_pairs = np.array([p[1] for p in UV_PAIRS], dtype=np.float64)
    x_axis = np.linspace(X_MIN, X_MAX, N_X_EVAL, dtype=np.float64)
    n_pairs = len(UV_PAIRS)
    return EvalGrid(
      u_flat=np.repeat(u_pairs, N_X_EVAL),
      v_flat=np.repeat(v_pairs, N_X_EVAL),
      x_flat=np.tile(x_axis, n_pairs),
      shape=(n_pairs, N_X_EVAL),
      conditional=True,
      x_axis=x_axis,
    )
  axis = np.linspace(_EPS, 1.0 - _EPS, UV_GRID_N + 2, dtype=np.float64)[1:-1]
  uu, vv = np.meshgrid(axis, axis, indexing="ij")
  u_flat = uu.reshape(-1)
  v_flat = vv.reshape(-1)
  return EvalGrid(
    u_flat=u_flat,
    v_flat=v_flat,
    x_flat=None,
    shape=(u_flat.shape[0],),
    conditional=False,
    x_axis=None,
  )


def ground_truth(family: str, scenario: str) -> dict[str, np.ndarray]:
  """Exact pdf/cdf/hfunc1/hfunc2 on :func:`eval_grid` for ``(family, scenario)``.

  Conditional: one ``Bicop`` per evaluation ``x`` (50), evaluated at the 25 uv
  pairs, stacked into ``(n_pairs, n_x)`` grids.  Unconditional: a single
  ``Bicop`` evaluated on the flattened uv grid.
  """
  fam = FAMILIES[family]
  spec = TAU_SCENARIOS[scenario]
  grid = eval_grid(scenario)

  if spec.conditional:
    assert spec.tau_of_x is not None and grid.x_axis is not None
    tau_x = spec.tau_of_x(grid.x_axis)
    uv = np.column_stack(
      [
        np.array([p[0] for p in UV_PAIRS], dtype=np.float64),
        np.array([p[1] for p in UV_PAIRS], dtype=np.float64),
      ]
    )
    cols: dict[str, list[np.ndarray]] = {q: [] for q in QUANTITIES}
    for tau in tau_x:
      cop = _bicop(fam, float(tau))
      cols["pdf"].append(np.asarray(cop.pdf(uv), dtype=np.float64))
      cols["cdf"].append(np.asarray(cop.cdf(uv), dtype=np.float64))
      cols["hfunc1"].append(np.asarray(cop.hfunc1(uv), dtype=np.float64))
      cols["hfunc2"].append(np.asarray(cop.hfunc2(uv), dtype=np.float64))
    return {q: np.stack(cols[q], axis=1) for q in QUANTITIES}

  assert spec.tau is not None
  cop = _bicop(fam, spec.tau)
  uv = np.column_stack([grid.u_flat, grid.v_flat])
  return {
    "pdf": np.asarray(cop.pdf(uv), dtype=np.float64),
    "cdf": np.asarray(cop.cdf(uv), dtype=np.float64),
    "hfunc1": np.asarray(cop.hfunc1(uv), dtype=np.float64),
    "hfunc2": np.asarray(cop.hfunc2(uv), dtype=np.float64),
  }


def sample(
  family: str, scenario: str, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
  """Draw ``n`` training points ``(u, v[, x])`` from the true copula.

  Conditional: ``x`` is a deterministic ``linspace`` and each row is sampled by
  inverse Rosenblatt, ``v = hinv1([u, w])`` under the row's own ``tau(x_i)``.
  Unconditional: ``Bicop.simulate`` with no covariate (``x is None``).
  """
  fam = FAMILIES[family]
  spec = TAU_SCENARIOS[scenario]
  rng = np.random.default_rng(seed)

  if not spec.conditional:
    assert spec.tau is not None
    cop = _bicop(fam, spec.tau)
    uv = np.asarray(
      cop.simulate(n, seeds=[int(s) for s in rng.integers(1, 2**31 - 1, 3)]),
      dtype=np.float64,
    )
    return uv[:, 0], uv[:, 1], None

  assert spec.tau_of_x is not None
  x = np.linspace(X_MIN, X_MAX, n, dtype=np.float64)
  tau_x = spec.tau_of_x(x)
  u = rng.uniform(_EPS, 1.0 - _EPS, size=n)
  w = rng.uniform(_EPS, 1.0 - _EPS, size=n)
  v = np.empty(n, dtype=np.float64)
  # ponytail: one Bicop per row because tau(x) is continuous and pyvinecopulib
  # does not vectorise hinv1 over row-specific parameters. O(n) Bicop builds is
  # negligible next to the TabPFN fit; if it ever bites, group by rounded tau.
  for i in range(n):
    cop = _bicop(fam, float(tau_x[i]))
    v[i] = float(np.asarray(cop.hinv1(np.array([[u[i], w[i]]]))).ravel()[0])
  return u, v, x
