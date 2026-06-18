"""
evaluate.py — before/after benchmark helpers for fine-tuned vs base TabPFN.

These compose :mod:`npcc.eval` with a known ground truth to quantify whether
fine-tuning improves *conditional* density estimation. Two reference scenarios:

- :func:`clayton_reference` — the notebook's out-of-prior Clayton copula with
  Kendall's tau linear in a covariate (the decisive generalisation check);
- any held-out :class:`~npcc.priors.PriorDraw`, whose ``true_density`` is exact.

:func:`conditional_report` evaluates an estimator's conditional density against a
true-density closure on a :class:`~npcc.eval.ConditionalGridSpec` and returns
ISE/IAE/KL. Estimators are passed as plain ``density_fn(u, v, x)`` callables to
keep this estimator-agnostic (wrap ``PFNRBicop.pdf`` with ``functools.partial``
for Sinkhorn / batch options).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pyvinecopulib as pv
from pyvinecopulib.families import clayton

from npcc.eval import (
  ConditionalGridSpec,
  conditional_density_grids,
  conditional_metrics,
)

DensityFn = Callable[[np.ndarray, np.ndarray, np.ndarray | None], np.ndarray]


@dataclass(frozen=True)
class ClaytonReference:
  """Clayton copula with tau(x) linear in a single covariate x in [x_min, x_max]."""

  tau_min: float = 0.1
  tau_max: float = 0.9
  x_min: float = 0.0
  x_max: float = 1.0

  def tau_of_x(self, x: np.ndarray) -> np.ndarray:
    frac = (np.asarray(x, float) - self.x_min) / (self.x_max - self.x_min)
    return self.tau_min + (self.tau_max - self.tau_min) * np.clip(
      frac, 0.0, 1.0
    )

  def _theta(self, x: np.ndarray) -> np.ndarray:
    tau = self.tau_of_x(x)
    return 2.0 * tau / (1.0 - tau)  # Clayton: tau = theta / (theta + 2)

  def simulate(
    self, n: int, rng: np.random.Generator
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x, u, v)`` with x spanning the covariate range."""
    x = np.linspace(self.x_min, self.x_max, n)
    theta = self._theta(x)
    u = rng.uniform(1e-6, 1.0 - 1e-6, n)
    w = rng.uniform(1e-6, 1.0 - 1e-6, n)
    v = np.empty(n)
    for i in range(n):
      bicop = pv.Bicop(family=clayton, parameters=np.array([[theta[i]]]))
      v[i] = np.asarray(bicop.hinv1(np.array([[u[i], w[i]]])))[0]
    return x, u, np.clip(v, 1e-6, 1.0 - 1e-6)

  def true_density(
    self, u: np.ndarray, v: np.ndarray, x: np.ndarray | None
  ) -> np.ndarray:
    """Exact conditional Clayton density ``c(u, v | x)`` (x required)."""
    if x is None:
      raise ValueError("ClaytonReference is conditional; x is required.")
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    x_flat = np.asarray(x, float).reshape(len(u), -1)[:, 0]
    theta = self._theta(x_flat)
    out = np.empty(len(u))
    for i in range(len(u)):
      bicop = pv.Bicop(family=clayton, parameters=np.array([[theta[i]]]))
      out[i] = np.asarray(bicop.pdf(np.array([[u[i], v[i]]])))[0]
    return out


def conditional_report(
  est_density_fn: DensityFn,
  true_density_fn: DensityFn,
  spec: ConditionalGridSpec,
) -> dict[str, float]:
  """ISE/IAE/KL of an estimated vs true conditional density over the grid."""
  true_grid = conditional_density_grids(true_density_fn, spec)
  est_grid = conditional_density_grids(est_density_fn, spec)
  metrics = conditional_metrics(true_grid, est_grid, reduce_over="all")
  return {k: float(v) for k, v in metrics.items()}
