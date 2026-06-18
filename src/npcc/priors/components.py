"""
components.py — copula components and mixtures for the conditional-copula prior.

A :class:`ParameterProcess` turns covariates ``X`` into per-row native copula
parameters, in one of three modes:

- ``"tau"``   — one :class:`CovariateLink` -> Kendall's tau -> native parameters
                (``tau_capable`` families);
- ``"native"``— one link per native parameter, each squashed into its bound
                (the "parameter as a function of covariates" mode);
- ``"none"``  — the parameter-free independence copula.

A :class:`CopulaComponent` evaluates the density and the conditional-inverse
``hinv1`` per row, grouping rows that share identical parameters into a single
vectorised ``pyvinecopulib`` call (so constant / unconditional parameters cost
one call). A :class:`CopulaMixture` is a Dirichlet-weighted sum of components;
its density is the weighted sum and its simulator draws a component per row.

By the uniform-margin identity ``f_{V|U,X} = f_{U|V,X} = c(u,v|x)``, the same
mixture is the target for both Rosenblatt directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from npcc.priors.families import FamilySpec, build_bicop, params_from_tau
from npcc.priors.links import CovariateLink, affine_to_interval


@dataclass(frozen=True)
class ParameterProcess:
  """Maps covariates to per-row native copula parameters for one family."""

  spec: FamilySpec
  mode: Literal["tau", "native", "none"]
  links: tuple[CovariateLink, ...]

  def params(self, x: np.ndarray) -> np.ndarray:
    """Per-row native parameters, shape ``(n, n_params)`` (``(n, 0)`` for indep)."""
    n = len(x)
    if self.mode == "none":
      return np.zeros((n, 0))
    if self.mode == "tau":
      tau = affine_to_interval(
        self.links[0](x), self.spec.tau_lo, self.spec.tau_hi
      )
      out = np.empty((n, self.spec.n_params))
      for i, t in enumerate(tau):
        out[i] = params_from_tau(self.spec, float(t)).ravel()
      return out
    cols = [
      affine_to_interval(link(x), lo, hi)
      for link, (lo, hi) in zip(self.links, self.spec.native_bounds)
    ]
    return np.column_stack(cols)


def _eval_per_row(
  spec: FamilySpec,
  params: np.ndarray,
  a: np.ndarray,
  b: np.ndarray,
  method: str,
) -> np.ndarray:
  """Evaluate ``bc.pdf`` or ``bc.hinv1`` at ``(a, b)`` with per-row parameters.

  Rows sharing identical parameters are grouped into one vectorised call, so a
  constant parameter (the unconditional case) costs a single ``pv`` call.
  """
  n = len(a)
  out = np.empty(n)
  if spec.is_independence:
    # c == 1: pdf is 1; the conditional-inverse of independence is hinv1(u, w) = w.
    out[:] = 1.0 if method == "pdf" else np.asarray(b, float)
    return out

  ab = np.column_stack([np.asarray(a, float), np.asarray(b, float)])
  key = np.round(params, 9)
  uniq, inverse = np.unique(key, axis=0, return_inverse=True)
  inverse = np.asarray(inverse).ravel()
  for g in range(len(uniq)):
    idx = np.flatnonzero(inverse == g)
    bicop = build_bicop(spec, uniq[g])
    out[idx] = np.asarray(getattr(bicop, method)(ab[idx]), float).ravel()
  return out


@dataclass(frozen=True)
class CopulaComponent:
  """A single family/rotation with a covariate-driven parameter process."""

  process: ParameterProcess

  def pdf(self, u: np.ndarray, v: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-row copula density ``c(u_i, v_i | x_i)``."""
    return _eval_per_row(self.process.spec, self.process.params(x), u, v, "pdf")

  def hinv1(self, u: np.ndarray, w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-row conditional inverse ``v_i = C^{-1}_{V|U}(w_i | u_i, x_i)``."""
    return _eval_per_row(
      self.process.spec, self.process.params(x), u, w, "hinv1"
    )


@dataclass(frozen=True)
class CopulaMixture:
  """Dirichlet-weighted mixture of :class:`CopulaComponent` (still a copula)."""

  components: tuple[CopulaComponent, ...]
  weights: np.ndarray

  def pdf(self, u: np.ndarray, v: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Mixture density ``sum_k w_k c_k(u, v | x)``."""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    x = np.asarray(x, float)
    total = np.zeros(len(u))
    for weight, comp in zip(self.weights, self.components):
      total = total + weight * comp.pdf(u, v, x)
    return total

  def simulate(
    self, u: np.ndarray, w: np.ndarray, x: np.ndarray, rng: np.random.Generator
  ) -> np.ndarray:
    """Draw ``v`` given uniform ``u``/``w``: pick a component per row, then ``hinv1``.

    Valid mixture sampling because every component has uniform margins, so
    ``u ~ U(0, 1)`` regardless of the chosen component and
    ``v | u ~ C_k^{-1}(w | u)`` yields joint density ``sum_k w_k c_k``.
    """
    n = len(u)
    out = np.empty(n)
    comp_idx = rng.choice(len(self.components), size=n, p=self.weights)
    for k, comp in enumerate(self.components):
      mask = comp_idx == k
      if mask.any():
        out[mask] = comp.hinv1(u[mask], w[mask], x[mask])
    return out
