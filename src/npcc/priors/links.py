"""
links.py — covariate links for the conditional-copula prior.

A :class:`CovariateLink` is the gampcc-style additive predictor

    eta(x) = intercept + x @ linear + sum_j basis_j(x) * coef_j,

with an optional random-Fourier-feature basis ``basis_j(x) = sin(omega_j . x +
phi_j)`` for smooth non-linear covariate effects. The scalar ``eta`` is mapped
to a copula quantity downstream:

- tau-mode: ``tau = lo + (hi - lo) * sigmoid(eta)`` into the family's signed
  Kendall's-tau window, then converted to native parameters;
- native-mode: one link per native parameter, each squashed into its
  ``(lo, hi)`` bound via the same :func:`affine_to_interval`.

A **constant** link (``linear == 0`` and no basis) makes the copula parameter
independent of ``x`` — i.e. an unconditional copula (constant Kendall's tau),
which is also what ``p == 0`` covariates produce automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CovariateLink:
  """Additive predictor ``eta(x)`` with an optional random-Fourier basis.

  Attributes
  ----------
  intercept
      Scalar offset.
  linear
      Linear coefficients, shape ``(p,)`` (all-zero for a constant link).
  freqs
      Random Fourier frequencies, shape ``(n_basis, p)``, or ``None``.
  phases
      Random phases, shape ``(n_basis,)``, or ``None``.
  basis_coef
      Basis coefficients, shape ``(n_basis,)``, or ``None``.
  """

  intercept: float
  linear: np.ndarray
  freqs: np.ndarray | None
  phases: np.ndarray | None
  basis_coef: np.ndarray | None

  def __call__(self, x: np.ndarray) -> np.ndarray:
    """Evaluate ``eta`` at covariate rows ``x`` of shape ``(n, p)`` -> ``(n,)``."""
    x = np.asarray(x, float)
    eta = self.intercept + x @ self.linear
    if self.freqs is not None and self.freqs.shape[0] > 0:
      features = np.sin(x @ self.freqs.T + self.phases)
      eta = eta + features @ self.basis_coef
    return np.asarray(eta, float)


def _sigmoid(z: np.ndarray) -> np.ndarray:
  # Numerically stable logistic via tanh.
  return 0.5 * (1.0 + np.tanh(0.5 * z))


def affine_to_interval(eta: np.ndarray, lo: float, hi: float) -> np.ndarray:
  """Monotonically map ``eta in R`` into the open interval ``(lo, hi)``."""
  return lo + (hi - lo) * _sigmoid(eta)
