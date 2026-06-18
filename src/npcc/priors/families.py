"""
families.py — copula-family registry for the conditional-copula prior.

Each :class:`FamilySpec` pins one ``(pyvinecopulib family, rotation)`` pair
together with everything the prior needs to turn a covariate-driven signal into
valid copula parameters:

- ``tau_capable`` families (gaussian, frank, clayton, gumbel, joe) can be
  parameterised through Kendall's ``tau`` via ``pv.Bicop.tau_to_parameters``
  (the gampcc / amortized-vines style). Archimedean families carry the
  dependence *sign* in the rotation (0/180 positive, 90/270 negative), so their
  signed ``tau`` window is one-sided and ``tau_uses_abs`` feeds the magnitude to
  ``tau_to_parameters``.
- every family is ``native_capable`` except frank (kept tau-only to avoid the
  degenerate ``theta -> 0`` independence point): its native parameters are
  drawn directly inside ``native_bounds`` (the "parameter as a function of the
  covariates without going through Kendall's tau" mode the project wants).
- the two-parameter families (student, BB1/6/7/8) are native-only; ``tau`` does
  not pin them uniquely (``tau_to_parameters`` raises for them in pv 0.7.5).
- ``indep`` is a zero-parameter regulariser (``c == 1``).

The bounds below were verified constructible (finite ``pdf``/``hinv1`` at every
range corner) against pyvinecopulib 0.7.5; see ``tests/test_priors.py`` for the
endpoint guard. This module is correctness-critical (it defines the learning
target); treat the windows and bounds as load-bearing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvinecopulib as pv

# pyvinecopulib (>=1.0) exposes the families as module-level constants; this is
# the type-checkable way to name them (the BicopFamily enum members are not
# class attributes in the shipped stubs).
from pyvinecopulib.families import (
  bb1,
  bb6,
  bb7,
  bb8,
  clayton,
  frank,
  gaussian,
  gumbel,
  indep,
  joe,
  student,
)

_ARCH_ROTATIONS = (0, 90, 180, 270)
"""Rotations supported by the archimedean / BB families (gaussian/frank/student
are rotation-0 only)."""


@dataclass(frozen=True)
class FamilySpec:
  """A copula family + rotation usable in the prior, with its parameter ranges.

  Attributes
  ----------
  name
      Registry key, e.g. ``"clayton_90"`` or ``"gaussian"``.
  family
      The underlying ``pyvinecopulib`` family.
  rotation
      Density rotation (0/90/180/270), matching pyvinecopulib's convention.
  n_params
      Number of native parameters (0 for independence, 1 or 2 otherwise).
  tau_capable
      Whether the family can be parameterised through Kendall's ``tau``.
  tau_uses_abs
      For archimedean families the rotation carries the sign, so the magnitude
      ``|tau|`` is fed to ``tau_to_parameters``; gaussian/frank pass signed tau.
  tau_lo, tau_hi
      Signed Kendall's-tau window for tau-mode draws (one-sided for rotated
      archimedean families).
  native_capable
      Whether native-parameter draws are offered for this family.
  native_bounds
      Per-native-parameter ``(lo, hi)`` ranges for native-mode draws.
  is_independence
      Marks the zero-parameter independence copula (``c == 1``).
  """

  name: str
  family: pv.BicopFamily
  rotation: int
  n_params: int
  tau_capable: bool
  tau_uses_abs: bool
  tau_lo: float
  tau_hi: float
  native_capable: bool
  native_bounds: tuple[tuple[float, float], ...]
  is_independence: bool = False


# A default-parameter Bicop per family, reused for the (parameter-free) scalar
# tau <-> parameter conversion. Rotation does not affect the magnitude map.
_TAU_REF: dict[pv.BicopFamily, pv.Bicop] = {}


def _tau_ref(family: pv.BicopFamily) -> pv.Bicop:
  bc = _TAU_REF.get(family)
  if bc is None:
    bc = pv.Bicop(family=family)
    _TAU_REF[family] = bc
  return bc


def params_from_tau(spec: FamilySpec, tau: float) -> np.ndarray:
  """Convert a signed per-row ``tau`` to native parameters, shape ``(n_params, 1)``.

  Only valid for ``tau_capable`` families. Archimedean families feed the
  magnitude ``|tau|`` (sign lives in the rotation); ``tau_to_parameters``
  additionally clamps to the family's valid range (e.g. Clayton ``theta <= 28``).
  """
  value = abs(tau) if spec.tau_uses_abs else tau
  return np.asarray(
    _tau_ref(spec.family).tau_to_parameters(float(value)), float
  )


def build_bicop(spec: FamilySpec, params: np.ndarray | None) -> pv.Bicop:
  """Construct the ``pv.Bicop`` for ``spec`` at the given native parameters.

  ``params`` is coerced to the required ``(n_params, 1)`` column shape;
  independence ignores it.
  """
  if spec.is_independence:
    return pv.Bicop(family=spec.family, rotation=spec.rotation)
  cols = np.asarray(params, float).reshape(-1, 1)
  return pv.Bicop(family=spec.family, rotation=spec.rotation, parameters=cols)


def _make_registry() -> tuple[FamilySpec, ...]:
  specs: list[FamilySpec] = []

  # Elliptical / Frank: rotation 0, signed dependence carried by the parameter.
  specs.append(
    FamilySpec(
      "gaussian",
      gaussian,
      0,
      1,
      True,
      False,
      -0.95,
      0.95,
      True,
      ((-0.95, 0.95),),
    )
  )
  specs.append(
    FamilySpec(
      "frank",
      frank,
      0,
      1,
      True,
      False,
      -0.95,
      0.95,
      False,
      ((-25.0, 25.0),),
    )
  )

  # One-parameter archimedean: sign via rotation, magnitude via tau or native.
  arch = (
    ("clayton", clayton, (0.05, 28.0)),
    ("gumbel", gumbel, (1.0, 17.0)),
    ("joe", joe, (1.0, 17.0)),
  )
  for fam_name, fam, bounds in arch:
    for rot in _ARCH_ROTATIONS:
      positive = rot in (0, 180)
      tau_lo, tau_hi = (0.02, 0.95) if positive else (-0.95, -0.02)
      specs.append(
        FamilySpec(
          f"{fam_name}_{rot}",
          fam,
          rot,
          1,
          True,
          True,
          tau_lo,
          tau_hi,
          True,
          (bounds,),
        )
      )

  # Student-t: rotation 0, native (rho, df); tau does not pin it uniquely.
  specs.append(
    FamilySpec(
      "student",
      student,
      0,
      2,
      False,
      False,
      0.0,
      0.0,
      True,
      ((-0.95, 0.95), (2.0, 30.0)),
    )
  )

  # Two-parameter BB families: native-only, all rotations.
  bb = (
    ("bb1", bb1, ((0.05, 6.0), (1.0, 6.0))),
    ("bb6", bb6, ((1.0, 5.0), (1.0, 5.0))),
    ("bb7", bb7, ((1.0, 5.0), (0.2, 5.0))),
    ("bb8", bb8, ((1.0, 7.5), (0.1, 0.99))),
  )
  for fam_name, fam, bounds in bb:
    for rot in _ARCH_ROTATIONS:
      specs.append(
        FamilySpec(
          f"{fam_name}_{rot}",
          fam,
          rot,
          2,
          False,
          False,
          0.0,
          0.0,
          True,
          bounds,
        )
      )

  # Independence regulariser.
  specs.append(
    FamilySpec(
      "indep",
      indep,
      0,
      0,
      False,
      False,
      0.0,
      0.0,
      False,
      (),
      is_independence=True,
    )
  )
  return tuple(specs)


DEFAULT_REGISTRY: tuple[FamilySpec, ...] = _make_registry()
"""The default rich family set: gaussian, frank, clayton/gumbel/joe (4 rotations
each), student-t, BB1/6/7/8 (4 rotations each), and independence."""
