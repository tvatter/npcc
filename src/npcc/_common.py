"""
_common.py — small array helpers shared across npcc modules.

All exports are private (leading underscore) and are not part of the
package's public API.  They live here only to avoid duplication between
:mod:`npcc.tabpfn_quantile_density1d`, :mod:`npcc.tabpfn_density1d`, and
:mod:`npcc.pfnr_bicop`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _as_2d(x: ArrayLike) -> np.ndarray:
  """Reshape a 1-D array to ``(n, 1)`` and pass 2-D through untouched.

  Anything else (0-D, 3-D, ...) is rejected.  Used to normalise the
  conditioning feature matrix ``W`` (or ``X``) at the boundary of every
  ``fit`` / ``density`` call so internal code can assume two dimensions.
  """
  arr = np.asarray(x, dtype=float)
  if arr.ndim == 1:
    arr = arr.reshape(-1, 1)
  if arr.ndim != 2:
    raise ValueError("Expected a 1D or 2D array.")
  return arr


def _check_uv(
  u: ArrayLike, v: ArrayLike, eps: float
) -> tuple[np.ndarray, np.ndarray]:
  """Validate copula coordinates and clip them away from ``{0, 1}``.

  Copulas live on the open unit square, so any input on or outside the
  boundary is a user error.  The valid points are then clipped into
  ``[eps, 1 - eps]`` so downstream logit transforms cannot blow up.
  """
  u_arr = np.asarray(u, dtype=float).reshape(-1)
  v_arr = np.asarray(v, dtype=float).reshape(-1)
  if u_arr.shape != v_arr.shape:
    raise ValueError("u and v must have the same shape.")
  if np.any((u_arr <= 0.0) | (u_arr >= 1.0)) or np.any(
    (v_arr <= 0.0) | (v_arr >= 1.0)
  ):
    raise ValueError("u and v must lie strictly inside (0, 1).")
  return (
    np.clip(u_arr, eps, 1.0 - eps),
    np.clip(v_arr, eps, 1.0 - eps),
  )


def _logit(p: np.ndarray) -> np.ndarray:
  """Numerically stable logit ``log(p) - log1p(-p)``."""
  return np.log(p) - np.log1p(-p)
