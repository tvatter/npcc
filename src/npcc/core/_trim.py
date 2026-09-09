"""The open unit interval: clamping into it, and mapping out of it.

Copula scores live in ``(0, 1)`` open, and both directions across that
boundary need care. :func:`check_uv` is this package's *domain* step, the
counterpart of pyvinecopulib's ``core._trim.trim``, and differs from it
for a reason: ``trim`` clamps silently at the working precision, while this
**rejects** a value at or outside ``{0, 1}`` before clamping to a caller-chosen
``eps``. A copula argument of exactly ``0`` or ``1`` is a caller error here
rather than a rounding artifact, since every score reaching an estimator comes
from a probability integral transform that cannot produce one.

:func:`logit` maps the other way, onto the whole real line, which is what the
inner distributional regressors are fitted on.
"""

from __future__ import annotations

import torch

__all__ = ["check_uv", "logit"]


def check_uv(
  u: torch.Tensor,
  v: torch.Tensor,
  eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Validate copula coordinates and clip them away from ``{0, 1}``."""
  u_t = u.reshape(-1)
  v_t = v.reshape(-1)

  if u_t.shape != v_t.shape:
    raise ValueError("u and v must have the same shape.")

  if torch.any((u_t <= 0.0) | (u_t >= 1.0)) or torch.any(
    (v_t <= 0.0) | (v_t >= 1.0)
  ):
    raise ValueError("u and v must lie strictly inside (0, 1).")

  return (
    torch.clamp(u_t, eps, 1.0 - eps),
    torch.clamp(v_t, eps, 1.0 - eps),
  )


def logit(p: torch.Tensor) -> torch.Tensor:
  """Numerically stable logit ``log(p) - log1p(-p)``."""
  return torch.log(p) - torch.log1p(-p)
