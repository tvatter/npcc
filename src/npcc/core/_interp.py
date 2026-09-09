"""Linear interpolation and finite differences over a sampled axis.

The quantile-table margins reconstruct a density, a distribution function and
a quantile function from a table of predicted quantiles, which is
interpolation in three directions plus one derivative. NumPy has
:func:`numpy.interp` and :func:`numpy.gradient` for the unbatched cases and
nothing for the batched ones; these are the Torch equivalents, on the device
the table was predicted on.

pyvinecopulib's own interpolation helpers do not cover this: its
``torch._bicop_interp`` is bilinear on the unit square for pair-copula density
grids, and ``torch._margin_kde1d_interp`` is cubic over kernel-density cells.
Both are private, and neither is a general one-dimensional linear
interpolator.
"""

from __future__ import annotations

import torch

__all__ = [
  "gradient_1d",
  "interp",
  "interp_batched_fp",
  "interp_batched_xp",
]


def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
  """1-D linear interpolation, analogue of :func:`numpy.interp`.

  ``xp`` must be sorted ascending.  Values of ``x`` outside
  ``[xp[0], xp[-1]]`` are clamped to the endpoints (flat extrapolation),
  matching NumPy's default behaviour.  Inputs are 1-D; output has the
  shape of ``x``.
  """
  n = xp.shape[0]
  idx = torch.searchsorted(xp, x).clamp(1, n - 1)
  x0 = xp[idx - 1]
  x1 = xp[idx]
  y0 = fp[idx - 1]
  y1 = fp[idx]
  denom = (x1 - x0).clamp_min(torch.finfo(xp.dtype).tiny)
  t = ((x - x0) / denom).clamp(0.0, 1.0)
  return y0 + t * (y1 - y0)


def interp_batched_xp(
  x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor
) -> torch.Tensor:
  """Per-row linear interpolation with row-specific ``xp`` and ``fp``.

  ``x`` is shape ``(n,)``, ``xp`` and ``fp`` are shape ``(n, k)`` with
  each row sorted ascending in ``xp``.  Output has shape ``(n,)``.
  Values of ``x[i]`` outside ``[xp[i, 0], xp[i, -1]]`` are clamped
  (flat extrapolation).
  """
  n, k = xp.shape
  xp = xp.contiguous()
  values = x.unsqueeze(1).contiguous()
  idx = torch.searchsorted(xp, values).squeeze(1).clamp(1, k - 1)
  rows = torch.arange(n, device=xp.device)
  x0 = xp[rows, idx - 1]
  x1 = xp[rows, idx]
  y0 = fp[rows, idx - 1]
  y1 = fp[rows, idx]
  denom = (x1 - x0).clamp_min(torch.finfo(xp.dtype).tiny)
  t = ((x - x0) / denom).clamp(0.0, 1.0)
  return y0 + t * (y1 - y0)


def interp_batched_fp(
  x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor
) -> torch.Tensor:
  """Per-row linear interpolation with shared ``xp`` and row-specific ``fp``.

  ``x`` is shape ``(n,)``, ``xp`` is shape ``(k,)`` (sorted), ``fp``
  is shape ``(n, k)``.  Output has shape ``(n,)``.
  """
  k = xp.shape[0]
  idx = torch.searchsorted(xp, x).clamp(1, k - 1)
  rows = torch.arange(x.shape[0], device=xp.device)
  x0 = xp[idx - 1]
  x1 = xp[idx]
  y0 = fp[rows, idx - 1]
  y1 = fp[rows, idx]
  denom = (x1 - x0).clamp_min(torch.finfo(xp.dtype).tiny)
  t = ((x - x0) / denom).clamp(0.0, 1.0)
  return y0 + t * (y1 - y0)


def gradient_1d(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
  """Central differences over the last axis, with one-sided edges.

  Mirrors :func:`numpy.gradient` for a 1-D coordinate ``x`` and a
  values tensor ``y`` whose last axis is sampled along ``x``.  Works
  for any leading shape: ``y`` of shape ``(..., k)`` returns ``(..., k)``.
  """
  k = x.shape[0]
  if k < 2:
    raise ValueError("x must have at least 2 points.")
  out = torch.empty_like(y)
  out[..., 0] = (y[..., 1] - y[..., 0]) / (x[1] - x[0])
  out[..., -1] = (y[..., -1] - y[..., -2]) / (x[-1] - x[-2])
  if k > 2:
    dx = x[2:] - x[:-2]
    out[..., 1:-1] = (y[..., 2:] - y[..., :-2]) / dx
  return out
