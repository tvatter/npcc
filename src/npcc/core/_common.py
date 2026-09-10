"""Small Torch helpers shared across NPCC modules."""

from __future__ import annotations

import torch


def _resolve_device(device: str | torch.device | None) -> torch.device:
  """Resolve ``None`` to ``cuda`` if available, else ``cpu``.

  A bare ``cuda`` device (no index) is normalised to ``cuda:<current index>``
  so it compares equal to the device tensors actually materialise on (e.g.
  ``cuda:0``); ``torch.device("cuda") != torch.device("cuda:0")`` otherwise.
  """
  if device is None:
    resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
    resolved = torch.device(device)

  if resolved.type == "cuda" and resolved.index is None:
    resolved = torch.device("cuda", torch.cuda.current_device())

  return resolved


def _check_uv(
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


def _logit(p: torch.Tensor) -> torch.Tensor:
  """Numerically stable logit ``log(p) - log1p(-p)``."""
  return torch.log(p) - torch.log1p(-p)


def _torch_interp(
  x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor
) -> torch.Tensor:
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


def _torch_interp_batched_xp(
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


def _torch_interp_batched_fp(
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


def _torch_gradient_1d(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
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
