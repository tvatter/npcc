"""
_common.py — small array helpers shared across npcc modules.

All exports are private (leading underscore) and are not part of the
package's public API.  They live here only to avoid duplication between
:mod:`npcc.tabpfn_quantile_distribution1d`,
:mod:`npcc.tabpfn_criterion_distribution1d`, and
:mod:`npcc.pfnr_bicop`.

The helpers are torch-aware: numeric inputs may be NumPy arrays or
torch tensors, and the corresponding output is a torch tensor on the
caller-supplied device.  See ``_normalize_inputs`` / ``_wrap_output``
for the round-trip pattern at public-API boundaries.
"""

from __future__ import annotations

import numpy as np
import torch

TensorLike = np.ndarray | torch.Tensor
"""Public-API numeric input type: NumPy array or torch tensor."""


def _resolve_device(device: str | torch.device | None) -> torch.device:
  """Resolve ``None`` to ``cuda`` if available, else ``cpu``."""
  if device is None:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return torch.device(device)


def _to_tensor(
  x: TensorLike,
  *,
  device: torch.device,
  dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
  """Convert ``x`` to a tensor on ``device`` with ``dtype``.

  No-op for tensors that already match; otherwise calls ``.to`` (for
  tensors) or ``torch.as_tensor`` (for NumPy / array-like).  Default
  dtype is ``float64`` to preserve numerical precision in the
  trapezoidal integrators and quantile-table inversions; ``float32``
  is used only at the TabPFN ``criterion`` boundary, where the head
  consumes the regressor's float32 logits.
  """
  if isinstance(x, torch.Tensor):
    return x.to(device=device, dtype=dtype)
  return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def _normalize_inputs(
  *inputs: TensorLike | None,
  device: torch.device,
) -> tuple[bool, list[torch.Tensor | None]]:
  """Coerce mixed NumPy / torch inputs to tensors on ``device``.

  Returns ``(return_as_torch, tensors)`` where ``return_as_torch`` is
  ``True`` iff at least one non-``None`` input was a torch tensor.
  ``None`` entries pass through to let callers preserve optional-arg
  semantics (e.g. ``x: TensorLike | None``).
  """
  return_as_torch = any(
    isinstance(x, torch.Tensor) for x in inputs if x is not None
  )
  tensors: list[torch.Tensor | None] = [
    None if x is None else _to_tensor(x, device=device) for x in inputs
  ]
  return return_as_torch, tensors


def _wrap_output(x: torch.Tensor, *, return_as_torch: bool) -> TensorLike:
  """Detach + move to CPU + cast to NumPy unless ``return_as_torch``."""
  if return_as_torch:
    return x
  return x.detach().cpu().numpy()


def _as_2d(
  x: TensorLike, *, device: torch.device | None = None
) -> torch.Tensor:
  """Reshape a 1-D input to ``(n, 1)`` and pass 2-D through untouched.

  Anything else (0-D, 3-D, ...) is rejected.  Used to normalise the
  conditioning feature matrix ``W`` (or ``X``) at the boundary of every
  ``fit`` / ``density`` call so internal code can assume two dimensions.
  """
  t = _to_tensor(x, device=_resolve_device(device))
  if t.ndim == 1:
    t = t.reshape(-1, 1)
  if t.ndim != 2:
    raise ValueError("Expected a 1D or 2D array.")
  return t


def _check_uv(
  u: TensorLike,
  v: TensorLike,
  eps: float,
  *,
  device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Validate copula coordinates and clip them away from ``{0, 1}``.

  Copulas live on the open unit square, so any input on or outside the
  boundary is a user error.  The valid points are then clipped into
  ``[eps, 1 - eps]`` so downstream logit transforms cannot blow up.
  """
  dev = _resolve_device(device)
  u_t = _to_tensor(u, device=dev).reshape(-1)
  v_t = _to_tensor(v, device=dev).reshape(-1)
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


def _logit(p: TensorLike) -> torch.Tensor:
  """Numerically stable logit ``log(p) - log1p(-p)``.

  When ``p`` is already a tensor, the output stays on its device;
  NumPy inputs land on the default device (auto-detected).
  """
  if isinstance(p, torch.Tensor):
    p_t = p
  else:
    p_t = _to_tensor(p, device=_resolve_device(None))
  return torch.log(p_t) - torch.log1p(-p_t)


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
  idx = torch.searchsorted(xp, x.unsqueeze(1)).squeeze(1).clamp(1, k - 1)
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
