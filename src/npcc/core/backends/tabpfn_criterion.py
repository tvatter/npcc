"""
tabpfn_criterion.py — native-head TabPFN backend (the default).

TabPFN's regressor is internally a classifier over a "bar distribution".
``predict(W, output_type="full")`` returns per-row logits over the bins
plus a ``criterion`` head exposing ``pdf`` / ``cdf`` / ``icdf`` that
evaluate the corresponding density / CDF / quantile at arbitrary points.
Reading the head directly avoids the quantile-table inversion and is
both faster and (currently) as accurate — hence this is the default
backend for TabPFN.

This backend subclasses the neutral
:class:`~npcc.core.conditional_distribution1d.ConditionalDistribution1D`
directly (it is native-evaluation, not quantile-table based) and
provides its own fast ``pdf_grid`` / ``cdf_grid`` overrides that predict
once per conditioning row.

Support transforms and the change-of-variables are as documented on the
base class: ``f_Y(y | w) = f_Z(T(y) | w) * |T'(y)|``; CDFs and quantiles
need no Jacobian.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

import numpy as np
import torch

from npcc.core._common import (
  TensorLike,
  _as_2d,
  _normalize_inputs,
  _wrap_output,
)
from npcc.core.backends.tabpfn_common import (
  _DEFAULT_MODEL_VERSION,
  ModelVersion,
  make_tabpfn_regressor,
)
from npcc.core.conditional_distribution1d import ConditionalDistribution1D


class _CriterionLike(Protocol):
  """Duck-typed view of TabPFN's ``output_type="full"`` distribution head.

  We call ``pdf``, ``cdf``, and ``icdf``; no need to depend on the
  concrete TabPFN class, which has changed name across versions.
  """

  def pdf(self, logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor: ...
  def cdf(self, logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor: ...
  def icdf(self, logits: torch.Tensor, left_prob: float) -> torch.Tensor: ...


def _coerce_logits_tensor(
  logits: object, device: torch.device | str
) -> torch.Tensor:
  """Convert TabPFN ``full`` logits to a float32 tensor on ``device``.

  TabPFN may return masked / invalid bins as ``None`` inside an object
  array; map those to ``-inf`` so a downstream softmax assigns them zero
  probability.
  """
  if isinstance(logits, torch.Tensor):
    return logits.to(device=device, dtype=torch.float32)
  arr = np.asarray(logits, dtype=object)
  safe = np.empty(arr.shape, dtype=np.float32)
  for idx, val in np.ndenumerate(arr):
    safe[idx] = -np.inf if val is None else float(val)
  return torch.as_tensor(safe, dtype=torch.float32, device=device)


class TabPFNCriterionBackend(ConditionalDistribution1D):
  """Conditional predictive distribution via TabPFN's native binned head.

  Parameters
  ----------
  transform, eps, device, batch_size
      Forwarded to :class:`ConditionalDistribution1D`.
  model_kwargs
      Forwarded to the ``TabPFNRegressor`` constructor.
  model_version
      TabPFN model version (default: v3).
  """

  model_: Any | None

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
    model_version: ModelVersion | None = _DEFAULT_MODEL_VERSION,
  ) -> None:
    super().__init__(
      transform=transform,
      eps=eps,
      device=device,
      batch_size=batch_size,
    )
    self.model_kwargs = dict(model_kwargs or {})
    self.model_kwargs.setdefault("device", str(self._device))
    self.model_version = model_version
    self.model_ = None

  def _fit_model(self, w: torch.Tensor, z: torch.Tensor) -> None:
    self.model_ = make_tabpfn_regressor(self.model_version, self.model_kwargs)
    self.model_.fit(w, z)

  # ------------------------------------------------------------------
  # Internal helpers.
  # ------------------------------------------------------------------

  def _predict_full(
    self, w_t: torch.Tensor
  ) -> tuple[torch.Tensor, _CriterionLike]:
    """Run a single ``output_type="full"`` forward pass.

    TabPFN's predict input must be on CPU; the returned logits land on
    TabPFN's internal device, which we coerce onto ``self._device``.
    """
    assert self.model_ is not None
    pred = self.model_.predict(w_t.detach().cpu(), output_type="full")
    logits = pred["logits"]
    criterion: _CriterionLike = pred["criterion"]
    return _coerce_logits_tensor(logits, device=self._device), criterion

  def _criterion_pdf_z(
    self,
    logits_t: torch.Tensor,
    criterion: _CriterionLike,
    z: torch.Tensor,
  ) -> torch.Tensor:
    """Evaluate ``criterion.pdf`` at z-space points; returns shape ``(n,)``."""
    z_eval = z.to(dtype=logits_t.dtype, device=logits_t.device).reshape(-1, 1)
    dens = criterion.pdf(logits_t, z_eval)
    return dens.reshape(-1).to(dtype=z.dtype)

  def _criterion_cdf_z(
    self,
    logits_t: torch.Tensor,
    criterion: _CriterionLike,
    z: torch.Tensor,
  ) -> torch.Tensor:
    """Evaluate ``criterion.cdf`` at the (already z-space) eval points."""
    z_eval = z.to(dtype=logits_t.dtype, device=logits_t.device).reshape(-1, 1)
    cdf = criterion.cdf(logits_t, z_eval)
    return cdf.reshape(-1).to(dtype=z.dtype)

  # ------------------------------------------------------------------
  # Public API: pdf / cdf / icdf + the *_grid Cartesian fast paths.
  # ------------------------------------------------------------------

  def pdf(
    self, w: TensorLike, y: TensorLike, *, batch_size: int | None = None
  ) -> TensorLike:
    """Return ``f(y_i | w_i)`` for each row ``i`` (chunked over rows)."""
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    return_as_torch, (w_in, y_in) = _normalize_inputs(w, y, device=self._device)
    assert w_in is not None and y_in is not None
    w_t = _as_2d(w_in, device=self._device)
    y_t = y_in.reshape(-1)
    if y_t.shape[0] != w_t.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_t)
    n = y_t.shape[0]
    parts: list[torch.Tensor] = []

    for start in range(0, n, effective_batch_size):
      end = min(start + effective_batch_size, n)
      logits_t, criterion = self._predict_full(w_t[start:end])
      parts.append(self._criterion_pdf_z(logits_t, criterion, z[start:end]))

    dens_z = torch.cat(parts) if parts else torch.empty(0, device=self._device)
    out = dens_z * self._jacobian_inverse(y_t)
    return _wrap_output(out, return_as_torch=return_as_torch)

  def pdf_grid(
    self,
    w: TensorLike,
    y_grid: TensorLike,
    *,
    batch_size: int | None = None,
  ) -> TensorLike:
    """Density on the Cartesian product of ``w`` rows and ``y_grid``.

    Returns shape ``(n_w, n_y)``.  Each ``w`` row triggers exactly one
    TabPFN forward pass; the same logits are evaluated against every
    ``y`` value.
    """
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    return_as_torch, (w_in, y_in) = _normalize_inputs(
      w, y_grid, device=self._device
    )
    assert w_in is not None and y_in is not None
    w_t = _as_2d(w_in, device=self._device)
    y_grid_t = y_in.reshape(-1)
    if y_grid_t.numel() == 0:
      raise ValueError("y_grid must contain at least one value.")

    z_grid_t = self._transform_y(y_grid_t)
    jac = self._jacobian_inverse(y_grid_t)

    chunks: list[torch.Tensor] = []
    n_grid = y_grid_t.shape[0]

    for start in range(0, w_t.shape[0], effective_batch_size):
      stop = min(start + effective_batch_size, w_t.shape[0])
      w_chunk = w_t[start:stop]
      logits_t, criterion = self._predict_full(w_chunk)

      logits_rep = logits_t.repeat_interleave(n_grid, dim=0)
      z_flat = (
        z_grid_t.repeat(w_chunk.shape[0])
        .to(dtype=logits_t.dtype, device=logits_t.device)
        .reshape(-1, 1)
      )

      pdf_z_flat = criterion.pdf(logits_rep, z_flat)
      if not isinstance(pdf_z_flat, torch.Tensor):
        pdf_z_flat = torch.as_tensor(
          pdf_z_flat, dtype=torch.float32, device=self._device
        )

      pdf_z = pdf_z_flat.reshape(w_chunk.shape[0], n_grid).to(
        dtype=y_grid_t.dtype
      )
      chunks.append(pdf_z * jac.unsqueeze(0))

    out = torch.cat(chunks, dim=0)
    return _wrap_output(out, return_as_torch=return_as_torch)

  def cdf(
    self, w: TensorLike, y: TensorLike, *, batch_size: int | None = None
  ) -> TensorLike:
    """Return ``F(y_i | w_i)`` per row via ``criterion.cdf`` (chunked)."""
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    return_as_torch, (w_in, y_in) = _normalize_inputs(w, y, device=self._device)
    assert w_in is not None and y_in is not None
    w_t = _as_2d(w_in, device=self._device)
    y_t = y_in.reshape(-1)
    if y_t.shape[0] != w_t.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_t)
    n = y_t.shape[0]
    parts: list[torch.Tensor] = []

    for start in range(0, n, effective_batch_size):
      end = min(start + effective_batch_size, n)
      logits_t, criterion = self._predict_full(w_t[start:end])
      parts.append(self._criterion_cdf_z(logits_t, criterion, z[start:end]))

    out = torch.cat(parts) if parts else torch.empty(0, device=self._device)
    return _wrap_output(out, return_as_torch=return_as_torch)

  def icdf(
    self, w: TensorLike, alphas: TensorLike, *, batch_size: int | None = None
  ) -> TensorLike:
    """Per-row conditional quantile ``F^{-1}(alphas_i | w_i)`` on the y-scale.

    The criterion's ``icdf`` is scalar-alpha, so we loop over rows within
    each chunk after a single batched ``predict(output_type="full")``
    forward pass — the forward pass dominates the cost.  The forward is
    chunked by ``batch_size`` to bound the logits transient.
    """
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    return_as_torch, (w_in, a_in) = _normalize_inputs(
      w, alphas, device=self._device
    )
    assert w_in is not None and a_in is not None
    w_t = _as_2d(w_in, device=self._device)
    alpha_t = a_in.reshape(-1)
    if alpha_t.shape[0] != w_t.shape[0]:
      raise ValueError("w and alphas have incompatible lengths.")
    if torch.any((alpha_t <= 0.0) | (alpha_t >= 1.0)):
      raise ValueError("alphas must lie strictly inside (0, 1).")

    n = alpha_t.shape[0]
    z_out = torch.empty(n, dtype=alpha_t.dtype, device=self._device)

    for start in range(0, n, effective_batch_size):
      end = min(start + effective_batch_size, n)
      logits_t, criterion = self._predict_full(w_t[start:end])
      for j in range(end - start):
        z_ij = criterion.icdf(
          logits_t[j : j + 1], float(alpha_t[start + j].item())
        )
        z_out[start + j] = z_ij.reshape(-1)[0].to(
          device=self._device, dtype=alpha_t.dtype
        )

    return _wrap_output(
      self._inverse_transform(z_out), return_as_torch=return_as_torch
    )

  def cdf_grid(
    self,
    w: TensorLike,
    y_grid: TensorLike,
    *,
    batch_size: int | None = None,
  ) -> TensorLike:
    """CDF on the Cartesian product of ``w`` rows and ``y_grid``.

    Returns shape ``(n_w, n_y)``.  One TabPFN forward pass per ``w`` row
    (chunked by ``batch_size``); the same logits are evaluated against
    every ``y`` value.
    """
    self._check_fitted()
    effective_batch_size = self._resolve_batch_size(batch_size)

    return_as_torch, (w_in, y_in) = _normalize_inputs(
      w, y_grid, device=self._device
    )
    assert w_in is not None and y_in is not None
    w_t = _as_2d(w_in, device=self._device)
    y_t = y_in.reshape(-1)
    if y_t.numel() == 0:
      raise ValueError("y_grid must contain at least one value.")

    n_y = y_t.shape[0]
    z_grid = self._transform_y(y_t)

    chunks: list[torch.Tensor] = []
    for start in range(0, w_t.shape[0], effective_batch_size):
      stop = min(start + effective_batch_size, w_t.shape[0])
      n_chunk = stop - start
      logits_t, criterion = self._predict_full(w_t[start:stop])
      logits_eval = logits_t.repeat_interleave(n_y, dim=0)
      z_flat = z_grid.repeat(n_chunk)
      cdf_z = self._criterion_cdf_z(logits_eval, criterion, z_flat)
      chunks.append(cdf_z.reshape(n_chunk, n_y))

    out = (
      torch.cat(chunks, dim=0)
      if chunks
      else torch.empty((0, n_y), dtype=torch.float64, device=self._device)
    )
    return _wrap_output(out, return_as_torch=return_as_torch)
