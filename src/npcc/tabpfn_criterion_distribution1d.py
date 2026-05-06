"""
tabpfn_criterion_distribution1d.py — univariate conditional predictive
distribution via TabPFN's native distribution head.

Approach
--------
TabPFN's regressor is internally a classifier over a set of pre-computed
"bar distribution" bins.  Calling

    pred = regressor.predict(W, output_type="full")

returns a dictionary with two pieces:

- ``pred["logits"]``: tensor of shape ``(n_w, n_bins)`` whose rows are
  the unnormalised log-densities over the bins.
- ``pred["criterion"]``: the ``BarDistribution``-like head used during
  training; it carries the bin edges and exposes ``pdf`` / ``cdf`` /
  ``icdf`` methods that turn logits into per-bin probability mass and
  evaluate the corresponding piecewise-linear PDF / CDF / quantile
  function at arbitrary points ``z``.

Compared to
:class:`npcc.tabpfn_quantile_distribution1d.TabPFNQuantileDistribution1D`
this avoids querying a quantile grid and inverting a numerical
derivative for the PDF, so it is faster and typically more accurate,
but it is specific to TabPFN's binned output.

Logit support transform
-----------------------
``U`` and ``V`` are copula scores in ``(0, 1)``.  When
``transform="logit"`` we fit on ``Z = logit(Y)`` and convert back via
the standard Jacobian for densities (CDFs and quantiles need no
correction since monotone transforms preserve them):

    f_Y(y | w) = f_Z(logit(y) | w) / (y * (1 - y)).

Cartesian-product evaluation (``pdf_grid`` / ``cdf_grid``)
----------------------------------------------------------
For diagnostics and grid-based copula plots one often needs the
density (or CDF) on the full Cartesian product of conditioning rows
``W`` and evaluation points ``y_grid``.  ``pdf_grid`` / ``cdf_grid``
exploit the fact that a single TabPFN forward pass per row of ``W``
is enough — the same logits are re-used across every ``y`` value —
and are materially faster than calling :py:meth:`pdf` /
:py:meth:`cdf` on the explicit tile.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from npcc._common import (
  TensorLike,
  _as_2d,
  _normalize_inputs,
  _wrap_output,
)
from npcc.tabpfn_distribution1d import TabPFNDistribution1D


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
  array; map those to ``-inf`` so a downstream softmax assigns them
  zero probability.
  """
  if isinstance(logits, torch.Tensor):
    return logits.to(device=device, dtype=torch.float32)
  arr = np.asarray(logits, dtype=object)
  safe = np.empty(arr.shape, dtype=np.float32)
  for idx, val in np.ndenumerate(arr):
    safe[idx] = -np.inf if val is None else float(val)
  return torch.as_tensor(safe, dtype=torch.float32, device=device)


class TabPFNCriterionDistribution1D(TabPFNDistribution1D):
  """Univariate conditional predictive distribution via TabPFN's binned head.

  See the module-level docstring for the algorithm.  The ``fit`` /
  ``pdf`` / ``cdf`` / ``icdf`` API mirrors
  :class:`TabPFNQuantileDistribution1D` so the two classes are drop-in
  interchangeable inside :class:`PFNRBicop`.
  """

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
    if self.model_ is None:
      raise RuntimeError("The model is not fitted.")
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
    """Evaluate ``criterion.pdf`` at z-space points; returns shape ``(n,)``.

    The criterion head consumes ``logits_t``'s dtype (typically
    float32); we down-cast ``z`` for the call and bring the result
    back to ``z``'s dtype (float64 by convention) for downstream use.
    """
    z_eval = z.to(dtype=logits_t.dtype, device=logits_t.device).reshape(-1, 1)
    dens = criterion.pdf(logits_t, z_eval)
    return dens.reshape(-1).to(dtype=z.dtype)

  def _criterion_cdf_z(
    self,
    logits_t: torch.Tensor,
    criterion: _CriterionLike,
    z: torch.Tensor,
  ) -> torch.Tensor:
    """Evaluate ``criterion.cdf`` at the (already z-space) eval points.

    The CDF of ``Y`` equals the CDF of ``Z = transform(Y)`` evaluated at
    ``transform(y)`` — no Jacobian for monotone transforms.
    """
    z_eval = z.to(dtype=logits_t.dtype, device=logits_t.device).reshape(-1, 1)
    cdf = criterion.cdf(logits_t, z_eval)
    return cdf.reshape(-1).to(dtype=z.dtype)

  # ------------------------------------------------------------------
  # Public API: pdf / cdf / icdf + the *_grid Cartesian fast paths.
  # ------------------------------------------------------------------

  def pdf(
    self, w: TensorLike, y: TensorLike, *, batch_size: int = 400
  ) -> TensorLike:
    """Return ``f(y_i | w_i)`` for each row ``i``.

    Inference is chunked into pieces of ``batch_size`` rows to bound
    GPU memory usage; results are concatenated and the inverse-Jacobian
    is applied at the end.
    """
    if self.model_ is None:
      raise RuntimeError("The model is not fitted.")
    if batch_size <= 0:
      raise ValueError("batch_size must be positive.")

    return_as_torch, (w_in, y_in) = _normalize_inputs(w, y, device=self._device)
    assert w_in is not None and y_in is not None
    w_t = _as_2d(w_in, device=self._device)
    y_t = y_in.reshape(-1)
    if y_t.shape[0] != w_t.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_t)
    n = y_t.shape[0]
    parts: list[torch.Tensor] = []

    for start in range(0, n, batch_size):
      end = min(start + batch_size, n)
      logits_t, criterion = self._predict_full(w_t[start:end])
      parts.append(self._criterion_pdf_z(logits_t, criterion, z[start:end]))

    dens_z = torch.cat(parts) if parts else torch.empty(0, device=self._device)
    out = dens_z * self._jacobian_inverse(y_t)
    return _wrap_output(out, return_as_torch=return_as_torch)

  def pdf_grid(self, w: TensorLike, y_grid: TensorLike) -> TensorLike:
    """Density on the Cartesian product of ``w`` rows and ``y_grid``.

    Returns shape ``(n_w, n_y)`` with ``out[i, j] = f(y_grid[j] | w[i])``.
    Each ``w`` row triggers exactly one TabPFN forward pass; the same
    logits are then evaluated against every ``y`` value.  This is the
    fast path for grid plots and copula visualisations.
    """
    if self.model_ is None:
      raise RuntimeError("The model is not fitted.")

    return_as_torch, (w_in, y_in) = _normalize_inputs(
      w, y_grid, device=self._device
    )
    assert w_in is not None and y_in is not None
    w_t = _as_2d(w_in, device=self._device)
    y_t = y_in.reshape(-1)
    n_w, n_y = w_t.shape[0], y_t.shape[0]

    logits_t, criterion = self._predict_full(w_t)
    logits_eval = logits_t.repeat_interleave(n_y, dim=0)
    y_tiled = y_t.tile(n_w)
    z = self._transform_y(y_tiled)
    dens_z = self._criterion_pdf_z(logits_eval, criterion, z)
    dens_y = dens_z * self._jacobian_inverse(y_tiled)
    return _wrap_output(
      dens_y.reshape(n_w, n_y), return_as_torch=return_as_torch
    )

  def cdf(
    self, w: TensorLike, y: TensorLike, *, batch_size: int = 400
  ) -> TensorLike:
    """Return ``F(y_i | w_i) = P(Y <= y_i | W = w_i)`` per row.

    Uses ``criterion.cdf`` directly on the binned distribution head.
    No Jacobian correction is needed: monotone transforms preserve the
    CDF, so ``F_Y(y | w) = F_Z(transform(y) | w)``.  Inference is
    chunked into ``batch_size`` rows.
    """
    if self.model_ is None:
      raise RuntimeError("The model is not fitted.")
    if batch_size <= 0:
      raise ValueError("batch_size must be positive.")

    return_as_torch, (w_in, y_in) = _normalize_inputs(w, y, device=self._device)
    assert w_in is not None and y_in is not None
    w_t = _as_2d(w_in, device=self._device)
    y_t = y_in.reshape(-1)
    if y_t.shape[0] != w_t.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_t)
    n = y_t.shape[0]
    parts: list[torch.Tensor] = []

    for start in range(0, n, batch_size):
      end = min(start + batch_size, n)
      logits_t, criterion = self._predict_full(w_t[start:end])
      parts.append(self._criterion_cdf_z(logits_t, criterion, z[start:end]))

    out = torch.cat(parts) if parts else torch.empty(0, device=self._device)
    return _wrap_output(out, return_as_torch=return_as_torch)

  def icdf(self, w: TensorLike, alphas: TensorLike) -> TensorLike:
    """Per-row conditional quantile ``F^{-1}(alphas_i | w_i)`` on the y-scale.

    For each row ``i``, returns ``y`` such that
    ``F(y | w_i) = alphas_i``.  Used by the Rosenblatt simulation
    recipe behind :py:meth:`PFNRBicop.tau`.

    The criterion's ``icdf`` is scalar-α, so we loop over rows after a
    single batched ``predict(output_type="full")`` forward pass — the
    forward pass dominates the cost.
    """
    if self.model_ is None:
      raise RuntimeError("The model is not fitted.")

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

    logits_t, criterion = self._predict_full(w_t)

    z_out = torch.empty(
      alpha_t.shape[0], dtype=alpha_t.dtype, device=self._device
    )
    for i in range(alpha_t.shape[0]):
      z_i = criterion.icdf(logits_t[i : i + 1], float(alpha_t[i].item()))
      z_out[i] = z_i.reshape(-1)[0].to(device=self._device, dtype=alpha_t.dtype)

    return _wrap_output(
      self._inverse_transform(z_out), return_as_torch=return_as_torch
    )

  def cdf_grid(self, w: TensorLike, y_grid: TensorLike) -> TensorLike:
    """CDF on the Cartesian product of ``w`` rows and ``y_grid`` values.

    Returns shape ``(n_w, n_y)`` with ``out[i, j] = F(y_grid[j] | w[i])``.
    One TabPFN forward pass per ``w`` row; the fast path for grid-based
    integration of the joint copula CDF.
    """
    if self.model_ is None:
      raise RuntimeError("The model is not fitted.")

    return_as_torch, (w_in, y_in) = _normalize_inputs(
      w, y_grid, device=self._device
    )
    assert w_in is not None and y_in is not None
    w_t = _as_2d(w_in, device=self._device)
    y_t = y_in.reshape(-1)
    n_w, n_y = w_t.shape[0], y_t.shape[0]

    logits_t, criterion = self._predict_full(w_t)
    logits_eval = logits_t.repeat_interleave(n_y, dim=0)
    y_tiled = y_t.tile(n_w)
    z = self._transform_y(y_tiled)
    cdf_z = self._criterion_cdf_z(logits_eval, criterion, z)
    return _wrap_output(
      cdf_z.reshape(n_w, n_y), return_as_torch=return_as_torch
    )
