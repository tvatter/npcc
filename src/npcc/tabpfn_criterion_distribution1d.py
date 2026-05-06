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

from typing import Protocol, Self

import numpy as np
import torch
from numpy.typing import ArrayLike
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from npcc._common import _as_2d
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
  """Convert TabPFN ``full`` logits to a float32 tensor.

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

  def fit(self, w: ArrayLike, y: ArrayLike) -> Self:
    """Fit a TabPFN-v2.5 regressor on ``(w, transform(y))``.

    Locked to TabPFN-v2.5 because v2.6 has reported regressions on
    tabular regression tasks.  Override via ``model_kwargs`` if needed.
    """
    w_arr = _as_2d(w)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if len(y_arr) != w_arr.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_arr)

    self.model_ = TabPFNRegressor.create_default_for_version(
      ModelVersion.V2_5, **self.model_kwargs
    )
    self.model_.fit(w_arr, z)
    return self

  # ------------------------------------------------------------------
  # Internal helpers.
  # ------------------------------------------------------------------

  def _predict_full(self, w: np.ndarray) -> tuple[torch.Tensor, _CriterionLike]:
    """Run a single ``output_type="full"`` forward pass."""
    if self.model_ is None:
      raise RuntimeError("The model is not fitted.")
    pred = self.model_.predict(w, output_type="full")
    logits = pred["logits"]
    criterion: _CriterionLike = pred["criterion"]
    device = (
      logits.device if isinstance(logits, torch.Tensor) else torch.device("cpu")
    )
    return _coerce_logits_tensor(logits, device=device), criterion

  def _criterion_pdf(
    self,
    logits_t: torch.Tensor,
    criterion: _CriterionLike,
    z: np.ndarray,
  ) -> np.ndarray:
    z_eval = torch.as_tensor(
      z[:, None].astype(np.float32),
      dtype=torch.float32,
      device=logits_t.device,
    )
    dens = criterion.pdf(logits_t, z_eval)
    return dens.reshape(-1).detach().cpu().numpy()

  def _criterion_cdf(
    self,
    logits_t: torch.Tensor,
    criterion: _CriterionLike,
    z: np.ndarray,
  ) -> np.ndarray:
    """Evaluate ``criterion.cdf`` at the (already z-space) eval points.

    The CDF of ``Y`` equals the CDF of ``Z = transform(Y)`` evaluated at
    ``transform(y)`` — no Jacobian for monotone transforms.
    """
    z_eval = torch.as_tensor(
      z[:, None].astype(np.float32),
      dtype=torch.float32,
      device=logits_t.device,
    )
    cdf = criterion.cdf(logits_t, z_eval)
    return cdf.reshape(-1).detach().cpu().numpy()

  # ------------------------------------------------------------------
  # Public API: pdf / cdf / icdf + the *_grid Cartesian fast paths.
  # ------------------------------------------------------------------

  def pdf(
    self, w: ArrayLike, y: ArrayLike, *, batch_size: int = 400
  ) -> np.ndarray:
    """Return ``f(y_i | w_i)`` for each row ``i``.

    Inference is chunked into pieces of ``batch_size`` rows to bound
    GPU memory usage; results are concatenated and the inverse-Jacobian
    is applied at the end.
    """
    if self.model_ is None:
      raise RuntimeError("The model is not fitted.")
    if batch_size <= 0:
      raise ValueError("batch_size must be positive.")

    w_arr = _as_2d(w)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if len(y_arr) != w_arr.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_arr)
    n = len(y_arr)
    out = np.zeros(n, dtype=float)

    for start in range(0, n, batch_size):
      end = min(start + batch_size, n)
      logits_t, criterion = self._predict_full(w_arr[start:end])
      out[start:end] = self._criterion_pdf(logits_t, criterion, z[start:end])

    return out * self._jacobian_inverse(y_arr)

  def pdf_grid(self, w: ArrayLike, y_grid: ArrayLike) -> np.ndarray:
    """Density on the Cartesian product of ``w`` rows and ``y_grid``.

    Returns shape ``(n_w, n_y)`` with ``out[i, j] = f(y_grid[j] | w[i])``.
    Each ``w`` row triggers exactly one TabPFN forward pass; the same
    logits are then evaluated against every ``y`` value.  This is the
    fast path for grid plots and copula visualisations.
    """
    if self.model_ is None:
      raise RuntimeError("The model is not fitted.")

    w_arr = _as_2d(w)
    y_arr = np.asarray(y_grid, dtype=float).reshape(-1)
    n_w, n_y = w_arr.shape[0], len(y_arr)

    logits_t, criterion = self._predict_full(w_arr)
    logits_eval = logits_t.repeat_interleave(n_y, dim=0)
    y_tiled = np.tile(y_arr, n_w)
    z = self._transform_y(y_tiled)
    dens_z = self._criterion_pdf(logits_eval, criterion, z)
    dens_y = dens_z * self._jacobian_inverse(y_tiled)
    return dens_y.reshape(n_w, n_y)

  def cdf(
    self, w: ArrayLike, y: ArrayLike, *, batch_size: int = 400
  ) -> np.ndarray:
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

    w_arr = _as_2d(w)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if len(y_arr) != w_arr.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_arr)
    n = len(y_arr)
    out = np.zeros(n, dtype=float)

    for start in range(0, n, batch_size):
      end = min(start + batch_size, n)
      logits_t, criterion = self._predict_full(w_arr[start:end])
      out[start:end] = self._criterion_cdf(logits_t, criterion, z[start:end])

    return out

  def icdf(self, w: ArrayLike, alphas: ArrayLike) -> np.ndarray:
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

    w_arr = _as_2d(w)
    alpha_arr = np.asarray(alphas, dtype=float).reshape(-1)
    if len(alpha_arr) != w_arr.shape[0]:
      raise ValueError("w and alphas have incompatible lengths.")
    if np.any((alpha_arr <= 0.0) | (alpha_arr >= 1.0)):
      raise ValueError("alphas must lie strictly inside (0, 1).")

    logits_t, criterion = self._predict_full(w_arr)

    z_out = np.empty(len(alpha_arr), dtype=float)
    for i, a in enumerate(alpha_arr):
      z_i = criterion.icdf(logits_t[i : i + 1], float(a))
      z_out[i] = float(z_i.detach().cpu().reshape(-1)[0])

    return self._inverse_transform(z_out)

  def cdf_grid(self, w: ArrayLike, y_grid: ArrayLike) -> np.ndarray:
    """CDF on the Cartesian product of ``w`` rows and ``y_grid`` values.

    Returns shape ``(n_w, n_y)`` with ``out[i, j] = F(y_grid[j] | w[i])``.
    One TabPFN forward pass per ``w`` row; the fast path for grid-based
    integration of the joint copula CDF.
    """
    if self.model_ is None:
      raise RuntimeError("The model is not fitted.")

    w_arr = _as_2d(w)
    y_arr = np.asarray(y_grid, dtype=float).reshape(-1)
    n_w, n_y = w_arr.shape[0], len(y_arr)

    logits_t, criterion = self._predict_full(w_arr)
    logits_eval = logits_t.repeat_interleave(n_y, dim=0)
    y_tiled = np.tile(y_arr, n_w)
    z = self._transform_y(y_tiled)
    cdf_z = self._criterion_cdf(logits_eval, criterion, z)
    return cdf_z.reshape(n_w, n_y)
