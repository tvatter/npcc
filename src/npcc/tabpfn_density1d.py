"""
tabpfn_density1d.py — univariate conditional density via TabPFN's
native distribution head.

Approach
--------
TabPFN's regressor is internally a classifier over a set of pre-computed
"bar distribution" bins.  Calling

    pred = regressor.predict(W, output_type="full")

returns a dictionary with two pieces:

- ``pred["logits"]``: tensor of shape ``(n_w, n_bins)`` whose rows are
  the unnormalised log-densities over the bins.
- ``pred["criterion"]``: the ``BarDistribution``-like head used during
  training; it carries the bin edges and exposes a ``pdf(logits, z)``
  method that converts logits to per-bin probability mass and then
  integrates against the bin shape to produce a density at arbitrary
  evaluation points ``z``.

The conditional density at ``y`` is therefore one direct call::

    f(y | w) = criterion.pdf(logits(w), z = transform(y)).

Compared to :class:`npcc.tabpfn_quantile_density1d.TabPFNQuantileDensity1D`
this avoids querying a quantile grid and inverting a numerical
derivative, so it is faster and typically more accurate, but it is
specific to TabPFN's binned output.

Logit support transform
-----------------------
``U`` and ``V`` are copula scores in ``(0, 1)``.  When
``transform="logit"`` we fit on ``Z = logit(Y)`` and convert back
via the standard Jacobian::

    f_Y(y | w) = f_Z(logit(y) | w) / (y * (1 - y)).

Cartesian-product evaluation (``density_grid``)
-----------------------------------------------
For diagnostics and grid-based copula plots one often needs the density
on the full Cartesian product of conditioning rows ``W`` and evaluation
points ``y_grid``.  ``density_grid`` exploits the fact that a single
TabPFN forward pass per row of ``W`` is enough — the same logits are
re-used across every ``y`` value — and is materially faster than
calling :py:meth:`density` on the explicit tile.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, Self

import numpy as np
import torch
from numpy.typing import ArrayLike
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from npcc._common import _as_2d, _logit


class _CriterionLike(Protocol):
  """Duck-typed view of TabPFN's ``output_type="full"`` distribution head.

  We only ever call ``pdf``; no need to depend on the concrete TabPFN
  class, which has changed name across versions.
  """

  def pdf(self, logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor: ...


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


class TabPFNDensity1D:
  """Univariate conditional density ``f(Y | W=w)`` from TabPFN logits.

  See the module-level docstring for the algorithm.  The ``fit`` /
  ``density`` API mirrors :class:`TabPFNQuantileDensity1D` so the two
  classes are drop-in interchangeable inside :class:`PFNRBicop`.

  Parameters
  ----------
  transform
      ``"identity"`` fits TabPFN directly on ``Y``.  ``"logit"`` fits on
      ``Z = logit(Y)`` and applies the inverse Jacobian on the way out;
      this is the only sensible choice when ``Y`` is bounded in
      ``(0, 1)``, which is always the case for copula scores.
  eps
      Clip distance from the boundary of ``(0, 1)`` used by the logit
      transform.
  model_kwargs
      Forwarded to :py:meth:`TabPFNRegressor.create_default_for_version`.
  """

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit"] = "logit",
    eps: float = 1e-6,
    model_kwargs: dict[str, Any] | None = None,
  ) -> None:
    self.transform = transform
    self.eps = eps
    self.model_kwargs = model_kwargs or {}
    self.model_: TabPFNRegressor | None = None

  def _transform_y(self, y: np.ndarray) -> np.ndarray:
    if self.transform == "identity":
      return y
    if self.transform == "logit":
      y_clip = np.clip(y, self.eps, 1.0 - self.eps)
      return _logit(y_clip)
    raise ValueError(f"Unknown transform: {self.transform}")

  def _jacobian_inverse(self, y: np.ndarray) -> np.ndarray:
    if self.transform == "identity":
      return np.ones_like(y)
    if self.transform == "logit":
      y_clip = np.clip(y, self.eps, 1.0 - self.eps)
      return 1.0 / (y_clip * (1.0 - y_clip))
    raise ValueError(f"Unknown transform: {self.transform}")

  def fit(self, w: ArrayLike, y: ArrayLike) -> Self:
    """Fit a TabPFN-v2.5 regressor on ``(w, transform(y))``.

    The estimator is locked to TabPFN-v2.5 because v2.6 has reported
    regressions on tabular regression tasks.  Override via
    ``model_kwargs`` if needed.
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
    """Evaluate ``criterion.pdf`` at the (already z-space) eval points."""
    z_eval = torch.as_tensor(
      z[:, None].astype(np.float32),
      dtype=torch.float32,
      device=logits_t.device,
    )
    dens = criterion.pdf(logits_t, z_eval)
    return dens.reshape(-1).detach().cpu().numpy()

  def density(
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

  def density_grid(self, w: ArrayLike, y_grid: ArrayLike) -> np.ndarray:
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
