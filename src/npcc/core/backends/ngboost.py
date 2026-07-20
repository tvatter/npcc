"""
ngboost.py — NGBoost parametric backend (extra).

NGBoost fits a parametric conditional distribution by natural-gradient
boosting.  ``pred_dist(X).dist`` is a SciPy frozen distribution whose
parameters are vectors (one per row), giving analytic ``pdf`` / ``cdf`` /
``ppf`` — no quantile-table inversion needed.  This backend therefore
subclasses the neutral
:class:`~npcc.core.conditional_distribution1d.ConditionalDistribution1D`
directly and overrides every primitive, including fast
predict-params-once-per-row ``pdf_grid`` / ``cdf_grid``.

The regressor models the *transformed* target ``Z = T(Y)``; densities are
mapped back with the inverse Jacobian and quantiles with the inverse
transform, exactly as documented on the base class.

Requires the ``ngboost`` extra (``pip install npcc[ngboost]``).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from ngboost import NGBRegressor
from ngboost.distns import Normal

from npcc.core._common import (
  TensorLike,
  _as_2d,
  _normalize_inputs,
  _wrap_output,
)
from npcc.core.conditional_distribution1d import ConditionalDistribution1D


class NGBoostBackend(ConditionalDistribution1D):
  """Conditional predictive distribution via an NGBoost parametric fit.

  Parameters
  ----------
  transform, eps, device, batch_size
      Forwarded to :class:`ConditionalDistribution1D`.
  dist
      NGBoost distribution class for ``Z`` (default: ``Normal``).
  **ngb_kwargs
      Forwarded to ``NGBRegressor`` (e.g. ``n_estimators``,
      ``learning_rate``).
  """

  model_: NGBRegressor | None

  def __init__(
    self,
    *,
    transform: Literal["identity", "logit", "probit"] = "logit",
    eps: float = 1e-6,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    dist: type | None = None,
    **ngb_kwargs: object,
  ) -> None:
    super().__init__(
      transform=transform,
      eps=eps,
      device=device,
      batch_size=batch_size,
    )
    self._dist = dist if dist is not None else Normal
    self.ngb_kwargs = dict(ngb_kwargs)
    self.model_ = None

  def _fit_model(self, w: torch.Tensor, z: torch.Tensor) -> None:
    model = NGBRegressor(Dist=self._dist, verbose=False, **self.ngb_kwargs)
    model.fit(w.detach().cpu().numpy(), z.detach().cpu().numpy())
    self.model_ = model

  def _frozen(self, w_t: torch.Tensor) -> Any:  # noqa: ANN401 - scipy frozen dist
    """Return the SciPy frozen distribution of ``Z`` for rows of ``w_t``."""
    assert self.model_ is not None
    return self.model_.pred_dist(w_t.detach().cpu().numpy()).dist

  def _to_device(self, arr: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(
      np.asarray(arr, dtype=float), dtype=torch.float64, device=self._device
    )

  def pdf(
    self, w: TensorLike, y: TensorLike, *, batch_size: int | None = None
  ) -> TensorLike:
    self._check_fitted()
    rt, (w_in, y_in) = _normalize_inputs(w, y, device=self._device)
    assert w_in is not None and y_in is not None
    w_t = _as_2d(w_in, device=self._device)
    y_t = y_in.reshape(-1)
    if y_t.shape[0] != w_t.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_t)
    f_z = self._to_device(self._frozen(w_t).pdf(z.detach().cpu().numpy()))
    out = f_z * self._jacobian_inverse(y_t)
    return _wrap_output(out, return_as_torch=rt)

  def cdf(
    self, w: TensorLike, y: TensorLike, *, batch_size: int | None = None
  ) -> TensorLike:
    self._check_fitted()
    rt, (w_in, y_in) = _normalize_inputs(w, y, device=self._device)
    assert w_in is not None and y_in is not None
    w_t = _as_2d(w_in, device=self._device)
    y_t = y_in.reshape(-1)
    if y_t.shape[0] != w_t.shape[0]:
      raise ValueError("w and y have incompatible lengths.")

    z = self._transform_y(y_t)
    out = self._to_device(self._frozen(w_t).cdf(z.detach().cpu().numpy()))
    return _wrap_output(out, return_as_torch=rt)

  def icdf(
    self, w: TensorLike, alphas: TensorLike, *, batch_size: int | None = None
  ) -> TensorLike:
    self._check_fitted()
    rt, (w_in, a_in) = _normalize_inputs(w, alphas, device=self._device)
    assert w_in is not None and a_in is not None
    w_t = _as_2d(w_in, device=self._device)
    alpha_t = a_in.reshape(-1)
    if alpha_t.shape[0] != w_t.shape[0]:
      raise ValueError("w and alphas have incompatible lengths.")
    if torch.any((alpha_t <= 0.0) | (alpha_t >= 1.0)):
      raise ValueError("alphas must lie strictly inside (0, 1).")

    z = self._to_device(self._frozen(w_t).ppf(alpha_t.detach().cpu().numpy()))
    return _wrap_output(self._inverse_transform(z), return_as_torch=rt)

  def pdf_grid(
    self, w: TensorLike, y_grid: TensorLike, *, batch_size: int | None = None
  ) -> TensorLike:
    self._check_fitted()
    rt, (w_in, y_in) = _normalize_inputs(w, y_grid, device=self._device)
    assert w_in is not None and y_in is not None
    w_t = _as_2d(w_in, device=self._device)
    y_grid_t = y_in.reshape(-1)
    if y_grid_t.numel() == 0:
      raise ValueError("y_grid must contain at least one value.")

    z_grid = self._transform_y(y_grid_t)
    jac = self._jacobian_inverse(y_grid_t)
    # params computed once per w row; broadcast against the z-grid:
    # (n_y, 1) vs (n_w,) -> (n_y, n_w) -> transpose -> (n_w, n_y).
    frozen = self._frozen(w_t)
    f_z = self._to_device(frozen.pdf(z_grid.detach().cpu().numpy()[:, None]).T)
    out = f_z * jac.unsqueeze(0)
    return _wrap_output(out, return_as_torch=rt)

  def cdf_grid(
    self, w: TensorLike, y_grid: TensorLike, *, batch_size: int | None = None
  ) -> TensorLike:
    self._check_fitted()
    rt, (w_in, y_in) = _normalize_inputs(w, y_grid, device=self._device)
    assert w_in is not None and y_in is not None
    w_t = _as_2d(w_in, device=self._device)
    y_grid_t = y_in.reshape(-1)
    if y_grid_t.numel() == 0:
      raise ValueError("y_grid must contain at least one value.")

    z_grid = self._transform_y(y_grid_t)
    frozen = self._frozen(w_t)
    out = self._to_device(frozen.cdf(z_grid.detach().cpu().numpy()[:, None]).T)
    return _wrap_output(out, return_as_torch=rt)
