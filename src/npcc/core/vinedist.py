"""Original-scale distributions built from Rosenblatt vine copulas."""

from __future__ import annotations

import torch
from pyvinecopulib.core import ControlsLike, VinedistBase


class RosenblattVinedist(VinedistBase[torch.Tensor]):
  """Original-scale distribution backed by a Rosenblatt vine.

  The distribution is constructed from an unfitted or fitted vine copula and
  one margin per variable. The inherited :meth:`fit` method first fits the
  existing margins, transforms the observations to the copula scale, and then
  fits the existing vine copula along its fixed structure.

  All observations, covariates, margins, and pair copulas are expected to use
  the same torch device.

  Notes
  -----
  This class intentionally does not declare ``vinecop_class`` or
  ``margin_class``. The generic :meth:`VinedistBase.from_data` construction
  path cannot create a non-simplified Rosenblatt vine correctly. Construct
  the required parts first and then call :meth:`fit`.
  """

  supports_weighted_copula: bool = False
  supports_fit_covariates: bool = True

  @classmethod
  def _coerce_fit_data(
    cls,
    y: torch.Tensor,
    weights: torch.Tensor | None,
    controls: ControlsLike | None,
  ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Keep fitting data in the caller-provided torch placement.

    Device and dtype conversion are intentionally omitted. The observations,
    covariates, margins, and copula are expected to have matching placement.
    """
    del cls, controls

    if not isinstance(y, torch.Tensor):
      raise TypeError("y must be a torch tensor.")

    return y, weights
