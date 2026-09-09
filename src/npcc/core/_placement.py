"""Bring an input array onto the placement an estimator evaluates on.

pyvinecopulib draws three separable steps over every input -- *placement*,
onto the dtype and device the object's own tensors live on; *layout*, the
shapes it admits; and *domain*, clamping copula arguments into the open unit
square -- and gives each one an overridable hook per level (``_prep``,
``_layout``, ``trim``), composed by ``_prep_args``.

Its default ``_prep``, from ``PlacementMixin``, *infers* the placement from
arrays the object already holds. That inference finds nothing here: none of
this package's estimators is a ``torch.nn.Module``, and none holds a bare
float tensor at rest -- a :class:`~npcc.core.bicop.RosenblattBicop` holds two
backend estimators, a ``torch.device`` and Python scalars. So the inherited
hook hands its argument straight back, which is indistinguishable from having
placed it, and a mis-placed input surfaces much later as a device mismatch
inside a concatenation.

:class:`TensorPlacement` replaces the inference with the answer this package
already carries: an explicit ``_device`` per estimator, and ``float64``
throughout. It is the counterpart of pyvinecopulib's own
``torch._placement.TensorPlacementMixin``, which reads a registered tensor off
an ``nn.Module`` instead, and it is mixed in **ahead** of the canonical base
for the same reason -- ``_prep`` has to resolve here rather than to the
array-API inference behind it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

__all__ = ["TensorPlacement", "resolve_device", "to_numpy"]


def resolve_device(device: str | torch.device | None) -> torch.device:
  """Resolve ``None`` to ``cuda`` if available, else ``cpu``.

  A bare ``cuda`` device (no index) is normalized to ``cuda:<current index>``
  so it compares equal to the device tensors actually materialize on (e.g.
  ``cuda:0``); ``torch.device("cuda") != torch.device("cuda:0")`` otherwise.

  Parameters
  ----------
  device : str, torch.device, or None
      The requested device, or ``None`` to pick the best available one.

  Returns
  -------
  torch.device
      The resolved device, with a CUDA index filled in.
  """
  if device is None:
    resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  else:
    resolved = torch.device(device)

  if resolved.type == "cuda" and resolved.index is None:
    resolved = torch.device("cuda", torch.cuda.current_device())

  return resolved


def to_numpy(a: Any) -> np.ndarray:  # noqa: ANN401 - any array type, converted
  """Host NumPy view of ``a`` -- placement's return trip.

  Every third-party model this package drives reads NumPy, so a tensor has to
  come back across that boundary. ``np.asarray`` alone raises on a tensor that
  requires grad and again on one that lives on an accelerator, so this
  detaches and transfers first, both through ``getattr`` since a NumPy array
  has neither method.

  Mirrors ``pyvinecopulib.core._placement.to_numpy``, which is private
  upstream; the one implementation here replaces the three spellings of it
  that the backends had grown.

  Parameters
  ----------
  a : array
      Values in any array namespace.

  Returns
  -------
  numpy.ndarray
      The same values, on the host, detached.
  """
  v: Any = a
  detach = getattr(v, "detach", None)
  if detach is not None:
    v = detach()
  cpu = getattr(v, "cpu", None)
  if cpu is not None:
    v = cpu()
  return np.asarray(v)


class TensorPlacement:
  """The ``_prep`` hook for an estimator placed on its own ``_device``.

  Mixed in **ahead** of the canonical pyvinecopulib base, so ``_prep``
  resolves here rather than to the array-API inference that base ships.

  Only ordinary private members belong on a mixin at that position: it lands
  ahead of everything behind it in the resulting MRO, so anything defined here
  that a base also defines would silently shadow it.
  """

  _device: torch.device

  def _prep(self, a: Any) -> torch.Tensor:  # noqa: ANN401 - any array, placed
    """Bring one input array onto this estimator's dtype and device.

    ``as_tensor`` rather than ``tensor`` or ``detach``, so a tensor that
    already matches is returned untouched and a gradient-carrying one stays in
    the graph.

    Parameters
    ----------
    a : array
        An input array in any namespace.

    Returns
    -------
    torch.Tensor
        The same values, ``float64``, on ``self._device``.
    """
    return torch.as_tensor(a, dtype=torch.float64, device=self._device)
