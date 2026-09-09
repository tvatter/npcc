"""Bring an input array onto the placement an estimator evaluates on.

pyvinecopulib draws three separable steps over every input -- *placement*,
onto the dtype and device the object's own tensors live on; *layout*, the
shapes it admits; and *domain*, clamping copula arguments into the open unit
square. The first two are overridable hooks per level, ``_prep`` and
``_layout``; the domain step is the module-level ``trim`` that ``_prep_args``
applies after them.

:class:`TensorPlacement` overrides ``_prep``. Three reasons it has to, none of
which is that the inherited inference fails -- :meth:`_set_placement` plants a
reference tensor precisely so that it does not:

1. **The MRO.** All four canonical bases already inherit ``PlacementMixin``,
   so a mixin placed *after* the base never wins the lookup at all. Mixing in
   ahead of the base is what makes the override reachable.
2. **Autograd.** ``torch.as_tensor`` carries a gradient across a dtype or
   device change; the inherited hook reaches placement through
   ``array_api_compat``'s ``xp.asarray``, which severs the graph. A float32
   tensor that requires grad comes back from ``place`` detached and from
   ``_prep`` still in the graph.
3. **Declared rather than read.** ``_prep`` states ``float64`` on ``_device``
   unconditionally and returns an annotated ``torch.Tensor``; the inherited
   one returns ``Any`` and adopts whatever the first float array the object
   happens to hold says.

It is the counterpart of pyvinecopulib's own
``torch._placement.TensorPlacementMixin``, which reads a registered tensor off
an ``nn.Module`` instead.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["TensorPlacement", "resolve_device"]


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


class TensorPlacement:
  """The ``_prep`` hook for an estimator placed on its own ``_device``.

  Mixed in **ahead** of the canonical pyvinecopulib base, so ``_prep``
  resolves here rather than to the array-API inference that base ships.

  Only ordinary private members belong on a mixin at that position: it lands
  ahead of everything behind it in the resulting MRO, so anything defined here
  that a base also defines would silently shadow it.

  Overriding ``_prep`` is not sufficient on its own, which is what
  :meth:`_set_placement` is for. ``pyvinecopulib.core.prepare_covariates``
  places a covariate matrix through the module-level ``place`` rather than
  through the object's ``_prep``, so the override never sees that path -- and
  ``place`` answers by looking for an array the object *holds*. Holding one is
  therefore the only way to be placed on the paths the base owns: the vine
  cascade's per-edge covariates, and ``BicopBase.loglik``.
  """

  _device: torch.device
  #: An empty ``float64`` tensor on :attr:`_device`, so that upstream's
  #: module-level ``place`` can read a placement off this object. See the
  #: class docstring for why holding one is necessary.
  _placement_ref: torch.Tensor

  def _set_placement(self, device: str | torch.device | None) -> None:
    """Record the device this estimator evaluates on, and a reference tensor.

    Parameters
    ----------
    device : str, torch.device, or None
        The requested device; ``None`` picks the best available one.

    Returns
    -------
    None
    """
    self._device = resolve_device(device)
    # Empty, so it costs nothing, and float64 so `place` adopts the working
    # precision rather than only the device. See the class docstring for why
    # holding one is necessary at all.
    self._placement_ref = torch.empty(
      0, dtype=torch.float64, device=self._device
    )

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
