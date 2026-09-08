"""Fixed-structure non-simplified vines built from Rosenblatt pair copulas."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import torch
from pyvinecopulib import RVineStructure
from pyvinecopulib.core import (
  BicopLike,
  NonSimplifiedContext,
  VinecopBase,
)

from npcc.core._common import _resolve_device
from npcc.core.bicop import RosenblattBicop


class RosenblattVinecop(VinecopBase[torch.Tensor]):
  """Fixed-structure non-simplified vine of Rosenblatt pair copulas.

  Each edge contains a fitted :class:`RosenblattBicop`. Higher-tree edges
  receive their vine conditioning-set values followed by any external
  covariates.

  Parameters
  ----------
  pair_copulas
    Fitted pair copulas arranged as ``[tree][edge]``. ``None`` creates an unfitted
    vine whose pairs can subsequently be estimated with :meth:`fit`.
  structure
    R-vine structure defining the edge layout and order.
  var_types
    Variable types in variable order. Only continuous variables are currently
    supported by :class:`RosenblattBicop`.
  device
    Device used for random sampling. When omitted for a fitted vine, it is
    inferred from the pair copulas.

  Notes
  -----
  Public numerical methods accept and return torch tensors. Quasi-random
  sampling uses pyvinecopulib's NumPy-based sampling utility internally and
  converts its result to torch at that boundary.
  """

  supports_covariates: bool = True
  bicop_class: ClassVar[type[RosenblattBicop]] = RosenblattBicop

  def __init__(
    self,
    pair_copulas: list[list[RosenblattBicop]] | None,
    structure: RVineStructure,
    *,
    var_types: list[str] | None = None,
    device: str | torch.device | None = None,
  ) -> None:
    self.pair_copulas: list[list[RosenblattBicop]] = []

    self._bind_vine(
      structure,
      NonSimplifiedContext(),
      var_types=var_types,
    )

    if pair_copulas is None:
      self._device = _resolve_device(device)
      return

    copied_pairs = self._copy_pair_copulas(pair_copulas)
    self._validate_pair_copulas(copied_pairs)

    self.pair_copulas = copied_pairs
    self._device = self._resolve_vine_device(
      copied_pairs,
      device=device,
    )

  @staticmethod
  def _copy_pair_copulas(
    pair_copulas: Sequence[Sequence[BicopLike[torch.Tensor]]],
  ) -> list[list[RosenblattBicop]]:
    """Validate pair implementations and copy the nested containers."""
    copied: list[list[RosenblattBicop]] = []

    for row in pair_copulas:
      copied_row: list[RosenblattBicop] = []

      for pair in row:
        if not isinstance(pair, RosenblattBicop):
          raise TypeError(
            "RosenblattVinecop only accepts RosenblattBicop pairs."
          )

        copied_row.append(pair)

      copied.append(copied_row)

    return copied

  def _validate_pair_copulas(
    self,
    pair_copulas: list[list[RosenblattBicop]],
  ) -> None:
    """Validate the tree and edge dimensions of a pair-copula matrix."""
    if len(pair_copulas) != self.trunc_lvl:
      raise ValueError(
        f"pair_copulas has {len(pair_copulas)} trees, "
        f"expected {self.trunc_lvl}."
      )

    for tree, row in enumerate(pair_copulas):
      expected_edges = self.d - tree - 1

      if len(row) != expected_edges:
        raise ValueError(
          f"pair_copulas tree {tree} has {len(row)} edges, "
          f"expected {expected_edges}."
        )

  def _resolve_vine_device(
    self,
    pair_copulas: list[list[RosenblattBicop]],
    *,
    device: str | torch.device | None = None,
  ) -> torch.device:
    """Infer and validate the common pair-copula device."""
    pair_devices = {pair._device for row in pair_copulas for pair in row}

    if len(pair_devices) > 1:
      devices = ", ".join(
        sorted(str(pair_device) for pair_device in pair_devices)
      )
      raise ValueError(
        f"All pair copulas must use the same device; found: {devices}."
      )

    requested_device = None if device is None else _resolve_device(device)

    if pair_devices:
      pair_device = next(iter(pair_devices))

      if requested_device is not None and requested_device != pair_device:
        raise ValueError(
          f"Requested vine device {requested_device} does not match "
          f"pair-copula device {pair_device}."
        )

      return pair_device

    return (
      requested_device
      if requested_device is not None
      else _resolve_device(None)
    )

  def get_pair_copula(
    self,
    tree: int,
    edge: int,
  ) -> RosenblattBicop:
    """Return the pair copula at a tree and edge position."""
    return self.pair_copulas[tree][edge]

  def set_pair_copulas(
    self,
    pair_copulas: list[list[BicopLike[torch.Tensor]]],
  ) -> None:
    """Install pair copulas produced by the inherited fit engine."""
    copied_pairs = self._copy_pair_copulas(pair_copulas)
    self._validate_pair_copulas(copied_pairs)

    self.pair_copulas = copied_pairs
    self._device = self._resolve_vine_device(copied_pairs)
    self._batched = None

  def _sample_uniform(
    self,
    n: int,
    qrng: bool,
    seeds: list[int],
  ) -> torch.Tensor:
    """Draw base uniforms for inherited vine sampling."""
    if qrng:
      from pyvinecopulib.utils import sample_uniform

      draws = sample_uniform(n, self.d, qrng=True, seeds=list(seeds))
      return torch.as_tensor(
        draws,
        dtype=torch.float64,
        device=self._device,
      )

    generator = torch.Generator(device=self._device)
    if seeds:
      generator.manual_seed(int(seeds[0]))
    else:
      generator.seed()

    return torch.rand(
      (n, self.d),
      generator=generator,
      dtype=torch.float64,
      device=self._device,
    )
