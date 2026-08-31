"""Fixed-structure non-simplified vines built from Rosenblatt pair copulas."""

from __future__ import annotations

from typing import Any, Literal, Self, cast

import torch

from pyvinecopulib import RVineStructure
from pyvinecopulib.core import NonSimplifiedContext, VinecopBase

from npcc.core._common import (
  TensorLike,
  _resolve_device,
  _to_tensor,
  _wrap_output,
  is_torch_array,
)
from npcc.core.bicop import RosenblattBicop
from npcc.core.quantile_table_distribution1d import QuantileGridConfig


class RosenblattVinecop(VinecopBase[TensorLike]):
  """A fixed-structure non-simplified vine of Rosenblatt pair copulas."""

  supports_covariates: bool = True

  def __init__(
    self,
    pair_copulas: list[list[RosenblattBicop]],
    structure: RVineStructure,
    *,
    device: str | torch.device | None = None,
  ) -> None:
    self._bind_vine(structure, NonSimplifiedContext())
    self.pair_copulas = [list(row) for row in pair_copulas]
    self._validate_pair_copulas()
    self._device = self._resolve_vine_device(device)

  @classmethod
  def from_data(
    cls,
    u: TensorLike,
    structure: RVineStructure,
    *,
    x: TensorLike | None = None,
    backend: str = "tabpfn-criterion",
    quantile_config: QuantileGridConfig | None = None,
    transform: Literal["identity", "logit", "probit"] = "logit",
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    # Backend-specific third-party parameters are intentionally heterogeneous.
    backend_kwargs: dict[str, Any] | None = None,
    sinkhorn_iters: int | None = None,
    projection_grid_size: int = 101,
  ) -> Self:
    """Fit Rosenblatt pair copulas along a fixed R-vine structure."""

    effective_device = _resolve_device(device)

    def fit_edge(
      tree: int,
      edge: int,
      u_edge: TensorLike,
      x_edge: TensorLike | None,
    ) -> RosenblattBicop:
      del tree, edge
      return RosenblattBicop(
        backend=backend,
        quantile_config=quantile_config,
        transform=transform,
        device=effective_device,
        batch_size=batch_size,
        backend_kwargs=backend_kwargs,
        sinkhorn_iters=sinkhorn_iters,
        projection_grid_size=projection_grid_size,
      ).fit(u_edge, x_edge)

    pair_copulas = cast(
      list[list[RosenblattBicop]],
      VinecopBase.fit(
        structure,
        u,
        fit_edge,
        context=NonSimplifiedContext(),
        x=x,
      ),
    )
    return cls(pair_copulas, structure, device=effective_device)

  def _validate_pair_copulas(self) -> None:
    expected_trees = self.trunc_lvl
    if len(self.pair_copulas) != expected_trees:
      raise ValueError(
        f"pair_copulas has {len(self.pair_copulas)} trees, "
        f"expected {expected_trees}."
      )

    for tree, row in enumerate(self.pair_copulas):
      expected_edges = self.d - tree - 1
      if len(row) != expected_edges:
        raise ValueError(
          f"pair_copulas tree {tree} has {len(row)} edges, "
          f"expected {expected_edges}."
        )

  def _resolve_vine_device(
    self,
    device: str | torch.device | None,
  ) -> torch.device:
    pair_devices = {pair._device for row in self.pair_copulas for pair in row}

    if len(pair_devices) > 1:
      devices = ", ".join(sorted(str(value) for value in pair_devices))
      raise ValueError(
        f"All pair copulas must use the same device; found: {devices}."
      )

    requested = None if device is None else _resolve_device(device)

    if pair_devices:
      pair_device = next(iter(pair_devices))
      if requested is not None and requested != pair_device:
        raise ValueError(
          f"Requested vine device {requested} does not match "
          f"pair-copula device {pair_device}."
        )
      return pair_device

    return _resolve_device(device)

  def _get_pair_copula(
    self,
    tree: int,
    edge: int,
  ) -> RosenblattBicop:
    return self.pair_copulas[tree][edge]

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

  def cdf(
    self,
    u: TensorLike,
    *,
    N: int = 10000,
    qrng: bool = True,
    num_threads: int = 1,
    seeds: list[int] | None = None,
    x: TensorLike | None = None,
    block_size: int = 4096,
    batched: bool | None = None,
  ) -> TensorLike:
    """Evaluate the Monte Carlo CDF while preserving the query array type."""

    return_as_torch = is_torch_array(u)
    u_t = _to_tensor(u, device=self._device)
    x_t = None if x is None else _to_tensor(x, device=self._device)

    out = super().cdf(
      u_t,
      N=N,
      qrng=qrng,
      num_threads=num_threads,
      seeds=seeds,
      x=x_t,
      block_size=block_size,
      batched=batched,
    )
    assert isinstance(out, torch.Tensor)
    return _wrap_output(out, return_as_torch=return_as_torch)

  def sample_conditional(
    self,
    u_cond: TensorLike,
    *,
    qrng: bool = False,
    num_threads: int = 1,
    seeds: list[int] | None = None,
    conditioning_set: list[int] | None = None,
    x: TensorLike | None = None,
  ) -> TensorLike:
    """Conditionally sample while preserving the conditioning array type."""

    return_as_torch = is_torch_array(u_cond)
    u_cond_t = _to_tensor(u_cond, device=self._device)
    x_t = None if x is None else _to_tensor(x, device=self._device)

    out = super().sample_conditional(
      u_cond_t,
      qrng=qrng,
      num_threads=num_threads,
      seeds=seeds,
      conditioning_set=conditioning_set,
      x=x_t,
    )
    assert isinstance(out, torch.Tensor)
    return _wrap_output(out, return_as_torch=return_as_torch)
