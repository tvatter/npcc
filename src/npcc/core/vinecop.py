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
  """Fixed-structure non-simplified vine of Rosenblatt pair copulas.

  Each edge is a fitted :class:`RosenblattBicop`. Higher-tree edges receive
  their vine conditioning-set values followed by any external covariates.
  The vine structure is fixed; automatic structure selection is not provided.

  Parameters
  ----------
  pair_copulas
    Fitted pair copulas arranged as ``[tree][edge]``. Tree ``t`` must contain
    ``dim - t - 1`` pairs.
  structure
    R-vine structure defining the edge layout, order, and truncation level.
  device
    Device used for random sampling. When omitted, it is inherited from the
    pair copulas. All pair copulas must use the same device.

  Notes
  -----
  NumPy and torch inputs are supported, but ``u`` and ``x`` must use the same
  array namespace within a call- Unconditional ``sample`` returns a float64
  torch tensor on ``device``.
  """

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
    """Fit Rosenblatt pair copulas along a fixed R-vine structure.

    Parameters
    ----------
    u
      Continuous pseudo-observations with shape ``(n, structure.dim)`` and
      values in the unit interval.
    structure
      Fixed R-vine structure. Its order and truncation level are retained.
    x
      Optional external covariates with shape ``(n, p)``. These are appended
      after each edge's internal conditioning-set values.
    backend
      Registered conditional-distribution backend used for every pair copula.
    quantile_config
      Quantile-grid and boundary configuration forwarded to every pair copula.
    transform
      Target-support transform used by every pair copula backend.
    device
      Shared pair copula and sampling device.
    batch_size
      Default backend inference chunk size.
    backend_kwargs
      Backend-specific constructor options forwarded to every pair.
    sinkhorn_iters
      Optional pair copula-level Sinkhorn projection iteration count.
    projection_grid_size
      Pair copula-level Sinkhorn projection grid size.

    Returns
    -------
    RosenblattVinecop
      Fitted non-simplified vine using the supplied structure.

    Raises
    ------
    ValueError
      If the observations, covariates, or structure dimensions are
      incompatible.

    Notes
    -----
    Tree-zero pair copulas receive external ``x`` only. A higher-tree pair copula
    receives ``[u_D, x]``, where ``u_D`` follows pyvinecopulib's conditioning-tree
    order. The method fits edges sequentially because later trees depend on
    h-functions from earlier fitted pair copulas.
    """

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
    """Evaluate the Monte Carlo CDF while preserving the query array type.

    The inherited CDF combines query points with samples generated on the vine's
    torch device. NumPy queries are temporarily converted to torch and converted
    back after evaluation. External-covariate CDF evaluation is unsupported by
    ``VinecopBase``.
    """

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
    """Conditionally sample while preserving the conditioning array type.

    The inherited sampler uses torch base uniforms. NumPy conditioning values are
    temporarily converted to the vine device and the resulting samples are
    converted back to NumPy.
    """

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
