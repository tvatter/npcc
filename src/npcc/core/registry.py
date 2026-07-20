"""
registry.py — name -> backend-factory registry for the inner
conditional distribution used by :class:`~npcc.core.bicop.RosenblattBicop`.

A *factory* is any callable that accepts the common construction kwargs
(``transform``, ``config``, ``device``, ``batch_size``) plus arbitrary
backend-specific keyword arguments, and returns a
:class:`~npcc.core.conditional_distribution1d.ConditionalDistribution1D`.

Built-in backends are registered lazily: the factory imports its backend
module (and, for the optional-extra backends, its third-party
dependency) only when the backend is actually requested, so ``import
npcc`` and :func:`available_backends` never import NGBoost / TabICL /
scikit-learn.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from typing import Any, Literal

from npcc.core.conditional_distribution1d import ConditionalDistribution1D
from npcc.core.quantile_table_distribution1d import QuantileGridConfig

_Transform = Literal["identity", "logit", "probit"]

BackendFactory = Callable[..., ConditionalDistribution1D]

_REGISTRY: dict[str, BackendFactory] = {}


def register_backend(name: str, factory: BackendFactory) -> None:
  """Register ``factory`` under ``name`` (overwrites any existing entry)."""
  _REGISTRY[name] = factory


def available_backends() -> list[str]:
  """Return the sorted list of registered backend names."""
  return sorted(_REGISTRY)


def create_backend(name: str, **kwargs: object) -> ConditionalDistribution1D:
  """Instantiate the backend registered under ``name``.

  Extra ``kwargs`` are forwarded to the factory (hence to the backend
  constructor); passing a keyword a backend does not accept raises
  ``TypeError`` from that constructor.
  """
  if name not in _REGISTRY:
    raise ValueError(
      f"Unknown backend {name!r}. Available: {available_backends()}."
    )
  return _REGISTRY[name](**kwargs)


def _missing_extra(name: str, extra: str) -> ImportError:
  return ImportError(
    f"The {name!r} backend requires an optional dependency. "
    f"Install it with `pip install npcc[{extra}]` (or `uv sync "
    f"--extra {extra}`)."
  )


# ------------------------------------------------------------------
# Built-in backends (lazy factories).
# ------------------------------------------------------------------


def _tabpfn_criterion_factory(
  *,
  transform: _Transform,
  config: QuantileGridConfig,
  device: str | torch.device | None,
  batch_size: int | None,
  **kw: Any,  # noqa: ANN401 - heterogeneous backend kwargs
) -> ConditionalDistribution1D:
  from npcc.core.backends.tabpfn_criterion import TabPFNCriterionBackend

  return TabPFNCriterionBackend(
    transform=transform,
    eps=config.eps,
    device=device,
    batch_size=batch_size,
    **kw,
  )


def _tabpfn_quantiles_factory(
  *,
  transform: _Transform,
  config: QuantileGridConfig,
  device: str | torch.device | None,
  batch_size: int | None,
  **kw: Any,  # noqa: ANN401 - heterogeneous backend kwargs
) -> ConditionalDistribution1D:
  from npcc.core.backends.tabpfn_quantile import TabPFNQuantileBackend

  return TabPFNQuantileBackend(
    transform=transform,
    config=config,
    device=device,
    batch_size=batch_size,
    **kw,
  )


def _ngboost_factory(
  *,
  transform: _Transform,
  config: QuantileGridConfig,
  device: str | torch.device | None,
  batch_size: int | None,
  **kw: Any,  # noqa: ANN401 - heterogeneous backend kwargs
) -> ConditionalDistribution1D:
  try:
    from npcc.core.backends.ngboost import NGBoostBackend
  except ImportError as exc:
    raise _missing_extra("ngboost", "ngboost") from exc

  return NGBoostBackend(
    transform=transform,
    eps=config.eps,
    device=device,
    batch_size=batch_size,
    **kw,
  )


def _quantile_gbm_factory(
  *,
  transform: _Transform,
  config: QuantileGridConfig,
  device: str | torch.device | None,
  batch_size: int | None,
  **kw: Any,  # noqa: ANN401 - heterogeneous backend kwargs
) -> ConditionalDistribution1D:
  try:
    from npcc.core.backends.quantile_gbm import QuantileGBMBackend
  except ImportError as exc:
    raise _missing_extra("gbm", "gbm") from exc

  return QuantileGBMBackend(
    transform=transform,
    config=config,
    device=device,
    batch_size=batch_size,
    **kw,
  )


def _tabicl_factory(
  *,
  transform: _Transform,
  config: QuantileGridConfig,
  device: str | torch.device | None,
  batch_size: int | None,
  **kw: Any,  # noqa: ANN401 - heterogeneous backend kwargs
) -> ConditionalDistribution1D:
  try:
    from npcc.core.backends.tabicl import TabICLBackend
  except ImportError as exc:
    raise _missing_extra("tabicl", "tabicl") from exc

  return TabICLBackend(
    transform=transform,
    config=config,
    device=device,
    batch_size=batch_size,
    **kw,
  )


register_backend("tabpfn-criterion", _tabpfn_criterion_factory)
register_backend("tabpfn-quantiles", _tabpfn_quantiles_factory)
register_backend("ngboost", _ngboost_factory)
register_backend("gbm", _quantile_gbm_factory)
register_backend("tabicl", _tabicl_factory)
