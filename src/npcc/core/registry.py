"""Name to backend-factory registry for the inner conditional distribution.

The registry :class:`~npcc.core.bicop.RosenblattBicop` selects its inner
conditional-density model from.

A *factory* is any callable that accepts the common construction kwargs
(``transform``, ``quantile_table_config``, ``eps``, ``device``,
``batch_size``) plus arbitrary backend-specific keyword arguments, and returns a
:class:`~npcc.core.margin.ConditionalMargin`.

Built-in backends are registered lazily: the factory imports its backend
module (and, for the optional-extra backends, its third-party
dependency) only when the backend is actually requested, so ``import
npcc`` and :func:`available_backends` never import NGBoost / TabICL /
scikit-learn.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from npcc.core.errors import (
  InvalidBackendKwargsError,
  MissingBackendDependencyError,
  UnknownBackendError,
)
from npcc.core.margin import ConditionalMargin
from npcc.core.margin_quantile_table import QuantileTableConfig

_Transform = Literal["identity", "logit", "probit"]

BackendFactory = Callable[..., ConditionalMargin]

_JSON_SCALARS = (bool, int, float, str)


@dataclass(frozen=True)
class BackendSpec:
  """A registered backend: its factory plus config-validation metadata.

  ``allowed_kwargs=None`` means the backend accepts arbitrary ``backend_kwargs``
  (only their JSON shape is checked) — used by hermetic test fakes and any
  opt-out backend. A concrete set restricts the keys a study config may
  pass, so typos fail at config-load time instead of hours into a run.
  ``nested_kwargs`` names keys whose value must itself be a table/dict (e.g.
  ``model_kwargs``); ``n_range`` is the backend's documented sample-size range.
  """

  factory: BackendFactory
  allowed_kwargs: frozenset[str] | None = None
  nested_kwargs: frozenset[str] = frozenset()
  kwarg_types: Mapping[str, type] = field(default_factory=dict)
  n_range: tuple[int | None, int | None] | None = None


_REGISTRY: dict[str, BackendSpec] = {}


def register_backend(
  name: str,
  factory: BackendFactory,
  *,
  allowed_kwargs: frozenset[str] | None = None,
  nested_kwargs: frozenset[str] = frozenset(),
  kwarg_types: Mapping[str, type] | None = None,
  n_range: tuple[int | None, int | None] | None = None,
) -> None:
  """Register ``factory`` under ``name`` (overwrites any existing entry)."""
  _REGISTRY[name] = BackendSpec(
    factory=factory,
    allowed_kwargs=allowed_kwargs,
    nested_kwargs=nested_kwargs,
    kwarg_types=dict(kwarg_types or {}),
    n_range=n_range,
  )


def available_backends() -> list[str]:
  """Return the sorted list of registered backend names."""
  return sorted(_REGISTRY)


def _spec(name: str) -> BackendSpec:
  try:
    return _REGISTRY[name]
  except KeyError:
    raise UnknownBackendError(
      f"Unknown backend {name!r}. Available: {available_backends()}."
    ) from None


def _is_json(value: object) -> bool:
  if value is None or isinstance(value, _JSON_SCALARS):
    return True
  if isinstance(value, (list, tuple)):
    return all(_is_json(v) for v in value)
  if isinstance(value, dict):
    return all(isinstance(k, str) and _is_json(v) for k, v in value.items())
  return False


def _matches_type(value: object, expected: type) -> bool:
  # bool is an int subclass; never accept it where a number is wanted.
  if expected in (int, float) and isinstance(value, bool):
    return False
  if expected is float:
    return isinstance(value, (int, float))
  return isinstance(value, expected)


def validate_backend_kwargs(name: str, kwargs: Mapping[str, object]) -> None:
  """Validate ``backend_kwargs`` for ``name`` against its :class:`BackendSpec`.

  Always checks JSON shape; when the spec declares ``allowed_kwargs`` it also
  rejects unknown keys and type-checks the ones listed in ``kwarg_types``.
  """
  spec = _spec(name)
  for key, value in kwargs.items():
    if not _is_json(value):
      raise InvalidBackendKwargsError(
        f"{name}.{key} must be JSON-serializable "
        f"(scalars, lists, dicts); got {type(value).__name__}."
      )
  if spec.allowed_kwargs is None:
    return
  unknown = sorted(set(kwargs) - spec.allowed_kwargs)
  if unknown:
    raise InvalidBackendKwargsError(
      f"{name}: unknown backend_kwargs {unknown}; "
      f"allowed: {sorted(spec.allowed_kwargs)}."
    )
  for key, value in kwargs.items():
    if key in spec.nested_kwargs:
      if not isinstance(value, dict):
        raise InvalidBackendKwargsError(f"{name}.{key} must be a table/dict.")
      continue
    expected = spec.kwarg_types.get(key)
    if expected is not None and not _matches_type(value, expected):
      raise InvalidBackendKwargsError(
        f"{name}.{key} must be {expected.__name__}, got {type(value).__name__}."
      )


def documented_n_range(name: str) -> tuple[int | None, int | None] | None:
  """Return the backend's documented ``(min_n, max_n)`` range, if declared."""
  return _spec(name).n_range


def create_backend(
  name: str,
  *,
  transform: _Transform,
  quantile_table_config: QuantileTableConfig,
  eps: float,
  device: str | torch.device | None,
  batch_size: int | None,
  backend_kwargs: Mapping[str, Any] | None = None,
) -> ConditionalMargin:
  """Instantiate the backend registered under ``name``.

  ``backend_kwargs`` are validated against the backend's :class:`BackendSpec`
  (unknown or mis-typed keys raise :class:`InvalidBackendKwargsError`) and then
  forwarded to the factory alongside the common construction kwargs.
  """
  spec = _spec(name)
  kwargs = dict(backend_kwargs or {})
  validate_backend_kwargs(name, kwargs)
  return spec.factory(
    transform=transform,
    quantile_table_config=quantile_table_config,
    eps=eps,
    device=device,
    batch_size=batch_size,
    **kwargs,
  )


def _missing_extra(name: str, extra: str) -> MissingBackendDependencyError:
  return MissingBackendDependencyError(
    f"The {name!r} backend requires an optional dependency. "
    f"Install it with `pip install npcc[{extra}]` (or `uv sync "
    f"--extra {extra}`)."
  )


def _coerce_model_version(kw: dict[str, Any]) -> None:
  """Turn a string ``model_version`` backend kwarg into the TabPFN enum."""
  if "model_version" in kw:
    from npcc.core.backends.tabpfn_common import ModelVersion

    version = kw["model_version"]
    if not isinstance(version, ModelVersion):
      kw["model_version"] = ModelVersion(version)


# ------------------------------------------------------------------
# Built-in backends (lazy factories).
# ------------------------------------------------------------------


def _tabpfn_criterion_factory(
  *,
  transform: _Transform,
  quantile_table_config: QuantileTableConfig,
  eps: float,
  device: str | torch.device | None,
  batch_size: int | None,
  **kw: Any,  # noqa: ANN401 - heterogeneous backend kwargs
) -> ConditionalMargin:
  from npcc.core.backends.tabpfn_criterion import TabPFNCriterionBackend

  del quantile_table_config
  _coerce_model_version(kw)
  return TabPFNCriterionBackend(
    transform=transform,
    eps=eps,
    device=device,
    batch_size=batch_size,
    **kw,
  )


def _tabpfn_quantiles_factory(
  *,
  transform: _Transform,
  quantile_table_config: QuantileTableConfig,
  eps: float,
  device: str | torch.device | None,
  batch_size: int | None,
  **kw: Any,  # noqa: ANN401 - heterogeneous backend kwargs
) -> ConditionalMargin:
  from npcc.core.backends.tabpfn_quantile import TabPFNQuantileBackend

  _coerce_model_version(kw)
  return TabPFNQuantileBackend(
    transform=transform,
    quantile_table_config=quantile_table_config,
    eps=eps,
    device=device,
    batch_size=batch_size,
    **kw,
  )


def _ngboost_factory(
  *,
  transform: _Transform,
  quantile_table_config: QuantileTableConfig,
  eps: float,
  device: str | torch.device | None,
  batch_size: int | None,
  **kw: Any,  # noqa: ANN401 - heterogeneous backend kwargs
) -> ConditionalMargin:
  try:
    from npcc.core.backends.ngboost import NGBoostBackend
  except ImportError as exc:
    raise _missing_extra("ngboost", "ngboost") from exc

  del quantile_table_config
  return NGBoostBackend(
    transform=transform,
    eps=eps,
    device=device,
    batch_size=batch_size,
    **kw,
  )


def _quantile_gbm_factory(
  *,
  transform: _Transform,
  quantile_table_config: QuantileTableConfig,
  eps: float,
  device: str | torch.device | None,
  batch_size: int | None,
  **kw: Any,  # noqa: ANN401 - heterogeneous backend kwargs
) -> ConditionalMargin:
  try:
    from npcc.core.backends.quantile_gbm import QuantileGBMBackend
  except ImportError as exc:
    raise _missing_extra("gbm", "gbm") from exc

  return QuantileGBMBackend(
    transform=transform,
    quantile_table_config=quantile_table_config,
    eps=eps,
    device=device,
    batch_size=batch_size,
    **kw,
  )


def _tabicl_factory(
  *,
  transform: _Transform,
  quantile_table_config: QuantileTableConfig,
  eps: float,
  device: str | torch.device | None,
  batch_size: int | None,
  **kw: Any,  # noqa: ANN401 - heterogeneous backend kwargs
) -> ConditionalMargin:
  try:
    from npcc.core.backends.tabicl import TabICLBackend
  except ImportError as exc:
    raise _missing_extra("tabicl", "tabicl") from exc

  return TabICLBackend(
    transform=transform,
    quantile_table_config=quantile_table_config,
    eps=eps,
    device=device,
    batch_size=batch_size,
    **kw,
  )


def _catboost_factory(
  *,
  transform: _Transform,
  quantile_table_config: QuantileTableConfig,
  eps: float,
  device: str | torch.device | None,
  batch_size: int | None,
  **kw: Any,  # noqa: ANN401 - heterogeneous backend kwargs
) -> ConditionalMargin:
  try:
    from npcc.core.backends.catboost import CatBoostBackend
  except ImportError as exc:
    raise _missing_extra("catboost", "catboost") from exc

  return CatBoostBackend(
    transform=transform,
    quantile_table_config=quantile_table_config,
    eps=eps,
    device=device,
    batch_size=batch_size,
    **kw,
  )


def _xgb_quantile_factory(
  *,
  transform: _Transform,
  quantile_table_config: QuantileTableConfig,
  eps: float,
  device: str | torch.device | None,
  batch_size: int | None,
  **kw: Any,  # noqa: ANN401 - heterogeneous backend kwargs
) -> ConditionalMargin:
  try:
    from npcc.core.backends.xgboost_quantile import XGBQuantileBackend
  except ImportError as exc:
    raise _missing_extra("xgb-quantile", "xgboost") from exc

  return XGBQuantileBackend(
    transform=transform,
    quantile_table_config=quantile_table_config,
    eps=eps,
    device=device,
    batch_size=batch_size,
    **kw,
  )


def _pytabkit_realmlp_factory(
  *,
  transform: _Transform,
  quantile_table_config: QuantileTableConfig,
  eps: float,
  device: str | torch.device | None,
  batch_size: int | None,
  **kw: Any,  # noqa: ANN401 - heterogeneous backend kwargs
) -> ConditionalMargin:
  try:
    from npcc.core.backends.pytabkit import PyTabKitRealMLPBackend
  except ImportError as exc:
    raise _missing_extra("pytabkit-realmlp", "pytabkit") from exc

  return PyTabKitRealMLPBackend(
    transform=transform,
    quantile_table_config=quantile_table_config,
    eps=eps,
    device=device,
    batch_size=batch_size,
    **kw,
  )


def _pytabkit_tabm_factory(
  *,
  transform: _Transform,
  quantile_table_config: QuantileTableConfig,
  eps: float,
  device: str | torch.device | None,
  batch_size: int | None,
  **kw: Any,  # noqa: ANN401 - heterogeneous backend kwargs
) -> ConditionalMargin:
  try:
    from npcc.core.backends.pytabkit import PyTabKitTabMBackend
  except ImportError as exc:
    raise _missing_extra("pytabkit-tabm", "pytabkit") from exc

  return PyTabKitTabMBackend(
    transform=transform,
    quantile_table_config=quantile_table_config,
    eps=eps,
    device=device,
    batch_size=batch_size,
    **kw,
  )


def _tabpfn_finetune_factory(
  *,
  transform: _Transform,
  quantile_table_config: QuantileTableConfig,
  eps: float,
  device: str | torch.device | None,
  batch_size: int | None,
  **kw: Any,  # noqa: ANN401 - heterogeneous backend kwargs
) -> ConditionalMargin:
  # Fine-tuning is native to the core `tabpfn` package (no extra).
  from npcc.core.backends.tabpfn_finetune import (
    FinetunedTabPFNCriterionBackend,
  )

  del quantile_table_config
  return FinetunedTabPFNCriterionBackend(
    transform=transform,
    eps=eps,
    device=device,
    batch_size=batch_size,
    **kw,
  )


def _tabicl_finetune_factory(
  *,
  transform: _Transform,
  quantile_table_config: QuantileTableConfig,
  eps: float,
  device: str | torch.device | None,
  batch_size: int | None,
  **kw: Any,  # noqa: ANN401 - heterogeneous backend kwargs
) -> ConditionalMargin:
  try:
    from npcc.core.backends.tabicl_finetune import FinetunedTabICLBackend
  except ImportError as exc:
    raise _missing_extra("tabicl-finetune", "tabicl") from exc

  return FinetunedTabICLBackend(
    transform=transform,
    quantile_table_config=quantile_table_config,
    eps=eps,
    device=device,
    batch_size=batch_size,
    **kw,
  )


def _nori_factory(
  *,
  transform: _Transform,
  quantile_table_config: QuantileTableConfig,
  eps: float,
  device: str | torch.device | None,
  batch_size: int | None,
  **kw: Any,  # noqa: ANN401 - heterogeneous backend kwargs
) -> ConditionalMargin:
  try:
    from npcc.core.backends.nori import NoriBackend
  except ImportError as exc:
    raise MissingBackendDependencyError(
      "The 'nori' backend requires synthefy-nori. Until the torch-uncapping "
      "fix reaches PyPI, install the fork into a cu128 env:\n"
      '  uv pip install "synthefy-nori @ '
      'git+https://github.com/tvatter/synthefy-nori.git@allow-newer-torch-cuda"'
    ) from exc

  return NoriBackend(
    transform=transform,
    quantile_table_config=quantile_table_config,
    eps=eps,
    device=device,
    batch_size=batch_size,
    **kw,
  )


# TabPFN model version (v2.5 / v3) is a hyperparameter, not a distinct
# backend: pass `backend_kwargs = { model_version = "v2.5" }` (default v3).
_TABPFN_KWARGS = frozenset({"model_kwargs", "model_version"})
_FINETUNE_LOSS_KWARGS = frozenset(
  {
    "epochs",
    "learning_rate",
    "early_stopping",
    "ce_loss_weight",
    "crps_loss_weight",
    "crls_loss_weight",
    "mse_loss_weight",
    "mae_loss_weight",
  }
)
_FINETUNE_LOSS_TYPES: dict[str, type] = {
  "epochs": int,
  "learning_rate": float,
  "early_stopping": bool,
  "ce_loss_weight": float,
  "crps_loss_weight": float,
  "crls_loss_weight": float,
  "mse_loss_weight": float,
  "mae_loss_weight": float,
}
_PYTABKIT_KWARGS = frozenset(
  {"hpo", "n_cv", "n_refit", "random_state", "n_epochs"}
)
_PYTABKIT_TYPES: dict[str, type] = {
  "hpo": bool,
  "n_cv": int,
  "n_refit": int,
  "n_epochs": int,
}

register_backend(
  "tabpfn-criterion",
  _tabpfn_criterion_factory,
  allowed_kwargs=_TABPFN_KWARGS,
  nested_kwargs=frozenset({"model_kwargs"}),
  n_range=(None, 100_000),
)
register_backend(
  "tabpfn-quantiles",
  _tabpfn_quantiles_factory,
  allowed_kwargs=_TABPFN_KWARGS,
  nested_kwargs=frozenset({"model_kwargs"}),
  n_range=(None, 100_000),
)
register_backend(
  "tabpfn-finetune",
  _tabpfn_finetune_factory,
  allowed_kwargs=_FINETUNE_LOSS_KWARGS,
  kwarg_types=_FINETUNE_LOSS_TYPES,
  n_range=(None, 50_000),
)
register_backend(
  "ngboost",
  _ngboost_factory,
  allowed_kwargs=frozenset(
    {"n_estimators", "learning_rate", "minibatch_frac", "random_state"}
  ),
  kwarg_types={"n_estimators": int, "learning_rate": float},
)
register_backend(
  "gbm",
  _quantile_gbm_factory,
  allowed_kwargs=frozenset(
    {
      "n_estimators",
      "max_depth",
      "learning_rate",
      "subsample",
      "min_samples_leaf",
      "random_state",
    }
  ),
  kwarg_types={
    "n_estimators": int,
    "max_depth": int,
    "learning_rate": float,
    "subsample": float,
  },
)
register_backend(
  "catboost",
  _catboost_factory,
  allowed_kwargs=frozenset(
    {
      "iterations",
      "depth",
      "learning_rate",
      "l2_leaf_reg",
      "random_seed",
      "border_count",
      "task_type",
    }
  ),
  kwarg_types={"iterations": int, "depth": int, "learning_rate": float},
)
# `xgb-quantile` is registered like any other backend, and is a poor choice on
# a study grid that reaches into the tails: XGBoost quantile trees cannot
# extrapolate, so the predicted conditional support collapses to roughly the
# inner [0.24, 0.96] of (0, 1) and the quantile->density inversion returns
# density exactly 0 outside it. On the study grid that zeroes ~44% of
# evaluation points holding ~16% of the true mass, so KL = E_true[log(truth/0)]
# blows up: pdf-KL ~8 against ~0.1-0.4 for every other backend. Restricted to
# in-support cells its KL is an ordinary ~0.5, so this is tail collapse rather
# than quantile crossing (rows are monotone-sorted) or a metric artifact. A
# tail-robust inversion -- extrapolated quantile tails, or a floored density --
# is the prerequisite to using it on unbounded supports. A registry entry is
# not a recommendation; a study excludes it by not naming it.
register_backend(
  "xgb-quantile",
  _xgb_quantile_factory,
  allowed_kwargs=frozenset(
    {
      "n_estimators",
      "tree_method",
      "max_depth",
      "learning_rate",
      "subsample",
      "colsample_bytree",
      "min_child_weight",
      "reg_alpha",
      "reg_lambda",
      "gamma",
      "n_jobs",
      "random_state",
    }
  ),
  kwarg_types={
    "n_estimators": int,
    "max_depth": int,
    "learning_rate": float,
    "subsample": float,
  },
)
register_backend(
  "pytabkit-realmlp",
  _pytabkit_realmlp_factory,
  allowed_kwargs=_PYTABKIT_KWARGS,
  kwarg_types=_PYTABKIT_TYPES,
)
register_backend(
  "pytabkit-tabm",
  _pytabkit_tabm_factory,
  allowed_kwargs=_PYTABKIT_KWARGS,
  kwarg_types=_PYTABKIT_TYPES,
)
# Nori's NoriRegressor kwargs are model-release-specific; left unrestricted.
register_backend("nori", _nori_factory)
register_backend(
  "tabicl",
  _tabicl_factory,
  allowed_kwargs=frozenset({"model_kwargs"}),
  nested_kwargs=frozenset({"model_kwargs"}),
  n_range=(300, 48_000),
)
register_backend(
  "tabicl-finetune",
  _tabicl_finetune_factory,
  allowed_kwargs=frozenset({"epochs", "learning_rate", "early_stopping"}),
  kwarg_types={"epochs": int, "learning_rate": float, "early_stopping": bool},
  n_range=(300, 48_000),
)
