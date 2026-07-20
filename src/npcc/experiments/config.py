"""Experiment configuration: the sweep grid and run-level options.

The grid is declared in a TOML file (loaded with the stdlib :mod:`tomllib`);
run-level options come from the CLI.  TOML has no ``null`` literal, so the
``normalize`` axis accepts the string ``"none"`` (or ``"off"``/``0``) to mean
"no Sinkhorn projection".
"""

from __future__ import annotations

import tomllib
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from npcc.core.conditional_distribution import SupportTransform
from npcc.core.providers import JSONValue, Recovery, TabICLConfig, TabPFNConfig
from npcc.core.quantile_inversion import QuantileInversionConfig
from npcc.experiments.scenarios import FAMILIES, TAU_SCENARIOS

PROVIDERS: tuple[str, ...] = ("tabpfn", "tabicl")


@dataclass(frozen=True)
class EstimatorSpec:
  """An estimator configuration fitted once per data cell."""

  label: str
  provider: str
  recovery: Recovery
  transform: SupportTransform
  provider_config: TabPFNConfig | TabICLConfig
  quantile_inversion: QuantileInversionConfig | None = None

  def __post_init__(self) -> None:
    if not self.label:
      raise ValueError("Estimator labels must be non-empty.")
    if self.provider not in PROVIDERS:
      raise ValueError(f"Unknown provider: {self.provider!r}.")
    if self.provider == "tabpfn" and not isinstance(
      self.provider_config, TabPFNConfig
    ):
      raise ValueError("TabPFN estimators require TabPFNConfig.")
    if self.provider == "tabicl" and not isinstance(
      self.provider_config, TabICLConfig
    ):
      raise ValueError("TabICL estimators require TabICLConfig.")
    if (
      self.provider == "tabicl"
      and self.recovery is not Recovery.QUANTILE_INVERSION
    ):
      raise ValueError("TabICL supports only quantile_inversion recovery.")
    if self.recovery is Recovery.NATIVE_DISTRIBUTION:
      if self.quantile_inversion is not None:
        raise ValueError(
          "quantile_inversion is invalid for native distribution recovery."
        )

  def canonical_dict(self) -> dict[str, object]:
    """Return the stable JSON-compatible estimator definition."""
    config: dict[str, object]
    if isinstance(self.provider_config, TabPFNConfig):
      config = {
        "model_version": self.provider_config.model_version,
        "n_estimators": self.provider_config.n_estimators,
        "ignore_pretraining_limits": (
          self.provider_config.ignore_pretraining_limits
        ),
        "extra_regressor_kwargs": dict(
          self.provider_config.extra_regressor_kwargs
        ),
      }
    else:
      config = {
        "checkpoint": self.provider_config.checkpoint,
        "n_estimators": self.provider_config.n_estimators,
        "ensemble_batch_size": self.provider_config.ensemble_batch_size,
        "cache_mode": self.provider_config.cache_mode,
        "extra_regressor_kwargs": dict(
          self.provider_config.extra_regressor_kwargs
        ),
      }
    inversion = None
    if self.quantile_inversion is not None:
      inversion = {
        "n_quantiles": self.quantile_inversion.n_quantiles,
        "alpha_min": self.quantile_inversion.alpha_min,
        "alpha_max": self.quantile_inversion.alpha_max,
        "min_qprime": self.quantile_inversion.min_qprime,
      }
    return {
      "provider": self.provider,
      "recovery": self.recovery.value,
      "transform": self.transform.value,
      "provider_config": config,
      "quantile_inversion": inversion,
    }

  @property
  def estimator_id(self) -> str:
    """Full SHA-256 of the canonical estimator definition."""
    encoded = json.dumps(
      self.canonical_dict(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()

  @property
  def model_id(self) -> str:
    """Provider-neutral upstream model identifier."""
    if isinstance(self.provider_config, TabPFNConfig):
      return self.provider_config.model_version
    return self.provider_config.checkpoint


@dataclass(frozen=True)
class Cell:
  """One data-generating cell of the sweep (its own sampled data + truth)."""

  family: str
  tau_scenario: str
  n: int
  rep: int


def _check_subset(name: str, values: list[Any], allowed: Iterable[str]) -> None:
  allowed_set = set(allowed)
  if not values:
    raise ValueError(f"{name} must be non-empty.")
  unknown = [v for v in values if v not in allowed_set]
  if unknown:
    raise ValueError(
      f"Unknown {name}: {unknown}. Allowed: {sorted(allowed_set)}."
    )


def _coerce_normalize(raw: list[Any]) -> list[int | None]:
  out: list[int | None] = []
  for x in raw:
    if x is None or (isinstance(x, str) and x.lower() in {"none", "off"}):
      out.append(None)
    elif isinstance(x, bool):  # bool is an int subclass; reject explicitly.
      raise ValueError(f"normalize entries must be int or 'none', got {x!r}.")
    elif isinstance(x, int) and x == 0:
      out.append(None)
    elif isinstance(x, int) and x > 0:
      out.append(x)
    else:
      raise ValueError(
        f"normalize entries must be a positive int or 'none', got {x!r}."
      )
  return out


@dataclass
class GridConfig:
  """The cartesian grid of axes to sweep."""

  families: list[str]
  tau_scenarios: list[str]
  estimators: list[EstimatorSpec]
  normalize: list[int | None]
  n: list[int]
  n_rep: int
  projection_grid_size: int = 30
  conditional_uv_grid_n: int = 20
  conditional_x_grid_n: int = 10
  surface_tau_levels: list[float] = field(
    default_factory=lambda: [0.1, 0.5, 0.9]
  )
  surface_families: list[str] = field(default_factory=lambda: ["clayton"])
  enable_tau_diagnostics: bool = True
  tau_diagnostic_n: int = 1000

  def __post_init__(self) -> None:
    _check_subset("families", self.families, FAMILIES)
    _check_subset("tau_scenarios", self.tau_scenarios, TAU_SCENARIOS)
    if not self.estimators:
      raise ValueError("estimators must be non-empty.")
    labels = [est.label for est in self.estimators]
    if len(labels) != len(set(labels)):
      raise ValueError("Estimator labels must be unique.")
    if not self.normalize:
      raise ValueError("normalize must be non-empty (e.g. [None]).")
    for entry in self.normalize:
      if entry is not None and (
        isinstance(entry, bool) or not isinstance(entry, int) or entry <= 0
      ):
        raise ValueError(
          "normalize entries must be None or a positive int "
          f"(got {entry!r}); the string 'none' is only for TOML input."
        )
    if not self.n or any(v <= 0 for v in self.n):
      raise ValueError("n must be a non-empty list of positive ints.")
    if self.n_rep <= 0:
      raise ValueError("n_rep must be a positive int.")
    if self.projection_grid_size < 2:
      raise ValueError("projection_grid_size must be >= 2.")
    if self.conditional_uv_grid_n < 2:
      raise ValueError("conditional_uv_grid_n must be >= 2.")
    if self.conditional_x_grid_n < 1:
      raise ValueError("conditional_x_grid_n must be >= 1.")
    if not self.surface_tau_levels:
      raise ValueError("surface_tau_levels must be non-empty.")
    for tau in self.surface_tau_levels:
      if tau <= 0.0 or tau >= 1.0:
        raise ValueError("surface_tau_levels entries must be in (0, 1).")
    if self.surface_families:
      _check_subset("surface_families", self.surface_families, FAMILIES)
    if self.tau_diagnostic_n < 10:
      raise ValueError("tau_diagnostic_n must be >= 10.")

  def estimator_specs(self) -> list[EstimatorSpec]:
    return list(self.estimators)

  def cells(self) -> list[Cell]:
    return [
      Cell(family=f, tau_scenario=s, n=n, rep=rep)
      for f in self.families
      for s in self.tau_scenarios
      for n in self.n
      for rep in range(self.n_rep)
    ]


@dataclass
class RunConfig:
  """Run-level options (not part of the swept grid)."""

  out: Path
  device: str | None = None
  workers: int = 1
  base_seed: int = 317
  log_level: str = "INFO"
  fmt: str = "parquet"

  def __post_init__(self) -> None:
    if self.workers < 1:
      raise ValueError("workers must be >= 1.")
    if self.fmt not in {"csv", "parquet"}:
      raise ValueError("fmt must be 'csv' or 'parquet'.")


def load_grid(path: str | Path) -> GridConfig:
  """Load a :class:`GridConfig` from the ``[grid]`` table of a TOML file."""
  with Path(path).open("rb") as fh:
    data = tomllib.load(fh)
  grid = data.get("grid", data)
  raw_estimators = grid.get("estimators", data.get("estimators"))
  if not isinstance(raw_estimators, list):
    raise ValueError("Define at least one [[estimators]] table.")
  try:
    return GridConfig(
      families=list(grid["families"]),
      tau_scenarios=list(grid["tau_scenarios"]),
      estimators=[_parse_estimator(est) for est in raw_estimators],
      normalize=_coerce_normalize(list(grid["normalize"])),
      n=[int(v) for v in grid["n"]],
      n_rep=int(grid["n_rep"]),
      projection_grid_size=int(grid.get("projection_grid_size", 30)),
      conditional_uv_grid_n=int(grid.get("conditional_uv_grid_n", 20)),
      conditional_x_grid_n=int(grid.get("conditional_x_grid_n", 10)),
      surface_tau_levels=[
        float(v) for v in grid.get("surface_tau_levels", [0.1, 0.5, 0.9])
      ],
      surface_families=[
        str(v) for v in grid.get("surface_families", ["clayton"])
      ],
      enable_tau_diagnostics=bool(grid.get("enable_tau_diagnostics", True)),
      tau_diagnostic_n=int(grid.get("tau_diagnostic_n", 1000)),
    )
  except KeyError as exc:
    raise ValueError(f"Missing required [grid] key: {exc}.") from exc


def _parse_estimator(est: dict[str, object]) -> EstimatorSpec:
  provider = str(est["provider"])
  raw_config = est.get("provider_config")
  if not isinstance(raw_config, dict):
    raise ValueError("Each estimator requires a [provider_config] table.")
  try:
    if provider == "tabpfn":
      allowed = {
        "model_version",
        "n_estimators",
        "ignore_pretraining_limits",
        "extra_regressor_kwargs",
      }
      unknown = set(raw_config).difference(allowed)
      if unknown:
        raise ValueError(f"Invalid tabpfn provider_config fields: {unknown}.")
      provider_config: TabPFNConfig | TabICLConfig = TabPFNConfig(
        model_version=str(raw_config.get("model_version", "v3")),
        n_estimators=cast(int | None, raw_config.get("n_estimators")),
        ignore_pretraining_limits=bool(
          raw_config.get("ignore_pretraining_limits", False)
        ),
        extra_regressor_kwargs=cast(
          dict[str, JSONValue], raw_config.get("extra_regressor_kwargs", {})
        ),
      )
    elif provider == "tabicl":
      allowed = {
        "checkpoint",
        "n_estimators",
        "ensemble_batch_size",
        "cache_mode",
        "extra_regressor_kwargs",
      }
      unknown = set(raw_config).difference(allowed)
      if unknown:
        raise ValueError(f"Invalid tabicl provider_config fields: {unknown}.")
      provider_config = TabICLConfig(
        checkpoint=str(
          raw_config.get("checkpoint", "tabicl-regressor-v2-20260212.ckpt")
        ),
        n_estimators=cast(int, raw_config.get("n_estimators", 8)),
        ensemble_batch_size=cast(int, raw_config.get("ensemble_batch_size", 8)),
        cache_mode=str(raw_config.get("cache_mode", "repr")),
        extra_regressor_kwargs=cast(
          dict[str, JSONValue], raw_config.get("extra_regressor_kwargs", {})
        ),
      )
    else:
      raise ValueError(f"Unknown provider: {provider!r}.")
    raw_inversion = est.get("quantile_inversion")
    inversion = None
    if isinstance(raw_inversion, dict):
      inversion = QuantileInversionConfig(
        n_quantiles=cast(int, raw_inversion.get("n_quantiles", 101)),
        alpha_min=cast(float, raw_inversion.get("alpha_min", 1e-3)),
        alpha_max=cast(float, raw_inversion.get("alpha_max", 1.0 - 1e-3)),
        min_qprime=cast(float, raw_inversion.get("min_qprime", 1e-6)),
      )
    return EstimatorSpec(
      label=str(est["label"]),
      provider=provider,
      recovery=Recovery(str(est["recovery"])),
      transform=SupportTransform(str(est["transform"])),
      provider_config=provider_config,
      quantile_inversion=inversion,
    )
  except TypeError as exc:
    raise ValueError(f"Invalid {provider} provider_config: {exc}.") from exc
