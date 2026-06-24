"""Experiment configuration: the sweep grid and run-level options.

The grid is declared in a TOML file (loaded with the stdlib :mod:`tomllib`);
run-level options come from the CLI.  TOML has no ``null`` literal, so the
``normalize`` axis accepts the string ``"none"`` (or ``"off"``/``0``) to mean
"no Sinkhorn projection".
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

from tabpfn.constants import ModelVersion

from npcc.experiments.scenarios import FAMILIES, TAU_SCENARIOS

TRANSFORMS: tuple[str, ...] = ("identity", "logit", "probit")
METHODS: tuple[str, ...] = ("criterion", "quantiles")
MODEL_VERSIONS: tuple[str, ...] = tuple(v.value for v in ModelVersion)
DEFAULT_MODEL_VERSION: str = ModelVersion.V3.value


@dataclass(frozen=True)
class EstimatorSpec:
  """An estimator configuration fitted once per data cell."""

  transform: str
  method: str
  model_version: str = DEFAULT_MODEL_VERSION


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
  transforms: list[str]
  methods: list[str]
  normalize: list[int | None]
  n: list[int]
  n_rep: int
  model_versions: list[str] = field(
    default_factory=lambda: [DEFAULT_MODEL_VERSION]
  )
  projection_grid_size: int = 30

  def __post_init__(self) -> None:
    _check_subset("families", self.families, FAMILIES)
    _check_subset("tau_scenarios", self.tau_scenarios, TAU_SCENARIOS)
    _check_subset("transforms", self.transforms, TRANSFORMS)
    _check_subset("methods", self.methods, METHODS)
    _check_subset("model_versions", self.model_versions, MODEL_VERSIONS)
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

  def estimator_specs(self) -> list[EstimatorSpec]:
    return [
      EstimatorSpec(transform=t, method=m, model_version=mv)
      for t, m, mv in product(
        self.transforms, self.methods, self.model_versions
      )
    ]

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
  fmt: str = "csv"

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
  try:
    return GridConfig(
      families=list(grid["families"]),
      tau_scenarios=list(grid["tau_scenarios"]),
      transforms=list(grid["transforms"]),
      methods=list(grid["methods"]),
      normalize=_coerce_normalize(list(grid["normalize"])),
      n=[int(v) for v in grid["n"]],
      n_rep=int(grid["n_rep"]),
      model_versions=[
        str(v) for v in grid.get("model_versions", [DEFAULT_MODEL_VERSION])
      ],
      projection_grid_size=int(grid.get("projection_grid_size", 30)),
    )
  except KeyError as exc:
    raise ValueError(f"Missing required [grid] key: {exc}.") from exc
