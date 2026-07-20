"""Tests for ``npcc.experiments.config`` (grid parsing + validation)."""

from __future__ import annotations

from pathlib import Path

import pytest

from npcc.experiments.config import (
  GridConfig,
  RunConfig,
  load_grid,
)

_TOML = """
[grid]
families = ["clayton", "gumbel"]
tau_scenarios = ["linear", "uncond50"]
transforms = ["logit", "identity"]
backends = ["tabpfn-criterion", "tabpfn-quantiles"]
normalize = ["none", 5]
n = [50, 100]
n_rep = 3
conditional_uv_grid_n = 4
conditional_x_grid_n = 2
surface_tau_levels = [0.1, 0.5]
surface_families = ["clayton"]
enable_tau_diagnostics = false
tau_diagnostic_n = 100
"""


def _write(tmp_path: Path, text: str) -> Path:
  p = tmp_path / "study.toml"
  p.write_text(text)
  return p


def test_load_grid_parses_and_coerces_normalize(tmp_path: Path) -> None:
  grid = load_grid(_write(tmp_path, _TOML))
  assert grid.families == ["clayton", "gumbel"]
  assert grid.normalize == [None, 5]
  assert grid.n_rep == 3
  assert grid.conditional_uv_grid_n == 4
  assert grid.conditional_x_grid_n == 2
  assert grid.surface_tau_levels == [0.1, 0.5]
  assert grid.surface_families == ["clayton"]
  assert grid.enable_tau_diagnostics is False
  assert grid.tau_diagnostic_n == 100


def test_cells_and_estimator_specs_are_cartesian(tmp_path: Path) -> None:
  grid = load_grid(_write(tmp_path, _TOML))
  # 2 families x 2 scenarios x 2 n x 3 rep
  assert len(grid.cells()) == 2 * 2 * 2 * 3
  # 2 transforms x 2 backends
  assert len(grid.estimator_specs()) == 4


def test_backends_multiply_estimator_specs(tmp_path: Path) -> None:
  text = _TOML.replace(
    'backends = ["tabpfn-criterion", "tabpfn-quantiles"]',
    'backends = ["tabpfn-criterion", "tabpfn-quantiles", "ngboost"]',
  )
  grid = load_grid(_write(tmp_path, text))
  assert grid.backends == ["tabpfn-criterion", "tabpfn-quantiles", "ngboost"]
  # 2 transforms x 3 backends
  specs = grid.estimator_specs()
  assert len(specs) == 6
  assert {s.backend for s in specs} == {
    "tabpfn-criterion",
    "tabpfn-quantiles",
    "ngboost",
  }


def test_unknown_backend_rejected(tmp_path: Path) -> None:
  text = _TOML.replace(
    'backends = ["tabpfn-criterion", "tabpfn-quantiles"]',
    'backends = ["tabpfn-criterion", "not-a-backend"]',
  )
  with pytest.raises(ValueError, match="Unknown backends"):
    load_grid(_write(tmp_path, text))


def test_normalize_zero_and_off_become_none(tmp_path: Path) -> None:
  text = _TOML.replace('normalize = ["none", 5]', 'normalize = [0, "off", 5]')
  grid = load_grid(_write(tmp_path, text))
  assert grid.normalize == [None, None, 5]


@pytest.mark.parametrize(
  "field,bad",
  [
    ("families", '["clayton", "nope"]'),
    ("tau_scenarios", '["linear", "nope"]'),
    ("transforms", '["logit", "nope"]'),
    ("backends", '["tabpfn-criterion", "nope"]'),
  ],
)
def test_unknown_axis_value_rejected(
  tmp_path: Path, field: str, bad: str
) -> None:
  text = _TOML
  # replace the relevant line's RHS
  for line in _TOML.splitlines():
    if line.startswith(f"{field} ="):
      text = _TOML.replace(line, f"{field} = {bad}")
  with pytest.raises(ValueError, match="Unknown"):
    load_grid(_write(tmp_path, text))


def test_negative_normalize_rejected(tmp_path: Path) -> None:
  text = _TOML.replace('normalize = ["none", 5]', "normalize = [-1]")
  with pytest.raises(ValueError, match="positive int"):
    load_grid(_write(tmp_path, text))


def test_small_tau_diagnostic_n_rejected(tmp_path: Path) -> None:
  text = _TOML.replace("tau_diagnostic_n = 100", "tau_diagnostic_n = 9")
  with pytest.raises(ValueError, match="tau_diagnostic_n"):
    load_grid(_write(tmp_path, text))


def test_missing_key_raises(tmp_path: Path) -> None:
  text = "\n".join(
    line for line in _TOML.splitlines() if not line.startswith("n_rep")
  )
  with pytest.raises(ValueError, match="Missing required"):
    load_grid(_write(tmp_path, text))


def test_shipped_study_config_loads() -> None:
  """The committed paper config must load and match the study design."""
  cfg_path = Path(__file__).resolve().parents[2] / "configs" / "study.toml"
  grid = load_grid(cfg_path)
  assert grid.families == ["clayton", "gumbel", "frank", "gaussian"]
  assert grid.tau_scenarios == ["linear", "quadratic", "sin"]
  assert grid.n == [200, 500, 1000]
  assert grid.n_rep == 5
  assert grid.transforms == ["logit", "probit"]
  assert grid.backends == [
    "tabpfn-criterion-v2.5",
    "tabpfn-criterion-v3",
    "tabpfn-quantiles-v2.5",
    "tabpfn-quantiles-v3",
    "nori",
    "tabicl",
    "ngboost",
    "xgb-quantile",
    "pytabkit-realmlp",
    "pytabkit-tabm",
  ]
  assert grid.normalize == [None]
  assert len(grid.cells()) == 4 * 3 * 3 * 5
  assert len(grid.estimator_specs()) == 2 * 10


def test_runconfig_validates_workers_and_fmt(tmp_path: Path) -> None:
  assert RunConfig(out=tmp_path).fmt == "parquet"
  with pytest.raises(ValueError, match="workers"):
    RunConfig(out=tmp_path, workers=0)
  with pytest.raises(ValueError, match="fmt"):
    RunConfig(out=tmp_path, fmt="xml")


def test_gridconfig_rejects_empty_axis() -> None:
  with pytest.raises(ValueError, match="non-empty"):
    GridConfig(
      families=[],
      tau_scenarios=["linear"],
      transforms=["logit"],
      backends=["tabpfn-criterion"],
      normalize=[None],
      n=[100],
      n_rep=1,
    )
