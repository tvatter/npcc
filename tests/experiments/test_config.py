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
methods = ["criterion", "quantiles"]
normalize = ["none", 5]
n = [50, 100]
n_rep = 3
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


def test_cells_and_estimator_specs_are_cartesian(tmp_path: Path) -> None:
  grid = load_grid(_write(tmp_path, _TOML))
  # 2 families x 2 scenarios x 2 n x 3 rep
  assert len(grid.cells()) == 2 * 2 * 2 * 3
  # 2 transforms x 2 methods x 1 (default) model version
  assert len(grid.estimator_specs()) == 4


def test_model_versions_default_to_v3(tmp_path: Path) -> None:
  grid = load_grid(_write(tmp_path, _TOML))
  assert grid.model_versions == ["v3"]
  assert all(s.model_version == "v3" for s in grid.estimator_specs())


def test_model_versions_multiply_estimator_specs(tmp_path: Path) -> None:
  text = _TOML + '\nmodel_versions = ["v2.5", "v3"]\n'
  grid = load_grid(_write(tmp_path, text))
  assert grid.model_versions == ["v2.5", "v3"]
  # 2 transforms x 2 methods x 2 model versions
  specs = grid.estimator_specs()
  assert len(specs) == 8
  assert {s.model_version for s in specs} == {"v2.5", "v3"}


def test_unknown_model_version_rejected(tmp_path: Path) -> None:
  text = _TOML + '\nmodel_versions = ["v2.5", "v99"]\n'
  with pytest.raises(ValueError, match="Unknown model_versions"):
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
    ("methods", '["criterion", "nope"]'),
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


def test_missing_key_raises(tmp_path: Path) -> None:
  text = "\n".join(
    line for line in _TOML.splitlines() if not line.startswith("n_rep")
  )
  with pytest.raises(ValueError, match="Missing required"):
    load_grid(_write(tmp_path, text))


def test_shipped_study_config_loads() -> None:
  """The committed example config must load and produce a non-empty grid."""
  cfg_path = Path(__file__).resolve().parents[2] / "configs" / "study.toml"
  grid = load_grid(cfg_path)
  assert grid.cells()
  assert grid.estimator_specs()


def test_runconfig_validates_workers_and_fmt(tmp_path: Path) -> None:
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
      methods=["criterion"],
      normalize=[None],
      n=[100],
      n_rep=1,
    )
