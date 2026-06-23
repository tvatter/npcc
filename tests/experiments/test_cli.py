"""CLI tests: argparse + end-to-end run writing artifacts (hermetic)."""

from __future__ import annotations

from pathlib import Path

from npcc.experiments import cli

_TOML = """
[grid]
families = ["clayton"]
tau_scenarios = ["linear"]
transforms = ["logit"]
methods = ["criterion"]
normalize = ["none"]
n = [30]
n_rep = 1
"""


def test_cli_end_to_end_writes_tables(
  patch_uniform: None, tmp_path: Path
) -> None:
  config = tmp_path / "study.toml"
  config.write_text(_TOML)
  out = tmp_path / "results"

  rc = cli.main(["--config", str(config), "--out", str(out), "--workers", "1"])
  assert rc == 0

  for name in ("metrics", "runtime", "summary", "runtime_summary"):
    path = out / f"{name}.csv"
    assert path.exists(), f"missing {name}.csv"
    assert path.stat().st_size > 0
  assert (out / "config.json").exists()


def test_cli_parser_requires_config_and_out() -> None:
  parser = cli._build_parser()
  ns = parser.parse_args(["--config", "g.toml", "--out", "o", "--workers", "4"])
  assert ns.fmt == "csv"
  assert ns.workers == 4
  assert Path(ns.config).name == "g.toml"
