"""Command-line entry point: ``npcc-simstudy --config study.toml --out results/``.

Loads the experiment grid from a TOML file, runs the study, and writes the
result tables (``metrics``, ``runtime``, ``summary``, ``runtime_summary``) plus
an echo of the resolved config to the output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from npcc.experiments.config import GridConfig, RunConfig, load_grid
from npcc.experiments.runner import aggregate_results, run_study

logger = logging.getLogger("npcc.experiments")


def _build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    prog="npcc-simstudy",
    description="Run the npcc TabPFN-Rosenblatt copula simulation study.",
  )
  p.add_argument(
    "--config",
    required=True,
    type=Path,
    help="TOML file with a [grid] table defining the sweep.",
  )
  p.add_argument(
    "--out",
    required=True,
    type=Path,
    help="Output directory for the result tables.",
  )
  p.add_argument(
    "--device",
    default=None,
    help="torch device (default: auto-select cuda if available, else cpu).",
  )
  p.add_argument(
    "--workers",
    type=int,
    default=1,
    help="Number of data cells to run concurrently (threads).",
  )
  p.add_argument("--base-seed", type=int, default=317)
  p.add_argument(
    "--log-level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
  )
  p.add_argument(
    "--format",
    dest="fmt",
    default="csv",
    choices=["csv", "parquet"],
    help="Output table format (parquet needs pyarrow).",
  )
  return p


def _write(df: pd.DataFrame, base: Path, fmt: str) -> None:
  if fmt == "parquet":
    df.to_parquet(base.with_suffix(".parquet"), index=False)
  else:
    df.to_csv(base.with_suffix(".csv"), index=False)


def _config_echo(grid: GridConfig, run: RunConfig, wall: float) -> dict:
  return {
    "grid": {
      "families": grid.families,
      "tau_scenarios": grid.tau_scenarios,
      "transforms": grid.transforms,
      "methods": grid.methods,
      "normalize": grid.normalize,
      "n": grid.n,
      "n_rep": grid.n_rep,
      "projection_grid_size": grid.projection_grid_size,
    },
    "run": {
      "device": run.device,
      "workers": run.workers,
      "base_seed": run.base_seed,
      "fmt": run.fmt,
    },
    "wall_seconds": wall,
  }


def main(argv: list[str] | None = None) -> int:
  args = _build_parser().parse_args(argv)
  logging.basicConfig(
    level=args.log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )

  grid = load_grid(args.config)
  run = RunConfig(
    out=args.out,
    device=args.device,
    workers=args.workers,
    base_seed=args.base_seed,
    log_level=args.log_level,
    fmt=args.fmt,
  )
  run.out.mkdir(parents=True, exist_ok=True)

  metrics_df, runtime_df, wall = run_study(grid, run)
  mc_summary, runtime_summary = aggregate_results(metrics_df, runtime_df)

  _write(metrics_df, run.out / "metrics", run.fmt)
  _write(runtime_df, run.out / "runtime", run.fmt)
  _write(mc_summary, run.out / "summary", run.fmt)
  _write(runtime_summary, run.out / "runtime_summary", run.fmt)
  (run.out / "config.json").write_text(
    json.dumps(_config_echo(grid, run, wall), indent=2)
  )

  logger.info("Wrote %d result tables to %s", 4, run.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
