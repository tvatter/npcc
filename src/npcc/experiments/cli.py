"""Command-line entry point: ``npcc-simstudy --config study.toml --out results/``.

Loads the experiment grid from a TOML file, runs the study, and writes the
raw result tables, paper-facing summaries, and an echo of the resolved
config to the output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

from npcc.experiments.config import GridConfig, RunConfig, load_grid
from npcc.experiments.runner import (
  _write_table,
  aggregate_study_outputs,
  run_study,
)

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
    "--resume",
    action="store_true",
    help="Reuse completed per-cell checkpoints in --out and run only the rest.",
  )
  p.add_argument(
    "--gpu-mem-fraction",
    type=float,
    default=None,
    help=(
      "Cap this process's CUDA memory to a fraction of total VRAM (0-1]. "
      "Leaves headroom for a display server sharing the GPU: a peak that "
      "would starve it raises a clean OOM instead of hanging the machine."
    ),
  )
  p.add_argument(
    "--log-level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
  )
  p.add_argument(
    "--format",
    dest="fmt",
    default="parquet",
    choices=["csv", "parquet"],
    help="Output table format (parquet needs pyarrow).",
  )
  return p


def _config_echo(
  grid: GridConfig, run: RunConfig, wall: float
) -> dict[str, Any]:
  return {
    "grid": {
      "families": grid.families,
      "tau_scenarios": grid.tau_scenarios,
      "estimators": [
        {
          "label": e.label,
          "estimator_id": e.estimator_id,
          **e.canonical_dict(),
        }
        for e in grid.estimators
      ],
      "normalize": grid.normalize,
      "n": grid.n,
      "n_rep": grid.n_rep,
      "projection_grid_size": grid.projection_grid_size,
      "conditional_uv_grid_n": grid.conditional_uv_grid_n,
      "conditional_x_grid_n": grid.conditional_x_grid_n,
      "surface_tau_levels": grid.surface_tau_levels,
      "surface_families": grid.surface_families,
      "enable_tau_diagnostics": grid.enable_tau_diagnostics,
      "tau_diagnostic_n": grid.tau_diagnostic_n,
    },
    "run": {
      "device": run.device,
      "workers": run.workers,
      "base_seed": run.base_seed,
      "fmt": run.fmt,
      "gpu_mem_fraction": run.gpu_mem_fraction,
    },
    "wall_seconds": wall,
  }


def main(argv: list[str] | None = None) -> int:
  # Reduce CUDA fragmentation across the many per-estimator fits (read lazily
  # by the caching allocator on first allocation, so setting it here is early
  # enough); also honors the value if already exported in the shell.
  os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    gpu_mem_fraction=args.gpu_mem_fraction,
  )
  run.out.mkdir(parents=True, exist_ok=True)

  metric_df, quantity_df, diagnostic_df, runtime_df, wall = run_study(
    grid, run, resume=args.resume
  )
  outputs = aggregate_study_outputs(metric_df, diagnostic_df, runtime_df)

  _write_table(outputs["summary_by_x"], run.out / "summary", run.fmt)
  for name, df in outputs.items():
    _write_table(df, run.out / name, run.fmt)
  _write_table(diagnostic_df, run.out / "diagnostics", run.fmt)
  _write_table(quantity_df, run.out / "quantities", run.fmt)
  _write_table(runtime_df, run.out / "runtime", run.fmt)
  (run.out / "config.json").write_text(
    json.dumps(_config_echo(grid, run, wall), indent=2)
  )

  logger.info("Wrote simulation-study result tables to %s", run.out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
