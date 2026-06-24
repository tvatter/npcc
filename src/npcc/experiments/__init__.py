"""Simulation-study tooling for npcc.

Requires the ``experiments`` extra (``pandas``, ``matplotlib``). The core
package (``import npcc``) does not import this subpackage, so it stays light.

Typical use::

    from npcc.experiments import GridConfig, RunConfig, run_study
    metrics, runtime, wall = run_study(grid, run)

or via the CLI: ``npcc-simstudy --config study.toml --out results/``.
"""

from __future__ import annotations

from npcc.experiments.config import GridConfig, RunConfig, load_grid
from npcc.experiments.runner import aggregate_results, run_study

__all__ = [
  "GridConfig",
  "RunConfig",
  "aggregate_results",
  "load_grid",
  "run_study",
]
