"""Run the simulation study: sweep the grid, score against truth, aggregate.

Each *data cell* ``(family, tau_scenario, n, rep)`` samples its own training set
and ground truth once; the estimator configs ``(transform, method)`` are fitted
on that shared data, and each fitted model is evaluated under every
``normalize`` (Sinkhorn) variant.  The ``normalize`` axis only affects the
density, so ``cdf`` / ``hfunc1`` / ``hfunc2`` are scored once per estimator
(labelled ``normalize="none"``) while ``pdf`` is scored per variant.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import Literal, cast

import numpy as np
import pandas as pd
import torch

from tabpfn.constants import ModelVersion

from npcc.core.pfnr_bicop import PFNRBicop
from npcc.experiments import metrics, scenarios
from npcc.experiments.config import Cell, EstimatorSpec, GridConfig, RunConfig
from npcc.experiments.scenarios import UV_PAIRS, EvalGrid

logger = logging.getLogger(__name__)


def _norm_label(norm: int | None) -> str:
  return "none" if norm is None else str(norm)


def _cell_seed(base_seed: int, cell: Cell) -> int:
  """Deterministic, axis-decorrelated seed for a data cell."""
  fam_idx = list(scenarios.FAMILIES).index(cell.family)
  scn_idx = list(scenarios.TAU_SCENARIOS).index(cell.tau_scenario)
  seq = np.random.SeedSequence([base_seed, fam_idx, scn_idx, cell.n, cell.rep])
  return int(seq.generate_state(1)[0])


def _rows_for_quantity(
  cell: Cell,
  est: EstimatorSpec,
  seed: int,
  quantity: str,
  truth: np.ndarray,
  pred: np.ndarray,
  grid: EvalGrid,
  *,
  normalize: str,
  include_kl: bool,
) -> list[dict]:
  """Metric rows for one quantity: per-(u,v) curve metrics (conditional) or a
  single grid-metric row (unconditional)."""
  base = {
    "family": cell.family,
    "tau_scenario": cell.tau_scenario,
    "n": cell.n,
    "rep": cell.rep,
    "seed": seed,
    "transform": est.transform,
    "method": est.method,
    "model_version": est.model_version,
    "normalize": normalize,
    "quantity": quantity,
  }
  if grid.conditional:
    assert grid.x_axis is not None
    pred_grid = pred.reshape(grid.shape)
    rows = []
    for pair_idx, (u_val, v_val) in enumerate(UV_PAIRS):
      stats = metrics.curve_metrics(
        truth[pair_idx], pred_grid[pair_idx], grid.x_axis, include_kl=include_kl
      )
      rows.append({**base, "u": float(u_val), "v": float(v_val), **stats})
    return rows

  stats = metrics.grid_metrics(truth, pred, include_kl=include_kl)
  return [{**base, "u": np.nan, "v": np.nan, **stats}]


def summarize_one_cell(
  cell: Cell,
  estimator_specs: list[EstimatorSpec],
  normalize: list[int | None],
  *,
  base_seed: int,
  device: str | None,
  projection_grid_size: int = 101,
) -> tuple[list[dict], list[dict]]:
  """Fit every estimator on one cell's data and score against truth."""
  seed = _cell_seed(base_seed, cell)

  u, v, x = scenarios.sample(cell.family, cell.tau_scenario, cell.n, seed)
  truth = scenarios.ground_truth(cell.family, cell.tau_scenario)
  grid = scenarios.eval_grid(cell.tau_scenario)

  metric_rows: list[dict] = []
  runtime_rows: list[dict] = []

  for est in estimator_specs:
    t0 = perf_counter()
    model = PFNRBicop(
      method=cast(Literal["criterion", "quantiles"], est.method),
      transform=cast(Literal["identity", "logit", "probit"], est.transform),
      device=device,
      projection_grid_size=projection_grid_size,
      model_version=ModelVersion(est.model_version),
    )
    model.fit(u, v, x)
    fit_time = perf_counter() - t0

    with torch.inference_mode():
      # Normalization does not affect cdf / hfunc — score them once.
      timings: dict[str, float] = {}
      single_preds: dict[str, np.ndarray] = {}
      for q, fn in (
        ("cdf", lambda: model.cdf(grid.u_flat, grid.v_flat, x=grid.x_flat)),
        (
          "hfunc1",
          lambda: model.hfunc1(grid.u_flat, grid.v_flat, x=grid.x_flat),
        ),
        (
          "hfunc2",
          lambda: model.hfunc2(grid.u_flat, grid.v_flat, x=grid.x_flat),
        ),
      ):
        t0 = perf_counter()
        single_preds[q] = np.asarray(fn(), dtype=np.float64)
        timings[q] = perf_counter() - t0

      for q in ("cdf", "hfunc1", "hfunc2"):
        metric_rows += _rows_for_quantity(
          cell,
          est,
          seed,
          q,
          truth[q],
          single_preds[q],
          grid,
          normalize="none",
          include_kl=False,
        )

      # pdf: one evaluation per normalization variant.
      pdf_time = 0.0
      for norm in normalize:
        t0 = perf_counter()
        pdf_hat = np.asarray(
          model.pdf(
            grid.u_flat, grid.v_flat, x=grid.x_flat, sinkhorn_iters=norm
          ),
          dtype=np.float64,
        )
        pdf_time += perf_counter() - t0
        metric_rows += _rows_for_quantity(
          cell,
          est,
          seed,
          "pdf",
          truth["pdf"],
          pdf_hat,
          grid,
          normalize=_norm_label(norm),
          include_kl=True,
        )

    runtime_rows.append(
      {
        "family": cell.family,
        "tau_scenario": cell.tau_scenario,
        "n": cell.n,
        "rep": cell.rep,
        "seed": seed,
        "transform": est.transform,
        "method": est.method,
        "model_version": est.model_version,
        "fit_time": fit_time,
        "pdf_time": pdf_time,
        "cdf_time": timings["cdf"],
        "h1_time": timings["hfunc1"],
        "h2_time": timings["hfunc2"],
        "total_estimator_time": fit_time + pdf_time + sum(timings.values()),
      }
    )

  return metric_rows, runtime_rows


def run_study(
  grid: GridConfig, run: RunConfig
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
  """Sweep the full grid; return (metrics, runtime) DataFrames and wall time."""
  cells = grid.cells()
  specs = grid.estimator_specs()
  logger.info(
    "Study: %d cells x %d estimators x %d normalize variants",
    len(cells),
    len(specs),
    len(grid.normalize),
  )

  metric_rows: list[dict] = []
  runtime_rows: list[dict] = []
  t0_wall = perf_counter()

  def _do(cell: Cell) -> tuple[list[dict], list[dict]]:
    logger.debug("cell start: %s", cell)
    out = summarize_one_cell(
      cell,
      specs,
      grid.normalize,
      base_seed=run.base_seed,
      device=run.device,
      projection_grid_size=grid.projection_grid_size,
    )
    logger.info("cell done: %s", cell)
    return out

  if run.workers <= 1:
    for cell in cells:
      m, r = _do(cell)
      metric_rows += m
      runtime_rows += r
  else:
    max_workers = min(run.workers, len(cells), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
      futures = [pool.submit(_do, cell) for cell in cells]
      for fut in as_completed(futures):
        m, r = fut.result()
        metric_rows += m
        runtime_rows += r

  wall = perf_counter() - t0_wall
  logger.info("Study finished in %.1fs", wall)

  sort_axes = [
    "family",
    "tau_scenario",
    "method",
    "transform",
    "model_version",
    "n",
    "rep",
  ]
  metrics_df = (
    pd.DataFrame(metric_rows)
    .sort_values([*sort_axes, "normalize", "quantity", "u", "v"])
    .reset_index(drop=True)
  )
  runtime_df = (
    pd.DataFrame(runtime_rows).sort_values(sort_axes).reset_index(drop=True)
  )
  return metrics_df, runtime_df, wall


def aggregate_results(
  metrics_df: pd.DataFrame, runtime_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Aggregate over repetitions, grouped by every axis.

  Returns a long Monte-Carlo summary (one row per axis x quantity x metric)
  and a runtime summary (one row per axis combination).
  """
  group = [
    "family",
    "tau_scenario",
    "n",
    "transform",
    "method",
    "model_version",
    "normalize",
    "quantity",
  ]
  chunks = []
  for metric in ("IAE", "ISE", "KL"):
    g = metrics_df.groupby(group, as_index=False, dropna=False).agg(
      rep_mean=(metric, "mean"),
      rep_std=(metric, "std"),
      rep_median=(metric, "median"),
      rep_p05=(metric, lambda s: float(np.nanquantile(s, 0.05))),
      rep_p95=(metric, lambda s: float(np.nanquantile(s, 0.95))),
      rep_max=(metric, "max"),
    )
    g["metric"] = metric
    chunks.append(g)
  mc_summary = pd.concat(chunks, ignore_index=True)

  rt_group = [
    "family",
    "tau_scenario",
    "n",
    "transform",
    "method",
    "model_version",
  ]
  runtime_summary = runtime_df.groupby(rt_group, as_index=False).agg(
    fit_time_mean=("fit_time", "mean"),
    fit_time_std=("fit_time", "std"),
    pdf_time_mean=("pdf_time", "mean"),
    cdf_time_mean=("cdf_time", "mean"),
    h1_time_mean=("h1_time", "mean"),
    h2_time_mean=("h2_time", "mean"),
    total_estimator_time_mean=("total_estimator_time", "mean"),
  )
  return mc_summary, runtime_summary
