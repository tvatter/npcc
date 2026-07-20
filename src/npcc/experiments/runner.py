"""Run the simulation study: sweep the grid, score, diagnose, aggregate.

Each data cell ``(family, tau_scenario, n, rep)`` samples its own training set
and ground truth once.  Estimator configs are fitted on that shared data.

Conditional scenarios are scored at fixed ``x`` values: errors are averaged over
the uv grid first, and only then summarized over Monte-Carlo repetitions or
conditioning values. This keeps KL as a fixed-``x`` copula-density metric.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from typing import Literal, cast

from npcc.core.foundation_model_bicop import FoundationModelBicop
from npcc.experiments import metrics, scenarios
from npcc.experiments.config import Cell, EstimatorSpec, GridConfig, RunConfig
from npcc.experiments.scenarios import EvalGrid

logger = logging.getLogger(__name__)

_EPS: float = 1e-12

# Avoid pandas/pyarrow Arrow-string inference crashes seen on Windows while
# building and sorting the large mixed string/numeric experiment tables.
pd.options.future.infer_string = False


def _norm_label(norm: int | None) -> str:
  return "none" if norm is None else str(norm)


def _data_frame(rows: list[dict]) -> pd.DataFrame:
  """Build a DataFrame without pandas' optional Arrow-backed string inference."""
  with pd.option_context("future.infer_string", False):
    return pd.DataFrame(rows)


def _nan_quantile(s: pd.Series, q: float) -> float:
  values = s.to_numpy(dtype=np.float64)
  if np.isnan(values).all():
    return np.nan
  return float(np.nanquantile(values, q))


def _cell_seed(base_seed: int, cell: Cell) -> int:
  """Deterministic, axis-decorrelated seed for a data cell."""
  fam_idx = list(scenarios.FAMILIES).index(cell.family)
  scn_idx = list(scenarios.TAU_SCENARIOS).index(cell.tau_scenario)
  seq = np.random.SeedSequence([base_seed, fam_idx, scn_idx, cell.n, cell.rep])
  return int(seq.generate_state(1)[0])


def _base_row(cell: Cell, est: EstimatorSpec, seed: int) -> dict[str, object]:
  return {
    "family": cell.family,
    "tau_scenario": cell.tau_scenario,
    "n": cell.n,
    "rep": cell.rep,
    "seed": seed,
    "estimator_id": est.estimator_id,
    "label": est.label,
    "provider": est.provider,
    "transform": est.transform.value,
    "recovery": est.recovery.value,
    "model_id": est.model_id,
  }


def _tau_values(scenario: str, x_axis: np.ndarray | None) -> np.ndarray:
  spec = scenarios.TAU_SCENARIOS[scenario]
  if spec.conditional:
    assert spec.tau_of_x is not None
    assert x_axis is not None
    return spec.tau_of_x(x_axis)
  assert spec.tau is not None
  return np.array([spec.tau], dtype=np.float64)


def _metric_rows_for_quantity(
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
) -> list[dict[str, object]]:
  """Metric rows after averaging pointwise errors over uv at fixed x."""
  base = _base_row(cell, est, seed)
  base.update({"normalize": normalize, "quantity": quantity})

  if grid.conditional:
    assert grid.x_axis is not None
    tau_x = _tau_values(cell.tau_scenario, grid.x_axis)
    truth_grid = truth.reshape(grid.shape)
    pred_grid = pred.reshape(grid.shape)
    rows: list[dict[str, object]] = []
    for x_idx, x_val in enumerate(grid.x_axis):
      y_true = truth_grid[:, x_idx]
      y_hat = pred_grid[:, x_idx]
      err = y_hat - y_true
      kl = np.nan
      if include_kl:
        y_true_pos = np.clip(y_true, _EPS, None)
        y_hat_pos = np.clip(y_hat, _EPS, None)
        kl = float(np.mean(y_true_pos * np.log(y_true_pos / y_hat_pos)))
      rows.append(
        {
          **base,
          "x": float(x_val),
          "tau_true": float(tau_x[x_idx]),
          "IAE": float(np.mean(np.abs(err))),
          "ISE": float(np.mean(err**2)),
          "KL": kl,
        }
      )
    return rows

  stats = metrics.grid_metrics(truth, pred, include_kl=include_kl)
  tau = _tau_values(cell.tau_scenario, None)[0]
  return [{**base, "x": np.nan, "tau_true": float(tau), **stats}]


def _surface_x_rows(scenario: str, tau_levels: list[float]) -> list[dict]:
  """Map target Kendall-tau levels to deterministic x slice(s)."""
  spec = scenarios.TAU_SCENARIOS[scenario]
  if not spec.conditional:
    assert spec.tau is not None
    return [
      {"target_tau": float(spec.tau), "x": np.nan, "tau_true": float(spec.tau)}
    ]

  assert spec.tau_of_x is not None
  x_dense = np.linspace(scenarios.X_MIN, scenarios.X_MAX, 2001)
  tau_dense = spec.tau_of_x(x_dense)
  rows: list[dict] = []
  for target in tau_levels:
    roots: list[float] = []
    delta = tau_dense - target
    exact = np.flatnonzero(np.isclose(delta, 0.0, atol=1e-6))
    roots.extend(float(x_dense[i]) for i in exact)
    for i in range(x_dense.shape[0] - 1):
      if delta[i] == 0.0 or delta[i] * delta[i + 1] > 0.0:
        continue
      lo = float(x_dense[i])
      hi = float(x_dense[i + 1])
      for _ in range(40):
        mid = (lo + hi) / 2.0
        if (float(spec.tau_of_x(np.array([lo]))[0]) - target) * (
          float(spec.tau_of_x(np.array([mid]))[0]) - target
        ) <= 0.0:
          hi = mid
        else:
          lo = mid
      roots.append((lo + hi) / 2.0)
    if not roots:
      roots.append(float(x_dense[int(np.argmin(np.abs(delta)))]))

    unique_roots: list[float] = []
    for root in sorted(roots):
      if not unique_roots or abs(root - unique_roots[-1]) > 1e-4:
        unique_roots.append(root)
    for root in unique_roots:
      tau_true = float(spec.tau_of_x(np.array([root]))[0])
      rows.append(
        {"target_tau": float(target), "x": root, "tau_true": tau_true}
      )
  return rows


def _quantity_rows(
  cell: Cell,
  est: EstimatorSpec,
  seed: int,
  quantity: str,
  truth: np.ndarray,
  pred: np.ndarray,
  grid: EvalGrid,
  *,
  normalize: str,
  target_tau: np.ndarray,
  tau_true: np.ndarray,
) -> list[dict[str, object]]:
  base = _base_row(cell, est, seed)
  truth_grid = truth.reshape(grid.shape)
  pred_grid = pred.reshape(grid.shape)
  rows: list[dict[str, object]] = []
  assert grid.x_axis is not None
  for pair_idx in range(grid.shape[0]):
    u_val = float(grid.u_flat[pair_idx * grid.x_axis.shape[0]])
    v_val = float(grid.v_flat[pair_idx * grid.x_axis.shape[0]])
    for x_idx, x_val in enumerate(grid.x_axis):
      rows.append(
        {
          **base,
          "quantity": quantity,
          "normalize": normalize,
          "target_tau": float(target_tau[x_idx]),
          "x": float(x_val),
          "tau_true": float(tau_true[x_idx]),
          "u": u_val,
          "v": v_val,
          "truth": float(truth_grid[pair_idx, x_idx]),
          "pred": float(pred_grid[pair_idx, x_idx]),
          "error": float(
            pred_grid[pair_idx, x_idx] - truth_grid[pair_idx, x_idx]
          ),
        }
      )
  return rows


def _diagnostic_rows(
  cell: Cell,
  est: EstimatorSpec,
  seed: int,
  model: FoundationModelBicop,
  pdf_by_norm: dict[str, np.ndarray],
  grid: EvalGrid,
  *,
  enable_tau_diagnostics: bool,
  tau_diagnostic_n: int,
) -> tuple[list[dict[str, object]], float]:
  """Marginal-density and dependence diagnostics at fixed x."""
  base = _base_row(cell, est, seed)
  rows: list[dict[str, object]] = []
  tau_time = 0.0
  if not grid.conditional:
    for norm, pdf_hat in pdf_by_norm.items():
      c = pdf_hat.reshape((grid.u_axis.shape[0], grid.v_axis.shape[0]))
      diag = metrics.marginal_diagnostics(c, grid.u_axis, grid.v_axis)
      tau_true = float(_tau_values(cell.tau_scenario, None)[0])
      tau_hat = np.nan
      if enable_tau_diagnostics:
        t0 = perf_counter()
        tau_hat = model.tau(n=tau_diagnostic_n)
        tau_time += perf_counter() - t0
      rows.append(
        {
          **base,
          "normalize": norm,
          "x": np.nan,
          "tau_true": tau_true,
          "tau_hat": tau_hat,
          "tau_abs_err": abs(tau_hat - tau_true),
          **diag,
        }
      )
    return rows, tau_time

  assert grid.x_axis is not None
  tau_x = _tau_values(cell.tau_scenario, grid.x_axis)
  tau_hat_by_x: dict[int, float] = {}
  if enable_tau_diagnostics:
    for idx, x_val in enumerate(grid.x_axis):
      t0 = perf_counter()
      tau_hat_by_x[idx] = model.tau(
        x_row=np.array([[float(x_val)]], dtype=np.float64),
        n=tau_diagnostic_n,
      )
      tau_time += perf_counter() - t0
  for norm, pdf_hat in pdf_by_norm.items():
    pdf_grid = pdf_hat.reshape(grid.shape)
    for x_idx, x_val in enumerate(grid.x_axis):
      c = pdf_grid[:, x_idx].reshape(
        (grid.u_axis.shape[0], grid.v_axis.shape[0])
      )
      diag = metrics.marginal_diagnostics(c, grid.u_axis, grid.v_axis)
      tau_hat = tau_hat_by_x[x_idx] if enable_tau_diagnostics else np.nan
      tau_true = float(tau_x[x_idx])
      rows.append(
        {
          **base,
          "normalize": norm,
          "x": float(x_val),
          "tau_true": tau_true,
          "tau_hat": tau_hat,
          "tau_abs_err": abs(tau_hat - tau_true),
          **diag,
        }
      )
  return rows, tau_time


def summarize_one_cell(
  cell: Cell,
  estimator_specs: list[EstimatorSpec],
  normalize: list[int | None],
  *,
  base_seed: int,
  device: str | None,
  projection_grid_size: int,
  conditional_uv_grid_n: int,
  conditional_x_grid_n: int,
  surface_tau_levels: list[float],
  surface_families: list[str],
  enable_tau_diagnostics: bool,
  tau_diagnostic_n: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
  """Fit every estimator on one cell's data and return all output rows."""
  seed = _cell_seed(base_seed, cell)

  u, v, x = scenarios.sample(cell.family, cell.tau_scenario, cell.n, seed)
  metric_grid = scenarios.eval_grid(
    cell.tau_scenario,
    conditional_uv_grid_n=conditional_uv_grid_n,
    conditional_x_grid_n=conditional_x_grid_n,
  )
  truth = scenarios.ground_truth(cell.family, cell.tau_scenario, metric_grid)

  surface_rows = _surface_x_rows(cell.tau_scenario, surface_tau_levels)
  surface_grid: EvalGrid | None = None
  surface_truth: dict[str, np.ndarray] | None = None
  surface_target_tau: np.ndarray | None = None
  surface_tau_true: np.ndarray | None = None
  if cell.family in surface_families and metric_grid.conditional:
    surface_x = np.array([row["x"] for row in surface_rows], dtype=np.float64)
    surface_grid = scenarios.eval_grid_for_x(
      cell.tau_scenario, surface_x, conditional_uv_grid_n=conditional_uv_grid_n
    )
    surface_truth = scenarios.ground_truth(
      cell.family, cell.tau_scenario, surface_grid
    )
    surface_target_tau = np.array(
      [row["target_tau"] for row in surface_rows], dtype=np.float64
    )
    surface_tau_true = np.array(
      [row["tau_true"] for row in surface_rows], dtype=np.float64
    )

  metric_rows: list[dict] = []
  quantity_rows: list[dict] = []
  diagnostic_rows: list[dict] = []
  runtime_rows: list[dict] = []

  for est in estimator_specs:
    t0 = perf_counter()
    estimator_seed = seed ^ int(est.estimator_id[:8], 16)
    model = FoundationModelBicop(
      provider=cast(Literal["tabpfn", "tabicl"], est.provider),
      recovery=est.recovery,
      provider_config=est.provider_config,
      quantile_inversion=est.quantile_inversion,
      transform=est.transform,
      device=device,
      projection_grid_size=projection_grid_size,
      random_state=estimator_seed,
    )
    model.fit(u, v, x)
    fit_time = perf_counter() - t0

    with torch.inference_mode():
      timings: dict[str, float] = {}
      single_preds: dict[str, np.ndarray] = {}
      for q, fn in (
        (
          "cdf",
          lambda: model.cdf(
            metric_grid.u_flat, metric_grid.v_flat, x=metric_grid.x_flat
          ),
        ),
        (
          "hfunc1",
          lambda: model.hfunc1(
            metric_grid.u_flat, metric_grid.v_flat, x=metric_grid.x_flat
          ),
        ),
        (
          "hfunc2",
          lambda: model.hfunc2(
            metric_grid.u_flat, metric_grid.v_flat, x=metric_grid.x_flat
          ),
        ),
      ):
        t0 = perf_counter()
        single_preds[q] = np.asarray(fn(), dtype=np.float64)
        timings[q] = perf_counter() - t0

      for q in ("cdf", "hfunc1", "hfunc2"):
        metric_rows += _metric_rows_for_quantity(
          cell,
          est,
          seed,
          q,
          truth[q],
          single_preds[q],
          metric_grid,
          normalize="none",
          include_kl=False,
        )

      pdf_time = 0.0
      pdf_by_norm: dict[str, np.ndarray] = {}
      for norm in normalize:
        norm_label = _norm_label(norm)
        t0 = perf_counter()
        pdf_hat = np.asarray(
          model.pdf(
            metric_grid.u_flat,
            metric_grid.v_flat,
            x=metric_grid.x_flat,
            sinkhorn_iters=norm,
          ),
          dtype=np.float64,
        )
        pdf_time += perf_counter() - t0
        pdf_by_norm[norm_label] = pdf_hat
        metric_rows += _metric_rows_for_quantity(
          cell,
          est,
          seed,
          "pdf",
          truth["pdf"],
          pdf_hat,
          metric_grid,
          normalize=norm_label,
          include_kl=True,
        )

      diag_rows, tau_time = _diagnostic_rows(
        cell,
        est,
        seed,
        model,
        pdf_by_norm,
        metric_grid,
        enable_tau_diagnostics=enable_tau_diagnostics,
        tau_diagnostic_n=tau_diagnostic_n,
      )
      diagnostic_rows += diag_rows

      surface_time = 0.0
      if (
        surface_grid is not None
        and surface_truth is not None
        and surface_target_tau is not None
        and surface_tau_true is not None
      ):
        surface_preds: dict[str, np.ndarray] = {}
        for q, fn in (
          (
            "cdf",
            lambda: model.cdf(
              surface_grid.u_flat, surface_grid.v_flat, x=surface_grid.x_flat
            ),
          ),
          (
            "hfunc1",
            lambda: model.hfunc1(
              surface_grid.u_flat, surface_grid.v_flat, x=surface_grid.x_flat
            ),
          ),
          (
            "hfunc2",
            lambda: model.hfunc2(
              surface_grid.u_flat, surface_grid.v_flat, x=surface_grid.x_flat
            ),
          ),
        ):
          t0 = perf_counter()
          surface_preds[q] = np.asarray(fn(), dtype=np.float64)
          surface_time += perf_counter() - t0
          quantity_rows += _quantity_rows(
            cell,
            est,
            seed,
            q,
            surface_truth[q],
            surface_preds[q],
            surface_grid,
            normalize="none",
            target_tau=surface_target_tau,
            tau_true=surface_tau_true,
          )

        for norm in normalize:
          norm_label = _norm_label(norm)
          t0 = perf_counter()
          pdf_hat = np.asarray(
            model.pdf(
              surface_grid.u_flat,
              surface_grid.v_flat,
              x=surface_grid.x_flat,
              sinkhorn_iters=norm,
            ),
            dtype=np.float64,
          )
          surface_time += perf_counter() - t0
          quantity_rows += _quantity_rows(
            cell,
            est,
            seed,
            "pdf",
            surface_truth["pdf"],
            pdf_hat,
            surface_grid,
            normalize=norm_label,
            target_tau=surface_target_tau,
            tau_true=surface_tau_true,
          )

    runtime_rows.append(
      {
        "family": cell.family,
        "tau_scenario": cell.tau_scenario,
        "n": cell.n,
        "rep": cell.rep,
        "seed": seed,
        "estimator_id": est.estimator_id,
        "label": est.label,
        "provider": est.provider,
        "transform": est.transform.value,
        "recovery": est.recovery.value,
        "model_id": model.provider.model_id,
        "fit_time": fit_time,
        "pdf_time": pdf_time,
        "cdf_time": timings["cdf"],
        "h1_time": timings["hfunc1"],
        "h2_time": timings["hfunc2"],
        "tau_time": tau_time,
        "surface_time": surface_time,
        "total_estimator_time": (
          fit_time + pdf_time + tau_time + surface_time + sum(timings.values())
        ),
      }
    )

  return metric_rows, quantity_rows, diagnostic_rows, runtime_rows


def run_study(
  grid: GridConfig, run: RunConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
  """Sweep the full grid; return metric, quantity, diagnostic, runtime tables."""
  cells = grid.cells()
  specs = grid.estimator_specs()
  logger.info(
    "Study: %d cells x %d estimators x %d normalize variants",
    len(cells),
    len(specs),
    len(grid.normalize),
  )

  metric_rows: list[dict] = []
  quantity_rows: list[dict] = []
  diagnostic_rows: list[dict] = []
  runtime_rows: list[dict] = []
  t0_wall = perf_counter()

  def _do(cell: Cell) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    logger.debug("cell start: %s", cell)
    out = summarize_one_cell(
      cell,
      specs,
      grid.normalize,
      base_seed=run.base_seed,
      device=run.device,
      projection_grid_size=grid.projection_grid_size,
      conditional_uv_grid_n=grid.conditional_uv_grid_n,
      conditional_x_grid_n=grid.conditional_x_grid_n,
      surface_tau_levels=grid.surface_tau_levels,
      surface_families=grid.surface_families,
      enable_tau_diagnostics=grid.enable_tau_diagnostics,
      tau_diagnostic_n=grid.tau_diagnostic_n,
    )
    logger.info("cell done: %s", cell)
    return out

  if run.workers <= 1:
    for cell in cells:
      m, q, d, r = _do(cell)
      metric_rows += m
      quantity_rows += q
      diagnostic_rows += d
      runtime_rows += r
  else:
    max_workers = min(run.workers, len(cells), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
      futures = [pool.submit(_do, cell) for cell in cells]
      for fut in as_completed(futures):
        m, q, d, r = fut.result()
        metric_rows += m
        quantity_rows += q
        diagnostic_rows += d
        runtime_rows += r

  wall = perf_counter() - t0_wall
  logger.info("Study finished in %.1fs", wall)

  sort_axes = [
    "family",
    "tau_scenario",
    "estimator_id",
    "label",
    "provider",
    "recovery",
    "transform",
    "model_id",
    "n",
    "rep",
  ]
  metric_df = (
    _data_frame(metric_rows)
    .sort_values([*sort_axes, "normalize", "quantity", "x"])
    .reset_index(drop=True)
  )
  quantity_df = _data_frame(quantity_rows)
  if not quantity_df.empty:
    quantity_df = quantity_df.sort_values(
      [*sort_axes, "normalize", "quantity", "target_tau", "x", "u", "v"]
    ).reset_index(drop=True)
  diagnostic_df = (
    _data_frame(diagnostic_rows)
    .sort_values([*sort_axes, "normalize", "x"])
    .reset_index(drop=True)
  )
  runtime_df = (
    _data_frame(runtime_rows).sort_values(sort_axes).reset_index(drop=True)
  )
  return metric_df, quantity_df, diagnostic_df, runtime_df, wall


_SUMMARY_STATS: tuple[str, ...] = (
  "rep_mean",
  "rep_std",
  "rep_median",
  "rep_p05",
  "rep_p95",
  "rep_max",
)
_METRIC_NAMES: tuple[str, ...] = ("IAE", "ISE", "KL")
_ESTIMATOR_AXES: tuple[str, ...] = (
  "family",
  "tau_scenario",
  "n",
  "estimator_id",
  "label",
  "provider",
  "transform",
  "recovery",
  "model_id",
)


def _summary_stats(
  df: pd.DataFrame, group: list[str], value_col: str
) -> pd.DataFrame:
  return df.groupby(group, as_index=False, dropna=False).agg(
    rep_mean=(value_col, "mean"),
    rep_std=(value_col, "std"),
    rep_median=(value_col, "median"),
    rep_p05=(value_col, lambda s: _nan_quantile(s, 0.05)),
    rep_p95=(value_col, lambda s: _nan_quantile(s, 0.95)),
    rep_max=(value_col, "max"),
  )


def _metric_summary(metric_df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
  chunks: list[pd.DataFrame] = []
  for metric in _METRIC_NAMES:
    g = _summary_stats(metric_df, group, metric)
    g["metric"] = metric
    chunks.append(g)
  return pd.concat(chunks, ignore_index=True)


def _summary_over_x(metric_df: pd.DataFrame) -> pd.DataFrame:
  group = [
    *_ESTIMATOR_AXES,
    "normalize",
    "quantity",
  ]
  rep_group = [*group, "rep"]
  chunks: list[pd.DataFrame] = []
  for metric in _METRIC_NAMES:
    averaged = metric_df.groupby(rep_group, as_index=False, dropna=False).agg(
      value=(metric, "mean")
    )
    g = _summary_stats(averaged, group, "value")
    g["metric"] = metric
    chunks.append(g)
  return pd.concat(chunks, ignore_index=True)


def _runtime_summary(runtime_df: pd.DataFrame) -> pd.DataFrame:
  rt_group = list(_ESTIMATOR_AXES)
  return runtime_df.groupby(rt_group, as_index=False, dropna=False).agg(
    fit_time_mean=("fit_time", "mean"),
    fit_time_std=("fit_time", "std"),
    pdf_time_mean=("pdf_time", "mean"),
    cdf_time_mean=("cdf_time", "mean"),
    h1_time_mean=("h1_time", "mean"),
    h2_time_mean=("h2_time", "mean"),
    tau_time_mean=("tau_time", "mean"),
    surface_time_mean=("surface_time", "mean"),
    total_estimator_time_mean=("total_estimator_time", "mean"),
  )


def _selection_summary(summary_over_x: pd.DataFrame) -> pd.DataFrame:
  ranked = summary_over_x[
    (summary_over_x["quantity"] == "pdf") & (summary_over_x["metric"] == "KL")
  ].copy()
  ranked = ranked.sort_values(
    ["rep_mean", "rep_median", "rep_p95", *_ESTIMATOR_AXES, "normalize"],
    na_position="last",
  ).reset_index(drop=True)
  ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=int))
  return ranked


def _tau_summary(diagnostic_df: pd.DataFrame) -> pd.DataFrame:
  tau_df = diagnostic_df[diagnostic_df["normalize"] == "none"]
  group = [*_ESTIMATOR_AXES, "x", "tau_true"]
  return _summary_stats(tau_df, group, "tau_abs_err").rename(
    columns={name: f"tau_abs_err_{name}" for name in _SUMMARY_STATS}
  )


def _projection_summary(
  metric_df: pd.DataFrame, diagnostic_df: pd.DataFrame
) -> pd.DataFrame:
  pair_axes = [*_ESTIMATOR_AXES, "rep", "x", "tau_true"]
  pdf = metric_df[metric_df["quantity"] == "pdf"]
  metric_wide = pdf.pivot_table(
    index=pair_axes,
    columns="normalize",
    values=list(_METRIC_NAMES),
    aggfunc="mean",
    dropna=False,
  )
  if "none" not in metric_wide.columns.get_level_values(1):
    return _data_frame([])

  diag_metrics = [
    "row_mean_abs_err",
    "row_max_abs_err",
    "col_mean_abs_err",
    "col_max_abs_err",
  ]
  diag_wide = diagnostic_df.pivot_table(
    index=pair_axes,
    columns="normalize",
    values=diag_metrics,
    aggfunc="mean",
    dropna=False,
  )

  rows: list[dict[str, object]] = []
  projected = sorted(
    label
    for label in metric_wide.columns.get_level_values(1).unique()
    if label != "none"
  )
  for projected_label in projected:
    for idx in metric_wide.index:
      row = dict(zip(pair_axes, idx, strict=True))
      row["projected_normalize"] = projected_label
      for metric in _METRIC_NAMES:
        none_val = metric_wide.loc[idx, (metric, "none")]
        proj_val = metric_wide.loc[idx, (metric, projected_label)]
        row[f"{metric}_none"] = float(none_val)
        row[f"{metric}_projected"] = float(proj_val)
        row[f"{metric}_delta"] = float(proj_val - none_val)
      for metric in diag_metrics:
        none_val = diag_wide.loc[idx, (metric, "none")]
        proj_val = diag_wide.loc[idx, (metric, projected_label)]
        row[f"{metric}_none"] = float(none_val)
        row[f"{metric}_projected"] = float(proj_val)
        row[f"{metric}_delta"] = float(proj_val - none_val)
      rows.append(row)

  paired = _data_frame(rows)
  if paired.empty:
    return paired

  group = [*_ESTIMATOR_AXES, "projected_normalize"]
  value_cols = [
    col
    for col in paired.columns
    if col.endswith("_none")
    or col.endswith("_projected")
    or col.endswith("_delta")
  ]
  return paired.groupby(group, as_index=False, dropna=False)[value_cols].mean()


def aggregate_results(
  metric_df: pd.DataFrame, runtime_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Aggregate fixed-x metrics over repetitions and summarize runtimes."""
  group = [
    *_ESTIMATOR_AXES,
    "normalize",
    "quantity",
    "x",
    "tau_true",
  ]
  return _metric_summary(metric_df, group), _runtime_summary(runtime_df)


def aggregate_study_outputs(
  metric_df: pd.DataFrame,
  diagnostic_df: pd.DataFrame,
  runtime_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
  """Build paper-facing study summaries from raw result tables."""
  summary_by_x, runtime_summary = aggregate_results(metric_df, runtime_df)
  summary_over_x = _summary_over_x(metric_df)
  return {
    "metrics_by_x": metric_df,
    "summary_by_x": summary_by_x,
    "summary_over_x": summary_over_x,
    "selection_summary": _selection_summary(summary_over_x),
    "projection_summary": _projection_summary(metric_df, diagnostic_df),
    "tau_summary": _tau_summary(diagnostic_df),
    "runtime_summary": runtime_summary,
  }
