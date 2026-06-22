# simulation/study.py

from __future__ import annotations

import os
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

from tabpfn.constants import ModelVersion

# ------------------------------------------------------------
# CONFIG: define which models to compare here
# ------------------------------------------------------------
MODEL_SPECS = {
    "V2.5": ModelVersion.V2_5,
    "V3": ModelVersion.V3,
}

# ------------------------------------------------------------
# CORE METRIC FUNCTION
# ------------------------------------------------------------
eps = 1e-12


def curve_metrics(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    x_grid: np.ndarray,
    include_kl: bool = False,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_hat = np.asarray(y_hat, dtype=np.float64)

    err = y_hat - y_true
    abs_err = np.abs(err)

    iae = float(np.trapezoid(abs_err, x_grid))
    ise = float(np.trapezoid(err ** 2, x_grid))

    kl = np.nan
    if include_kl:
        y_true_pos = np.clip(y_true, eps, None)
        y_hat_pos = np.clip(y_hat, eps, None)

        true_mass = float(np.trapezoid(y_true_pos, x_grid))
        hat_mass = float(np.trapezoid(y_hat_pos, x_grid))

        p = y_true_pos / max(true_mass, eps)
        q = y_hat_pos / max(hat_mass, eps)

        kl = float(np.trapezoid(p * np.log(p / q), x_grid))

    return {
        "IAE": iae,
        "ISE": ise,
        "KL": kl,
    }


# ------------------------------------------------------------
# SINGLE REPLICATION
# ------------------------------------------------------------
def summarize_one_rep(
    n: int,
    rep: int,
    *,
    base_seed: int,
    device,
    sample_scenario_1,
    curve_metrics,
    PFNRBicop,
    U_EVAL_FLAT,
    V_EVAL_FLAT,
    X_EVAL_FLAT,
    TRUE_PDF_GRID,
    TRUE_CDF_GRID,
    TRUE_HFUNC1_GRID,
    TRUE_HFUNC2_GRID,
    uv_pairs,
    x_eval,
    N_PAIRS,
    N_X,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = base_seed + 10000 * n + rep
    t0_total = perf_counter()

    # ------------------------------------------------------------
    # Sample one training dataset. Same sampled data is reused for
    # all model versions.
    # ------------------------------------------------------------
    t0_sample = perf_counter()
    df_rep = sample_scenario_1(n=n, seed=seed)
    sample_seconds = perf_counter() - t0_sample

    u_train = df_rep["u"].values
    v_train = df_rep["v"].values
    x_train = df_rep["x"].values

    metric_rows = []
    pointwise_rows = []
    timing_rows = []

    for model_name, model_version in MODEL_SPECS.items():
        t0_model = perf_counter()

        # --------------------------------------------------------
        # Fit model
        # --------------------------------------------------------
        t0_fit = perf_counter()
        model = PFNRBicop(
            device=device,
            model_version=model_version,
        )
        model.fit(u_train, v_train, x_train)
        fit_seconds = perf_counter() - t0_fit

        # --------------------------------------------------------
        # Evaluate model on flattened grid
        # --------------------------------------------------------
        with torch.inference_mode():
            t0 = perf_counter()
            pdf_hat_flat = np.asarray(
                model.pdf(U_EVAL_FLAT, V_EVAL_FLAT, x=X_EVAL_FLAT),
                dtype=np.float64,
            )
            eval_pdf_seconds = perf_counter() - t0

            t0 = perf_counter()
            cdf_hat_flat = np.asarray(
                model.cdf(U_EVAL_FLAT, V_EVAL_FLAT, x=X_EVAL_FLAT),
                dtype=np.float64,
            )
            eval_cdf_seconds = perf_counter() - t0

            t0 = perf_counter()
            h1_hat_flat = np.asarray(
                model.hfunc1(U_EVAL_FLAT, V_EVAL_FLAT, x=X_EVAL_FLAT),
                dtype=np.float64,
            )
            eval_h1_seconds = perf_counter() - t0

            t0 = perf_counter()
            h2_hat_flat = np.asarray(
                model.hfunc2(U_EVAL_FLAT, V_EVAL_FLAT, x=X_EVAL_FLAT),
                dtype=np.float64,
            )
            eval_h2_seconds = perf_counter() - t0

        # --------------------------------------------------------
        # Reshape flattened predictions into:
        #   rows = uv pairs
        #   columns = x grid
        # --------------------------------------------------------
        pdf_hat_grid = pdf_hat_flat.reshape(N_PAIRS, N_X)
        cdf_hat_grid = cdf_hat_flat.reshape(N_PAIRS, N_X)
        h1_hat_grid = h1_hat_flat.reshape(N_PAIRS, N_X)
        h2_hat_grid = h2_hat_flat.reshape(N_PAIRS, N_X)

        # --------------------------------------------------------
        # Per-uv-pair integrated metrics and pointwise curves
        # --------------------------------------------------------
        for pair_idx, (u_val, v_val) in enumerate(uv_pairs):
            truth_and_hat = [
                ("pdf", TRUE_PDF_GRID[pair_idx], pdf_hat_grid[pair_idx], True),
                ("cdf", TRUE_CDF_GRID[pair_idx], cdf_hat_grid[pair_idx], False),
                ("hfunc1", TRUE_HFUNC1_GRID[pair_idx], h1_hat_grid[pair_idx], False),
                ("hfunc2", TRUE_HFUNC2_GRID[pair_idx], h2_hat_grid[pair_idx], False),
            ]

            for quantity, y_true_curve, y_hat_curve, include_kl in truth_and_hat:
                # ------------------------------------------------
                # Integrated curve metrics over x
                # IAE / ISE for all quantities.
                # KL only for pdf.
                # ------------------------------------------------
                stats = curve_metrics(
                    y_true=y_true_curve,
                    y_hat=y_hat_curve,
                    x_grid=x_eval,
                    include_kl=include_kl,
                )

                stats.update({
                    "model": model_name,
                    "n": n,
                    "rep": rep,
                    "seed": seed,
                    "quantity": quantity,
                    "u": float(u_val),
                    "v": float(v_val),
                })

                metric_rows.append(stats)

                # ------------------------------------------------
                # Pointwise values over x.
                # Needed for plotting conditional curves:
                #   x -> c_hat(u, v | x)
                # ------------------------------------------------
                err = y_hat_curve - y_true_curve

                for x_idx, x_val in enumerate(x_eval):
                    pointwise_rows.append({
                        "model": model_name,
                        "n": n,
                        "rep": rep,
                        "seed": seed,
                        "quantity": quantity,
                        "u": float(u_val),
                        "v": float(v_val),
                        "x_idx": int(x_idx),
                        "x": float(x_val),
                        "y_true": float(y_true_curve[x_idx]),
                        "y_hat": float(y_hat_curve[x_idx]),
                        "error": float(err[x_idx]),
                        "abs_error": float(abs(err[x_idx])),
                        "sq_error": float(err[x_idx] ** 2),
                    })

        timing_rows.append({
            "model": model_name,
            "n": n,
            "rep": rep,
            "seed": seed,
            "sample_seconds": sample_seconds,
            "fit_time": fit_seconds,
            "pdf_time": eval_pdf_seconds,
            "cdf_time": eval_cdf_seconds,
            "h1_time": eval_h1_seconds,
            "h2_time": eval_h2_seconds,
            "total_model_time": perf_counter() - t0_model,
        })

    total_rep_time = perf_counter() - t0_total

    metric_results = pd.DataFrame(metric_rows)
    pointwise_results = pd.DataFrame(pointwise_rows)
    timing = pd.DataFrame(timing_rows)
    timing["total_rep_time"] = total_rep_time

    return metric_results, pointwise_results, timing


# ------------------------------------------------------------
# RUN ALL REPS
# ------------------------------------------------------------
def run_repetitions(
    n_list: List[int],
    n_rep: int,
    *,
    base_seed: int,
    num_workers: int,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    tasks = [(n, rep) for n in n_list for rep in range(n_rep)]

    metric_chunks = []
    pointwise_chunks = []
    runtime_chunks = []

    t0_wall = perf_counter()

    if num_workers <= 1:
        for n, rep in tasks:
            metrics, pointwise, timing = summarize_one_rep(
                n=n,
                rep=rep,
                base_seed=base_seed,
                **kwargs,
            )

            metric_chunks.append(metrics)
            pointwise_chunks.append(pointwise)
            runtime_chunks.append(timing)

    else:
        max_workers = min(num_workers, len(tasks), os.cpu_count() or 1)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    summarize_one_rep,
                    n=n,
                    rep=rep,
                    base_seed=base_seed,
                    **kwargs,
                )
                for n, rep in tasks
            ]

            for fut in as_completed(futures):
                metrics, pointwise, timing = fut.result()

                metric_chunks.append(metrics)
                pointwise_chunks.append(pointwise)
                runtime_chunks.append(timing)

    wall_seconds = perf_counter() - t0_wall

    rep_results = (
        pd.concat(metric_chunks, ignore_index=True)
        .sort_values(["model", "n", "rep", "quantity", "u", "v"])
        .reset_index(drop=True)
    )

    pointwise_results = (
        pd.concat(pointwise_chunks, ignore_index=True)
        .sort_values(["model", "n", "rep", "quantity", "u", "v", "x_idx"])
        .reset_index(drop=True)
    )

    runtime = (
        pd.concat(runtime_chunks, ignore_index=True)
        .sort_values(["model", "n", "rep"])
        .reset_index(drop=True)
    )

    return rep_results, pointwise_results, runtime, wall_seconds


# ------------------------------------------------------------
# AGGREGATION
# ------------------------------------------------------------
def aggregate_results(
    rep_results: pd.DataFrame,
    runtime: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate Monte Carlo results and runtime results.

    Returns
    -------
    mc_summary_long
        Long-format summary by model, n, quantity, metric.
    mc_summary_wide
        Wide-format comparison table with one column per model.
    runtime_summary_long
        Long-format runtime summary by model and n.
    runtime_summary_wide
        Wide-format runtime comparison table with one column per model.
    """

    metric_names = ["IAE", "ISE", "KL"]

    # ------------------------------------------------------------
    # Long-format MC summary
    # ------------------------------------------------------------
    mc_long_chunks = []

    for metric in metric_names:
        g = (
            rep_results
            .groupby(["model", "n", "quantity"], as_index=False)
            .agg(
                rep_mean=(metric, "mean"),
                rep_std=(metric, "std"),
                rep_median=(metric, "median"),
                rep_p05=(metric, lambda s: float(np.nanquantile(s, 0.05))),
                rep_p95=(metric, lambda s: float(np.nanquantile(s, 0.95))),
                rep_max=(metric, "max"),
            )
        )
        g["metric"] = metric
        mc_long_chunks.append(g)

    mc_summary_long = pd.concat(mc_long_chunks, ignore_index=True)

    mc_summary_long = mc_summary_long[
        [
            "model",
            "n",
            "quantity",
            "metric",
            "rep_mean",
            "rep_std",
            "rep_median",
            "rep_p05",
            "rep_p95",
            "rep_max",
        ]
    ]

    # ------------------------------------------------------------
    # Wide-format MC summary
    # ------------------------------------------------------------
    mc_summary_wide = mc_summary_long.pivot_table(
        index=["n", "quantity", "metric"],
        columns="model",
        values="rep_mean",
    ).reset_index()

    # Flatten possible column index
    mc_summary_wide.columns.name = None

    # Add direct comparison columns if both versions exist
    if {"V2.5", "V3"}.issubset(mc_summary_wide.columns):
        mc_summary_wide["V3_minus_V2.5"] = (
            mc_summary_wide["V3"] - mc_summary_wide["V2.5"]
        )

        mc_summary_wide["V3_over_V2.5"] = np.divide(
            mc_summary_wide["V3"],
            mc_summary_wide["V2.5"],
            out=np.full(len(mc_summary_wide), np.nan, dtype=np.float64),
            where=np.asarray(mc_summary_wide["V2.5"]) != 0,
        )

        mc_summary_wide["better_model"] = np.where(
            mc_summary_wide["V3"] < mc_summary_wide["V2.5"],
            "V3",
            "V2.5",
        )

    # ------------------------------------------------------------
    # Long-format runtime summary
    # ------------------------------------------------------------
    runtime_summary_long = (
        runtime
        .groupby(["model", "n"], as_index=False)
        .agg(
            fit_time_mean=("fit_time", "mean"),
            fit_time_std=("fit_time", "std"),
            fit_time_p95=("fit_time", lambda s: float(np.nanquantile(s, 0.95))),

            pdf_time_mean=("pdf_time", "mean"),
            pdf_time_std=("pdf_time", "std"),
            pdf_time_p95=("pdf_time", lambda s: float(np.nanquantile(s, 0.95))),

            cdf_time_mean=("cdf_time", "mean"),
            cdf_time_std=("cdf_time", "std"),
            cdf_time_p95=("cdf_time", lambda s: float(np.nanquantile(s, 0.95))),

            h1_time_mean=("h1_time", "mean"),
            h1_time_std=("h1_time", "std"),
            h1_time_p95=("h1_time", lambda s: float(np.nanquantile(s, 0.95))),

            h2_time_mean=("h2_time", "mean"),
            h2_time_std=("h2_time", "std"),
            h2_time_p95=("h2_time", lambda s: float(np.nanquantile(s, 0.95))),

            total_model_time_mean=("total_model_time", "mean"),
            total_model_time_std=("total_model_time", "std"),
            total_model_time_p95=(
                "total_model_time",
                lambda s: float(np.nanquantile(s, 0.95)),
            ),
        )
    )

    # ------------------------------------------------------------
    # Wide-format runtime summary
    # ------------------------------------------------------------
    runtime_summary_wide = runtime_summary_long.pivot_table(
        index="n",
        columns="model",
        values=[
            "fit_time_mean",
            "pdf_time_mean",
            "cdf_time_mean",
            "h1_time_mean",
            "h2_time_mean",
            "total_model_time_mean",
        ],
    )

    # Flatten multi-index columns:
    # ("fit_time_mean", "V2.5") -> "fit_time_mean_V2.5"
    runtime_summary_wide.columns = [
        f"{metric}_{model}" for metric, model in runtime_summary_wide.columns
    ]
    runtime_summary_wide = runtime_summary_wide.reset_index()

    if {
        "total_model_time_mean_V2.5",
        "total_model_time_mean_V3",
    }.issubset(runtime_summary_wide.columns):
        runtime_summary_wide["total_time_V3_minus_V2.5"] = (
            runtime_summary_wide["total_model_time_mean_V3"]
            - runtime_summary_wide["total_model_time_mean_V2.5"]
        )

        runtime_summary_wide["total_time_V3_over_V2.5"] = np.divide(
            runtime_summary_wide["total_model_time_mean_V3"],
            runtime_summary_wide["total_model_time_mean_V2.5"],
            out=np.full(len(runtime_summary_wide), np.nan, dtype=np.float64),
            where=np.asarray(runtime_summary_wide["total_model_time_mean_V2.5"]) != 0,
        )

    return (
        mc_summary_long,
        mc_summary_wide,
        runtime_summary_long,
        runtime_summary_wide,
    )



def _uv_grid_axes(uv_pairs, figsize_per_panel=(3.2, 3.0), sharey=True):
    """
    Create axes arranged according to the uv grid.

    Smallest u is left.
    Smallest v is bottom.
    Largest u is right.
    Largest v is top.
    """
    u_vals = sorted({float(u) for u, _ in uv_pairs})
    v_vals = sorted({float(v) for _, v in uv_pairs})

    n_u = len(u_vals)
    n_v = len(v_vals)

    fig, axes = plt.subplots(
        n_v,
        n_u,
        figsize=(figsize_per_panel[0] * n_u, figsize_per_panel[1] * n_v),
        sharey=sharey,
        squeeze=False,
    )

    axis_map = {}

    for u_idx, u_val in enumerate(u_vals):
        for v_idx, v_val in enumerate(v_vals):
            # matplotlib row 0 is top, so invert v index
            row = n_v - 1 - v_idx
            col = u_idx
            axis_map[(u_val, v_val)] = axes[row, col]

    return fig, axes, axis_map, u_vals, v_vals


def plot_metric_uv_boxgrid(
    rep_results,
    *,
    quantity: str,
    metric: str,
    uv_pairs,
    model_order=None,
    n_order=None,
    ylim=None,
    title=None,
    figsize_per_panel=(3.2, 3.0),
    showfliers=True,
):
    """
    Plot one metric as boxplots over repetitions, separately for every uv pair.

    Parameters
    ----------
    rep_results
        DataFrame with columns:
        model, n, rep, quantity, u, v, and metric columns such as IAE, ISE, KL.
    quantity
        One of: "pdf", "cdf", "hfunc1", "hfunc2".
    metric
        One of: "IAE", "ISE", "KL".
        KL is usually meaningful only for quantity="pdf".
    uv_pairs
        List of (u, v) pairs defining the uv grid.
    model_order
        Optional order of model labels, e.g. ["V2.5", "V3"].
    n_order
        Optional order of sample sizes.
    ylim
        Optional tuple, e.g. (0.0, 0.1).
    title
        Optional figure title.
    """

    if rep_results.empty:
        print("rep_results is empty.")
        return None, None

    required_cols = {"model", "n", "quantity", "u", "v", metric}
    missing = required_cols.difference(rep_results.columns)
    if missing:
        raise ValueError(f"rep_results is missing required columns: {sorted(missing)}")

    df = rep_results.loc[
        (rep_results["quantity"] == quantity)
        & np.isfinite(rep_results[metric])
    ].copy()

    if df.empty:
        print(f"No data found for quantity={quantity!r}, metric={metric!r}.")
        return None, None

    # Normalize u/v to float for robust matching
    df["u"] = df["u"].astype(float)
    df["v"] = df["v"].astype(float)

    uv_pairs_float = [(float(u), float(v)) for u, v in uv_pairs]

    if model_order is None:
        model_order = list(df["model"].dropna().unique())

    if n_order is None:
        n_order = sorted(df["n"].dropna().unique())

    groups = [
        (model_name, n)
        for n in n_order
        for model_name in model_order
    ]

    group_labels = [f"{model}, n={n}" for model, n in groups]
    n_groups = len(groups)

    fig, axes, axis_map, u_vals, v_vals = _uv_grid_axes(
        uv_pairs_float,
        figsize_per_panel=figsize_per_panel,
        sharey=True,
    )

    colors = plt.cm.tab20(np.linspace(0, 1, max(n_groups, 1)))

    for u_val, v_val in uv_pairs_float:
        ax = axis_map[(u_val, v_val)]

        data = []
        positions = []

        for i, (model_name, n) in enumerate(groups):
            vals = df.loc[
                (df["u"] == u_val)
                & (df["v"] == v_val)
                & (df["model"] == model_name)
                & (df["n"] == n),
                metric,
            ].dropna().values

            data.append(vals)
            positions.append(i + 1)

        # If all groups empty, mark subplot and continue
        if all(len(vals) == 0 for vals in data):
            ax.text(
                0.5,
                0.5,
                "no data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"u={u_val:g}, v={v_val:g}")
            continue

        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.65,
            patch_artist=True,
            showfliers=showfliers,
            manage_ticks=False,
        )

        for i, patch in enumerate(box["boxes"]):
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.65)

        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.4)

        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)

        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_title(f"u={u_val:g}, v={v_val:g}", fontsize=10)
        ax.grid(axis="y", alpha=0.25)

        # Avoid overcrowding subplot x labels
        ax.set_xticks([])

    # Axis labels by grid position
    for u_val in u_vals:
        ax_bottom = axis_map[(u_val, v_vals[0])]
        ax_bottom.set_xlabel(f"u={u_val:g}")

    for v_val in v_vals:
        ax_left = axis_map[(u_vals[0], v_val)]
        ax_left.set_ylabel(f"v={v_val:g}\n{metric}")

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[i],
            lw=8,
            alpha=0.65,
            label=group_labels[i],
        )
        for i in range(n_groups)
    ]

    axes[0, -1].legend(
        handles=legend_handles,
        title="Model / sample size",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )

    if title is None:
        title = f"{quantity} {metric} by (u, v) Pair"

    fig.suptitle(
        title,
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 0.88, 0.96])

    return fig, axes

def plot_all_uv_metric_boxgrids(
    rep_results,
    *,
    uv_pairs,
    model_order=None,
    metrics_by_quantity=None,
    ylim_by_metric=None,
):
    if metrics_by_quantity is None:
        metrics_by_quantity = {
            "pdf": ["IAE", "ISE", "KL"],
            "cdf": ["IAE", "ISE"],
            "hfunc1": ["IAE", "ISE"],
            "hfunc2": ["IAE", "ISE"],
        }

    if ylim_by_metric is None:
        ylim_by_metric = {
            "IAE": (0.0, 0.1),
            "ISE": (0.0, 0.1),
            "KL": (0.0, 0.1),
        }

    figs_axes = {}

    for quantity, metrics in metrics_by_quantity.items():
        for metric in metrics:
            if metric not in rep_results.columns:
                continue

            df_sub = rep_results.loc[
                (rep_results["quantity"] == quantity)
                & rep_results[metric].notna()
            ]

            if df_sub.empty:
                continue

            fig, axes = plot_metric_uv_boxgrid(
                rep_results,
                quantity=quantity,
                metric=metric,
                uv_pairs=uv_pairs,
                model_order=model_order,
                ylim=ylim_by_metric.get(metric),
                title=f"{quantity} {metric} by (u, v) Pair",
            )

            figs_axes[(quantity, metric)] = (fig, axes)

    return figs_axes



def average_over_uv_pairs(
    rep_results: pd.DataFrame,
    *,
    metrics_by_quantity: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """
    Average metric values over all (u, v) pairs for each
    model, n, rep, and quantity.

    Expected rep_results columns:
      model, n, rep, quantity, u, v, IAE, ISE, KL, ...

    Returns
    -------
    avg_results
        Long-ish DataFrame with one row per model/n/rep/quantity
        and averaged metric columns.
    """
    if metrics_by_quantity is None:
        metrics_by_quantity = {
            "pdf": ["IAE", "ISE", "KL"],
            "cdf": ["IAE", "ISE"],
            "hfunc1": ["IAE", "ISE"],
            "hfunc2": ["IAE", "ISE"],
        }

    required = {"model", "n", "rep", "quantity"}
    missing = required.difference(rep_results.columns)
    if missing:
        raise ValueError(f"rep_results is missing required columns: {sorted(missing)}")

    chunks = []

    for quantity, metrics in metrics_by_quantity.items():
        df_q = rep_results.loc[rep_results["quantity"] == quantity].copy()

        if df_q.empty:
            continue

        available_metrics = [
            m for m in metrics
            if m in df_q.columns and not df_q[m].dropna().empty
        ]

        if not available_metrics:
            continue

        avg_q = (
            df_q
            .groupby(["model", "n", "rep", "quantity"], as_index=False)
            .agg({m: "mean" for m in available_metrics})
        )

        chunks.append(avg_q)

    if not chunks:
        return pd.DataFrame()

    return pd.concat(chunks, ignore_index=True)


def plot_avg_uv_metric_boxplot(
    rep_results: pd.DataFrame,
    *,
    quantity: str,
    metric: str,
    model_order: list[str] | None = None,
    n_order: list[int] | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 5),
    showfliers: bool = True,
):
    """
    Boxplot for a metric averaged over all (u, v) pairs.

    x-axis:
        sample size n

    grouped/color:
        model version

    boxplot distribution:
        repetitions
    """
    avg_results = average_over_uv_pairs(rep_results)

    if avg_results.empty:
        print("No averaged uv results available.")
        return None, None

    if metric not in avg_results.columns:
        print(f"Metric {metric!r} not found in averaged results.")
        return None, None

    df = avg_results.loc[
        (avg_results["quantity"] == quantity)
        & avg_results[metric].notna()
    ].copy()

    if df.empty:
        print(f"No data for quantity={quantity!r}, metric={metric!r}.")
        return None, None

    if model_order is None:
        model_order = list(df["model"].dropna().unique())

    if n_order is None:
        n_order = sorted(df["n"].dropna().unique())

    fig, ax = plt.subplots(figsize=figsize)

    base_positions = np.arange(len(n_order), dtype=float)
    n_models = len(model_order)

    group_width = 0.75
    box_width = (group_width / max(n_models, 1)) * 0.85

    if n_models == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(
            -group_width / 2 + box_width / 2,
            group_width / 2 - box_width / 2,
            n_models,
        )

    colors = plt.cm.tab10(np.linspace(0, 1, max(n_models, 1)))

    for i, model_name in enumerate(model_order):
        grouped = []

        for n in n_order:
            vals = df.loc[
                (df["model"] == model_name)
                & (df["n"] == n),
                metric,
            ].dropna().values

            grouped.append(vals)

        if all(len(vals) == 0 for vals in grouped):
            continue

        positions = base_positions + offsets[i]

        box = ax.boxplot(
            grouped,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=showfliers,
            manage_ticks=False,
        )

        for patch in box["boxes"]:
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.65)

        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

    ax.set_xticks(base_positions)
    ax.set_xticklabels([str(n) for n in n_order])

    ax.set_xlabel("sample size n")
    ax.set_ylabel(f"uv-averaged {metric}")

    if title is None:
        title = f"{quantity} {metric}, averaged over (u, v) pairs"

    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    if ylim is not None:
        ax.set_ylim(*ylim)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[i],
            lw=8,
            alpha=0.65,
            label=model_order[i],
        )
        for i in range(n_models)
    ]

    ax.legend(
        handles=legend_handles,
        title="model",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )

    fig.tight_layout()
    return fig, ax

def plot_all_avg_uv_metric_boxplots(
    rep_results: pd.DataFrame,
    *,
    model_order: list[str] | None = None,
    n_order: list[int] | None = None,
    metrics_by_quantity: dict[str, list[str]] | None = None,
    ylim_by_metric: dict[str, tuple[float, float]] | tuple[float, float] | None = None,
    figsize: tuple[float, float] = (10, 5),
    showfliers: bool = True,
):
    """
    Plot average-over-uv boxplots for all requested quantity/metric combinations.

    Parameters
    ----------
    ylim_by_metric
        Either:
          - dict, e.g. {"IAE": (0, 0.1), "ISE": (0, 0.1), "KL": (0, 0.1)}
          - tuple, e.g. (0, 0.1), applied to all metrics
          - None
    """
    if metrics_by_quantity is None:
        metrics_by_quantity = {
            "pdf": ["IAE", "ISE", "KL"],
            "cdf": ["IAE", "ISE"],
            "hfunc1": ["IAE", "ISE"],
            "hfunc2": ["IAE", "ISE"],
        }

    figs_axes = {}

    for quantity, metrics in metrics_by_quantity.items():
        for metric in metrics:
            if metric not in rep_results.columns:
                continue

            df_sub = rep_results.loc[
                (rep_results["quantity"] == quantity)
                & rep_results[metric].notna()
            ]

            if df_sub.empty:
                continue

            if isinstance(ylim_by_metric, dict):
                ylim = ylim_by_metric.get(metric)
            else:
                ylim = ylim_by_metric

            fig, ax = plot_avg_uv_metric_boxplot(
                rep_results,
                quantity=quantity,
                metric=metric,
                model_order=model_order,
                n_order=n_order,
                ylim=ylim,
                title=f"{quantity} {metric}, averaged over all (u, v) pairs",
                figsize=figsize,
                showfliers=showfliers,
            )

            figs_axes[(quantity, metric)] = (fig, ax)

    return figs_axes


def plot_runtime_components_boxplot(
    runtime,
    *,
    component_cols=None,
    model_order=None,
    n_order=None,
    ylim=None,
    figsize=(16, 6),
    showfliers=True,
    title="Runtime Components by Model Version and Sample Size",
):
    """
    Grouped boxplot for runtime components.

    x-axis:
        runtime component, e.g. fit, pdf, cdf, hfunc1, hfunc2, total

    grouped/color:
        model version + sample size n

    boxplot distribution:
        repetitions

    Parameters
    ----------
    runtime
        DataFrame with columns:
        model, n, fit_time, pdf_time, cdf_time, h1_time, h2_time, total_model_time.
    component_cols
        Optional list of runtime columns to plot.
    model_order
        Optional fixed model order, e.g. ["V2.5", "V3"].
    n_order
        Optional fixed sample-size order.
    ylim
        Optional y-axis limits, e.g. (0.0, 60.0).
    figsize
        Matplotlib figure size.
    showfliers
        Whether to show boxplot outliers.
    title
        Plot title.

    Returns
    -------
    fig, ax
    """
    if runtime.empty:
        print("No runtime data. Run the Monte Carlo experiment first.")
        return None, None

    if component_cols is None:
        component_cols = [
            "fit_time",
            "pdf_time",
            "cdf_time",
            "h1_time",
            "h2_time",
            "total_model_time",
        ]

    available_cols = [c for c in component_cols if c in runtime.columns]

    if not available_cols:
        print("No component timing columns found in runtime.")
        return None, None

    required_cols = {"model", "n"}
    missing = required_cols.difference(runtime.columns)
    if missing:
        raise ValueError(f"runtime is missing required columns: {sorted(missing)}")

    if model_order is None:
        model_order = list(runtime["model"].dropna().unique())

    if n_order is None:
        n_order = sorted(runtime["n"].dropna().unique())

    groups = [
        (model_name, n)
        for n in n_order
        for model_name in model_order
    ]

    group_count = len(groups)

    fig, ax = plt.subplots(figsize=figsize)

    base_positions = np.arange(len(available_cols), dtype=float)
    group_width = 0.85
    box_width = (group_width / max(group_count, 1)) * 0.85

    if group_count == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(
            -group_width / 2 + box_width / 2,
            group_width / 2 - box_width / 2,
            group_count,
        )

    colors = plt.cm.tab20(np.linspace(0, 1, max(group_count, 1)))

    for i, (model_name, n) in enumerate(groups):
        mask = (
            (runtime["model"] == model_name)
            & (runtime["n"] == n)
        )

        grouped = [
            runtime.loc[mask, col].dropna().values
            for col in available_cols
        ]

        if all(len(vals) == 0 for vals in grouped):
            continue

        positions = base_positions + offsets[i]

        box = ax.boxplot(
            grouped,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=showfliers,
            manage_ticks=False,
        )

        for patch in box["boxes"]:
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.6)

        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

    pretty_labels = [
        c.replace("_time", "")
         .replace("total_model", "total")
         .replace("h1", "hfunc1")
         .replace("h2", "hfunc2")
         .replace("_", " ")
        for c in available_cols
    ]

    ax.set_xticks(base_positions)
    ax.set_xticklabels(pretty_labels, rotation=20, ha="right")

    ax.set_ylabel("seconds")
    ax.set_xlabel("component")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    if ylim is not None:
        ax.set_ylim(*ylim)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[i],
            lw=8,
            alpha=0.6,
            label=f"{model_name}, n={n}",
        )
        for i, (model_name, n) in enumerate(groups)
    ]

    ax.legend(
        handles=legend_handles,
        title="Model / sample size",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )

    fig.tight_layout()

    return fig, ax

def _conditional_curve_uv_axes(
    uv_pairs,
    *,
    figsize_per_panel=(3.3, 3.0),
    sharey=True,
):
    """
    Create axes arranged according to the uv grid.

    Smallest u -> left.
    Largest u  -> right.
    Smallest v -> bottom.
    Largest v  -> top.
    """
    uv_pairs_float = [(float(u), float(v)) for u, v in uv_pairs]

    u_vals = sorted({u for u, _ in uv_pairs_float})
    v_vals = sorted({v for _, v in uv_pairs_float})

    n_u = len(u_vals)
    n_v = len(v_vals)

    fig, axes = plt.subplots(
        n_v,
        n_u,
        figsize=(figsize_per_panel[0] * n_u, figsize_per_panel[1] * n_v),
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )

    axis_map = {}

    for u_idx, u_val in enumerate(u_vals):
        for v_idx, v_val in enumerate(v_vals):
            row = n_v - 1 - v_idx
            col = u_idx
            axis_map[(u_val, v_val)] = axes[row, col]

    return fig, axes, axis_map, u_vals, v_vals

def plot_conditional_curves_by_uv(
    pointwise_results: pd.DataFrame,
    *,
    uv_pairs,
    quantity: str = "pdf",
    x_col: str = "x",
    model_order: list[str] | None = None,
    n: int | None = None,
    interval: tuple[float, float] = (0.05, 0.95),
    ylim: tuple[float, float] | None = None,
    figsize_per_panel: tuple[float, float] = (3.3, 3.0),
    title: str | None = None,
    show_true: bool = True,
):
    """
    Plot fitted conditional curves over x for every (u, v) pair.

    For each model:
      - solid line = mean y_hat over repetitions
      - band = central interval over repetitions, default 5%-95%

    Expected pointwise_results columns:
      model, n, rep, quantity, u, v, x, y_true, y_hat
    """

    if pointwise_results.empty:
        print("pointwise_results is empty.")
        return None, None

    required_cols = {
        "model",
        "n",
        "rep",
        "quantity",
        "u",
        "v",
        x_col,
        "y_true",
        "y_hat",
    }

    missing = required_cols.difference(pointwise_results.columns)
    if missing:
        raise ValueError(
            f"pointwise_results is missing required columns: {sorted(missing)}"
        )

    df = pointwise_results.loc[
        pointwise_results["quantity"] == quantity
    ].copy()

    if n is not None:
        df = df.loc[df["n"] == n].copy()

    if df.empty:
        print(f"No data for quantity={quantity!r}, n={n!r}.")
        return None, None

    df["u"] = df["u"].astype(float)
    df["v"] = df["v"].astype(float)

    uv_pairs_float = [(float(u), float(v)) for u, v in uv_pairs]

    if model_order is None:
        model_order = list(df["model"].dropna().unique())

    q_low, q_high = interval

    pred_summary = (
        df
        .groupby(["model", "u", "v", x_col], as_index=False)
        .agg(
            mean_hat=("y_hat", "mean"),
            low_hat=("y_hat", lambda s: float(np.quantile(s, q_low))),
            high_hat=("y_hat", lambda s: float(np.quantile(s, q_high))),
        )
    )

    true_summary = (
        df
        .groupby(["u", "v", x_col], as_index=False)
        .agg(true=("y_true", "mean"))
    )

    fig, axes, axis_map, u_vals, v_vals = _conditional_curve_uv_axes(
        uv_pairs_float,
        figsize_per_panel=figsize_per_panel,
        sharey=True,
    )

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(model_order), 1)))

    for u_val, v_val in uv_pairs_float:
        ax = axis_map[(u_val, v_val)]

        if show_true:
            true_sub = true_summary.loc[
                (true_summary["u"] == u_val)
                & (true_summary["v"] == v_val)
            ].sort_values(x_col)

            if not true_sub.empty:
                ax.plot(
                    true_sub[x_col].values,
                    true_sub["true"].values,
                    color="black",
                    linestyle="dashed",
                    linewidth=1.4,
                    label="True",
                )

        for i, model_name in enumerate(model_order):
            sub = pred_summary.loc[
                (pred_summary["model"] == model_name)
                & (pred_summary["u"] == u_val)
                & (pred_summary["v"] == v_val)
            ].sort_values(x_col)

            if sub.empty:
                continue

            ax.plot(
                sub[x_col].values,
                sub["mean_hat"].values,
                color=colors[i],
                linewidth=1.5,
                label=model_name,
            )

            ax.fill_between(
                sub[x_col].values,
                sub["low_hat"].values,
                sub["high_hat"].values,
                color=colors[i],
                alpha=0.18,
                linewidth=0,
            )

        ax.set_title(f"u={u_val:g}, v={v_val:g}", fontsize=10)
        ax.grid(alpha=0.25)

        if ylim is not None:
            ax.set_ylim(*ylim)

    for u_val in u_vals:
        axis_map[(u_val, v_vals[0])].set_xlabel(f"{x_col}\nu={u_val:g}")

    for v_val in v_vals:
        axis_map[(u_vals[0], v_val)].set_ylabel(f"v={v_val:g}\n{quantity}")

    handles, labels = [], []
    for ax in axes.flatten():
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=max(1, len(labels)),
        frameon=False,
    )

    if title is None:
        title = f"{quantity}: fitted conditional curves over {x_col}"

    if n is not None:
        title += f" (n={n})"

    fig.suptitle(
        title,
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    return fig, axes