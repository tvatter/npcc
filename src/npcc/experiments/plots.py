"""Plotting helpers for the simulation study (matplotlib only).

These consume the tidy tables produced by :func:`npcc.experiments.run_study`
and :func:`npcc.experiments.aggregate_results`.  They are deliberately small and
generic: pick a ``(family, tau_scenario, quantity, metric)`` slice and compare
across any axis (``method`` / ``transform`` / ``normalize`` / ...) via ``hue``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


def _slice(
  df: pd.DataFrame, family: str, tau_scenario: str, quantity: str
) -> pd.DataFrame:
  return df[
    (df["family"] == family)
    & (df["tau_scenario"] == tau_scenario)
    & (df["quantity"] == quantity)
  ]


def mean_metric_lineplot(
  mc_summary: pd.DataFrame,
  *,
  family: str,
  tau_scenario: str,
  quantity: str,
  metric: str,
  hue: str = "method",
  ax: Axes | None = None,
) -> Axes:
  """Plot the across-rep mean of ``metric`` vs ``n``, one line per ``hue`` level.

  ``mc_summary`` is the long table from :func:`aggregate_results`.  Error bands
  use ``rep_std``.
  """
  ax = ax or plt.subplots(figsize=(5.0, 3.5))[1]
  sub = _slice(mc_summary, family, tau_scenario, quantity)
  sub = sub[sub["metric"] == metric]
  for level, g in sub.groupby(hue):
    g = g.sort_values("n")
    ax.plot(g["n"], g["rep_mean"], marker="o", label=f"{hue}={level}")
    ax.fill_between(
      g["n"],
      g["rep_mean"] - g["rep_std"],
      g["rep_mean"] + g["rep_std"],
      alpha=0.15,
    )
  ax.set_xlabel("n")
  ax.set_ylabel(f"{metric} ({quantity})")
  ax.set_title(f"{family} / {tau_scenario}")
  ax.legend(fontsize="small")
  return ax


def metric_boxplot(
  metrics_df: pd.DataFrame,
  *,
  family: str,
  tau_scenario: str,
  quantity: str,
  metric: str,
  n: int,
  by: str = "method",
  ax: Axes | None = None,
) -> Axes:
  """Boxplot of the per-rep ``metric`` distribution at sample size ``n``,
  one box per level of axis ``by``."""
  ax = ax or plt.subplots(figsize=(5.0, 3.5))[1]
  sub = _slice(metrics_df, family, tau_scenario, quantity)
  sub = sub[sub["n"] == n].dropna(subset=[metric])
  levels = sorted(sub[by].unique(), key=str)
  data = [sub[sub[by] == lvl][metric].to_numpy() for lvl in levels]
  ax.boxplot(data, tick_labels=[str(lvl) for lvl in levels])
  ax.set_xlabel(by)
  ax.set_ylabel(f"{metric} ({quantity})")
  ax.set_title(f"{family} / {tau_scenario} / n={n}")
  return ax
