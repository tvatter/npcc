"""Plotting helpers for the simulation study (matplotlib only).

These consume the tidy tables produced by :func:`npcc.experiments.run_study`
and :func:`npcc.experiments.aggregate_results`.  They are deliberately small and
generic: pick a ``(family, tau_scenario, quantity, metric)`` slice and compare
across any axis (``backend`` / ``transform`` / ``normalize`` / ...) via ``hue``.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch


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
  hue: str = "label",
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
  by: str = "label",
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


def metric_boxplot_by_n(
  metrics_df: pd.DataFrame,
  *,
  family: str,
  tau_scenario: str,
  quantity: str,
  metric: str,
  label: str | Sequence[str] | None = None,
  normalize: str | Sequence[str] | None = None,
  ax: Axes | None = None,
) -> Axes:
  """Plot per-repetition metric boxplots by sample size and model identity.

  For conditional scenarios, metrics are averaged over conditioning values
  within each repetition before plotting. Each model identity is the
  ``(label, normalize)`` combination (``label`` already encodes the backend,
  transform, and hyperparameters). A selector may be ``None`` to include all
  values, a string to select one value, or a sequence of strings to select
  multiple values.
  """
  ax = ax or plt.subplots(figsize=(6.0, 3.5))[1]
  sub = _slice(metrics_df, family, tau_scenario, quantity).dropna(
    subset=[metric]
  )
  model_columns = ["label", "normalize"]
  selectors = {
    "label": label,
    "normalize": normalize,
  }
  for column, selector in selectors.items():
    if selector is None:
      continue
    values = [selector] if isinstance(selector, str) else list(selector)
    sub = sub[sub[column].isin(values)]

  per_rep = sub.groupby(
    ["n", *model_columns, "rep"], as_index=False, dropna=False
  )[metric].mean()
  sample_sizes = sorted(per_rep["n"].unique())
  models = sorted(
    per_rep[model_columns].drop_duplicates().itertuples(index=False, name=None),
    key=lambda values: tuple(str(value) for value in values),
  )

  n_models = len(models)
  group_width = 0.8
  box_width = group_width / max(n_models, 1)
  color_map = plt.get_cmap("tab10")
  handles: list[Patch] = []
  for model_index, model_values in enumerate(models):
    offset = (model_index - (n_models - 1) / 2) * box_width
    positions = [index + offset for index in range(len(sample_sizes))]
    model_rows = per_rep
    for column, value in zip(model_columns, model_values, strict=True):
      model_rows = model_rows[model_rows[column] == value]
    data = [
      model_rows[model_rows["n"] == n][metric].to_numpy() for n in sample_sizes
    ]
    color = color_map(model_index % 10)
    boxes = ax.boxplot(
      data,
      positions=positions,
      widths=box_width * 0.9,
      patch_artist=True,
      manage_ticks=False,
    )
    for box in boxes["boxes"]:
      box.set_facecolor(color)
    label = ", ".join(
      f"{column}={value}"
      for column, value in zip(model_columns, model_values, strict=True)
    )
    handles.append(Patch(facecolor=color, label=label))

  ax.set_xticks(range(len(sample_sizes)), [str(n) for n in sample_sizes])
  ax.set_xlabel("Sample size")
  ax.set_ylabel(metric)
  ax.set_title(f"{family} / {tau_scenario} / {quantity}")
  if handles:
    ax.legend(
      handles=handles,
      title="Model",
      fontsize="small",
      loc="center left",
      bbox_to_anchor=(1.02, 0.5),
    )
  return ax
