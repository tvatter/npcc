from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from npcc.experiments.plots import metric_boxplot_by_n


def test_metric_boxplot_by_n_groups_repetitions_by_model_and_n() -> None:
  rows = []
  for n in (100, 200):
    for recovery in ("native_distribution", "quantile_inversion"):
      for rep in (0, 1, 2):
        for x in (0.25, 0.75):
          rows.append(
            {
              "family": "clayton",
              "tau_scenario": "linear",
              "quantity": "pdf",
              "n": n,
              "label": recovery,
              "provider": "tabpfn",
              "recovery": recovery,
              "model_id": "v3",
              "transform": "logit",
              "normalize": "none",
              "rep": rep,
              "x": x,
              "ISE": n / 100 + rep + x,
            }
          )
  metrics_df = pd.DataFrame(rows)

  ax = metric_boxplot_by_n(
    metrics_df,
    family="clayton",
    tau_scenario="linear",
    quantity="pdf",
    metric="ISE",
  )

  assert [tick.get_text() for tick in ax.get_xticklabels()] == ["100", "200"]
  assert ax.get_xlabel() == "Sample size"
  assert ax.get_ylabel() == "ISE"
  assert len(ax.patches) == 4
  legend = ax.get_legend()
  assert legend is not None
  legend_anchor = legend.get_bbox_to_anchor().transformed(
    ax.transAxes.inverted()
  )
  assert legend_anchor.x0 == pytest.approx(1.02)
  assert [text.get_text() for text in legend.get_texts()] == [
    "label=native_distribution, provider=tabpfn, "
    "recovery=native_distribution, model_id=v3, transform=logit, normalize=none",
    "label=quantile_inversion, provider=tabpfn, "
    "recovery=quantile_inversion, model_id=v3, transform=logit, normalize=none",
  ]


def test_metric_boxplot_by_n_ignores_missing_metric_values() -> None:
  metrics_df = pd.DataFrame(
    {
      "family": ["clayton", "clayton"],
      "tau_scenario": ["constant", "constant"],
      "quantity": ["cdf", "cdf"],
      "n": [100, 100],
      "label": ["native", "native"],
      "provider": ["tabpfn", "tabpfn"],
      "recovery": ["native_distribution", "native_distribution"],
      "model_id": ["v3", "v3"],
      "transform": ["logit", "logit"],
      "normalize": ["none", "none"],
      "rep": [0, 1],
      "IAE": [0.2, float("nan")],
    }
  )

  ax = metric_boxplot_by_n(
    metrics_df,
    family="clayton",
    tau_scenario="constant",
    quantity="cdf",
    metric="IAE",
  )

  median = ax.lines[4].get_ydata()
  assert median == pytest.approx([0.2, 0.2])


def test_metric_boxplot_by_n_filters_each_model_attribute() -> None:
  metrics_df = pd.DataFrame(
    [
      {
        "family": "clayton",
        "tau_scenario": "constant",
        "quantity": "pdf",
        "n": 100,
        "label": f"{recovery}-{version}",
        "provider": "tabpfn",
        "recovery": recovery,
        "model_id": version,
        "transform": transform,
        "normalize": normalize,
        "rep": 0,
        "ISE": 0.1,
      }
      for recovery in ("native_distribution", "quantile_inversion")
      for version in ("v2.5", "v3")
      for transform in ("identity", "logit")
      for normalize in ("none", "5")
    ]
  )

  ax = metric_boxplot_by_n(
    metrics_df,
    family="clayton",
    tau_scenario="constant",
    quantity="pdf",
    metric="ISE",
    recovery="native_distribution",
    model_id=["v2.5", "v3"],
    transform=("logit",),
    normalize="5",
  )

  assert len(ax.patches) == 2
  legend = ax.get_legend()
  assert legend is not None
  assert [text.get_text() for text in legend.get_texts()] == [
    "label=native_distribution-v2.5, provider=tabpfn, "
    "recovery=native_distribution, model_id=v2.5, transform=logit, normalize=5",
    "label=native_distribution-v3, provider=tabpfn, "
    "recovery=native_distribution, model_id=v3, transform=logit, normalize=5",
  ]
