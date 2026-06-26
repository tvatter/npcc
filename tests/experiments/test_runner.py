"""End-to-end runner tests using the fake TabPFN regressor (hermetic).

Two small studies are each run once (module-scoped fixtures) and shared across
assertions.  Sinkhorn is exercised only on the cheap unconditional path; the
conditional-projection correctness is covered by the core test suite.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from npcc.experiments.config import GridConfig, RunConfig
from npcc.experiments.runner import aggregate_results, run_study
from tests.conftest import (
  _TABPFN_REGRESSOR_TARGETS,
  _UniformQuantileRegressor,
)


def _run(grid: GridConfig) -> tuple[pd.DataFrame, pd.DataFrame, float]:
  mp = pytest.MonkeyPatch()
  for target in _TABPFN_REGRESSOR_TARGETS:
    mp.setattr(target, _UniformQuantileRegressor)
  try:
    return run_study(grid, RunConfig(out=Path("unused"), workers=1))
  finally:
    mp.undo()


@pytest.fixture(scope="module")
def coverage_study() -> tuple[pd.DataFrame, pd.DataFrame, float]:
  """All axes except Sinkhorn (no projection → fast)."""
  grid = GridConfig(
    families=["clayton"],
    tau_scenarios=["linear", "uncond50"],
    transforms=["logit", "identity"],
    methods=["criterion", "quantiles"],
    normalize=[None],
    n=[20],
    n_rep=1,
    projection_grid_size=8,
  )
  return _run(grid)


@pytest.fixture(scope="module")
def model_version_study() -> tuple[pd.DataFrame, pd.DataFrame, float]:
  """Two model versions on the same data cell (hermetic fake ignores them)."""
  grid = GridConfig(
    families=["clayton"],
    tau_scenarios=["uncond50"],
    transforms=["logit"],
    methods=["criterion"],
    normalize=[None],
    n=[20],
    n_rep=1,
    model_versions=["v2.5", "v3"],
    projection_grid_size=8,
  )
  return _run(grid)


@pytest.fixture(scope="module")
def norm_study() -> Iterator[pd.DataFrame]:
  """Sinkhorn on the unconditional path only (single grid, cheap)."""
  grid = GridConfig(
    families=["clayton"],
    tau_scenarios=["uncond50"],
    transforms=["logit"],
    methods=["criterion"],
    normalize=[None, 2],
    n=[20],
    n_rep=1,
    projection_grid_size=8,
  )
  yield _run(grid)[0]


def test_run_study_covers_all_axes(
  coverage_study: tuple[pd.DataFrame, pd.DataFrame, float],
) -> None:
  metrics_df, _, wall = coverage_study
  assert wall >= 0.0
  expected_cols = {
    "family",
    "tau_scenario",
    "n",
    "rep",
    "seed",
    "transform",
    "method",
    "normalize",
    "quantity",
    "u",
    "v",
    "IAE",
    "ISE",
    "KL",
  }
  assert expected_cols.issubset(metrics_df.columns)
  assert set(metrics_df["method"].unique()) == {"criterion", "quantiles"}
  assert set(metrics_df["transform"].unique()) == {"logit", "identity"}
  assert set(metrics_df["quantity"].unique()) == {
    "pdf",
    "cdf",
    "hfunc1",
    "hfunc2",
  }
  assert set(metrics_df["tau_scenario"].unique()) == {"linear", "uncond50"}


def test_conditional_and_unconditional_rows_differ_on_uv(
  coverage_study: tuple[pd.DataFrame, pd.DataFrame, float],
) -> None:
  metrics_df = coverage_study[0]
  cond = metrics_df[metrics_df["tau_scenario"] == "linear"]
  uncond = metrics_df[metrics_df["tau_scenario"] == "uncond50"]
  # Conditional rows carry per-(u,v) coordinates; unconditional aggregate over
  # the uv grid and leave u/v as NaN.
  assert cond["u"].notna().all()
  assert np.isnan(uncond["u"]).all()


def test_aggregate_results_groups_over_reps(
  coverage_study: tuple[pd.DataFrame, pd.DataFrame, float],
) -> None:
  metrics_df, runtime_df, _ = coverage_study
  mc_summary, runtime_summary = aggregate_results(metrics_df, runtime_df)
  assert set(mc_summary["metric"].unique()) == {"IAE", "ISE", "KL"}
  assert not runtime_summary.empty
  assert {"fit_time_mean", "pdf_time_mean"}.issubset(runtime_summary.columns)


def test_model_version_axis_labels_rows_and_aggregates(
  model_version_study: tuple[pd.DataFrame, pd.DataFrame, float],
) -> None:
  metrics_df, runtime_df, _ = model_version_study
  assert "model_version" in metrics_df.columns
  assert "model_version" in runtime_df.columns
  assert set(metrics_df["model_version"].unique()) == {"v2.5", "v3"}
  mc_summary, runtime_summary = aggregate_results(metrics_df, runtime_df)
  assert "model_version" in mc_summary.columns
  assert set(runtime_summary["model_version"].unique()) == {"v2.5", "v3"}


def test_normalize_axis_applies_only_to_pdf(norm_study: pd.DataFrame) -> None:
  pdf = norm_study[norm_study["quantity"] == "pdf"]
  non_pdf = norm_study[norm_study["quantity"] != "pdf"]
  assert set(pdf["normalize"].unique()) == {"none", "2"}
  assert set(non_pdf["normalize"].unique()) == {"none"}
