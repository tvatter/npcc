"""Smoke tests for the optional non-TabPFN backends (gbm / ngboost / tabicl).

Each test ``importorskip``s its backend dependency, so the suite still
passes in environments where the extra is not installed.  GBM and NGBoost
fit locally and quickly; TabICL is a foundation model whose first fit
downloads weights, so it is additionally guarded against download/runtime
failures (skipped, like the real-TabPFN smoke).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from npcc.core.bicop import RosenblattBicop
from npcc.core.quantile_table_distribution1d import QuantileGridConfig


def _gaussian_copula_sample(
  n: int, rho: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
  """Draw ``n`` (u, v) pairs from a Gaussian copula with correlation rho."""
  scipy_stats = pytest.importorskip("scipy.stats")
  rng = np.random.default_rng(seed)
  z = rng.multivariate_normal(
    mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]], size=n
  )
  u = scipy_stats.norm.cdf(z[:, 0])
  v = scipy_stats.norm.cdf(z[:, 1])
  return np.clip(u, 1e-3, 1 - 1e-3), np.clip(v, 1e-3, 1 - 1e-3)


def _check_fitted_model(m: RosenblattBicop) -> None:
  pts = np.array([0.3, 0.5, 0.7])
  pdf = np.asarray(m.pdf(pts, pts))
  assert np.all(np.isfinite(pdf)) and np.all(pdf >= 0.0)

  h = np.asarray(m.hfunc1(pts, pts))
  assert np.all((h >= 0.0) & (h <= 1.0))

  grid = np.asarray(
    m.pdf_grid(np.linspace(0.2, 0.8, 4), np.linspace(0.2, 0.8, 5))
  )
  assert grid.shape == (4, 5)
  assert np.all(grid >= 0.0)

  tau = m.tau(n=200)
  assert isinstance(tau, float)
  assert -1.0 <= tau <= 1.0


def test_gbm_backend_smoke() -> None:
  pytest.importorskip("sklearn")
  u, v = _gaussian_copula_sample(80, rho=0.6, seed=0)
  m = RosenblattBicop(
    backend="gbm",
    quantile_config=QuantileGridConfig(n_quantiles=11),
    backend_kwargs={"n_estimators": 20, "max_depth": 2},
  ).fit(u, v)
  _check_fitted_model(m)


def test_ngboost_backend_smoke() -> None:
  pytest.importorskip("ngboost")
  u, v = _gaussian_copula_sample(80, rho=0.6, seed=1)
  m = RosenblattBicop(
    backend="ngboost",
    backend_kwargs={"n_estimators": 40},
  ).fit(u, v)
  _check_fitted_model(m)


def test_catboost_backend_smoke() -> None:
  pytest.importorskip("catboost")
  u, v = _gaussian_copula_sample(80, rho=0.6, seed=3)
  m = RosenblattBicop(
    backend="catboost",
    backend_kwargs={"iterations": 100},
    quantile_config=QuantileGridConfig(n_quantiles=21),
  ).fit(u, v)
  _check_fitted_model(m)


def test_pytabkit_realmlp_backend_smoke() -> None:
  pytest.importorskip("pytabkit")
  u, v = _gaussian_copula_sample(80, rho=0.6, seed=5)
  m = RosenblattBicop(
    backend="pytabkit-realmlp",
    backend_kwargs={"n_epochs": 8},
    quantile_config=QuantileGridConfig(n_quantiles=21),
  ).fit(u, v)
  _check_fitted_model(m)


def test_nori_backend_smoke() -> None:
  pytest.importorskip("synthefy_nori")
  u, v = _gaussian_copula_sample(80, rho=0.6, seed=7)
  try:
    m = RosenblattBicop(
      backend="nori",
      quantile_config=QuantileGridConfig(n_quantiles=41),
    ).fit(u, v)
  except Exception as exc:  # noqa: BLE001 - weight download / runtime issues
    pytest.skip(f"Nori unavailable at runtime: {exc}")
  _check_fitted_model(m)


def test_tabpfn_finetune_backend_smoke() -> None:
  # Opt-in: fine-tuning is a GPU training loop (minutes), too slow for the
  # default suite. Enable with NPCC_RUN_FINETUNE=1.
  if not os.environ.get("NPCC_RUN_FINETUNE"):
    pytest.skip("set NPCC_RUN_FINETUNE=1 to run the fine-tune smoke")
  torch = pytest.importorskip("torch")
  if not torch.cuda.is_available():
    pytest.skip("fine-tuning needs a GPU")
  if not os.environ.get("TABPFN_TOKEN"):
    pytest.skip("no TABPFN_TOKEN")
  u, v = _gaussian_copula_sample(120, rho=0.6, seed=6)
  try:
    m = RosenblattBicop(
      backend="tabpfn-finetune",
      backend_kwargs={"epochs": 1, "early_stopping": False},
    ).fit(u, v)
    pdf = np.asarray(m.pdf(np.array([0.5]), np.array([0.5])))
  except Exception as exc:  # noqa: BLE001 - license/download/runtime issues
    pytest.skip(f"TabPFN fine-tune unavailable at runtime: {exc}")
  assert np.all(np.isfinite(pdf)) and np.all(pdf >= 0.0)


def test_tabicl_backend_smoke() -> None:
  pytest.importorskip("tabicl")
  u, v = _gaussian_copula_sample(80, rho=0.6, seed=2)
  try:
    m = RosenblattBicop(
      backend="tabicl",
      backend_kwargs={"model_kwargs": {"n_estimators": 1}},
    ).fit(u, v)
    pdf = np.asarray(m.pdf(np.array([0.3, 0.5]), np.array([0.4, 0.6])))
  except Exception as exc:  # noqa: BLE001 - weight download / runtime issues
    pytest.skip(f"TabICL unavailable at runtime: {exc}")
  assert np.all(np.isfinite(pdf)) and np.all(pdf >= 0.0)
