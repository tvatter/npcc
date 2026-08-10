"""
tabpfn_common.py — shared TabPFN construction seam.

Both TabPFN backends (criterion and quantile) build their regressor
through :func:`make_tabpfn_regressor`, so this is the single place the
concrete ``TabPFNRegressor`` symbol is imported and instantiated — and
therefore the single monkeypatch target in the tests.
"""

from __future__ import annotations

from typing import Any

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

_DEFAULT_MODEL_VERSION = ModelVersion.V3
"""Default TabPFN model version shared by both TabPFN backends.

Centralized here so a version bump is a one-line change.
"""


def make_tabpfn_regressor(
  model_version: ModelVersion | None,
  model_kwargs: dict[str, Any],
) -> TabPFNRegressor:
  """Construct a ``TabPFNRegressor`` for ``model_version`` / ``model_kwargs``.

  ``model_version=None`` builds a bare regressor; otherwise the versioned
  default constructor is used.  ``TabPFNRegressor`` is looked up as a
  module global so the tests can monkeypatch it here.
  """
  if model_version is None:
    return TabPFNRegressor(**model_kwargs)
  return TabPFNRegressor.create_default_for_version(
    model_version,
    **model_kwargs,
  )
