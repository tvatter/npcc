"""Typed exception taxonomy for npcc.

A single :class:`NpccError` root, with a stdlib base mixed in only where it is
semantically correct — a missing optional dependency *is* an
``ImportError``, so tooling that already catches ``ImportError`` keeps working.
The config errors are plain :class:`NpccError` subclasses so callers can catch
the npcc types precisely.
"""

from __future__ import annotations


class NpccError(Exception):
  """Base class for every error npcc raises on purpose."""


class BackendError(NpccError):
  """A problem selecting or configuring an inner distributional backend."""


class UnknownBackendError(BackendError):
  """The requested backend name is not registered."""


class MissingBackendDependencyError(BackendError, ImportError):
  """An optional backend's third-party dependency is not installed."""


class EstimatorConfigError(NpccError):
  """An estimator entry in the study configuration is invalid."""


class InvalidBackendKwargsError(EstimatorConfigError):
  """``backend_kwargs`` has unknown or mis-typed keys for the backend."""
