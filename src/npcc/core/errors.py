"""Public exceptions raised by foundation-model copula estimators."""


class ProviderConfigurationError(ValueError):
  """A foundation-model provider was configured inconsistently."""


class UnsupportedRecoveryError(ProviderConfigurationError):
  """The selected provider does not implement the requested recovery."""


class MissingProviderDependencyError(ImportError):
  """An optional provider dependency is not installed."""


class InvalidProviderOutputError(RuntimeError):
  """A provider returned malformed or non-finite predictions."""


class NotFittedError(RuntimeError):
  """An operation requires a successfully fitted estimator."""


class SinkhornConvergenceError(RuntimeError):
  """Sinkhorn projection did not reach its requested tolerance."""


class FoundationModelRangeWarning(UserWarning):
  """Training data are outside a provider's documented sample range."""
