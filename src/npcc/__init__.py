"""Neural Pair-Copula Constructions.

Every public name is re-exported here and from :mod:`npcc.core`, which is a
supported import path in its own right.
"""

from npcc.core.backends.tabpfn_criterion import TabPFNCriterionBackend
from npcc.core.backends.tabpfn_quantile import TabPFNQuantileBackend
from npcc.core.bicop import RosenblattBicop
from npcc.core.controls import (
  FitControlsRosenblattBicop,
  FitControlsRosenblattVinecop,
  Transform,
)
from npcc.core.margin import ConditionalMargin
from npcc.core.vinecop import RosenblattVinecop
from npcc.core.vinedist import RosenblattVinedist
from npcc.core.errors import (
  BackendError,
  EstimatorConfigError,
  InvalidBackendKwargsError,
  MissingBackendDependencyError,
  NpccError,
  UnknownBackendError,
)
from npcc.core.margin_quantile_table import (
  QuantileTableConfig,
  QuantileTableDistribution1D,
)
from npcc.core.registry import (
  BackendSpec,
  available_backends,
  create_backend,
  documented_n_range,
  register_backend,
  validate_backend_kwargs,
)

__all__ = [
  "BackendError",
  "BackendSpec",
  "ConditionalMargin",
  "EstimatorConfigError",
  "FitControlsRosenblattBicop",
  "FitControlsRosenblattVinecop",
  "InvalidBackendKwargsError",
  "MissingBackendDependencyError",
  "NpccError",
  "QuantileTableConfig",
  "QuantileTableDistribution1D",
  "RosenblattBicop",
  "RosenblattVinecop",
  "RosenblattVinedist",
  "TabPFNCriterionBackend",
  "TabPFNQuantileBackend",
  "Transform",
  "UnknownBackendError",
  "available_backends",
  "create_backend",
  "documented_n_range",
  "register_backend",
  "validate_backend_kwargs",
]
