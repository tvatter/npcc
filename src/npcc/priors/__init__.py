"""Rich prior over conditional bivariate copula densities for TabPFN
meta-training.

The public entry points are :class:`ConditionalCopulaPrior` (the hyperprior),
:func:`sample_spec` / :class:`PriorDraw` (one realised conditional copula with a
known density), and :func:`sample_pool` (a list of TabPFN-ready ``(W, y)``
meta-training datasets). The family registry is exposed via
:data:`DEFAULT_REGISTRY` and :class:`FamilySpec`.
"""

from npcc.priors.datasets import (
  direction_datasets,
  sample_eval_dataset,
  sample_pool,
)
from npcc.priors.families import DEFAULT_REGISTRY, FamilySpec
from npcc.priors.prior import (
  ConditionalCopulaPrior,
  PriorDraw,
  XSampler,
  sample_spec,
)

__all__ = [
  "DEFAULT_REGISTRY",
  "ConditionalCopulaPrior",
  "FamilySpec",
  "PriorDraw",
  "XSampler",
  "direction_datasets",
  "sample_eval_dataset",
  "sample_pool",
  "sample_spec",
]
