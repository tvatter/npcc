"""
datasets.py — turn prior draws into TabPFN meta-training datasets.

Each synthetic conditional copula yields one (or two) regression datasets whose
feature/target layout mirrors ``PFNRBicop._features`` exactly — features are
``[conditioner, x]`` (column-stacked, conditioner first) and the target is the
other copula score:

- direction 1: ``W = [u | X]``, ``y = v``  (the ``V | U, X`` Rosenblatt fit);
- direction 2: ``W = [v | X]``, ``y = u``  (the ``U | V, X`` reverse fit).

By the uniform-margin identity ``f_{V|U,X} = f_{U|V,X} = c(u,v|x)`` both
directions share the same target functional, so emitting both (the default)
doubles the meta-training signal at no extra simulation cost. Targets stay on
the raw ``(0, 1)`` copula scale; the fine-tuning harness applies the support
transform so it matches the inference-time ``PFNRBicop(transform=...)``.
"""

from __future__ import annotations

import numpy as np

from npcc.priors.prior import ConditionalCopulaPrior, PriorDraw, sample_spec


def _features(first: np.ndarray, x: np.ndarray) -> np.ndarray:
  """``[first | x]`` column-stack, matching ``PFNRBicop._features`` (``[first]``
  when there are no covariates)."""
  if x.shape[1] == 0:
    return first.reshape(-1, 1)
  return np.column_stack([first, x])


def direction_datasets(
  x: np.ndarray, u: np.ndarray, v: np.ndarray, *, both_directions: bool = True
) -> list[tuple[np.ndarray, np.ndarray]]:
  """Build the ``(W, y)`` dataset(s) for one simulated ``(X, u, v)``."""
  datasets = [(_features(u, x), v)]
  if both_directions:
    datasets.append((_features(v, x), u))
  return datasets


def sample_eval_dataset(
  prior: ConditionalCopulaPrior,
  rng: np.random.Generator,
  rows: int | None = None,
) -> tuple[PriorDraw, np.ndarray, np.ndarray, np.ndarray]:
  """Draw one dataset and return ``(draw, X, u, v)`` so the known ``c(u,v|x)`` is
  available via ``draw.true_density`` for evaluation."""
  draw = sample_spec(prior, rng)
  x, u, v = draw.simulate(rng, n=rows)
  return draw, x, u, v


def sample_pool(
  prior: ConditionalCopulaPrior,
  n_datasets: int,
  rows: int,
  rng: np.random.Generator,
  *,
  both_directions: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
  """Sample a pool of ``(W, y)`` meta-training datasets from the prior.

  Returns ``n_datasets`` draws expanded into ``1`` or ``2`` directions each
  (``2 * n_datasets`` datasets when ``both_directions`` is True).
  """
  if n_datasets < 1:
    raise ValueError("n_datasets must be >= 1.")
  if rows < 2:
    raise ValueError("rows must be >= 2.")
  pool: list[tuple[np.ndarray, np.ndarray]] = []
  for _ in range(n_datasets):
    draw = sample_spec(prior, rng)
    x, u, v = draw.simulate(rng, n=rows)
    pool.extend(direction_datasets(x, u, v, both_directions=both_directions))
  return pool
