"""
config.py — configuration for TabPFN meta-training on the conditional-copula prior.

Light-weight (no torch / tabpfn imports) so it can be constructed and validated
in hermetic tests. :meth:`FinetuneConfig.smoke` returns a tiny CPU preset that
runs the full pipeline end-to-end in seconds (for CI and local sanity).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class FinetuneConfig:
  """Hyperparameters for :func:`npcc.finetune.meta_train`.

  Attributes
  ----------
  n_datasets
      Size of the prior pool (distinct synthetic conditional copulas). With
      ``both_directions`` each contributes two meta-datasets.
  rows_per_dataset
      Rows simulated per synthetic dataset (kept well under TabPFN's 50k limit).
  query_fraction
      Fraction of each dataset used as the query (loss) split; the rest is the
      in-context support.
  both_directions
      Emit both Rosenblatt directions (``V|U,X`` and ``U|V,X``) per draw.
  transform
      Support transform applied to the regression target before preprocessing.
      MUST match the inference-time ``PFNRBicop(transform=...)``.
  epochs, learning_rate, weight_decay, grad_clip_value, warmup_frac
      Optimisation schedule (AdamW + cosine warmup, gradient clipping).
  ce_loss_weight, crps_loss_weight, crls_loss_weight, mse_loss_weight
      Weights of the composite TabPFN regression loss. Defaults favour density
      quality (bar-distribution NLL + CRPS) over point accuracy (MSE off).
  n_estimators_finetune
      In-context ensemble size during fine-tuning.
  device
      Torch device string (``"cuda"`` for real runs; ``"cpu"`` for the smoke).
  ignore_pretraining_limits
      Lift TabPFN's sample-size guards (needed for batched fine-tuning).
  use_activation_checkpointing
      Trade compute for memory in the transformer (recommended on GPU).
  random_state
      Seed for the prior pool, preprocessing, and per-epoch splits/shuffles.
  output_dir, checkpoint_name
      Where the fine-tuned checkpoint is written (name must contain ``"v2.5"``).
  max_steps
      Optional hard cap on optimizer steps (used by the smoke preset).
  log_every
      Emit a progress line (step, loss, elapsed, ETA) every this many steps
      (0 disables intra-epoch logging).
  eps
      Boundary clip shared with the support transform.
  """

  n_datasets: int = 512
  rows_per_dataset: int = 2000
  query_fraction: float = 0.2
  both_directions: bool = True
  transform: Literal["identity", "logit", "probit"] = "logit"
  epochs: int = 20
  learning_rate: float = 1e-5
  weight_decay: float = 0.01
  grad_clip_value: float | None = 1.0
  warmup_frac: float = 0.1
  ce_loss_weight: float = 1.0
  crps_loss_weight: float = 1.0
  crls_loss_weight: float = 0.0
  mse_loss_weight: float = 0.0
  n_estimators_finetune: int = 2
  max_data_size: int | None = None
  device: str = "cuda"
  ignore_pretraining_limits: bool = True
  use_activation_checkpointing: bool = True
  random_state: int = 0
  output_dir: Path = Path("runs/default")
  checkpoint_name: str = "finetuned_v2.5.pth"
  max_steps: int | None = None
  log_every: int = 50
  eps: float = 1e-6

  def __post_init__(self) -> None:
    self.output_dir = Path(self.output_dir)
    if self.n_datasets < 1:
      raise ValueError("n_datasets must be >= 1.")
    if self.rows_per_dataset < 4:
      raise ValueError("rows_per_dataset must be >= 4.")
    if not 0.0 < self.query_fraction < 1.0:
      raise ValueError("query_fraction must be in (0, 1).")
    if self.epochs < 1:
      raise ValueError("epochs must be >= 1.")
    if self.n_estimators_finetune < 1:
      raise ValueError("n_estimators_finetune must be >= 1.")
    if "v2.5" not in self.checkpoint_name:
      raise ValueError(
        "checkpoint_name must contain 'v2.5' (TabPFN resolves the architecture "
        "from the filename)."
      )

  @classmethod
  def smoke(cls, output_dir: str | Path = "runs/smoke") -> FinetuneConfig:
    """A tiny CPU configuration that exercises the whole pipeline in seconds."""
    return cls(
      log_every=1,
      n_datasets=2,
      rows_per_dataset=64,
      query_fraction=0.5,
      epochs=1,
      n_estimators_finetune=1,
      device="cpu",
      use_activation_checkpointing=False,
      max_steps=2,
      output_dir=Path(output_dir),
    )
