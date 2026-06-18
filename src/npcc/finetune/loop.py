"""
loop.py — meta-train (fine-tune) TabPFN-v2.5 on the conditional-copula prior.

Unlike ``FinetunedTabPFNRegressor`` (which fine-tunes on a *single* dataset split
into context/query), this drives the low-level TabPFN fine-tuning building blocks
over a *list* of synthetic datasets sampled from :class:`ConditionalCopulaPrior`
— i.e. true prior-fitting / meta-training. Each prior draw becomes one
meta-dataset; one optimizer step per dataset (the collator enforces batch size 1).

The per-step wiring mirrors ``FinetunedTabPFNRegressor`` exactly (verified
against the installed package): set the per-dataset bar distribution, condition
on the support split via ``fit_from_preprocessed``, forward the query split,
stack the per-estimator logits to ``[B*E, Q, L]`` and feed
``_compute_regression_loss`` (bar-distribution NLL + CRPS by default).
"""

from __future__ import annotations

import logging
import math
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.finetuning import RegressorBatch
from tabpfn.finetuning.data_util import (
  get_preprocessed_dataset_chunks,
  meta_dataset_collator,
)
from tabpfn.finetuning.finetuned_regressor import _compute_regression_loss
from tabpfn.finetuning.train_util import (
  get_and_init_optimizer,
  get_cosine_schedule_with_warmup,
)
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from npcc.finetune.config import FinetuneConfig
from npcc.finetune.save_load import save_finetuned
from npcc.priors import ConditionalCopulaPrior, sample_pool

logger = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
  """Human-readable duration, e.g. ``2h03m07s`` / ``5m12s`` / ``9s``."""
  total = int(seconds)
  h, rem = divmod(total, 3600)
  m, s = divmod(rem, 60)
  if h:
    return f"{h}h{m:02d}m{s:02d}s"
  if m:
    return f"{m}m{s:02d}s"
  return f"{s}s"


def _transform_targets(y: np.ndarray, transform: str, eps: float) -> np.ndarray:
  """Apply the support transform to copula-scale targets (matches ``_transform_y``)."""
  z = torch.as_tensor(np.asarray(y, float), dtype=torch.float64)
  if transform == "identity":
    pass
  elif transform == "logit":
    zc = z.clamp(eps, 1.0 - eps)
    z = torch.log(zc / (1.0 - zc))
  elif transform == "probit":
    zc = z.clamp(eps, 1.0 - eps)
    z = math.sqrt(2.0) * torch.erfinv(2.0 * zc - 1.0)
  else:
    raise ValueError(f"Unknown transform: {transform}")
  return z.numpy()


def build_base_regressor(cfg: FinetuneConfig) -> TabPFNRegressor:
  """Instantiate the base TabPFN-v2.5 regressor in batched fine-tuning mode."""
  return TabPFNRegressor.create_default_for_version(
    ModelVersion.V2_5,
    device=cfg.device,
    n_estimators=cfg.n_estimators_finetune,
    ignore_pretraining_limits=cfg.ignore_pretraining_limits,
    fit_mode="batched",
    differentiable_input=False,
  )


def _batch_loss(
  reg: TabPFNRegressor,
  batch: RegressorBatch,
  cfg: FinetuneConfig,
  device: torch.device,
) -> torch.Tensor:
  """Replicate ``FinetunedTabPFNRegressor._forward_with_loss`` for one batch."""
  _, per_estim_logits, _ = reg.forward(batch.X_query)
  logits_qbel = torch.stack(per_estim_logits, dim=2)
  q, b, e, ln = logits_qbel.shape
  logits_bql = logits_qbel.permute(1, 2, 0, 3).reshape(b * e, q, ln)
  targets_bq = batch.y_query.repeat(b * e, 1).to(device)
  return _compute_regression_loss(
    logits_BQL=logits_bql,
    targets_BQ=targets_bq,
    bardist_loss_fn=batch.znorm_space_bardist,
    ce_loss_weight=cfg.ce_loss_weight,
    crps_loss_weight=cfg.crps_loss_weight,
    crls_loss_weight=cfg.crls_loss_weight,
    mse_loss_weight=cfg.mse_loss_weight,
  )


def meta_train(
  cfg: FinetuneConfig, prior: ConditionalCopulaPrior | None = None
) -> Path:
  """Meta-train TabPFN on the prior and save the fine-tuned checkpoint.

  Returns the path to the saved ``.pth`` (loadable via
  ``PFNRBicop(finetuned_path=...)``).
  """
  prior = prior or ConditionalCopulaPrior()
  device = torch.device(cfg.device)
  rng = np.random.default_rng(cfg.random_state)

  reg = build_base_regressor(cfg)
  reg._initialize_model_variables()
  model = reg.model_
  model.to(device)
  if len(reg.models_) != 1:
    raise RuntimeError(
      "expected a single shared transformer (save_tabpfn_model precondition); "
      f"got {len(reg.models_)}."
    )
  if cfg.use_activation_checkpointing:
    # setattr: the model's custom __setattr__ is not typed for this flag.
    setattr(model, "recompute_layer", True)

  optimizer = get_and_init_optimizer(
    model.parameters(), cfg.learning_rate, cfg.weight_decay, device=cfg.device
  )

  pool = sample_pool(
    prior,
    cfg.n_datasets,
    cfg.rows_per_dataset,
    rng,
    both_directions=cfg.both_directions,
  )
  x_raw = [w for w, _ in pool]
  y_raw = [_transform_targets(y, cfg.transform, cfg.eps) for _, y in pool]
  query_size = max(2, int(round(cfg.rows_per_dataset * cfg.query_fraction)))

  scheduler: LambdaLR | None = None
  best_loss = math.inf
  best_state: dict[str, torch.Tensor] | None = None
  global_step = 0
  total_steps = 0
  start_time = time.monotonic()

  for epoch in range(cfg.epochs):
    seed = cfg.random_state + epoch
    split_fn = partial(
      train_test_split, test_size=query_size, random_state=seed
    )
    collection = get_preprocessed_dataset_chunks(
      calling_instance=reg,
      X_raw=x_raw,
      y_raw=y_raw,
      split_fn=split_fn,
      max_data_size=cfg.max_data_size,
      model_type="regressor",
      equal_split_size=False,
      data_shuffle_seed=seed,
      preprocessing_random_state=cfg.random_state,
    )
    loader = DataLoader(
      collection,
      batch_size=1,
      collate_fn=meta_dataset_collator,
      shuffle=True,
      generator=torch.Generator().manual_seed(seed),
    )
    if scheduler is None:
      total_steps = max(1, len(loader) * cfg.epochs)
      if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
      scheduler = LambdaLR(
        optimizer,
        get_cosine_schedule_with_warmup(
          total_steps, int(total_steps * cfg.warmup_frac)
        ),
      )
      logger.info(
        "meta-training %d datasets x %d epochs = %d steps on %s",
        len(loader),
        cfg.epochs,
        total_steps,
        cfg.device,
      )

    epoch_loss, n_batches = 0.0, 0
    for batch in loader:
      optimizer.zero_grad()
      reg.raw_space_bardist_ = batch.raw_space_bardist
      reg.znorm_space_bardist_ = batch.znorm_space_bardist
      reg.fit_from_preprocessed(
        batch.X_context, batch.y_context, batch.cat_indices, batch.configs
      )
      loss = _batch_loss(reg, batch, cfg, device)
      loss.backward()
      if cfg.grad_clip_value is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_value)
      optimizer.step()
      scheduler.step()
      step_loss = float(loss.detach().item())
      epoch_loss += step_loss
      n_batches += 1
      global_step += 1
      if cfg.log_every and global_step % cfg.log_every == 0:
        elapsed = time.monotonic() - start_time
        eta = elapsed / global_step * (total_steps - global_step)
        logger.info(
          "step %d/%d (%.0f%%) | loss %.4f | elapsed %s | eta %s",
          global_step,
          total_steps,
          100.0 * global_step / total_steps,
          step_loss,
          _fmt_duration(elapsed),
          _fmt_duration(eta),
        )
      if cfg.max_steps is not None and global_step >= cfg.max_steps:
        break

    mean_loss = epoch_loss / max(1, n_batches)
    logger.info(
      "epoch %d/%d done: mean_loss=%.4f (steps=%d, elapsed=%s)",
      epoch + 1,
      cfg.epochs,
      mean_loss,
      global_step,
      _fmt_duration(time.monotonic() - start_time),
    )
    if mean_loss < best_loss:
      best_loss = mean_loss
      best_state = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
      }
    if cfg.max_steps is not None and global_step >= cfg.max_steps:
      break

  if best_state is not None:
    model.load_state_dict(best_state)
  return save_finetuned(reg, cfg.output_dir / cfg.checkpoint_name)
