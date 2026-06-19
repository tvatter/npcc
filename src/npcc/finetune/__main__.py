"""
CLI for TabPFN meta-training on the conditional-copula prior.

Examples
--------
    uv run python -m npcc.finetune --smoke --output-dir /tmp/npcc_smoke
    uv run python -m npcc.finetune --output-dir runs/exp1 --n-datasets 512 --epochs 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from npcc.finetune.config import FinetuneConfig
from npcc.finetune.loop import meta_train
from npcc.priors import ConditionalCopulaPrior


def _build_config(args: argparse.Namespace) -> FinetuneConfig:
  if args.smoke:
    cfg = FinetuneConfig.smoke(output_dir=args.output_dir)
    cfg.freeze_backbone = args.freeze_backbone
    return cfg
  return FinetuneConfig(
    n_datasets=args.n_datasets,
    rows_per_dataset=args.rows_per_dataset,
    epochs=args.epochs,
    learning_rate=args.learning_rate,
    n_estimators_finetune=args.n_estimators,
    freeze_backbone=args.freeze_backbone,
    device=args.device,
    transform=args.transform,
    random_state=args.seed,
    output_dir=Path(args.output_dir),
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--smoke", action="store_true", help="tiny CPU sanity run"
  )
  parser.add_argument("--output-dir", default="runs/default")
  parser.add_argument("--n-datasets", type=int, default=512)
  parser.add_argument("--rows-per-dataset", type=int, default=2000)
  parser.add_argument("--epochs", type=int, default=20)
  parser.add_argument("--learning-rate", type=float, default=1e-5)
  parser.add_argument("--n-estimators", type=int, default=2)
  parser.add_argument(
    "--freeze-backbone",
    action="store_true",
    help="train only the decoder head (parameter-efficient fine-tuning)",
  )
  parser.add_argument("--device", default="cuda")
  parser.add_argument(
    "--transform", default="logit", choices=["identity", "logit", "probit"]
  )
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
  cfg = _build_config(args)
  path = meta_train(cfg, ConditionalCopulaPrior())
  logging.getLogger(__name__).info("saved fine-tuned checkpoint to %s", path)


if __name__ == "__main__":
  main()
