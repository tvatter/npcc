"""TabPFN meta-training on the conditional-copula prior (opt-in, GPU for real runs).

:class:`FinetuneConfig` is light-weight and always importable;
:func:`meta_train` / :func:`build_base_regressor` / :func:`save_finetuned` are
loaded lazily so that ``import npcc.finetune`` does not eagerly pull the heavy
``tabpfn.finetuning`` machinery until it is actually used.
"""

from __future__ import annotations

from typing import Any

from npcc.finetune.config import FinetuneConfig

__all__ = [
  "FinetuneConfig",
  "build_base_regressor",
  "meta_train",
  "save_finetuned",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401 — module __getattr__ idiom
  if name in {"meta_train", "build_base_regressor"}:
    from npcc.finetune import loop

    return getattr(loop, name)
  if name == "save_finetuned":
    from npcc.finetune.save_load import save_finetuned

    return save_finetuned
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
