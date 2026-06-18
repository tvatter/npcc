"""
save_load.py — persist a fine-tuned TabPFN regressor for reuse by ``PFNRBicop``.

Wraps ``tabpfn.model_loading.save_tabpfn_model`` (which writes the model weights
*and* the bar-distribution criterion into one ``.pth``), enforcing the
``"v2.5"``-in-filename rule: TabPFN's ``_resolve_model_version`` reads the
architecture version from the checkpoint filename, so a name without ``"v2.5"``
would silently load as V2.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
  from tabpfn import TabPFNClassifier, TabPFNRegressor


def save_finetuned(regressor: object, path: str | Path) -> Path:
  """Save ``regressor`` to ``path`` (filename must contain ``"v2.5"``)."""
  from tabpfn.model_loading import save_tabpfn_model

  path = Path(path)
  if "v2.5" not in path.name:
    raise ValueError(
      f"checkpoint filename {path.name!r} must contain 'v2.5'; otherwise TabPFN "
      "loads it as the V2 architecture and breaks the v2.5 lock."
    )
  path.parent.mkdir(parents=True, exist_ok=True)
  save_tabpfn_model(
    cast("TabPFNRegressor | TabPFNClassifier", regressor), str(path)
  )
  return path
