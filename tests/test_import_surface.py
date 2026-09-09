"""The import surface: which layer may reach which, and what `import npcc` costs.

Two invariants, both of which the package documents and neither of which
anything checked before.

``npcc.core.registry``'s module docstring promises that "``import npcc`` and
:func:`available_backends` never import NGBoost / TabICL / scikit-learn",
which is what makes the optional extras optional. Nothing enforced it, and one
eager import anywhere in the chain -- a convenience re-export, a type
annotation moved out from behind ``TYPE_CHECKING`` -- would turn an extra into
a hard dependency without failing anything.

And the layers: ``npcc.experiments`` is behind its own extra and sits above
``npcc.core``, so the edge points one way. A stray import the other way would
put pandas and matplotlib on the path of every ``import npcc``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src" / "npcc"

#: Third-party packages reachable **only** through an optional extra.
#: Importing `npcc` must not pull in any of them.
#:
#: `matplotlib`, `pandas` and `sklearn` are absent on purpose, and it is worth
#: saying why: each is also a transitive hard dependency -- `matplotlib` of
#: `pyvinecopulib`, `pandas` and `sklearn` of `tabpfn` -- so `import npcc`
#: loads them whatever this package does, and asserting otherwise would pin a
#: fact about someone else's dependency tree.
_OPTIONAL = (
  "catboost",
  "ngboost",
  "pytabkit",
  "synthefy_nori",
  "tabicl",
  "xgboost",
)


def _imported_by(statements: str) -> set[str]:
  """Top-level third-party modules present after running ``statements``.

  A fresh interpreter, because `sys.modules` in this one already holds
  everything the test suite imported.

  Parameters
  ----------
  statements : str
      Python source to run before the check.

  Returns
  -------
  set of str
      The names from ``_OPTIONAL`` that ended up imported.
  """
  probe = (
    f"{statements}\n"
    "import sys\n"
    f"present = [n for n in {_OPTIONAL!r} if n in sys.modules]\n"
    "print(' '.join(present))\n"
  )
  result = subprocess.run(
    [sys.executable, "-c", probe],
    capture_output=True,
    text=True,
    check=True,
    cwd=_ROOT,
  )
  return set(result.stdout.split())


def test_importing_npcc_needs_no_optional_extra() -> None:
  assert _imported_by("import npcc") == set()


def test_listing_the_backends_needs_no_optional_extra() -> None:
  """Naming what is available must not construct any of it."""
  assert _imported_by("import npcc; npcc.available_backends()") == set()


def test_validating_backend_kwargs_needs_no_optional_extra() -> None:
  """Building fit controls validates the backend name, and stops there."""
  assert (
    _imported_by(
      "from npcc import FitControlsRosenblattBicop\n"
      "FitControlsRosenblattBicop(backend='ngboost')"
    )
    == set()
  )


def _all_imports(path: Path) -> set[str]:
  """Every module named by an import in ``path``, at any scope, dotted.

  Parameters
  ----------
  path : pathlib.Path
      The Python file to read.

  Returns
  -------
  set of str
      Full dotted module names, so a layer prefix can be matched.
  """
  tree = ast.parse(path.read_text())
  names: set[str] = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      names.update(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
      names.add(node.module)
  return names


def _module_scope_imports(path: Path) -> set[str]:
  """Top-level module names imported at module scope in ``path``.

  Parameters
  ----------
  path : pathlib.Path
      The Python file to read.

  Returns
  -------
  set of str
      One entry per module-scope import, as its first dotted component.
  """
  tree = ast.parse(path.read_text())
  nested = {
    node
    for parent in ast.walk(tree)
    if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
    for node in ast.walk(parent)
  }
  names: set[str] = set()
  for node in ast.walk(tree):
    if node in nested:
      continue
    if isinstance(node, ast.Import):
      names.update(alias.name.split(".")[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
      names.add(node.module.split(".")[0])
  return names


def test_the_registry_imports_no_backend_module_at_module_scope() -> None:
  """The deferral that makes the extras optional lives in the factories.

  Each adapter module imports its own third-party package at module scope --
  which is right, the module being that package's adapter -- so what keeps
  `import npcc` cheap is entirely that the registry reaches those modules only
  from inside a factory body. One module-scope import here promotes every
  extra to a hard dependency at once.
  """
  eager = {
    name
    for name in _module_scope_imports(_SRC / "core" / "registry.py")
    if name == "npcc"
  }
  registry = (_SRC / "core" / "registry.py").read_text()
  tree = ast.parse(registry)
  nested = {
    node
    for parent in ast.walk(tree)
    if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
    for node in ast.walk(parent)
  }
  module_scope_backends = {
    node.module
    for node in ast.walk(tree)
    if isinstance(node, ast.ImportFrom)
    and node not in nested
    and (node.module or "").startswith("npcc.core.backends")
  }

  assert module_scope_backends == set()
  assert eager <= {"npcc"}


@pytest.mark.parametrize(
  "path",
  sorted((_SRC / "core" / "backends").glob("*.py")),
  ids=lambda p: p.name,
)
def test_every_adapter_is_reached_only_through_a_factory(path: Path) -> None:
  """An adapter that imports an extra must not be imported eagerly anywhere.

  ``npcc/__init__.py`` re-exports the two TabPFN adapters, which is why they
  are exempt: `tabpfn` is a hard dependency, not an extra.
  """
  if path.name in {"__init__.py", "tabpfn_common.py"}:
    pytest.skip("no third-party import of its own")

  needs_extra = _module_scope_imports(path) & set(_OPTIONAL)
  if not needs_extra:
    pytest.skip("imports no optional extra")

  module = f"npcc.core.backends.{path.stem}"
  eager = [
    other
    for other in sorted(_SRC.rglob("*.py"))
    if other != path and module in _module_scope_imports(other)
  ]

  assert eager == [], f"{module} imported at module scope by {eager}"


@pytest.mark.parametrize(
  "path",
  sorted((_SRC / "core").rglob("*.py")),
  ids=lambda p: str(p.relative_to(_SRC)),
)
def test_core_never_imports_the_layer_above_it(path: Path) -> None:
  """``npcc.experiments`` is above ``npcc.core``; the edge points one way.

  Read from the AST rather than as a substring: a docstring or comment naming
  the layer above is documentation -- a branch in ``core`` may exist because of
  a data shape the experiments package produces, and saying so is useful --
  while an ``import`` is the dependency this forbids.
  """
  imported = {
    name
    for name in _all_imports(path)
    if name == "npcc.experiments" or name.startswith("npcc.experiments.")
  }

  assert imported == set(), (
    f"{path.relative_to(_ROOT)} imports {sorted(imported)}"
  )


def test_every_core_name_is_also_reachable_from_the_top() -> None:
  """The two public surfaces agree, so either spelling works.

  ``npcc`` carries more -- the concrete backend leaves, which live in
  ``npcc.core.backends`` rather than in ``npcc.core`` itself.
  """
  import npcc
  import npcc.core

  assert set(npcc.core.__all__) <= set(npcc.__all__)
  assert set(npcc.__all__) - set(npcc.core.__all__) == {
    "TabPFNCriterionBackend",
    "TabPFNQuantileBackend",
  }


@pytest.mark.parametrize("module", ["npcc", "npcc.core"])
def test_the_public_lists_are_sorted_and_resolve(module: str) -> None:
  """A name in ``__all__`` that does not resolve is a broken re-export."""
  import importlib

  imported = importlib.import_module(module)
  names = list(imported.__all__)

  assert names == sorted(names)
  for name in names:
    assert hasattr(imported, name), name
