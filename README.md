# Neural Pair-Copulas Constructions (NPCCs)

Minimal setup notes for this project.

## uv dependency groups

- `dev`: linting, formatting, type checking, and tests (`ruff`, `ty`, `pytest`, `pytest-cov`, `pytest-xdist`).
- `interactive`: notebook/data/plotting tooling (`pandas`, `jupyter`, `matplotlib`, etc.).
- `uv` defaults include both groups (`default-groups = ["dev", "interactive"]`), so `uv sync` installs them.

## Package extras (PyTorch variant)

- Available extras: `cpu`, `cu126`, `cu128`, `cu130`.
- Extras are mutually exclusive (only one at a time).
- Each extra selects a matching PyTorch index/source.

Examples:

```bash
uv sync --extra cpu
uv sync --extra cu128
```

## Typical commands

```bash
uv run ruff check . --select ANN
uv run ruff check . --select ANN --fix
uv run ruff format .
uv run ty check
uv run pytest tests/ -v -n auto
uv run pytest tests/ --cov=src/ome --cov-report=term-missing -v -n auto
```
