# CLAUDE.md

## Project

Neural Pair-Copulas Constructions (NPCCs) for multivariate conditional density estimation.
Python >= 3.11, managed with `uv`.

## Commands

```bash
# Install (pick one extra: cpu, cu126, cu128, cu130)
uv sync --extra cpu

# Lint / format
uv run ruff check . --select ANN --fix
uv run ruff format .

# Type check
uv run ty check

# Tests
uv run pytest tests/ -v -n auto
uv run pytest tests/ --cov=src/npcc --cov-report=term-missing -v -n auto
```

## Code style

- **Formatter**: ruff — line length 80, indent width 2, target Python 3.11.
- **Linter**: ruff with `ANN` rules enabled — all public functions and methods must have type annotations.
- **Type checker**: ty — always run before committing; zero errors required.
- Use modern Python syntax: `X | Y` unions, `list[T]` / `dict[K, V]` built-in generics, `match` statements where appropriate.
- Prefer `pathlib.Path` over `os.path`.
- No `Any` unless unavoidable and explicitly annotated with a comment explaining why.

## Testing

- All new code must have corresponding tests under `tests/`.
- Tests run in parallel (`-n auto`); do not rely on shared mutable state.
- Use `pytest` fixtures; avoid bare `assert` on floats — use `pytest.approx` or numpy equivalents.

## Dependencies

- Add runtime dependencies via `uv add <pkg>`.
- Add dev/tooling dependencies via `uv add --group dev <pkg>`.
- Do not pin exact versions unless a compatibility issue requires it; prefer lower bounds (`>=`).
- PyTorch extras (`cpu`, `cu126`, `cu128`, `cu130`) are mutually exclusive — never mix them.

## General rules

- No commented-out code committed to the repo.
- No `print` statements in library code — use `logging`.
- Keep functions small and focused; avoid side effects in pure computation functions.
- Do not add backwards-compatibility shims or feature flags unless explicitly requested.
