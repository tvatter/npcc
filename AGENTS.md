# AGENTS.md

Normative engineering spec for contributors and coding agents working on this
repository: scope, invariants, module boundaries, public APIs, and acceptance
criteria.

## Project overview

`npcc` ("Neural Pair-Copulas Constructions") estimates **conditional bivariate
copula densities** nonparametrically, backed by
[TabPFN](https://github.com/PriorLabs/TabPFN). The eventual goal is
pair-copula constructions for multivariate conditional density estimation; the
package currently implements the bivariate building block.

The method (see `README.md` for the full derivation) rests on the **Rosenblatt
factorisation**. For copula scores `U, V ∈ (0, 1)` and covariates `X ∈ ℝ^p`, the
uniform-margin property collapses the bivariate conditional density into a single
univariate conditional density:

```text
c(u, v | x) = f_{V | U, X}(v | u, x).
```

So estimating a conditional bicop density reduces to estimating a univariate
conditional density, which a flexible distributional regressor like TabPFN
provides. The package exposes:

- one outer estimator — **`PFNRBicop`** (TabPFN-Rosenblatt bicop);
- two interchangeable inner univariate-conditional-predictive-distribution
  wrappers — **`TabPFNCriterionDistribution1D`** (TabPFN's native binned
  distribution head; default) and **`TabPFNQuantileDistribution1D`** (numerical
  inversion of the predicted quantile table; model-agnostic) — both subclassing
  the abstract **`TabPFNDistribution1D`**.

Key modeling choices:

- **Symmetric variant** (`symmetric=True`, default): the single-direction
  factorisation is ordering-dependent (it satisfies `∫ ĉ dv = 1` by construction
  but generally not `∫ ĉ du = 1`), so the symmetric estimator also fits the
  reverse direction `f_{U | V, X}` and averages, reducing directional bias.
- **Support transforms**: copula scores live in `(0, 1)`, so the inner estimator
  fits TabPFN on a transformed target `Z = T(Y)` and back-transforms densities by
  the change-of-variables Jacobian. `transform="logit"` (default), `"probit"`,
  or `"identity"`.
- **Marginal projection**: neither variant imposes exact uniform copula margins;
  the optional Sinkhorn / iterative-proportional-fitting projection
  (`sinkhorn_iters`) corrects the density on a grid to restore both margins.

**Numerical backend.** PyTorch in **float64** (chosen for precision in the
trapezoidal integrators and quantile-table inversions); `float32` is used only
at the TabPFN `criterion` boundary, where the head consumes the regressor's
float32 logits. Public methods accept NumPy arrays or torch tensors and return
the same type the caller passed in (see `_common._normalize_inputs` /
`_wrap_output`); device follows the input or resolves to CUDA-if-available.

Two important remarks about the repository's design philosophy:

- This repository is under active development. For new features and ongoing
  design work, internal and public APIs may change when doing so improves
  correctness, clarity, or architecture. Do not preserve backward compatibility
  by default unless the task explicitly requires it.
- This repository is quantitatively sensitive: small changes can produce
  mathematically incorrect behavior even when the code looks structurally sound.
  Treat all conventions, patterns, and documented formulas (transforms and their
  Jacobians, the Rosenblatt factorisation, h-function conventions, the Sinkhorn
  projection) as correctness-critical.

`CLAUDE.md` and `.github/copilot-instructions.md` (if present) are thin pointers
into this file.

## Scope

In scope:

- conditional bivariate copula *density* estimation (not copula fitting or
  sampling), backed by TabPFN;
- the Rosenblatt factorisation and the symmetric (both-directions) averaging;
- the two density-recovery methods (`criterion` and `quantiles`), interchangeable
  via `PFNRBicop(method=...)`, sharing the same outer API;
- the support transforms (`logit` / `probit` / `identity`) and their Jacobians;
- pointwise and Cartesian-grid evaluation of the density (`pdf` / `pdf_grid`),
  joint CDF (`cdf` / `cdf_grid`), and the log-density (`log_pdf`);
- the h-functions (`hfunc1` / `hfunc2`, `pyvinecopulib` convention), Kendall's
  `tau`, the conditional-CDF diagnostic, the `pyvinecopulib`-compatible
  `as_bicop` adapter, and the `plot` helper (thin `pyvinecopulib` / `matplotlib`
  wrapper);
- the optional Sinkhorn / IPF projection (`sinkhorn_iters`) for uniform margins;
- the standalone univariate conditional predictive distributions
  (`TabPFNDistribution1D` subclasses), usable outside the copula context;
- a **rich prior over conditional copula densities** (`npcc.priors`):
  mixtures over pyvinecopulib families (incl. Student-t and BB) with rotations,
  τ(x) Fisher-z links *and* native-parameter(x) links, and the unconditional
  (constant-parameter) case — deterministic given a seed; defines the
  meta-training target;
- **TabPFN meta-training / fine-tuning** on that prior (`npcc.finetune`):
  specialising the inner regressor's weights to copula-shaped data, saved as a
  checkpoint loaded back via `PFNRBicop(finetuned_path=...)`;
- **evaluation-only diagnostics** (`npcc.eval`): ISE/IAE/KL density + h-function
  metrics, conditional-grid evaluation, and margin calibration.

Out of scope (for now):

- dimensions `> 2`, full vine / pair-copula constructions (the eventual NPCC);
- copula family fitting and sampling;
- tabular reporting and *benchmark* harnesses (these live in `notebooks/` and
  `scripts/`, not the library — but the training **prior** and **eval** metrics
  are in-scope library code, since they define and measure the learning target);
- backends other than TabPFN for the inner estimator (the `quantiles` method is
  recovery-agnostic, but the wrappers target TabPFN).

## Package structure

All package code lives in `src/npcc/`.

```text
src/npcc/
  __init__.py     # Curated public API (re-exports + __all__)
  _common.py      # Private torch-aware array helpers shared across modules:
                  #   device/tensor coercion (_resolve_device/_to_tensor),
                  #   _normalize_inputs/_wrap_output round-trip, _as_2d/_check_uv,
                  #   _logit, _torch_interp(_batched)/_torch_gradient_1d
  tabpfn_distribution1d.py            # TabPFNDistribution1D (ABC): support
                  #   transforms + Jacobians, shared fit, abstract pdf/cdf/icdf
  tabpfn_criterion_distribution1d.py  # TabPFNCriterionDistribution1D: TabPFN
                  #   native binned head; adds pdf_grid/cdf_grid fast paths
  tabpfn_quantile_distribution1d.py   # QuantileGridConfig +
                  #   TabPFNQuantileDistribution1D: quantile-table inversion
  pfnr_bicop.py   # PFNRBicop outer estimator (+ _sinkhorn_project, _BicopAdapter);
                  #   accepts finetuned_path -> threaded to both inner directions
  eval.py         # Evaluation-only diagnostics (ISE/IAE/KL, conditional grids,
                  #   margin calibration); estimator-free, numpy only
  priors/         # Rich prior over conditional copula densities (numpy + pv):
    families.py   #   FamilySpec registry (tau<->param + native-param bounds)
    links.py      #   CovariateLink (Fisher-z / native links) + affine_to_interval
    components.py #   ParameterProcess, CopulaComponent, CopulaMixture
    prior.py      #   ConditionalCopulaPrior + sample_spec -> PriorDraw
    datasets.py   #   sample_pool / direction_datasets (TabPFN-ready (W, y) lists)
  finetune/       # TabPFN meta-training harness (opt-in; GPU for real runs):
    config.py     #   FinetuneConfig (+ .smoke() CPU preset)
    loop.py       #   meta_train(): low-level get_preprocessed_dataset_chunks loop
    save_load.py  #   save_finetuned() (enforces 'v2.5' in the filename)
    evaluate.py   #   ClaytonReference + conditional_report (before/after metrics)
    __main__.py   #   CLI: `uv run python -m npcc.finetune [--smoke]`
```

Test suite location: `tests/` (one test module per source module/subpackage,
plus `conftest.py` providing TabPFN fakes/fixtures). Worked end-to-end demos and
*benchmark* harnesses live in `notebooks/` and `scripts/`, not in the library.

### Module boundaries and public API

- `_common` owns torch-aware array plumbing only; every export is private
  (leading `_`) and not part of the public API. Depended on by the estimator
  modules; depends on nothing in the package.
- `tabpfn_distribution1d.TabPFNDistribution1D` is the `abc.ABC` base: it owns the
  support transforms (and their inverses/Jacobians), the construction-time
  fields (`transform` / `eps` / `device` / `model_kwargs` / `model_`), and the
  shared `fit`; `pdf` / `cdf` / `icdf` are abstract.
- `tabpfn_criterion_distribution1d` and `tabpfn_quantile_distribution1d` are the
  two interchangeable inner estimators. Only the criterion subclass exposes the
  `pdf_grid` / `cdf_grid` Cartesian-product fast paths (one TabPFN forward pass
  per conditioning row, reused across all evaluation points).
- `pfnr_bicop.PFNRBicop` composes one inner distribution (or two, when
  `symmetric=True`) and owns the Rosenblatt factorisation, symmetric averaging,
  the Sinkhorn projection (`_sinkhorn_project`), the trapezoidal CDF integrators,
  the h-functions, `tau`, the diagnostics, the `pyvinecopulib`-compatible
  `as_bicop` / `plot` surface, and the `finetuned_path` weight-loading seam.
- `priors` owns the conditional-copula prior; it depends only on `numpy` +
  `pyvinecopulib` (no torch), is deterministic given a `numpy` `Generator`, and is
  never imported by the estimators. `eval` owns evaluation-only diagnostics
  (numpy only) and is never imported by the estimators either.
- `finetune` owns the meta-training harness; it lazily imports the heavy
  `tabpfn.finetuning` internals (so `import npcc.finetune` stays cheap until
  `meta_train` is used) and consumes the prior. Its only library-facing contract
  is the checkpoint path read by `PFNRBicop(finetuned_path=...)`.

The curated public API (`__init__.__all__`) re-exports the estimators
(`PFNRBicop`, `TabPFNDistribution1D` and its two subclasses, `QuantileGridConfig`),
the prior entry points (`ConditionalCopulaPrior`, `PriorDraw`, `sample_spec`,
`sample_pool`), and the `eval` diagnostics. `npcc.finetune` is intentionally
*not* re-exported at top level (keeps `import npcc` light); use
`from npcc.finetune import meta_train`. `_common` is internal. Tests import from
the module/subpackage paths and may reach internals directly (`_common` helpers,
`_sinkhorn_project`, `priors`/`finetune` privates) to pin numerical behavior.

### Mathematical conventions and notation

These relations are correctness-critical (see the "quantitatively sensitive"
remark above) and mirror `README.md`:

- `U, V ∈ (0, 1)` — copula scores; `X ∈ ℝ^p` — covariates; `x=None` ⇒
  unconditional fit.
- Rosenblatt factorisation: `c(u, v | x) = f_{V | U, X}(v | u, x)`, with features
  `W = [u, x]` fed to the inner regressor (or `[v, x]` for the reverse direction).
- Symmetric variant: `ĉ_S(u, v | x) = ½·f_{V|U,X}(v|u,x) + ½·f_{U|V,X}(u|v,x)`.
- Uniform-margin identity: `f_{V|U,X}(v|u,x) = f_{U|V,X}(u|v,x) = c(u,v|x)` (the
  conditioner's marginal density is 1). So both Rosenblatt directions share the
  *same* target functional — one sampled `c(u,v|x)` trains both directions, and
  one fine-tuned checkpoint serves both inner regressors.
- Support transform `Z = T(Y)`, back-transform `f_Y(y|w) = f_Z(T(y)|w)·|dT/dy|`.
  CDFs and quantiles need no correction (monotone transforms preserve them):
    - `logit`: `f_Y = f_Z(logit(y)) / (y·(1 − y))`;
    - `probit`: `f_Y = f_Z(Φ⁻¹(y)) / φ(Φ⁻¹(y))`;
    - `identity`: `f_Y = f_Z(y)`.
- h-functions follow the `pyvinecopulib` convention — `h_i` conditions on the
  `i`-th argument: `hfunc1(u, v | x) = ∂C/∂u = F_{V|U,X}(v|u,x)` (always
  available); `hfunc2(u, v | x) = ∂C/∂v = F_{U|V,X}(u|v,x)` (requires
  `symmetric=True`).
- Margins: `∫ ĉ(u, v|x) dv = 1` holds by construction; `∫ ĉ du = 1` does not
  (unless symmetric, and even then only approximately). The optional Sinkhorn /
  IPF projection (`sinkhorn_iters`) enforces both on a grid — for `pdf_grid`
  directly on the evaluated grid; for pointwise `pdf` on an internal projection
  grid (criterion: TabPFN bar-distribution borders; quantiles: the predefined
  quantile grid) then interpolated back to the query points.
- `tau(x_row)` uses `pyvinecopulib`'s recipe (quasi-random `ghalton` +
  inverse-Rosenblatt + `wdm`), matching `pv.KernelBicop::parameters_to_tau`.

Runtime constraints worth knowing: `PFNRBicop` is locked to **TabPFN-v2.5** (v2.6
has reported regressions; override via `model_kwargs`); TabPFN refuses `> 1000`
samples on CPU unless a CUDA build is used or the limit is lifted; the first
`fit` downloads the checkpoint and requires `TABPFN_TOKEN` (loaded from `.env`).

Two correctness-critical contracts for fine-tuning:

- **Checkpoint filename must contain `"v2.5"`** — TabPFN's `_resolve_model_version`
  reads the architecture version from the filename and silently falls back to V2
  otherwise. `save_finetuned` enforces this on save and `TabPFNDistribution1D.fit`
  warns on load.
- **Transform consistency** — `FinetuneConfig.transform` must equal the
  inference-time `PFNRBicop(transform=...)`; the prior emits raw `(0,1)` targets
  and the harness applies the support transform, so a mismatch yields silently
  wrong densities. Real fine-tuning runs require a GPU; the `--smoke` preset runs
  the whole pipeline on CPU in seconds.

## Tooling

- Python 3.11+, environment via `uv`. Install with exactly one PyTorch extra:
  `uv sync --extra cpu` (or `cu126` / `cu128` / `cu130` / `cu132` — mutually
  exclusive, never mix).
- Lint + format: `ruff` (line length 80, indent width 2, target `py311`, `ANN`
  ruleset enabled — all public functions/methods must be annotated; `notebooks/`
  are exempt from `ANN`).
- Type check: `ty` (zero errors required).
- Unit test: `pytest` with `pytest-cov` and `pytest-xdist` for parallelism.

Always prefix Python tools with `uv run`.

Validation sequence for any behavior change (run in this order):

```bash
uv run ruff format .
uv run ruff check . --fix
uv run ruff check . --select ANN --fix
uv run ty check
uv run pytest tests/ --cov=src/npcc --cov-report=term-missing -v -n auto
```

Coverage must stay at or above the current level; new code must come with focused
tests, not blanket exclusions. The default unit-test surface is hermetic: TabPFN
is faked in `tests/conftest.py`, and the live integration test
(`tests/test_pfnr_bicop.py::test_real_tabpfn_smoke`) auto-skips when
`TABPFN_TOKEN` is unset or the PriorLabs license endpoint is unreachable.

Other useful commands:

```bash
uv venv
uv sync
# Run the demo notebook end-to-end
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/pfnr_bicop_demo.ipynb
# Fine-tuning: tiny CPU smoke (seconds) / real GPU run
uv run python -m npcc.finetune --smoke --output-dir /tmp/npcc_smoke
uv run python -m npcc.finetune --output-dir runs/exp1 --n-datasets 512 --epochs 20
```

For performance work, profile first and optimize only demonstrated hotspots;
preserve all quantitative semantics and documented invariants.

## Working on this repo

### Inspection order

Before making changes, inspect in this order:

1. `AGENTS.md` — project overview, scope, package structure, working guidelines,
   module boundaries.
2. `README.md` — high-level motivation, the Rosenblatt derivation, transforms,
   the two recovery methods, and the public API surface.
3. `src/npcc/...` — implementation and local patterns.
4. `tests/...` — expected behavior, edge cases, and the TabPFN fakes.

Prefer matching existing local patterns over introducing new ones.

### Definition of done

For behavior changes or new features:

- keep code compact, and diffs minimal and scoped to the task;
- avoid unnecessary API churn, but do not preserve backward compatibility by
  default unless the task explicitly requires it;
- add or update focused tests, but keep them compact; prefer
  extending/refactoring existing helpers, fixtures, and parametrized tests over
  duplicating logic;
- update docstrings for public behavior changes;
- run the validation sequence from [Tooling](#tooling);
- do not introduce undocumented conventions.

### Coding conventions

- Use modern Python features and idioms (3.11+): `X | Y` unions, `list[T]` /
  `dict[K, V]` built-in generics, `match` where appropriate; prefer
  `pathlib.Path` over `os.path`. Prefer readability and correctness over
  cleverness or terseness.
- Public functions and methods must include type annotations (`ANN`) and
  meaningful docstrings (purpose, params, returns, raised errors). No `Any`
  unless unavoidable and annotated with a comment explaining why.
- No commented-out code committed to the repo; no `print` in library code — use
  `logging`. Keep functions small and focused; avoid side effects in pure
  computation functions.
- Source modules (`src/npcc/**`) import explicit symbols (avoid module-object
  imports for local paths).
- Tests live under `tests/`, must be parallel-safe (no shared mutable state under
  `-n auto`), use `pytest` fixtures, and avoid bare `assert` on floats — use
  `pytest.approx` or numpy equivalents.
- Dependencies: add runtime deps via `uv add <pkg>`, dev/tooling via
  `uv add --group dev <pkg>`; prefer lower bounds (`>=`) over exact pins; never
  mix PyTorch extras. `pyvinecopulib` is pinned to its **`dev` branch (1.0.0)**
  via `[tool.uv.sources]` (built from source — needs a C++ toolchain); it
  installs as the major-version cleanup with a restructured module layout.
- pyvinecopulib copula families are named via the **module-level constants**
  (`from pyvinecopulib.families import clayton, gaussian, ...`), not
  `pv.BicopFamily.clayton` — the shipped stubs expose members as module
  constants, so only the former type-checks under `ty`. (Stub gap reported
  upstream: vinecopulib/pyvinecopulib#223.)
- Match existing local naming, typing, dataclass, and testing patterns unless
  there is a clear reason to change them.

### Maintaining this file

If a coding agent repeatedly misses a durable repository convention, or if code
review repeatedly corrects the same kind of mistake, update `AGENTS.md` rather
than relying on undocumented tribal knowledge. Do not add ephemeral,
user-specific, or machine-local preferences here.
