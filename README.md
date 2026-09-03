# Neural Pair-Copulas Constructions (NPCCs)

A Python library for **conditional pair copula and fixed-structure vine density estimation**
built on top of *any* distributional-regression backend.  The package
exposes one outer estimator — `RosenblattBicop` — whose inner
univariate-conditional-predictive-distribution model is **pluggable**
(selected by name via `backend=`).  TabPFN is the default backend, but the
construction only needs a regressor that can produce a conditional
predictive distribution (`pdf`, `cdf`, `icdf`), so classical and
foundation-model regressors slot in on equal footing.

---

## Rosenblatt conditional bivariate copula

Let $(U, V, X)$ denote observations with $U, V \in (0, 1)$ and
$X \in \mathbb{R}^p$.  We want to estimate the conditional copula
density $c(u, v \mid x)$.

### Rosenblatt factorisation

Because $U$ is uniform on the copula scale, the Rosenblatt construction
collapses the bivariate density into a single univariate conditional
density,

$$c(u, v \mid x) = f_{V \mid U, X}(v \mid u, x).$$

So estimating $c$ reduces to estimating $f_{V \mid U, X}$, which is
exactly what a flexible **distributional regressor** provides.  The
features fed to the regressor are simply $W = [u, x]$ (or $[v, x]$ for
the reverse direction).

### Symmetric averaging

A single Rosenblatt direction is ordering-dependent: it satisfies
$\int_0^1 \hat c (u, v \mid x) \, dv = 1$ by construction but generally
not $\int_0^1 \hat c (u, v \mid x) \, du = 1$.  To reduce this
directional bias the estimator always fits both directions and
averages,

$$\hat c(u, v \mid x)
= \tfrac{1}{2} \, \hat f_{V \mid U, X}(v \mid u, x)
+ \tfrac{1}{2} \, \hat f_{U \mid V, X}(u \mid v, x).$$

This still does not impose exact uniform copula margins; if you need
that, enable the optional iterative-proportional-fitting / Sinkhorn
projection via `sinkhorn_iters`.

### Support transforms

Both copula scores live in $(0, 1)$.  The inner estimator can fit the
backend on a transformed target

$$Z = T(Y),$$

and convert the density back via the change-of-variables formula

$$f_Y(y \mid w) = f_Z(T(y) \mid w) \left|\tfrac{dT(y)}{dy}\right|.$$

Supported transforms: `transform="logit"` (default,
$\tfrac{1}{y(1-y)}$), `transform="probit"`
($\tfrac{1}{\phi(\Phi^{-1}(y))}$), and `transform="identity"`.  The
transform machinery lives on the backend-neutral base class, so every
backend gets it for free.

---

## Pluggable backends

The inner conditional-density model is any registered backend, chosen by
name:

| `backend=`            | Underlying model                     | Extra to install       |
| --------------------- | ------------------------------------ | ---------------------- |
| `"tabpfn-criterion"`  | TabPFN native binned head (default)  | — (core)               |
| `"tabpfn-quantiles"`  | TabPFN quantile output, inverted     | — (core)               |
| `"ngboost"`           | NGBoost parametric (analytic)        | `npcc[ngboost]`        |
| `"gbm"`               | scikit-learn quantile GBM            | `npcc[gbm]`            |
| `"tabicl"`            | TabICL foundation model              | `npcc[tabicl]`         |

`available_backends()` lists them; `register_backend(name, factory)` adds
your own.

### Two base classes

Every backend implements the abstract
`ConditionalDistribution1D` interface — `fit(w, y)`, `pdf(w, y)`,
`cdf(w, y)`, `icdf(w, alphas)`, plus the Cartesian-grid fast paths
`pdf_grid(w, y_grid)` / `cdf_grid(w, y_grid)`.  There are two ways to
implement it:

- **Quantile-table backends** subclass `QuantileTableDistribution1D` and
  implement a single hook, `_predict_quantiles(w, alphas) -> (n, K)`.
  Everything else — the chunked, memory-safe quantile-table prediction,
  the monotone re-sort, and the predict-once-per-row pdf/cdf/icdf/grid
  inversion — is inherited.  This is the universal fast path used by
  `tabpfn-quantiles`, `gbm`, and `tabicl`.  Recovery details: pdf via
  $f(y \mid w) = 1 / Q'(\alpha)$ at $\alpha = F(y \mid w)$; cdf/icdf by
  linear interpolation in the sorted quantile table; $Q'$ floored to a
  positive constant for stability.

- **Native-evaluation backends** subclass `ConditionalDistribution1D`
  directly and evaluate the predictive distribution at arbitrary points.
  `tabpfn-criterion` reads TabPFN's `criterion` head
  (`predict(W, output_type="full")` → logits + `pdf`/`cdf`/`icdf`); it is
  faster than, and (currently) as accurate as, the quantile read-out —
  hence the **default**.  `ngboost` uses the analytic SciPy frozen
  distribution (`pred_dist(W).dist`).

### Speed

Grid methods (`pdf_grid` / `cdf_grid`), which drive the Sinkhorn
projection and plotting, predict **at most once per conditioning row** and
evaluate every grid point by interpolation — never one forward pass per
grid cell.  Inference is chunked by `batch_size` (device-aware default:
400 on CPU, 2000 on CUDA) to bound memory.

---

## Public API

`RosenblattBicop` is the main entry point.  `batch_size` defaults are
device-aware and overridable model-wide
(`RosenblattBicop(..., batch_size=...)`) or per call.

| Method | What it returns |
| --- | --- |
| `fit(u, v, x=None)` | Fits both Rosenblatt directions.  `x=None` → unconditional fit. |
| `pdf(u, v, x=None, *, batch_size=None, sinkhorn_iters=None)` | Pointwise $\hat c(u_i, v_i \mid x_i)$. |
| `log_pdf(u, v, x=None, *, batch_size=None, sinkhorn_iters=None)` | $\log$ of `pdf`, floored at the smallest positive float. |
| `pdf_grid(u_grid, v_grid, x_row=None, *, batch_size=None, sinkhorn_iters=None)` | Cartesian-grid density `out[i, j] = c(u_grid[i], v_grid[j] | x_row)`.  Available for every backend. |
| `cdf(u, v, x=None, *, n_int=12, batch_size=None)` | Pointwise joint CDF, trapezoidal in $s$ and $t$. |
| `cdf_grid(u_grid, v_grid, x_row=None, *, n_int=64)` | Cartesian-grid joint CDF.  Available for every backend. |
| `hfunc1(uv, x=None)` | $h_1 = \partial C / \partial u = F_{V \mid U, X}(v \mid u, x)$ (conditions on the first argument; matches `pyvinecopulib`). |
| `hfunc2(uv, x=None)` | $h_2 = \partial C / \partial v = F_{U \mid V, X}(u \mid v, x)$. |
| `tau(x_row=None, *, n=1000, seeds=None)` | Kendall's $\tau(x)$ via [pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib)'s recipe: `ghalton(n, 2)` + inverse-Rosenblatt + `wdm`. |
| `as_bicop(x_row=None)` | A `pyvinecopulib`-compatible adapter (`var_types = ["c", "c"]`, `pdf(uv)`). |
| `plot(*, x_row=None, plot_type="contour", margin_type="norm", ...)` | Contour/surface plot via `pyvinecopulib`'s plotter (lazy-imports `matplotlib`). |

Exported names: `RosenblattBicop`, `RosenblattVinecop`, `RosenblattVinedist`, `BackendMargin`,
the abstract `ConditionalDistribution1D` and `QuantileTableDistribution1D` base classes, 
`QuantileGridConfig`, the `TabPFNCriterionBackend` / `TabPFNQuantileBackend` leaves, 
and the registry helpers `create_backend` / `register_backend` / `available_backends`.

### Quick start

```python
from dotenv import load_dotenv
load_dotenv()  # picks up TABPFN_TOKEN from .env

import numpy as np
import pyvinecopulib as pv

from npcc import RosenblattBicop

# Sample from a Clayton bicop
clayton = pv.Bicop(
    family=pv.BicopFamily.clayton,
    parameters=np.asarray([[3.0]], dtype=np.float64),
)
u = clayton.sample(n=1000, seeds=[2, 2, 4])

# Fit the Rosenblatt copula (default backend="tabpfn-criterion")
model = RosenblattBicop()
model.fit(u)

# Pointwise density
print(model.pdf(np.array([[0.3, 0.4], [0.5, 0.6]])))

# Cartesian-grid density (fast path, any backend)
u_grid = np.linspace(0.05, 0.95, 30)
v_grid = np.linspace(0.05, 0.95, 30)
grid = model.pdf_grid(u_grid, v_grid)   # shape (30, 30)

# Joint CDF and h-functions (pyvinecopulib convention: h_i conditions on i-th arg)
C = model.cdf_grid(u_grid, v_grid)            # shape (30, 30)
h1 = model.hfunc1(np.array([[0.3, 0.4], [0.5, 0.6]]))   # F_{V|U,X}
h2 = model.hfunc2(np.array([[0.3, 0.4], [0.5, 0.6]]))   # F_{U|V,X}

# BicopBase API: native inverse h-functions, sampling, and log-likelihood
v = model.hinv1(np.array([[0.3, 0.25], [0.5, 0.75]]))
sampled = model.sample(100, seeds=[42])  # float64 torch tensor
log_likelihood = model.loglik(u)

# Kendall's tau via the pyvinecopulib quasi-random recipe.
tau = model.tau()   # Clayton(theta=3) analytic: theta / (theta + 2) = 0.6

# Plot via pyvinecopulib's helper (matplotlib)
model.plot(plot_type="contour", margin_type="norm")
```

To switch backends (TabPFN read-outs need no extra):

```python
model = RosenblattBicop(backend="tabpfn-quantiles")

# Non-TabPFN backends (install the matching extra); backend-specific
# hyperparameters go in backend_kwargs.
model = RosenblattBicop(backend="ngboost", backend_kwargs={"n_estimators": 500})
model = RosenblattBicop(backend="gbm", backend_kwargs={"max_depth": 3})
model = RosenblattBicop(backend="tabicl")
```

To pin the TabPFN model version, pass it through `backend_kwargs`:

```python
from tabpfn.constants import ModelVersion
model = RosenblattBicop(
    backend="tabpfn-criterion",
    backend_kwargs={"model_version": ModelVersion.V3},
)
```

To pass a covariate matrix:

```python
model.fit(np.column_stack([u, v]), x=X_train)               # X_train shape (n, p)
model.pdf(np.column_stack([u_query, v_query]), x_query)     # x_query shape (n_query, p)
```

---

## Fixed-structure multivariate vine

`RosenblattVinecop` composes fitted `RosenblattBicop` modules along a
caller-supplied `pyvinecopulib.RVineStructure`. The structure fixes the variable
order, edge layout, and truncation level; this estimator does not perform
automatic structure selection.

The vine is non-simplified. For an edge with conditioned variables
\(a_e, b_e\) and conditioning set \(D_e\), the pair copula receives

\[
   x_e = [u_{D_e}, x],
\]

where \(u_{D_e}\) contains the internal vine conditioning variables and \(x\)
contains optional external covariates. Consequently, the fitted vine density is

\[
   \hat c(u_1, \ldots, u_d \mid x)
   =
   \prod_e
   \hat c_{a_e,b_e;D_e}
   \left(
      u_{a_e\mid D_e},
      u_{b_e\mid D_e}
      \mid
      u_{D_e}, x
   \right).
\]

For tree-zero edges, \(D_e\) is empty, so those pairs receive only the external
covariates. Higher-tree pair copulas receive the conditioning-set values first and the
external coavariates last.

```python
import numpy as np
import pyvinecopulib as pv
import torch

from npcc import RosenblattVinecop

# Continuous pseudo-observations and optional external covariates.
u_train = rng.uniform(0.05, 0.95, size=(500, 3))
x_train = rng.normal(size=(500, 2))

# The order and truncation level are fixed by the supplied structure.
structure = pv.RVineStructure.from_order([1, 2, 3])

vine = RosenblattVinecop.from_data(
   u_train,
   structure,
   x=x_train,
   backend="tabpfn-criterion",
   device="cpu",
)

u_query = rng.uniform(0.05, 0.95, size=(2, 3))
x_query = rng.normal(size=(2, 2))

# NumPy inputs produce NumPy outputs.
density = vine.pdf(u_query, x=x_query)
independent = vine.rosenblatt(u_query, x=x_query)
recovered = vine.inverse_rosenblatt(independent, x=x_query)

# Conditional sample() preserves the covariate array type. Unconditional
# sample() returns a float64 torch tensor on the configured device.
samples = vine.sample(2, x=x_query, seeds=[42])
unconditional_samples = vine.sample(2, seeds=[42])

# With order [1, 2, 3], a one-column conditioning matrix conditions on
# variable 3, the current order tail.
u_cond = np.array([[0.3], [0.7]])
conditional_samples = vine.sample_conditional(
   u_cond,
   x=x_query,
   seeds=[42],
)
```

The initial vine integration supports continuous fixed structures only.
Automatic structure selection and discrete variables are not (yet) implemented.
Within ordinary evaluator calls, `u` and `x` must both be NumPy arrays or both
be torch tensors. A joint CDF with external covariates is not currently
available because it would require a separate Monte Carlo sample for every
covariate row. For non-simplified vines, `sample_conditional()` can condition
only on variables already forming the tail of the structure order.

---

## Original-scale vine distribution

`BackendMargin` adapts any registered distributional-regression backend to
pyvinecopulib's `MarginBase` interface. `RosenblattVinedist` fits one independent
backend margin per response column, transforms the observations through their
conditional marginal CDFs, and fits a `RosenblattVinecop` to the resulting
pseudo observations.

For observations \(Y=(Y_1,\ldots,Y_d)\) and optional covariates \(X\), the

\[
   U_j = F_j(X_j),
\]

and the resulting conditional joint density is

\[
   f(y_1,\ldots,y_d)
   =
   c\!\left(
      F_1(y_1\mid x),\ldots.F_d(y_d\mid x)
      \mid x
   \right)
   \prod_{j=1}^d f_j(y_j\mid x)
\]

```python
import numpy as np
import pyvinecopulib as pv

from npcc import RosenblattVinedist

rng = np.random.default_rng(42)
y_train = rng.normal(size=(500, 3))
x_train = rng.normal(size=(500, 2))

structure = pv.RVineStructure.from_order([1, 2, 3])

dist = Rosenblatt.from_data(
   y_train,
   x=x_train,
   structure=structure,
   margin_backend="tabpfn-criterion",
   pair_backend="tabpfn-criterion",
   device="cpu",
)

y_query = rng.normal(size=(5, 3))
x_query = rng.normal(size=(5, 2))

density = dist.pdf(y_query, x=x_query)
independent = dist.rosenblatt(y_query, x=x_query)
recovered = dist.inverse_rosenblatt(independent, x=x_query)
samples = dist.samples(5, x=x_query, seeds=[42])
```

Margin and pair-copula backends are configured independently through
`margin_backend` / `margin_backend_kwargs` and
`pair_backend` / `pair_backend_kwargs`.

The initial implementation supports continuous real-valued margins only.
Every margin uses the identity target transform, while pair copulas default to
the logit transform on the unit interval. A fixed `RVineStructure` is required.
Custom margins, observation weights, automatic structure selection, and
variable names are not currently supported.

---

## Notebooks

Worked demos live under [`notebooks/`](notebooks/) (Clayton demo,
conditional copula, Sinkhorn projection, and the simulation study).  They
require a `TABPFN_TOKEN` (see below) to run against the real TabPFN model.

---

## Setup

### Install

```bash
# Pick exactly one PyTorch flavour: cpu, cu126, cu128, cu130, cu132
uv sync --extra cpu

# Optionally add non-TabPFN backends:
uv sync --extra cpu --extra ngboost --extra gbm --extra tabicl
```

The package depends on `numpy>=2.0`, `pyvinecopulib>=0.8.0`, and
`tabpfn>=8.0`.  TabPFN pulls in PyTorch transitively; the flavour extras
just pin its build.

### Authenticate TabPFN (one-time)

`tabpfn` runs locally but authenticates once via a token from the
PriorLabs portal.

1. Go to <https://ux.priorlabs.ai>, log in, accept the `priorlabs-1-1`
   license on the **Licenses** tab, and copy your API key from the
   **Account** tab.
2. Drop it into a `.env` file at the repo root:

   ```
   TABPFN_TOKEN="..."
   ```

3. Call `dotenv.load_dotenv()` before fitting, or `export TABPFN_TOKEN=...`
   in your shell.

The first `fit(...)` downloads the TabPFN-v3 regressor checkpoint from
HuggingFace into the platform cache (Linux default: `~/.cache/tabpfn/`;
override with `TABPFN_MODEL_CACHE_DIR`).  Subsequent runs are offline.
(Only relevant to the TabPFN backends.)

### CPU sample-size cap

TabPFN refuses to fit on more than 1000 samples on CPU by default.  On
larger samples either use a CUDA build, set
`TABPFN_ALLOW_CPU_LARGE_DATASET=1`, or pass
`backend_kwargs={"model_kwargs": {"ignore_pretraining_limits": True}}`.

---

## Commands

```bash
# Lint + format (ANN ruleset → public functions must have annotations)
uv run ruff check . --select ANN --fix
uv run ruff format .

# Type check (zero errors required)
uv run ty check

# Tests
uv run pytest tests/ -v -n auto
uv run pytest tests/ --cov=src/npcc --cov-report=term-missing -v -n auto
```

The suite is hermetic by default: TabPFN is faked via a monkeypatched
regressor, and pluggable-backend behaviour is proven end-to-end with a
TabPFN-free in-process backend.  A few tests hit the real models
(`test_real_tabpfn_smoke`, the TabICL smoke) and skip automatically when
their dependency/credentials are absent.
