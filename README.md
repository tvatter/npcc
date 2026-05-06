# Neural Pair-Copulas Constructions (NPCCs)

A Python library for **conditional bivariate copula density estimation**
backed by [TabPFN](https://github.com/PriorLabs/TabPFN).  The package
exposes one outer estimator — `PFNRBicop` — and two interchangeable
inner univariate-conditional-density estimators that read the density
out of a fitted TabPFN regressor in two complementary ways.

---

## TabPFN-Rosenblatt conditional bivariate copula

Let $(U, V, X)$ denote observations with $U, V \in (0, 1)$ and
$X \in \mathbb{R}^p$.  We want to estimate the conditional copula
density $c(u, v \mid x)$.

### Rosenblatt factorisation

Because $U$ is uniform on the copula scale, the Rosenblatt construction
collapses the bivariate density into a single univariate conditional
density,

$$c(u, v \mid x) = f_{V \mid U, X}(v \mid u, x).$$

So estimating $c$ reduces to estimating $f_{V \mid U, X}$, which is
exactly what a flexible distributional regressor like TabPFN provides.
The features fed to the regressor are simply $W = [u, x]$ (or $[v, x]$
for the reverse direction).

### Symmetric variant

The naive estimator factorises in a single direction and is therefore
ordering-dependent: it satisfies $\int_0^1 \hat c (u, v \mid x) \, dv =
1$ by construction but generally not $\int_0^1 \hat c (u, v \mid x) \,
du = 1$.  Setting `symmetric=True` (the default) also fits the reverse
direction $f_{U \mid V, X}$ and averages,

$$\hat c_S(u, v \mid x)
= \tfrac{1}{2} \, \hat f_{V \mid U, X}(v \mid u, x)
+ \tfrac{1}{2} \, \hat f_{U \mid V, X}(u \mid v, x),$$

which reduces directional bias.  It still does not impose exact uniform
copula margins; if you need that, evaluate the density on a grid and
apply iterative-proportional-fitting / Sinkhorn projection.

### Logit transform

Both copula scores live in $(0, 1)$.  By default the inner density
estimator fits TabPFN on $Z = \mathrm{logit}(Y)$ — the unbounded image —
and converts back via the standard Jacobian,

$$f_Y(y \mid w) = f_Z(\mathrm{logit}(y) \mid w) \, \tfrac{1}{y(1-y)}.$$

This is generally numerically better behaved than estimating a density
on a bounded interval.

---

## Two density-recovery methods

Both classes expose the same `fit(w, y)` / `density(w, y)` API and are
drop-in interchangeable inside `PFNRBicop` via the `method=` argument.

### `TabPFNDensity1D` — TabPFN's native distribution head (default)

TabPFN's regressor is internally a classifier over a binned
distribution.  Calling `predict(W, output_type="full")` returns logits
over the bins plus a `criterion` object whose `pdf(logits, z)` method
evaluates the density at arbitrary points,

$$f(y \mid w) = \mathrm{criterion.pdf}\bigl( \mathrm{logits}(w), \, z = T(y) \bigr).$$

A single forward pass is needed per row of $W$.  The
`density_grid(w, y_grid)` method exploits this — one forward pass per
$w$ row, then evaluate at every $y$ value — to produce the full
Cartesian-product density matrix in a single shot.

### `TabPFNQuantileDensity1D` — numerical slope inversion

Asks TabPFN for the conditional quantile function $Q(\alpha \mid w)$
on a grid of $\alpha$ values, then recovers

$$f(y \mid w) = 1 / Q'(\alpha) \quad \text{at} \quad \alpha = F(y \mid w).$$

The class enforces monotonicity by sorting (rearrangement), clips the
slope to a positive floor to avoid singularities at quantile plateaus,
and uses linear interpolation in both lookup steps.  Slower and less
direct than the criterion approach, but model-agnostic.

---

## Public API

`PFNRBicop` is the main entry point.  Its key methods:

| Method                              | What it returns                                                                  |
| ----------------------------------- | -------------------------------------------------------------------------------- |
| `fit(u, v, x=None)`                 | Fits the inner density estimator(s).  `x=None` → unconditional fit.              |
| `density(u, v, x=None)`             | Pointwise $\hat c(u_i, v_i \mid x_i)$ (vectorised over rows).                    |
| `log_density(u, v, x=None)`         | $\log$ of `density`, floored at the smallest positive float.                     |
| `density_grid(u_grid, v_grid, x_row=None)` | Cartesian-grid density `out[i, j] = c(u_grid[i], v_grid[j] | x_row)`.  Requires `method="criterion"`. |
| `cdf(u, v, x=None, *, n_int=64)`    | Pointwise joint CDF $\hat C(u_i, v_i \mid x_i)$, trapezoidal in $s$ (and $t$ for symmetric). |
| `cdf_grid(u_grid, v_grid, x_row=None, *, n_int=64)` | Cartesian-grid joint CDF.  Requires `method="criterion"`.            |
| `hfunc1(u, v, x=None)`              | $h_1(u, v \mid x) = \partial C / \partial u = F_{V \mid U, X}(v \mid u, x)$.  Always available.  Conditions on the first argument (matches `pyvinecopulib`). |
| `hfunc2(u, v, x=None)`              | $h_2(u, v \mid x) = \partial C / \partial v = F_{U \mid V, X}(u \mid v, x)$.  Requires `symmetric=True`. |
| `tau(x_row=None, *, n=1000, seeds=None)` | Kendall's $\tau(x)$ via [pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib)'s recipe: quasi-random `ghalton(n, 2)` + inverse-Rosenblatt + `wdm`. Matches `pv.KernelBicop::parameters_to_tau`. |
| `conditional_cdf_v_given_u(u, v_grid, x=None)` | Diagnostic: $C_{V \mid U, X}(v \mid u_i)$ broadcast over `v_grid`.  Wraps `hfunc2`. |
| `as_bicop(x_row=None)`              | Returns a [`pyvinecopulib`](https://github.com/vinecopulib/pyvinecopulib)-compatible adapter (exposes `var_types = ["c", "c"]` and `pdf(uv)`). |
| `plot(*, x_row=None, plot_type="contour", margin_type="norm", ...)` | Renders a contour or surface plot via `pyvinecopulib`'s plotter (lazy-imports `matplotlib`). |

The two inner classes (`TabPFNDensity1D`, `TabPFNQuantileDensity1D`) are
also exported and can be used directly for univariate conditional
density estimation outside the copula context.

### Quick start

```python
from dotenv import load_dotenv
load_dotenv()  # picks up TABPFN_TOKEN from .env

import numpy as np
import pyvinecopulib as pv

from npcc import PFNRBicop

# Simulate from a Clayton bicop
clayton = pv.Bicop(
    family=pv.BicopFamily.clayton,
    parameters=np.asarray([[3.0]], dtype=np.float64),
)
u = clayton.simulate(n=1000, seeds=[2, 2, 4])

# Fit the TabPFN-Rosenblatt copula (defaults: symmetric=True, method="criterion")
model = PFNRBicop()
model.fit(u[:, 0], u[:, 1])

# Pointwise density
print(model.density(np.array([0.3, 0.5]), np.array([0.4, 0.6])))

# Cartesian-grid density (fast path, criterion method only)
u_grid = np.linspace(0.05, 0.95, 30)
v_grid = np.linspace(0.05, 0.95, 30)
grid = model.density_grid(u_grid, v_grid)   # shape (30, 30)

# Joint CDF and h-functions (pyvinecopulib convention: h_i conditions on i-th arg)
C = model.cdf_grid(u_grid, v_grid)            # shape (30, 30)
h1 = model.hfunc1(np.array([0.3, 0.5]), np.array([0.4, 0.6]))   # F_{V|U,X}
# h2 = model.hfunc2(...)  # F_{U|V,X}, requires symmetric=True

# Kendall's tau via the pyvinecopulib quasi-random recipe.
tau = model.tau()
# Clayton(theta=3) analytic: theta / (theta + 2) = 0.6.

# Plot via pyvinecopulib's helper (matplotlib)
model.plot(plot_type="contour", margin_type="norm")
```

To switch to the quantile-based method:

```python
model = PFNRBicop(method="quantiles")
```

To pass a covariate matrix:

```python
model.fit(u, v, x=X_train)               # X_train shape (n, p)
model.density(u_query, v_query, x_query) # x_query shape (n_query, p)
```

To plot at a specific covariate row:

```python
model.plot(x_row=np.array([[1.5, -0.5]]))
```

---

## Notebook

A worked end-to-end demo lives at
[`notebooks/pfnr_bicop_demo.ipynb`](notebooks/pfnr_bicop_demo.ipynb).
It simulates from a Clayton copula, fits `PFNRBicop`, compares against
the `pv.tll` benchmark on a regular grid (ISE / IAE / KL), renders
contour plots for the truth, `tll`, and `PFNRBicop`, then prints
Kendall's tau and a side-by-side joint-CDF heatmap.

```bash
uv run jupyter lab notebooks/pfnr_bicop_demo.ipynb
```

---

## Setup

### Install

```bash
# Pick exactly one of: cpu, cu126, cu128, cu130 (PyTorch flavour)
uv sync --extra cpu
```

The package depends on `numpy>=2.0`, `pyvinecopulib>=0.7.5`, and
`tabpfn>=2.0`.  TabPFN pulls in PyTorch transitively; the extras above
just pin its build (CPU-only or one of the CUDA variants).

### Authenticate TabPFN (one-time)

`tabpfn` runs locally but authenticates once via a token from the
PriorLabs portal.

1. Go to <https://ux.priorlabs.ai>, log in (or register), accept the
   `priorlabs-1-1` license on the **Licenses** tab, and copy your API
   key from the **Account** tab.
2. Drop it into a `.env` file at the repo root:

   ```
   TABPFN_TOKEN="..."
   ```

3. Make sure your code calls `dotenv.load_dotenv()` before instantiating
   `PFNRBicop`.  Alternatively, just `export TABPFN_TOKEN=...` in your
   shell.

The first call to `model.fit(...)` downloads the TabPFN-v2.5 regressor
checkpoint from HuggingFace into the platform cache directory (Linux
default: `~/.cache/tabpfn/`; override with `TABPFN_MODEL_CACHE_DIR`).
Subsequent runs are fully offline.

> **Why v2.5?**  `PFNRBicop` is locked to TabPFN-v2.5 because v2.6 has
> reported regressions on tabular regression tasks.  Override via
> `model_kwargs={...}` if needed.

### CPU sample-size cap

TabPFN refuses to fit on more than 1000 samples on CPU by default.  On
larger samples either use a CUDA build, set the
`TABPFN_ALLOW_CPU_LARGE_DATASET=1` environment variable, or pass
`model_kwargs={"ignore_pretraining_limits": True}` to `PFNRBicop`.

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

# Run the demo notebook end-to-end
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/pfnr_bicop_demo.ipynb
```

The test suite contains an integration test
(`tests/test_pfnr_bicop.py::test_real_tabpfn_smoke`) that hits the real
TabPFN-v2.5 model.  It is skipped automatically when `TABPFN_TOKEN` is
unset or when the PriorLabs license endpoint is unreachable, so the
default unit-test surface stays hermetic.
