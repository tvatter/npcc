# Neural Pair-Copulas Constructions (NPCCs)

A Python library for multivariate conditional density estimation via
neural vine copulas, using differentiable nonparametric pair-copula components.

## Model: Gaussian-Copula Kernel Bicop (`GCBicop`)

`GCBicop` implements a nonparametric bivariate pair-copula model.  The
density is represented as a positive mixture of separable Gaussian-copula
kernel basis functions:

```text
c(u, v) = Σ_{i,j} W_{ij} k_u(u; U_i, ρ_{u,i}) k_v(v; V_j, ρ_{v,j})
```

where the weights satisfy `W_{ij} ≥ 0` and `Σ_{i,j} W_{ij} = 1`, and
each one-dimensional basis function is a Gaussian-copula kernel:

```text
k(u; U_i, ρ_i) = φ(a_i) / (s_i φ(z)),

  z  = Φ⁻¹(u),  z_i = Φ⁻¹(U_i),  s_i = √(1 − ρ_i²),  a_i = (z − ρ_i z_i) / s_i
```

The primitive of each basis function is available in closed form:

```text
K(u; U_i, ρ_i) = ∫₀ᵘ k(t; U_i, ρ_i) dt = Φ(a_i)
```

This yields closed-form expressions for the density, CDF, and h-functions—
making `GCBicop` a natural building block for neural vine copulas where
tree-by-tree gradient-based training requires differentiable pair-copula
components with explicit h-function evaluations.

### Parameterization

| Component    | Form                                                        |
|--------------|-------------------------------------------------------------|
| Weights      | `W_{ij} = exp(L_{ij}) / Σ_{k,l} exp(L_{kl})` (softmax)   |
| Correlations | `ρ_{u,i} = ρ_max tanh(η_{u,i})`, one parameter per center  |
| Grid centers | Uniform or probit spacing in (0, 1)                         |

### Public API

```python
from npcc.gcbicop import GCBicop

model = GCBicop(m_u=25, m_v=25, rho_u_init=0.5, rho_v_init=0.5)

model.pdf(UV)                        # copula density,  shape [B]
model.cdf(UV)                        # copula CDF,       shape [B]
model.hfunc1(UV, normalized=False)   # ∫₀ᵘ c(s,v) ds,   shape [B]
model.hfunc2(UV, normalized=False)   # ∫₀ᵛ c(u,t) dt,   shape [B]
model.margin_u(u)                    # induced u-margin, shape [B]
model.margin_v(v)                    # induced v-margin, shape [B]
model.log_pdf(UV)                    # log density
model.nll(UV)                        # negative log-likelihood (sum)
model.marginal_penalty()             # soft uniformity penalty for training
```

## Setup

### uv dependency groups

- `dev`: linting, formatting, type checking, and tests (`ruff`, `ty`, `pytest`, `pytest-cov`, `pytest-xdist`).
- `interactive`: notebook/data/plotting tooling (`pandas`, `jupyter`, `matplotlib`, etc.).
- `uv` defaults include both groups (`default-groups = ["dev", "interactive"]`), so `uv sync` installs them.

### Package extras (PyTorch variant)

- Available extras: `cpu`, `cu126`, `cu128`, `cu130`.
- Extras are mutually exclusive (only one at a time).
- Each extra selects a matching PyTorch index/source.

```bash
uv sync --extra cpu
uv sync --extra cu128
```

## Commands

```bash
# Lint / format
uv run ruff check . --select ANN --fix
uv run ruff format .

# Type check
uv run ty check

# Tests
uv run pytest tests/ -v -n auto
uv run pytest tests/ --cov=src/npcc --cov-report=term-missing -v -n auto
```
