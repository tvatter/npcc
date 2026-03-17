# Neural Pair-Copulas Constructions (NPCCs)

A Python library for multivariate conditional density estimation via
neural vine copulas, using differentiable nonparametric pair-copula components.

## Models

Two related bivariate pair-copula families are provided.  Both use a
positive softmax-weighted mixture of separable basis functions and expose
identical public APIs, making them interchangeable building blocks in
neural vine copulas.

---

### `GCKBicop` — Gaussian-Copula-Kernel Bicop

Kernels are defined directly on the copula scale `(0, 1)`.  The density is:

```text
c(u, v) = sum_{i,j} W_{ij} k_u(u; U_i, rho_{u,i}) k_v(v; V_j, rho_{v,j})
```

Each 1-D basis function is the conditional density of a Gaussian copula:

```text
k(u; U_i, rho_i) = phi(a_i) / (s_i * phi(z)),

  z   = Phi^{-1}(u),  z_i = Phi^{-1}(U_i),
  s_i = sqrt(1 - rho_i^2),  a_i = (z - rho_i * z_i) / s_i
```

Its primitive is available in closed form:

```text
K(u; U_i, rho_i) = int_0^u k(t; U_i, rho_i) dt = Phi(a_i)
```

| Parameter    | Form                                                       |
|--------------|------------------------------------------------------------|
| Weights      | `W_{ij} = exp(L_{ij}) / sum exp(L_{kl})`  (softmax)        |
| Correlations | `rho_{u,i} = rho_max * tanh(eta_{u,i})`, one per center    |
| Grid centers | `Phi(linspace(-3.25, 3.25, m))` in `(0, 1)`               |

---

### `GTKBicop` — Gaussian-Transformation-Kernel Bicop

First transforms to Gaussian margins `z1 = Phi^{-1}(u)`, `z2 = Phi^{-1}(v)`,
then models a density in Z-space with separable Gaussian kernels, and maps
back to the copula scale via the Jacobian correction:

```text
c(u, v) = f_Z(Phi^{-1}(u), Phi^{-1}(v)) / (phi(Phi^{-1}(u)) * phi(Phi^{-1}(v)))
```

The Z-space density is:

```text
f_Z(z1, z2) = sum_{i,j} W_{ij} b_u(z1; mu_i, sigma_{u,i}) b_v(z2; nu_j, sigma_{v,j})

b(z; mu, sigma) = (1/sigma) * phi((z - mu) / sigma)
```

Its primitive is `B(z; mu, sigma) = Phi((z - mu) / sigma)`, which yields
closed-form CDF and h-function evaluations exactly as with `GCKBicop`.

**Why GTKBicop?**  Smoothing in the unbounded Z-space is often better
behaved in the tails than smoothing on the compact `(0, 1)` scale.  This
is closer in spirit to TLL-type nonparametric copula estimators.

| Parameter    | Form                                                            |
|--------------|-----------------------------------------------------------------|
| Weights      | `W_{ij} = exp(L_{ij}) / sum exp(L_{kl})`  (softmax)             |
| Scales       | `sigma_{u,i} = sigma_min + softplus(eta_{u,i})`, one per center |
| Grid centers | `linspace(-3.25, 3.25, m)` in Z-space                           |

---

### Closed-form quantities (both models)

Because the kernel primitives are explicit, all of the following are
available in closed form:

```text
C(u, v)     = sum_{i,j} W_{ij} Fu_i(u) Fv_j(v)
H1^raw(u,v) = sum_{i,j} W_{ij} Fu_i(u) fv_j(v)   (int_0^u c(s,v) ds)
H2^raw(u,v) = sum_{i,j} W_{ij} fu_i(u) Fv_j(v)   (int_0^v c(u,t) dt)
m_u(u)      = sum_i alpha_i fu_i(u)               (induced u-margin)
m_v(v)      = sum_j beta_j  fv_j(v)               (induced v-margin)
```

This makes both classes natural building blocks for neural vine copulas,
where tree-by-tree gradient-based training requires differentiable
pair-copula components with explicit h-function evaluations.

---

## Public API

Both classes share the same interface:

```python
from npcc import GCKBicop, GTKBicop

# GCKBicop — copula-scale kernels
model = GCKBicop(m_u=25, m_v=25)

# GTKBicop — Z-space Gaussian kernels
# gtk = GTKBicop(m_u=25, m_v=25)

model.pdf(UV)                        # copula density,  shape [B]
model.cdf(UV)                        # copula CDF,       shape [B]
model.hfunc1(UV, normalized=False)   # int_0^u c(s,v) ds, shape [B]
model.hfunc2(UV, normalized=False)   # int_0^v c(u,t) dt, shape [B]
model.margin_u(u)                    # induced u-margin,  shape [B]
model.margin_v(v)                    # induced v-margin,  shape [B]
model.log_pdf(UV)                    # log density
model.nll(UV)                        # mean negative log-likelihood
model.marginal_penalty()             # soft uniformity penalty
model.logit_smoothness_penalty()     # smooth logit surface penalty

# GCKBicop only
model.rho_smoothness_penalty()         # smooth rho-profile penalty

# GTKBicop only
# model.scale_smoothness_penalty()       # smooth sigma-profile penalty
```

A typical training loop:

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.05)
for _ in range(500):
    optimizer.zero_grad()
    loss = (
        model.nll(UV)
        + 0.1 * model.marginal_penalty()
        + 0.5 * model.logit_smoothness_penalty()
    )
    loss.backward()
    optimizer.step()
```

---

## Setup

### uv dependency groups

- `dev`: linting, formatting, type checking, and tests.
- `interactive`: marimo notebooks, pandas, matplotlib, etc.
- `uv` defaults include both groups, so `uv sync` installs them.

### Package extras (PyTorch variant)

Extras are mutually exclusive — pick one:

```bash
uv sync --extra cpu
uv sync --extra cu128
```

---

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

# Demo notebook
uv run marimo edit notebooks/gcbicop_demo.py
```
