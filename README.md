# NPCC

Neural Pair-Copula Constructions for multivariate conditional density
estimation. The foundation-model estimator fits both Rosenblatt directions and
averages them to reduce ordering bias; optional Sinkhorn projection enforces
approximately uniform copula margins.

## Foundation-model providers

`FoundationModelBicop` requires an explicit provider and recovery strategy:

```python
from npcc import FoundationModelBicop, Recovery, TabPFNConfig

model = FoundationModelBicop(
  provider="tabpfn",
  recovery=Recovery.NATIVE_DISTRIBUTION,
  provider_config=TabPFNConfig(model_version="v3"),
)
model.fit(u, v, x)
density = model.pdf(u_test, v_test, x_test)
```

TabPFN supports both `Recovery.NATIVE_DISTRIBUTION` and
`Recovery.QUANTILE_INVERSION`. TabICL exposes conditional quantiles, so it
supports only quantile inversion:

```python
from npcc import FoundationModelBicop, Recovery, TabICLConfig

model = FoundationModelBicop(
  provider="tabicl",
  recovery=Recovery.QUANTILE_INVERSION,
  provider_config=TabICLConfig(),
)
```

Quantile inversion predicts a trimmed conditional quantile table, repairs
crossing quantiles by monotone rearrangement, and interpolates it to obtain the
PDF, CDF, and inverse CDF. Configure it with `QuantileInversionConfig`.

Provider selection never falls back automatically: changing providers changes
the statistical estimator and must be explicit. Missing dependencies and model
loading failures include provider-specific installation and cache guidance.

## Installation

Choose one PyTorch extra and the required model providers:

```bash
uv sync --extra cpu --extra tabpfn
uv sync --extra cpu --extra tabicl
uv sync --extra cpu --extra foundation-models
```

TabPFN may require authentication before its checkpoint can be downloaded.
TabICL similarly loads its configured checkpoint on first use. For reproducible
or offline runs, populate the corresponding model cache in advance.

## Simulation study

The shipped study declares explicit estimator records rather than constructing
a Cartesian product implicitly. Each result row contains a mandatory human
label, provider-neutral `model_id`, and a stable full SHA-256 `estimator_id`.

```bash
uv run npcc-simstudy \
  --config configs/study.toml \
  --out results \
  --workers 4
```

The compact five-estimator probit example is in
[`notebooks/Simulation_Study.ipynb`](notebooks/Simulation_Study.ipynb). Further
examples are in
[`notebooks/foundation_model_bicop_demo.ipynb`](notebooks/foundation_model_bicop_demo.ipynb),
[`notebooks/conditional_copula.ipynb`](notebooks/conditional_copula.ipynb), and
[`notebooks/sinkhorn_projection_unconditional.ipynb`](notebooks/sinkhorn_projection_unconditional.ipynb).

## Development

```bash
uv run ruff format .
uv run ruff check . --select ANN
uv run ty check
uv run pytest tests/ -v -n auto
```

Normal tests are offline and mock provider inference. Live TabPFN and TabICL
smoke tests are opt-in and must use already cached model weights.
