"""Shared test fixtures and helpers for GCKBicop and GTKBicop tests."""

import torch


def random_uv(n: int = 100, seed: int = 0) -> torch.Tensor:
  g = torch.Generator()
  g.manual_seed(seed)
  return torch.rand(n, 2, generator=g) * 0.9 + 0.05


def unit_grid(n: int = 50) -> torch.Tensor:
  """Uniform grid on (0, 1) for integration checks."""
  return torch.linspace(0.02, 0.98, n)
