import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
  import marimo as mo

  return (mo,)


@app.cell
def _(mo):
  mo.md(r"""
    # GCKBicop & GTKBicop demo

    This notebook fits both `GCKBicop` and `GTKBicop` to data simulated from a
    **Clayton bivariate copula** and compares the learned densities side by side.

    Steps:
    1. Simulate pseudo-observations from a Clayton bicop via `pyvinecopulib`.
    2. Fit `GCKBicop` (Gaussian-copula kernels on copula scale) and `GTKBicop`
       (Gaussian kernels in Z-space) by minimising NLL + regularisation.
    3. Plot contour and surface comparisons on a shared evaluation grid.
    """)
  return


@app.cell
def _():
  import numpy as np
  import pyvinecopulib as pv
  import matplotlib.pyplot as plt
  import torch
  import torch.optim as optim
  import contextlib

  from npcc import GCKBicop, GTKBicop

  return GCKBicop, GTKBicop, contextlib, np, optim, plt, pv, torch


@app.cell
def _(np, pv):
  rho = 2.0
  cop = pv.Bicop(
    family=pv.BicopFamily.clayton,
    parameters=np.array([[rho]]),
  )
  u = cop.simulate(n=2000, seeds=[2, 2, 4])
  return cop, rho, u


@app.cell
def _(plt, rho, u):
  # contour + scatter overlay
  plt.scatter(u[:, 0], u[:, 1], alpha=0.2, s=4)
  plt.xlabel("U(0,1)")
  plt.ylabel("U(0,1)")
  plt.title("Clayton copula  θ = %.2f" % rho)
  return


@app.cell
def _():
  n_steps = 500
  lr = 0.1
  grid_size = 30
  # Penalty weights – set any to 0.0 to disable.
  lam_marg = 1.0
  lam_logits = 0.5
  lam_param = 2.0  # rho (GCK) or scale (GTK) smoothness
  return grid_size, lam_logits, lam_marg, lam_param, lr, n_steps


@app.cell
def _(
  GCKBicop,
  GTKBicop,
  grid_size,
  lam_logits,
  lam_marg,
  lam_param,
  lr,
  n_steps,
  optim,
  torch,
  u,
):
  UV = torch.tensor(u, dtype=torch.float32)

  models = {
    "GCK": GCKBicop(m_u=grid_size, m_v=grid_size),
    "GTK": GTKBicop(m_u=grid_size, m_v=grid_size),
  }
  optimizers = {
    "GCK": optim.Adam(models["GCK"].parameters(), lr=lr),
    "GTK": optim.Adam(models["GTK"].parameters(), lr=lr),
  }
  losses = {name: [] for name in models.keys()}
  for _step in range(n_steps):
    for name, model in models.items():
      optimizers[name].zero_grad()
      loss = (
        model.nll(UV)
        + lam_marg * model.marginal_penalty()
        + lam_logits * model.logit_smoothness_penalty()
      )
      if name == "GCK":
        loss += lam_param * model.rho_smoothness_penalty()
      else:
        loss += lam_param * model.scale_smoothness_penalty()
      loss.backward()
      optimizers[name].step()
      losses[name].append(loss.item())
  return losses, models


@app.cell
def _(losses, plt):
  # Loss curves for both models
  fig_loss, ax_loss = plt.subplots(figsize=(6, 2.5))
  for n, l in losses.items():
    ax_loss.plot(l, label=n)
  ax_loss.set_xlabel("step")
  ax_loss.set_ylabel("NLL + penalties")
  ax_loss.set_title("Training loss")
  ax_loss.legend()
  fig_loss.tight_layout()
  plt.show()
  return


@app.cell
def _(contextlib, np, plt, torch):
  class BicopWrapper:
    """
    Thin wrapper around a fitted GCKBicop or GTKBicop that exposes the
    interface expected by pyvinecopulib's plotting helpers:
      - ``var_types = ["c", "c"]``
      - ``pdf(u)`` accepting a NumPy array of shape [n, 2]
    """

    var_types = ["c", "c"]

    def __init__(self, model: object) -> None:
      self._model = model

    def pdf(self, u: np.ndarray) -> np.ndarray:
      UV = torch.tensor(u, dtype=torch.float32)
      with torch.no_grad():
        return self._model.pdf(UV).numpy()  # type: ignore[union-attr]

  def capture_last_fig(fn, *args, **kwargs):
    before = set(plt.get_fignums())
    fn(*args, **kwargs)
    after = set(plt.get_fignums())
    new_nums = sorted(after - before)
    return [plt.figure(n) for n in new_nums]

  @contextlib.contextmanager
  def suppress_show():
    old_show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
      yield
    finally:
      plt.show = old_show

  def bicop_plot(cop):
    cop.plot(type="contour", margin_type="norm")

  return BicopWrapper, suppress_show


@app.cell
def _(BicopWrapper, cop, mo, models, plt, pv, suppress_show, u):
  with suppress_show():
    plt.figure()
    # True contour
    cop.plot(type="contour", margin_type="norm")

    # TLL benchmark
    plt.figure()
    tll = pv.Bicop(
      family=pv.BicopFamily.tll,
    )
    tll.fit(u)
    tll.plot(type="contour", margin_type="norm")

    for m in models.values():
      plt.figure()
      pv._python_helpers.bicop.bicop_plot(
        BicopWrapper(m), plot_type="contour", margin_type="norm"
      )

  figs = [plt.figure(n) for n in plt.get_fignums()]
  mo.hstack([mo.as_html(fig) for fig in figs], widths="equal")
  return


if __name__ == "__main__":
  app.run()
