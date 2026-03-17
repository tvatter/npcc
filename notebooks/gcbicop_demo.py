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
    # GCBicop demo

    This notebook fits a `GCBicop` model to data simulated from a **Clayton bivariate copula** and compares the learned density to the true density side by side.

    Steps:
    1. Simulate pseudo-observations from a Clayton bicop via `pyvinecopulib`.
    2. Fit `GCBicop` by minimising the negative log-likelihood plus a
       small marginal uniformity penalty.
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

    from npcc.gcbicop import GCBicop

    return GCBicop, np, optim, plt, pv, torch


@app.cell
def _(np, pv):
    rho = 0.65
    cop = pv.Bicop(
      family=pv.BicopFamily.clayton,
      parameters=np.array([[rho]]),
    )
    u = cop.simulate(n=2000, seeds=[2, 2, 4])
    return cop, rho, u


@app.cell
def _(cop, plt, rho, u):
    # pyvinecopulib built-in contour + scatter overlay
    cop.plot(type="contour", margin_type="norm")
    plt.scatter(u[:, 0], u[:, 1], alpha=0.2, s=4)
    plt.xlabel("U(0,1)")
    plt.ylabel("U(0,1)")
    plt.title("Clayton copula  θ = %.2f" % rho)
    return


@app.cell
def _(GCBicop, np, torch):
    class GCBicopWrapper:
      """
      Thin wrapper around a fitted GCBicop that exposes the interface
      expected by pyvinecopulib's plotting helpers:
        - ``var_types = ["c", "c"]``
        - ``pdf(u)`` accepting a NumPy array of shape [n, 2]
      """

      var_types = ["c", "c"]

      def __init__(self, model: GCBicop) -> None:
        self._model = model

      def pdf(self, u: np.ndarray) -> np.ndarray:
        UV = torch.tensor(u, dtype=torch.float32)
        with torch.no_grad():
          return self._model.pdf(UV).numpy()

    return (GCBicopWrapper,)


@app.cell
def _():
    n_steps = 500
    lr = 0.05
    # Small weight on the marginal uniformity penalty.
    # Set to 0.0 to train with pure NLL.
    penalty_weight = 0.1
    return lr, n_steps, penalty_weight


@app.cell
def _(GCBicop, lr, n_steps, optim, penalty_weight, torch, u):
    UV = torch.tensor(u, dtype=torch.float32)

    model = GCBicop(m_u=20, m_v=20)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses: list[float] = []
    for step in range(n_steps):
      optimizer.zero_grad()
      loss = model.nll(UV) + penalty_weight * model.marginal_penalty()
      loss.backward()
      optimizer.step()
      losses.append(loss.item())
    return losses, model


@app.cell
def _(losses: list[float], plt):
    # Loss curve – should decrease smoothly
    fig_loss, ax_loss = plt.subplots(figsize=(5, 2.5))
    ax_loss.plot(losses)
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("NLL + penalty")
    ax_loss.set_title("Training loss")
    fig_loss.tight_layout()
    plt.show()
    return


@app.cell
def _(GCBicopWrapper, model, pv):
    model_wrapper = GCBicopWrapper(model)
    pv._python_helpers.bicop.bicop_plot(
      model_wrapper, plot_type="contour", margin_type="norm"
    )
    return


if __name__ == "__main__":
    app.run()
