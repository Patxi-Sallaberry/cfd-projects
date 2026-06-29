# Phase 2 — Physics-Informed Neural Networks (PINNs)
*Planned · Dec 2026 – Mar 2027 (LiU)*

**Goal.** Train neural networks that learn the **flow field** (velocity, pressure) by embedding the
governing **PDEs (Navier-Stokes) directly in the loss** — not just the integrated coefficients of
Phase 1. The network is differentiated w.r.t. its *inputs* (x, y, t) via autograd to evaluate the
PDE residual.

**Plan.** Start from a canonical case (1-D heat equation, then Burgers) to master the PINN
machinery, before moving to flow around the cylinder / NACA 0012. Likely tooling: PyTorch and/or
DeepXDE.

**Already in place.** The full PyTorch foundation built in Phase 1 — see
[`../../docs/pytorch_guide.md`](../../docs/pytorch_guide.md) (autograd is the key building block here).
