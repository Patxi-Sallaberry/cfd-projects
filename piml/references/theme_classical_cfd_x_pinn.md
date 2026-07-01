# Theme — bridging classical CFD/numerics with PINNs

A recurring, fertile idea across the literature (and a strong **Phase-3 candidate**): rather than train a
PINN by *naïve* residual minimization, **borrow decades of numerical-analysis wisdom** — pressure
correction, conservation, domain decomposition, variational forms, adaptive meshing, time-marching — and
fold it into the PINN's **loss, sampling, or architecture**. A classical numerical idea often makes a
better loss term (or a better-conditioned formulation) than a plain PDE residual.

All references below are detailed in [`bibliography.md`](bibliography.md); the mechanics are in
[`pinn_playbook.md`](pinn_playbook.md).

## The map: classical idea → its PINN counterpart
| Classical CFD / numerics idea | PINN counterpart | Reference |
|---|---|---|
| **SIMPLE** pressure–velocity coupling (pressure correction) | a derived *coupling-correction* loss term | **SIMPLE-PINN** — Wei et al. (2026), arXiv:2603.24013 |
| **Spectral methods** + pressure correction | spectral analysis in the loss for sharp interfaces | **SP-PINN** — Ding et al. (2025), Appl. Math. Mech. 46(2) |
| **Finite-volume conservation** on cells | conservative residual on subdomains | **cPINN** — Jagtap et al. (2020) |
| **Domain decomposition** (Schwarz, etc.) | space/space-time subdomains stitched with interface losses | **XPINN** — Jagtap & Karniadakis (2020) |
| **FEM / spectral-element** weak (variational) form | integrate the residual against test functions | **VPINN / hp-VPINN** — Kharazmi et al. (2019, 2021) |
| **Parareal** time-parallel integration | coarse/fine time decomposition | **PPINN** — Meng et al. (2020) |
| **Mixed FEM** (introduce auxiliary variables) | first-order system → lower-order autograd | **FO-PINN** — Gladstone et al. (2022) |
| **Adaptive mesh refinement (AMR)** | residual-based adaptive resampling of collocation | **RAR** in DeepXDE — Lu et al. (2021) |
| **Streamfunction–vorticity / staggered grids** (divergence-free by construction) | output ψ so `∇·u = 0` is exact | NSFnets (Jin et al. 2021); *this repo §2.6* |

## Why this pattern is powerful
- **Better conditioning, not just more compute.** Naïve residual PINNs stall on stiff/coupled problems
  (Krishnapriyan et al. 2021). Importing a scheme that *already* handles the stiffness (pressure coupling,
  conservation, causality/time-marching) attacks the root cause.
- **Physics/constraints for free.** A stream-function output makes continuity exact; a conservative form
  guarantees flux balance — the network no longer has to *learn* what a formulation can *guarantee*.
- **Interpretability & trust.** Tying the PINN to a known algorithm gives a validation path and a mental
  model — the opposite of a black box.

## This repo's own evidence
- **Stream-function formulation** (§2.6): `u=ψ_y, v=−ψ_x` made continuity exact and doubled the recovered
  lift — the streamfunction–vorticity idea, ported to a PINN.
- **Adam → L-BFGS** two-stage optimization (classical quasi-Newton polish) — with the caveat that L-BFGS
  needs a deterministic batch (we saw it *break* generalization on a frozen per-point-α batch).
- **Validate against the classical tool**: the vortex-**panel method** was the right reference for
  inviscid lift, and a reminder that sometimes the classical method simply *is* the better tool (§2.6).

## Phase-3 hooks (concrete ideas)
1. **SIMPLE-PINN on the airfoil**: apply the pressure-correction coupling loss to the NACA 4412 case and
   compare against the panel method + the plain ψ PINN.
2. **Conservative / variational form** for the transonic or high-Re regime where naïve residuals struggle.
3. **Operator-learning bridge**: combine a classical solver's structure with **DeepONet/FNO** to amortize
   over angle of attack / Reynolds (the principled successor to the §2.6 parametric PINN).

> Maintenance: when a new "classical-idea-into-PINN" paper is added to the bibliography, add its row to
> the map above.
