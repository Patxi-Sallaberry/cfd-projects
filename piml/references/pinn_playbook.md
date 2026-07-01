# PINN playbook — practical guide (methods, pitfalls, fixes)

Actionable knowledge for building physics-informed neural networks, grounded in the literature
(see [`bibliography.md`](bibliography.md)) **and** in this repo's own Phase-2 experiments. Citations use
author (year); full details in the bibliography.

> **For Claude Code:** treat this as the working reference when writing or debugging PINNs in this repo.
> Prefer these patterns; when a user hits a symptom, map it to the matching row in *Failure modes* below.

---

## 1. What a PINN is — and when *not* to use one
A PINN represents the solution field with a neural network and puts the **governing equation's residual
into the loss** (via autodiff w.r.t. the *inputs*), plus initial/boundary/data terms. With little or no
data, minimizing the residual yields the solution. (Raissi et al. 2019)

**Use a PINN when:** the problem is an **inverse / data-assimilation** task (unknown parameter, field
reconstruction from sparse data — Raissi et al. 2020 "Hidden Fluid Mechanics"), a **high-dimensional**
PDE, an irregular domain, or you need a **mesh-free, differentiable** surrogate. This is where PINNs beat
classical solvers.

**Do *not* reach for a PINN when** a mature classical method already solves the forward problem cheaply
and accurately. → *This repo's own lesson (§2.6):* for **exact inviscid 2-D lift**, a **vortex-panel
method** runs in milliseconds and nails `Cl(α)`; the PINN under-predicted the lift slope. Knowing which
tool fits which job is the point.

## 2. The core recipe
```
loss = w_pde * mean(residual(u, u_x, u_xx, ...)²)      # PDE, on collocation points
     + w_bc  * mean(BC²) + w_ic * mean(IC²)            # boundary / initial conditions
     + w_data* mean((u - u_meas)²)                     # optional data (required for inverse problems)
```
- Derivatives come from `torch.autograd.grad(..., create_graph=True)` (differentiate **output w.r.t.
  input**), nested for higher order. Keep each input coordinate as a **separate tensor** to differentiate per-direction.
- Use a **smooth activation** (`tanh`, `sin`, `swish`) — never ReLU when you need 2nd derivatives (ReLU'' ≡ 0).

## 3. Failure modes → fixes (diagnostic table)
| Symptom | Likely cause | Fix (reference) |
|---|---|---|
| Loss drops but solution wrong / trivial | one loss term dominates the gradient | **weight the terms**: learning-rate annealing (Wang, Teng, Perdikaris 2021) or **NTK weighting** (Wang, Yu, Perdikaris 2022); or self-adaptive weights |
| Can't fit sharp fronts / high frequencies | **spectral bias** of MLPs | **Fourier feature** input embedding; **adaptive activations** (Jagtap et al. 2020) |
| Fails on convection/reaction/stiff or long-time | ill-conditioned landscape, non-causal fitting | **curriculum / time-marching**, "**respect causality**", sequence-to-sequence (Krishnapriyan et al. 2021; Wang, Sankaran, Wang, Perdikaris 2023) |
| Stalls at a mediocre residual | Adam plateaus (1st-order) | **polish with L-BFGS** (2nd-order) after Adam — the standard two-stage recipe (Liu & Nocedal 1989). *Caveat (this repo):* L-BFGS needs a **deterministic** loss — freeze the batch, or it over-fits & can break generalization |
| Under-resolved where the solution varies | uniform collocation too sparse there | **residual-based adaptive refinement (RAR)** — add points where the residual is large (Lu et al. 2021, DeepXDE) |
| BCs only approximately satisfied | soft penalty too weak | **hard constraints**: bake the BC into an ansatz `u = g(x) + B(x)·NN(x)` so it holds exactly (Lagaris et al. 1998) |
| Huge domain / multiscale, one net can't cope | limited capacity per region | **domain decomposition**: XPINN / cPINN (Jagtap et al. 2020), hp-VPINN (Kharazmi et al. 2021) |

## 4. Formulation choices that matter (hard-won)
- **Normalize inputs/outputs** to ~O(1). *This repo (§2.6):* feeding α in radians (~0.2) made `tanh`
  barely react; rescaling α to [−1, 1] fixed it. Same for spatial coords and target magnitudes.
- **Choose the dependent variable to bake in a constraint.** *This repo:* switching from velocity
  `(u,v)` to a **stream function ψ** (`u=ψ_y, v=−ψ_x`) makes continuity `u_x+v_y=0` *exact* and lets you
  pin the body as a streamline — it doubled the recovered lift. (cf. NSFnets, Jin et al. 2021, which
  offers both velocity–pressure and vorticity–stream-function forms.)
- **Global vs local quantities.** Some targets (e.g. **circulation → lift**) are *global* (a contour
  integral), weakly constrained by pointwise residuals. *This repo:* re-parametrizing ψ as
  `freestream + Γ(α)·vortex + NN` did **not** help — re-parametrization doesn't change the constrained
  optimum; the network just left `Γ≈0`. Lesson: a formulation change only helps if it changes *what is
  represented*, not merely *how*.
- **Reduce autograd order** with a first-order system when 2nd derivatives are costly/unstable (FO-PINN,
  Gladstone et al. 2022).
- **Weighting is a real knob**, not a detail: start terms at comparable magnitudes; if one residual is
  ~10³× another, its square dominates and the rest is ignored.
- **Borrow a classical scheme into the loss.** Incompressible NS is stiff because of velocity–pressure
  coupling; classical CFD handles it with **SIMPLE**-type pressure-correction. **SIMPLE-PINN** (Wei et
  al. 2026) derives a velocity–pressure *coupling correction loss* from that algorithm → precise
  data-free flow (lid-driven cavity). General lesson: a decades-old numerical idea can become a better
  loss term than a naïve residual.

## 5. Validation & verification (non-negotiable)
- A low loss **does not** guarantee the right answer (trivial solutions exist). **Always validate**
  against a reference: analytic solution, a classical solver (finite-difference / FEM / **panel method**),
  or held-out data. *This repo* validated PINNs against exact solutions (heat, Kovasznay), a
  finite-difference solver (Burgers), and a vortex-panel method (airfoil lift).
- Report honestly: capturing the *trend* (linear `Cl(α)`, correct zero-lift angle) but missing the
  *magnitude* is a real, publishable finding — say so.

## 6. Beyond a single PINN — operator learning (for parametric problems)
A PINN solves **one** case per training. To cover a *family* (varying angle of attack, Re, geometry, or
initial condition), learn the **solution operator**:
- **DeepONet** (Lu et al. 2021) — branch net (input function) × trunk net (query point).
- **FNO** (Li et al. 2021) — global convolution in Fourier space; excellent for parametric/turbulent PDEs.
- These *amortize*: train once, evaluate any instance instantly — the principled version of this repo's
  "parametric PINN" idea. Natural **next step** for the roadmap.

## 7. Software
- **DeepXDE** (Python) — batteries-included PINNs/operators; on this repo's Phase-2 roadmap.
- **NVIDIA Modulus** — industrial-scale framework. **NeuralPDE.jl** (SciML/Julia). **jaxpi** (JAX) —
  implements the "Expert's Guide" best practices.

## 8. CFD-specific notes
- **High Reynolds / turbulence is hard** for PINNs: the boundary layer scales as ~1/√Re and a plain NS
  PINN won't resolve it. *This repo (§2.5):* solve the **right model per regime** — inviscid/potential
  for the visible high-Re outer flow, viscous NS only at low (laminar) Re. Turbulence closure with PINNs
  remains open research.
- Good CFD entry points: **NSFnets** (Jin et al. 2021), heat transfer (Cai et al. 2021), data-driven
  reconstruction (Raissi et al. 2020).
- **Multiphase / two-phase flow**: sharp gas–liquid interfaces stress the MLP's spectral bias. A
  **spectrum-based PINN (SP-PINN)** with a pressure-correction module (Ding et al. 2025) targets exactly
  the interface/velocity-peak regions — a concrete pairing of the spectral-bias fix (row 2 above) with
  a physics-based correction.

---
*Maintenance:* when a Phase-2/3 experiment teaches a new lesson, add a row/bullet here and cross-link the
relevant README section. This file is meant to grow with the project.
