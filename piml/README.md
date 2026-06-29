# PIML — CFD × Machine Learning roadmap

Building a hybrid **CFD × Physics-Informed Machine Learning** workflow on the NACA 0012 airfoil.
Each phase is self-contained with its own README.

| Phase | Folder | What | Status |
|---|---|---|---|
| 0 | [`phase0_post_processor`](./phase0_post_processor) | Python CFD post-processor + **V&V** (reference polar vs my own Fluent run) | ✅ |
| 1 | [`phase1_surrogate`](./phase1_surrogate) | **PyTorch surrogate** (α, Re) → (Cl, Cd), 1D & 2D + browser demo | ✅ |
| 2 | [`phase2_pinns`](./phase2_pinns) | **Physics-Informed NNs** — ODE, PDEs, inverse problem, up to 2-D **Navier–Stokes** | ✅ |
| 3 | [`phase3_optimization`](./phase3_optimization) | Shape **optimization** using the surrogate + dashboard | ◻ planned |

**Live demo (Phase 1):** https://patxi-sallaberry.github.io/cfd-projects/

New to PyTorch? The full guide is at [`../docs/pytorch_guide.md`](../docs/pytorch_guide.md).

### The thread tying it together
CFD is accurate but slow. A **surrogate** learns its input→output mapping and predicts instantly,
making **optimization** (thousands of evaluations) feasible. **PINNs** go further by baking the
physics (Navier-Stokes) into the model itself.
