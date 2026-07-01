# CFD × PIML — Patxi Sallaberry

Aerodynamics meets machine learning. This repository gathers my self-driven work toward becoming a
**CFD × Physics-Informed Machine Learning** engineer: high-fidelity CFD simulations, and neural
**surrogate models** that learn from them to predict aerodynamics in real time.

*INSA Toulouse (Mechanical Engineering) · Erasmus @ Linköping University — Fall 2026.*

---

## 🚀 Live demo — runs entirely in your browser

### → https://patxi-sallaberry.github.io/cfd-projects/

- **🏎️ [Formula Student wing aero explorer](https://patxi-sallaberry.github.io/cfd-projects/#fs)** —
  set a wing element's angle and the car speed → **downforce, drag and efficiency** instantly, plus the
  optimal angle. A neural surrogate (NACA 4412 inverted) + a finite-wing induced-drag model, fully
  client-side. *(A fast 2-D estimate for early design trade-offs — not a 3-D CFD.)*
- **🛩️ NACA 0012 surrogate** — move the angle (α) and Reynolds (Re) sliders and watch lift/drag/L-D
  update; the airfoil tilts with α. A PyTorch model exported to JavaScript.

---

## 🗺️ Repository map

```
cfd-projects/
├── piml/                       # CFD × ML roadmap (the core of this repo)
│   ├── phase0_post_processor/    ✅ CFD post-processor + Verification & Validation
│   ├── phase1_surrogate/         ✅ neural surrogate  (α, Re) → (Cl, Cd)
│   ├── phase2_pinns/             ✅ physics-informed neural networks (ODE → Navier–Stokes → airfoil)
│   ├── phase3_optimization/      ◻  shape optimization + dashboard (next)
│   ├── fs_wing_surrogate/        🏎️ Formula Student wing aero explorer (live tool)
│   └── references/               📚 PINN / Scientific-ML knowledge base (bibliography + playbook)
├── web/                        # the live demo site (Vite + React)
├── naca0012-airfoil-CFD/       # standalone CFD study — NACA 0012 airfoil
├── front-wing-CFD/             # standalone CFD study — Formula Student front wing
└── docs/                       # method memos + a full zero→expert PyTorch guide
```

---

## 🧠 PIML roadmap

| Phase | Focus | Status |
|---|---|---|
| **0** | CFD post-processor + Verification & Validation | ✅ done |
| **1** | Surrogate model (α, Re) → (Cl, Cd) — PyTorch, runs in the browser | ✅ done |
| **2** | Physics-Informed Neural Networks (PINNs) — up to 2-D Navier–Stokes | ✅ done |
| **3** | Shape optimization + dashboard | ◻ planned |

→ Start here: [`piml/`](./piml)

---

## 🌀 CFD studies (ANSYS Fluent 2026 R1 · k-ω SST)

- [`naca0012-airfoil-CFD/`](./naca0012-airfoil-CFD) — symmetric NACA 0012 airfoil, α = 15°, Re ≈ 4·10⁵.
- [`front-wing-CFD/`](./front-wing-CFD) — Formula Student dual-element front wing (NACA 4412 inverted), Cl/Cd ≈ 3.19.

---

## 🛠️ Tools

| Domain | Stack |
|---|---|
| CAD | Fusion 360, CATIA V5 |
| CFD | ANSYS Fluent 2026 R1, ANSYS Meshing |
| ML | Python · PyTorch · NeuralFoil / AeroSandbox |
| Web | Vite · React · TypeScript · Tailwind CSS · GSAP |

## 📚 Docs

- [`docs/pytorch_guide.md`](./docs/pytorch_guide.md) — a self-contained zero → expert PyTorch guide.
- [`piml/references/`](./piml/references) — curated **PINN / Scientific-ML knowledge base** (annotated bibliography + a practical playbook).
- [`docs/fluent_reports_NACA0012.md`](./docs/fluent_reports_NACA0012.md) · [`docs/fluent_domain_mesh_NACA0012.md`](./docs/fluent_domain_mesh_NACA0012.md) — Fluent setup memos.

---

*Patxi Sallaberry — INSA Toulouse · Erasmus @ Linköping University (LiU), Fall 2026*
*[LinkedIn](https://linkedin.com/in/patxi-sallaberry) · [GitHub](https://github.com/Patxi-Sallaberry)*
