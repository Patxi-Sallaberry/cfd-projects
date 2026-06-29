# CFD × PIML — Patxi Sallaberry

Aerodynamics meets machine learning. This repository gathers my self-driven work toward becoming a
**CFD × Physics-Informed Machine Learning** engineer: high-fidelity CFD simulations, and neural
**surrogate models** that learn from them to predict aerodynamics in real time.

*INSA Toulouse (Mechanical Engineering) · Erasmus @ Linköping University — Fall 2026.*

---

## 🚀 Live demo

**A neural surrogate of the NACA 0012 polar, running in your browser:**

### → https://patxi-sallaberry.github.io/cfd-projects/

Move the angle-of-attack (α) and Reynolds (Re) sliders and watch lift, drag and L/D update
instantly — the airfoil tilts with α, and a PyTorch model (exported to JavaScript) runs the
prediction **fully client-side**.

---

## 🗺️ Repository map

```
cfd-projects/
├── piml/                       # CFD × ML roadmap (the core of this repo)
│   ├── phase0_post_processor/    ✅ CFD post-processor + Verification & Validation
│   ├── phase1_surrogate/         ✅ neural surrogate  (α, Re) → (Cl, Cd)
│   ├── phase2_pinns/             ◻  physics-informed neural networks (next)
│   └── phase3_optimization/      ◻  shape optimization + dashboard
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
| **2** | Physics-Informed Neural Networks (PINNs) | 🚧 in progress |
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
- [`docs/fluent_reports_NACA0012.md`](./docs/fluent_reports_NACA0012.md) · [`docs/fluent_domain_mesh_NACA0012.md`](./docs/fluent_domain_mesh_NACA0012.md) — Fluent setup memos.

---

*Patxi Sallaberry — INSA Toulouse · Erasmus @ Linköping University (LiU), Fall 2026*
*[LinkedIn](https://linkedin.com/in/patxi-sallaberry) · [GitHub](https://github.com/Patxi-Sallaberry)*
