# CLAUDE.md — cfd-projects

Configuration file for Claude Code. Provides project context, conventions, and guardrails for AI-assisted development in this repository.

## Project Overview

**Repo:** `Patxi-Sallaberry/cfd-projects`
**Goal:** CFD portfolio + Physics-Informed Machine Learning (PIML) roadmap
**Owner:** Patxi Sallaberry — Mechanical Engineering student, INSA Toulouse (Erasmus LiU Aug 2026–Jan 2027)
**Career target:** CFD × PIML engineer → hybrid aero/simulation consultant

## Repository Structure

```
cfd-projects/
├── piml/                        # CFD × PIML roadmap (core of the repo)
│   ├── phase0_post_processor/     ✅ CFD post-processor + Verification & Validation
│   ├── phase1_surrogate/          ✅ neural surrogate (α, Re) → (Cl, Cd)
│   ├── phase2_pinns/              ✅ PINNs (ODE → Navier–Stokes → airfoil → parametric + panel V&V)
│   ├── phase3_optimization/       ◻  shape optimization (next)
│   ├── fs_wing_surrogate/         🏎️ Formula Student wing aero explorer (live tool)
│   └── references/                📚 PINN/SciML knowledge base (bibliography + playbook)
├── web/                         # live demo site (Vite + React) → GitHub Pages
├── naca0012-airfoil-CFD/        # standalone CFD study — NACA 0012 airfoil
├── front-wing-CFD/              # standalone CFD study — Formula Student front wing
└── docs/                        # method memos + zero→expert PyTorch guide
```

## Current status (July 2026)

**Phases 0–2 are done ✅** — CFD post-processor + V&V, browser-deployed surrogate + live site, and the
full PINN suite (ODE → heat → inverse → Burgers → 2-D Navier–Stokes → airfoil → **parametric PINN**
validated against a **vortex-panel method**, plus a GPU-ready trainer). **Phase 3 (shape optimization)
is next.** See the roadmap table below.

**Stack:** Python 3 · PyTorch (CPU) · NumPy · Matplotlib · (Phase 0/1 also Pandas).
**Venv:** repo-root `.venv` — `source .venv/bin/activate`.
**PINN knowledge base:** [`piml/references/`](piml/references/) — read before answering PINN questions.

## Coding Conventions

- **Language:** Python 3, type hints encouraged
- **Style:** PEP 8, docstrings on all public functions
- **Notebooks:** clear cell outputs before committing
- **Figures:** save to `outputs/` as PNG, 150 dpi minimum, tight layout
- **Data:** raw CFD exports in `data/`, never overwrite originals
- **Commits:** `feat:`, `fix:`, `data:`, `docs:` prefixes

## CFD Reference Data

NACA 0012 — k-omega SST — Re ≈ 400k — V∞ = 30 m/s

| α (°) | Ux (m/s) | Uy (m/s) | Expected Cl trend |
|--------|----------|----------|-------------------|
| 0      | 30.00    | 0.00     | ~0                |
| 5      | 29.89    | −2.61    | ~0.5              |
| 10     | 29.54    | −5.21    | ~0.9              |
| 15     | 28.98    | −7.76    | ~1.2 (near stall) |

Formula: `Cl = L / (0.5 * rho * V² * A)` where A = chord × span

## Key Paths (Windows host)

- Fluent project: `C:\Users\salla\cao\NACA0012-CFD.wbpj`
- Formula Student: `C:\Users\salla\OneDrive\Bureau\cao\formula_student_CFD\`
- GitHub: `~/projects/cfd-projects/`
- Agency (separate): `~/websites/`

## PIML Roadmap

| Phase | Period             | Key libraries              |
|-------|--------------------|----------------------------|
| 0     | Jul–Aug 2026       | NumPy, Pandas, Matplotlib  |
| 1     | Sep–Nov 2026 (LiU) | + PyTorch                  |
| 2     | Dec 2026–Mar 2027  | + DeepXDE (PINNs)          |
| 3     | Apr 2027–Jun 2028  | + FastAPI, Next.js          |

## PINN / Scientific-ML knowledge base

A curated, **verified** reference corpus on PINNs and physics-applied ML lives in
[`piml/references/`](piml/references/) — read it before answering PINN/SciML questions or writing PINN
code, and cite from it rather than from memory:
- [`piml/references/bibliography.md`](piml/references/bibliography.md) — annotated, grouped citations.
- [`piml/references/pinn_playbook.md`](piml/references/pinn_playbook.md) — practical methods + a
  **failure-mode → fix** table, grounded in the Phase-2 experiments.

When a Phase-2/3 experiment yields a new lesson, extend `pinn_playbook.md`. Never invent a citation —
verify (web search or a source in the repo) before adding one.

## Claude Code Usage Notes

- Activate the venv before running Python: `source .venv/bin/activate` (repo-root `.venv`; the older
  `piml/phase0_post_processor/.venv` path is stale). Background runs need **absolute** paths.
- Root access available (`root@Patxi`) — no sudo needed
- Do not touch `~/websites/` directory (separate web agency project)
- Prefer modular `src/` functions over monolithic notebooks
- When generating plots, always include axis labels, legend, and title
- Heavy PINN runs → launch in the **background** with `python -u` (unbuffered logs)
