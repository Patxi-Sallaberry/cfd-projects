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
├── naca0012/              # First CFD project (ANSYS Fluent, June 2026)
│   ├── geometry/          # STEP files, Fusion 360 exports
│   ├── results/           # Cl/Cd tables, screenshots
│   └── README.md
├── formula_student/       # Front wing 2-element (WIP, v2 geometry)
│   └── ...
└── piml/                  # Physics-Informed ML roadmap
    ├── phase0_post_processor/   # ← ACTIVE (Jul–Aug 2026)
    ├── phase1_surrogate/
    ├── phase2_pinns/
    └── phase3_optimization/
```

## Active Phase: Phase 0 — Post-Processor

**Period:** July–August 2026 (pre-Erasmus sprint)
**Stack:** Python 3.x · NumPy 2.5 · Pandas 3.0.3 · Matplotlib 3.11
**Venv:** `piml/phase0_post_processor/.venv` (already created)
**Data source:** ANSYS Fluent 2026 R1 NACA 0012 export (Cl/Cd vs α)

### Current tasks
- [ ] Ingest `naca0012_clcd.csv` into Pandas DataFrame
- [ ] Plot Cl vs α, Cd vs α, polar Cl/Cd
- [ ] Export figures to `outputs/` for GitHub README
- [ ] Write `src/postprocessor.py` as reusable module
- [ ] Finalize Phase 0 README for LiU Formula Student outreach

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

## Claude Code Usage Notes

- Always activate venv before running Python: `source piml/phase0_post_processor/.venv/bin/activate`
- Root access available (`root@Patxi`) — no sudo needed
- Do not touch `~/websites/` directory (separate web agency project)
- Prefer modular `src/` functions over monolithic notebooks
- When generating plots, always include axis labels, legend, and title
