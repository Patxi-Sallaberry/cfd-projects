# Phase 0 — CFD Post-Processor & Python Stack
*PIML Roadmap · July–August 2026*

## Objective

Build a Python post-processor on real NACA 0012 CFD data (ANSYS Fluent 2026 R1) as the foundation of the Physics-Informed ML roadmap. Output: clean Cl/Cd vs α plots, structured CSV data, and a notebook ready to feed Phase 1 surrogate models.

## Context

This phase bridges the CFD work completed in June 2026 (NACA 0012, k-omega SST, ~56k polyhedral cells) and the ML stack introduced at LiU in Phase 1. The goal is to be comfortable with the scientific Python stack and to have exploitable, versioned CFD data before Erasmus departure (August 2026).

## Stack

| Library    | Version | Role                          |
|------------|---------|-------------------------------|
| Python     | 3.x     | Base runtime                  |
| NumPy      | 2.5     | Array ops, angle decomposition |
| Pandas     | 3.0.3   | Data ingestion & structuring  |
| Matplotlib | 3.11    | Cl/Cd plots, pressure contours |

*PyTorch added in Phase 1 · DeepXDE in Phase 2 · FastAPI + Next.js in Phase 3*

## CFD Data Source

**Simulation:** NACA 0012, chord 200 mm, span 300 mm
**Solver:** ANSYS Fluent 2026 R1 — k-omega SST, steady, Re ≈ 400k
**Project:** `C:\Users\salla\cao\NACA0012-CFD.wbpj`

| α (°) | Ux (m/s) | Uy (m/s) |
|--------|----------|----------|
| 0      | 30.00    | 0.00     |
| 5      | 29.89    | −2.61    |
| 10     | 29.54    | −5.21    |
| 15     | 28.98    | −7.76    |

Cl/Cd exported from Fluent Report Definitions across α = 0°, 5°, 10°, 15°.

## Deliverables

- `data/naca0012_clcd.csv` — raw Fluent export
- `notebooks/01_clcd_analysis.ipynb` — Cl/Cd vs α curves + polar
- `src/postprocessor.py` — reusable functions (load, plot, export)
- `outputs/` — PNG figures for GitHub README and LiU Formula Student outreach

## Directory Structure

```
piml/
├── phase0_post_processor/
│   ├── data/
│   ├── notebooks/
│   ├── src/
│   ├── outputs/
│   └── requirements.txt
├── phase1_surrogate/      # LiU — Sep–Nov 2026
├── phase2_pinns/          # Dec 2026–Mar 2027
└── phase3_optimization/   # Apr 2027–Jun 2028
```

## Phase Roadmap

| Phase | Period          | Focus                                      |
|-------|-----------------|--------------------------------------------|
| **0** | Jul–Aug 2026    | Python stack + CFD Post-Processor (here)   |
| 1     | Sep–Nov 2026    | PyTorch + Surrogate Model NACA v1 (LiU)    |
| 2     | Dec–Mar 2027    | PINNs via DeepXDE — cylinder → NACA 0012  |
| 3     | Apr 2027–Jun 2028 | Shape optimization + dashboard + publication |

## How to Run

```bash
# Clone & setup
git clone https://github.com/Patxi-Sallaberry/cfd-projects.git
cd cfd-projects/piml/phase0_post_processor

# Create venv & install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run post-processor
python src/postprocessor.py

# Or open notebook
jupyter notebook notebooks/01_clcd_analysis.ipynb
```

## Next Step → Phase 1

At LiU (September 2026): train a first surrogate model (MLP or GPR) mapping α → (Cl, Cd) on the Phase 0 dataset. Contact LiU Formula Student team before August 2026 to coordinate.
