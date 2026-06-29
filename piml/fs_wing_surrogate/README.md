# FS Wing Aero Explorer — surrogate

A Formula-Student-oriented tool: predict a wing element's **downforce, drag and efficiency** in real
time and find the optimal angle — running live in the browser, part of the project site.

**Live:** https://patxi-sallaberry.github.io/cfd-projects/#fs

## What it does

- **Airfoil:** NACA 4412 mounted **inverted** (cambered → downforce), the profile of my
  [`front-wing-CFD`](../../front-wing-CFD) study.
- A 2-D surrogate `(angle, Re) → (Cl, Cd)` (NeuralFoil data, validation **R² ≈ 0.9999**) runs
  **client-side** (same tech as Phase 1).
- From the car speed it computes the Reynolds number, then downforce/drag in **Newtons** for a
  representative wing element, adds a **finite-wing induced-drag** term `Cd_i = Cl²/(π·AR·e)`, and
  reports **downforce, drag and L/D**.
- An **optimizer** sweeps the angle to find the **max downforce** and the **best efficiency** setting.

## Honesty / limits

This is a **fast first-order estimate** (2-D airfoil surrogate + induced-drag correction), ideal for
early trade-offs and setup exploration. It is **not** a 3-D CFD of the full wing — no ground effect,
no multi-element slot flow. For final numbers, run 3-D CFD (see [`front-wing-CFD`](../../front-wing-CFD)).
Stating this clearly is part of the engineering: know what your tool can and cannot say.

## Physical quantities

| Symbol | Meaning | Unit |
|---|---|---|
| **α** | wing angle of attack | degrees (°) |
| **V** | car speed | m/s |
| **Re** | Reynolds number, `ρ·V·c/μ` | dimensionless |
| **Cl, Cd** | 2-D lift / drag coefficients | dimensionless |
| **AR** | aspect ratio, `span / chord` | dimensionless |
| **Cd_i** | induced drag, `Cl²/(π·AR·e)` (e = Oswald factor ≈ 0.85) | dimensionless |
| **downforce / drag** | `Cl·½ρV²·S` / `Cd_total·½ρV²·S` | N |

Representative wing element: chord 0.30 m × span 1.20 m (AR = 4), ρ = 1.225 kg/m³.

## Build / regenerate the model

```bash
cd piml/fs_wing_surrogate
pip install neuralfoil aerosandbox torch
python src/build_fs_surrogate.py
# -> trains the surrogate and exports web/public/fs_wing_model.json (loaded by the web tool)
```

The web tool is [`web/src/components/FsAeroExplorer.tsx`](../../web/src/components/FsAeroExplorer.tsx);
it loads `fs_wing_model.json` and runs the inference in the browser (forward pass in `web/src/lib/model.ts`).
