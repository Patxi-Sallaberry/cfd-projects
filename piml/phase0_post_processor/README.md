# Phase 0 — CFD Post-Processor (V&V)
*PIML Roadmap · July–August 2026*

A Python post-processor for **NACA 0012** aerodynamic coefficients, built on a
**Verification & Validation (V&V)** workflow: a documented, reproducible *reference*
polar is compared against my own **ANSYS Fluent** simulations, and the deviation is
quantified. This phase is the data foundation for the Phase 1 surrogate models.

> **Why V&V?** Training a surrogate (Phase 1) or a PINN (Phase 2) on noisy data teaches
> the model my errors. A clean, cited ground truth lets me separate *model* error from
> *data* error — and honestly report how my own CFD compares to it. As shown below, this
> workflow immediately **caught a setup bug** in my first Fluent run.

---

## V&V result — debugging my Fluent setup, iteration by iteration

![NACA 0012 — reference vs my Fluent run](results/figures/naca0012_vv_polar.png)

**Iteration 2** (Force Vectors corrected) vs the reference polar:

| α | Cl (ref) | Cl (mine) | ΔCl | Cd (ref) | Cd (mine) | ΔCd |
|---|---|---|---|---|---|---|
| 0° | ≈ 0 | −0.007 | — | 0.007 | 0.196 | +2792 % |
| 5° | +0.623 | +0.075 | −88 % | 0.011 | 0.189 | +1607 % |
| 10° | +1.035 | +0.156 | −85 % | 0.021 | 0.166 | +697 % |
| 15° | +1.219 | +0.231 | −81 % | 0.052 | 0.126 | +144 % |

### Iteration 1 → 2: what the Force Vector fix changed
My first run reported **negative, growing-negative Cl** — impossible on a *symmetric* airfoil.
Root cause: the drag direction was `(cos α, +sin α)` while the inlet flow is `(cos α, ∓sin α)`,
so the `sin α` sign was inverted. Fixing it to `drag = (cos α, −sin α, 0)`,
`lift = (sin α, cos α, 0)` (see [`docs/fluent_reports_NACA0012.md`](../../docs/fluent_reports_NACA0012.md))
**restored the correct Cl sign and trend** ✓.

### What is still wrong (and the proof it isn't the Force Vector)
The magnitudes remain non-physical: **Cl is 5–8× too low** and **Cd is ~20–25× too high and still
*decreases* with α** (impossible — drag grows with incidence). The smoking gun is **α = 0°**, where
there is *no* Force Vector rotation at all: `Cd(0°) = 0.1955`, essentially unchanged from the
flawed run's `0.1957`. So the huge drag was never about the Force Vector — it comes from:

1. **Domain blockage.** `y ∈ [−0.3, 0.3]` is only ±1.5 chords tall: the airfoil chokes the channel,
   suppressing lift and inflating drag.
2. **Coarse mesh.** 56k cells, no inflation layers, y⁺ unresolved, continuity ≈ 9·10⁻⁴ (> 10⁻⁴).

**Conclusion:** still *not* usable for Phase 1 training — kept as a documented V&V iteration. Next
run: domain ±10–20 chords + inflation-layer mesh (y⁺ ≈ 1). Raw artifacts archived under
[`results/figures/fluent_runs/`](results/figures/fluent_runs).

---

## Reference data (ground truth)

Full provenance in [`data/SOURCES.md`](data/SOURCES.md):

| | |
|---|---|
| Tool | **NeuralFoil 0.3.2** (`xlarge`), via AeroSandbox — an **XFOIL surrogate** |
| Conditions | NACA 0012 · **Re = 4·10⁵** · Mach 0 · Ncrit 9 |
| Reproducible | `python src/generate_reference.py` regenerates `data/naca0012_reference.csv` |

> ⚠️ **Integrity note.** The reference is a numerical **prediction** (XFOIL-class), *not*
> experimental data and *not* my Fluent results. My runs are shown as overlaid points labelled
> *"Fluent (mine)"* — never presented as the reference.

## My CFD model (ANSYS Fluent)

Static-pressure field on my own NACA 0012 model (α = 15°) — *my simulation, coarse mesh*:

![My Fluent NACA 0012 — static pressure, α=15°](../../naca0012-airfoil-CFD/naca0012_pressure_alpha15_view1.png)

> Chord 200 mm · span 300 mm · k-ω SST · steady · Re ≈ 400 000. See
> [`naca0012-airfoil-CFD/`](../../naca0012-airfoil-CFD).

---

## How Cl / Cd are computed

Coefficients come from **forces**, not from the inlet velocity components:

```
Cl = L / (q∞ · S_ref)      Cd = D / (q∞ · S_ref)      q∞ = ½ · ρ · V∞²
```

with `ρ = 1.225`, `V∞ = 30 m/s`, `S_ref = chord × span = 0.06 m²`. In this run I read the Cl/Cd
coefficients directly from Fluent's *Report Definitions* (the force path above is kept as a
fallback in `postprocessor.py`).

## Stack

| Library | Version | Role |
|---|---|---|
| Python | 3.12 | Base runtime |
| NumPy | 2.5 | Array ops |
| Pandas | 3.0.3 | Data ingestion & structuring |
| Matplotlib | 3.11 | Polar / coefficient plots |
| NeuralFoil + AeroSandbox | 0.3.2 / 4.2.9 | Reference polar generation only |

## Directory structure

```
phase0_post_processor/
├── data/
│   ├── naca0012_reference.csv   # ground truth (generated, cited)
│   ├── naca0012_fluent.csv      # my Fluent run (flawed, kept for V&V)
│   └── SOURCES.md               # provenance & caveats
├── src/
│   ├── generate_reference.py    # regenerates the reference polar
│   └── postprocessor.py         # 3-panel V&V plot + error report
├── results/figures/
│   ├── naca0012_vv_polar.png    # reference vs Fluent (Cl, Cd, L/D)
│   └── fluent_runs/             # raw convergence/residual plots per α
└── requirements.txt
```

## How to run

```bash
cd cfd-projects/piml/phase0_post_processor

# (optional) regenerate the reference polar — needs: pip install neuralfoil aerosandbox
python src/generate_reference.py

# 3-panel V&V plot + console error table
python src/postprocessor.py
```

---

## Roadmap

| Phase | Period | Focus |
|---|---|---|
| **0** | Jul–Aug 2026 | Python stack + CFD post-processor, V&V *(here)* |
| 1 | Sep–Nov 2026 | PyTorch + surrogate model α → (Cl, Cd) — LiU |
| 2 | Dec 2026–Mar 2027 | PINNs via DeepXDE — cylinder → NACA 0012 |
| 3 | Apr 2027–Jun 2028 | Shape optimization + dashboard + write-up |

## Next step

Re-run the NACA 0012 sweep with corrected Force Vectors, a larger domain and an inflation-layer
mesh; fill `data/naca0012_fluent.csv` with the corrected Cl/Cd and re-run `postprocessor.py`.
Once the Fluent points land on the reference polar, the validated dataset feeds the Phase 1
surrogate (MLP / GPR mapping α → Cl, Cd) at LiU.
