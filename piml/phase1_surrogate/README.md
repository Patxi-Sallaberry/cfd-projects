# Phase 1 — Surrogate Model (α → Cl, Cd)
*PIML Roadmap · Sep–Nov 2026 · LiU*

A first **surrogate model**: a small neural network (PyTorch MLP) that learns the NACA 0012
polar `α → (Cl, Cd)` from data and predicts it **instantly** — replacing an expensive CFD run
with a learned function. This is the "hello world" of ML-for-CFD and the basis for Phase 2 (PINNs).

---

## Result

![Surrogate fit — Cl and Cd vs alpha](results/figures/surrogate_fit.png)

| Output | Test RMSE | Test R² |
|---|---|---|
| Cl | 0.0008 | 1.000 |
| Cd | 0.00002 | 1.000 |

The surrogate reproduces the polar on **held-out test points** it never saw during training.

> **Why so perfect?** The data is a smooth, noiseless function (a NeuralFoil/XFOIL polar), so a tiny
> MLP fits it almost exactly. This is a *learning milestone*, not a hard ML problem — the real
> difficulty comes later with **noisy/sparse** CFD data and **higher-dimensional** inputs (see Next step).

## Data

`data/naca0012_surrogate_dataset.csv` — 89 points, α ∈ [−6°, 16°] (step 0.25°), Re = 4·10⁵,
generated with **NeuralFoil** (XFOIL surrogate) via `src/make_dataset.py`.

> ⚠️ **Provenance.** Training data is an XFOIL-class prediction, *not* my own CFD (my Phase 0 Fluent
> run was non-physical — see `../phase0_post_processor`). Using a clean, cited ground truth is the
> right choice for learning the ML pipeline.

## Model

`src/train_surrogate.py`: MLP `1 → 64 → 64 → 2` (tanh), Adam, MSE loss on standardized data,
80/20 train/test split. Trained weights saved to `results/surrogate_naca0012.pt`.

## How to run

```bash
cd cfd-projects/piml/phase1_surrogate
pip install -r requirements.txt          # torch + neuralfoil + aerosandbox
python src/make_dataset.py               # (re)generate the dataset
python src/train_surrogate.py            # train + evaluate + plot
```

## Next step

- Add **Reynolds** as a second input → 2-D surrogate `(α, Re) → (Cl, Cd)`.
- Then **Phase 2**: physics-informed networks (PINNs) that learn the *flow field* under PDE
  constraints, not just the integrated coefficients.
