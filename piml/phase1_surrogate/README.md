# Phase 1 — Surrogate Model (α → Cl, Cd)
*PIML Roadmap · Sep–Nov 2026 · LiU*

A first **surrogate model**: a small neural network (PyTorch MLP) that learns the NACA 0012
polar `α → (Cl, Cd)` from data and predicts it **instantly** — replacing an expensive CFD run with
a learned function. This README is also a **self-contained study sheet**: it explains every concept
and every line of `src/train_surrogate.py`.

> New to PyTorch itself? Read the companion guide [`docs/pytorch_guide.md`](../../docs/pytorch_guide.md)
> (zero → expert), then come back here for the applied walkthrough.

## Physical quantities

| Symbol | Meaning | Unit |
|---|---|---|
| **α** | Angle of attack — tilt of the airfoil into the oncoming flow | degrees (°) |
| **Re** | Reynolds number — ratio of inertial to viscous forces, `Re = ρ·V·c/μ` | dimensionless |
| **Cl** | Lift coefficient — non-dimensional lift, `Cl = L / (½ρV²·S)` | dimensionless |
| **Cd** | Drag coefficient — non-dimensional drag, `Cd = D / (½ρV²·S)` | dimensionless |
| **L/D** | Lift-to-drag ratio — aerodynamic efficiency (higher = better) | dimensionless |

*Reference conditions for the dataset: free-stream speed V = 30 m/s, chord c = 0.20 m, air density
ρ = 1.225 kg/m³ → Re ≈ 4·10⁵.*

---

## Result

![Surrogate fit — Cl and Cd vs alpha](results/figures/surrogate_fit.png)

| Output | Validation RMSE | Validation R² |
|---|---|---|
| Cl | 0.0026 | 1.000 |
| Cd | 0.00007 | 1.000 |

The surrogate reproduces the polar on **held-out validation points** it never saw during training.

> **Why so perfect?** The data is a smooth, noiseless function (a NeuralFoil/XFOIL polar), so a tiny
> MLP fits it almost exactly. This is a *learning milestone*, not a hard ML problem — the real
> difficulty comes with **noisy/sparse** data and **higher-dimensional** inputs (see Next step).

> ⚠️ **Provenance.** Training data is an XFOIL-class prediction, *not* my own CFD (my Phase 0 Fluent
> run was non-physical — see `../phase0_post_processor`). A clean, cited ground truth is the right
> choice for learning the ML pipeline.

---

## Training dynamics: validation & early stopping

The training loop (v2) tracks **two losses per epoch** — train and validation —, decays the learning
rate with a **scheduler (StepLR)**, and keeps the **best** model (early stopping). Concepts detailed
in [`docs/pytorch_guide.md`](../../docs/pytorch_guide.md) §11 (scheduler), §12 (canonical loop), §14
(regularization).

![Learning curves](results/figures/learning_curves.png)

**How to read this curve:**
- **train (blue) and validation (orange) overlap** → **no overfitting**. If it overfitted, orange
  would lift *above* blue; here the data is clean (noiseless), so the model generalizes perfectly
  (R² = 1.000 on validation).
- **Spikes early, then smoothing.** While `lr = 1e-2`, Adam (full-batch) occasionally takes steps
  that **overshoot** → spikes (present in **both** curves together → an optimization artifact, **not**
  overfitting). The **`StepLR` scheduler** halves `lr` every 1000 epochs: the spikes **vanish after
  ~epoch 1000**, once `lr` is reduced. Measured gain: max spike cut **~21×** vs a constant `lr`.
- **Why early stopping helps here:** we **store the best weights** (minimum val-loss ≈ 2·10⁻⁵) and
  **restore** them at the end → the delivered model is never one from a spike.

Loop pseudo-code (detail in Block 5 below):
```python
best_val = inf
for epoch in range(EPOCHS):
    model.train();  opt.zero_grad()                    # 1) training
    loss = mse(model(Xtr), Ytr);  loss.backward();  opt.step()
    sched.step()                                        #    decay the learning rate
    model.eval()                                        # 2) validation (no gradients)
    with torch.no_grad():  vloss = mse(model(Xva), Yva)
    if vloss < best_val:                                # 3) early stopping: keep the best
        best_val, best_state, patience = vloss, copy(model.state_dict()), 0
    else:
        patience += 1
        if patience >= PATIENCE:  break                # val stalled -> stop
model.load_state_dict(best_state)                       # restore the BEST (not the last)
```

---

# Understanding the model (study sheet)

## 1. The idea in one sentence

We learn a **function** `f : α → (Cl, Cd)` from examples, to predict the aerodynamic coefficients
**instantly**, without re-running a (slow) CFD simulation.

This is **supervised learning** (we have the right answers to train on) and **regression** (the output
is a continuous number, not a category).

```
Without ML:   α  →  [CFD: hours]          →  Cl, Cd
Surrogate:    α  →  [network: milliseconds] →  Cl, Cd
```

## 2. Where ML actually runs

| | Role |
|---|---|
| **GitHub** | A **cupboard**: stores the code and results, keeps history, shares. It **runs nothing.** |
| **Your machine** | Where ML **runs**: on your computer's **CPU**, via Python. |

The real cycle:
```
1. write the code           (on the machine)
2. RUN it: python ...       (it computes on the machine: CPU, or GPU if available)
3. it produces results + figures
4. push to GitHub           (save / share)
```
A laptop is enough here. You'd move to **Google Colab** (free GPU, in-browser) or a cluster only for
**big** models.

## 3. The full pipeline

```
   data (89 pairs α→Cl,Cd)
        │
        ▼
   train (70%) / validation (30%) split
        │
        ▼
   normalization (standardize, fit on train only)
        │
        ▼
   MLP model  1→64→64→2   (~4400 parameters to tune)
        │
        ▼
   training loop (up to 5000 epochs):
        [train] forward → MSE → backprop → Adam
        [val]   validation loss (no gradients)
        [early stopping] keep the best weights
        │
        ▼
   final evaluation: RMSE, R² (on validation)
        │
        ▼
   inference: give an α, get (Cl, Cd) instantly
```

## 4. The code line by line

### Block 1 — The data (inputs → outputs)
```python
X = df[["alpha_deg"]].values.astype("float32")   # input  (N, 1)
Y = df[["Cl", "Cd"]].values.astype("float32")    # outputs (N, 2)
```
- `X` = the **feature** (input): angle of attack α [°]. Shape `(N, 1)` = N examples, 1 variable.
- `Y` = the **targets**: Cl and Cd [dimensionless]. Shape `(N, 2)`.
- `float32` = 32-bit floats, the standard for networks (fast, precise enough).

### Block 2 — Train / validation split
```python
idx = rng.permutation(N)
n_val = int(0.3 * N)
val_idx, tr_idx = idx[:n_val], idx[n_val:]
```
- Shuffle the indices and reserve **30%** of points for **validation**.
- **Why it's essential:** scoring the model on what it learned could hide **memorization without
  generalization** (= **overfitting**). The validation set, *never seen during training*, measures
  the **true** predictive ability — and drives **early stopping**.

### Block 3 — Normalization
```python
xm, xs = Xtr.mean(0), Xtr.std(0)        # mean / std
ym, ys = Ytr.mean(0), Ytr.std(0)
norm_x = lambda a: (a - xm) / xs        # standardize
```
- Bring each variable to **mean 0, std 1** (standardization).
- **Why:** α ranges −6 to 16, Cd ranges 0.006 to 0.07 — very different scales. Gradient descent
  converges **much better** when everything is on the same scale.
- **Important:** mean/std computed on the **train set only**. Otherwise information from validation
  would "leak" into training (= **data leakage**) and the measured performance would be inflated.

### Block 4 — The model (an MLP)
```python
model = nn.Sequential(
    nn.Linear(1, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, 2),
)
```
- `nn.Linear(1, 64)` = a **layer** of 64 **neurons**. Each neuron computes `y = w·x + b` (weighted
  sum + bias). Here 1 input → 64 outputs.
- `nn.Tanh()` = a non-linear **activation**. **Without it**, stacking linear layers would stay linear
  → unable to learn a curve. The non-linearity is what lets it model the polar.
- Architecture `1 → 64 → 64 → 2`; the last layer outputs (Cl, Cd).
- **Parameters:** `1·64+64` + `64·64+64` + `64·2+2` = 128 + 4160 + 130 ≈ **4,400** numbers to tune.
- **Universal approximation theorem:** an MLP with enough neurons approximates any continuous
  function → hence its ability to learn the polar.

### Block 4b — Loss and optimizer
```python
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)
```
- `MSELoss` = **mean squared error** `mean((pred − true)²)`, **the number to minimize**.
- `Adam` = the **optimizer** that updates the weights. `lr=1e-2` = **learning rate** (step size).
- `weight_decay=1e-5` = light **L2 regularization** (penalizes large weights). Small here since the
  data is clean; mostly a good habit.
- `StepLR` = a **scheduler**: it **multiplies `lr` by 0.5 every 1000 epochs**. High `lr` early (fast
  convergence) → low `lr` late (stable steps, no spikes).

### Block 5 — The training loop (train + validation + early stopping)
```python
best_val, best_state, since_best = inf, None, 0
for epoch in range(EPOCHS):
    # (a) training
    model.train(); opt.zero_grad()
    loss = loss_fn(model(Xtr_t), Ytr_t); loss.backward(); opt.step()
    sched.step()                                  # decay the learning rate
    # (b) validation (no gradients)
    model.eval()
    with torch.no_grad():
        vloss = loss_fn(model(Xva_t), Yva_t).item()
    # (c) early stopping: store the best model
    if vloss < best_val - 1e-7:
        best_val, since_best = vloss, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        since_best += 1
        if since_best >= PATIENCE: break
model.load_state_dict(best_state)        # restore the BEST (not the last)
```
Each epoch has **two phases**:
1. **Training** (`model.train()`): the usual 5 steps (zero_grad → forward → loss → backward → step).
2. **Validation** (`model.eval()` + `torch.no_grad()`): just compute the loss on the held-out points,
   **without** backprop or updates.
3. **Early stopping:** if `val_loss` beats its record, **save a copy** of the weights (`best_state`).
   If it stalls for `PATIENCE` epochs, **stop**. At the end, **restore the best** model — not the last
   (which may sit on an instability spike).

> `model.train()` / `model.eval()` switch the behaviour of Dropout/BatchNorm (no effect here, but the
> canonical habit).

### Block 6 — Final evaluation (on validation)
```python
model.eval()
with torch.no_grad():
    Pva = denorm_y(model(Xva_t).numpy())
rmse = np.sqrt(np.mean((pred - true) ** 2))
r2   = 1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2)
```
- `torch.no_grad()`: no gradients at evaluation → faster.
- `denorm_y`: back to physical units (real Cl, Cd).
- **RMSE** = typical error in physical units; **R²** = goodness of fit (**1 = perfect**, 0 = no better
  than the mean), computed on **validation** → measures generalization.

### Block 7 — Inference + figure
```python
a_dense = np.linspace(X.min(), X.max(), 300)...
p_dense = denorm_y(model(torch.tensor(norm_x(a_dense))).numpy())
```
- We ask the model to **predict** on 300 fine angles = **inference**: an α → (Cl, Cd) immediately.
  We save the figure and the **weights** (`results/surrogate_naca0012.pt`).

## 5. Glossary

| Term | Short definition |
|---|---|
| **Supervised learning** | Learn from *labelled* examples (input + correct answer). |
| **Regression** | Predict a *continuous* value (vs classification = a category). |
| **Feature / target** | Input variable / value to predict. |
| **Train / validation** | Training data / data held out to measure generalization (and drive early stopping). |
| **Overfitting** | The model memorizes the train set but generalizes poorly (train ≪ val gap). |
| **Early stopping** | Keep the model at the minimum val-loss, not the last epoch. |
| **Learning curves** | train_loss and val_loss plotted per epoch; their gap reveals overfitting. |
| **Normalization** | Put variables on the same scale (mean 0, std 1). |
| **Data leakage** | Validation info leaking into training → inflated performance. |
| **MLP** | Multi-layer perceptron: a network of layers of neurons. |
| **Neuron** | Computes `w·x + b` (weighted sum + bias). |
| **Activation (tanh)** | A non-linearity that lets the network learn curves. |
| **Parameters / weights** | The numbers (`w`, `b`) that training adjusts. |
| **Loss (MSE)** | Error to minimize: mean of squared errors. |
| **Epoch** | One full pass over the training data. |
| **Forward pass** | Computing input → output through the network. |
| **Backpropagation** | Computing the gradients of the loss w.r.t. the weights. |
| **Gradient descent / Adam** | Algorithm that updates the weights to lower the loss. |
| **Learning rate** | Step size of the weight update. |
| **Scheduler** | Evolves the learning rate over epochs (e.g. periodic halving) to stabilize late training. |
| **Inference** | Using the trained model to predict on new inputs. |
| **RMSE / R²** | Metrics: typical error / goodness of fit (1 = perfect). |

---

# Extension — 2-D surrogate: (α, Re) → (Cl, Cd)

Instead of a single curve, the model learns a **whole family of polars** (315 points: 45 angles ×
7 Reynolds numbers from 10⁵ to 1.5·10⁶) and can **interpolate** to a Reynolds number it never saw.

**New technique — `log(Re)`.** Re spans a factor of ~15 → we feed `log10(Re)` before normalizing,
otherwise its huge range would drown the information. This is **feature engineering**. Network input:
`[alpha, log10(Re)]`, architecture `2 → 64 → 64 → 2`.

## The learned family of polars

![Family of polars](results/figures/surrogate_2d_family.png)

Dots = data, lines = surrogate. The model captures the whole family, **physics included**: higher Re
→ later **stall** and lower **Cd**. Validation: R² = 0.999 (Cl), 0.994 (Cd).

## The proof: interpolation to an unseen Re

![Interpolation Re=3.5e5](results/figures/surrogate_2d_interpolation.png)

At **Re = 3.5·10⁵** (absent from the training grid), the surrogate (red) matches the NeuralFoil
reference (dashed): **RMSE Cl = 0.008, Cd = 0.0023**. *That* is the power of a surrogate — a fresh
operating point predicted **instantly**.

> ⚠️ **Honesty:** this is **interpolation** (between known Re). Outside the 10⁵–1.5·10⁶ range it would
> be **extrapolation**, far less reliable — a model cannot invent what it never came close to.

## Understanding the 2-D version: what changes vs 1-D

Going from 1-D to 2-D required only **three real changes**; everything else (normalization, loop,
scheduler, early stopping) is **identical**.

**1. Build the `[α, log10(Re)]` input**
```python
X = np.column_stack([A, np.log10(Re)]).astype("float32")   # (N, 2)
```
`column_stack` glues two columns → each row is `[angle, log10(Reynolds)]`. **Why `log10` of Re?** Re
spans ~15× (10⁵ → 1.5·10⁶). In log it becomes a regular scale (5 → 6) where each Reynolds step counts
equally; otherwise its huge dynamic range would drown the fine differences. This is **feature
engineering**.

**2. A 2-neuron input layer**
```python
nn.Linear(2, 64)   # instead of nn.Linear(1, 64)
```
The only architectural change. Normalization becomes *vectorial*: `Xtr.mean(0)` / `Xtr.std(0)` now
return **2 values** (one per column) → each input is scaled to its own range.

**3. An inference helper**
```python
def pred(a, re):
    Xq = nx(np.column_stack([a, np.log10(np.full_like(a, re))]).astype("float32"))
    with torch.no_grad():
        return dny(model(torch.tensor(Xq)).numpy())
```
We rebuild the **same** input `[α, log10(Re)]` (with the **same** scaler `nx`) to predict. The
`torch.no_grad()` is required: at inference there's no graph → otherwise `.numpy()` fails on a tensor
that still tracks the gradient.

**The interpolation demo** (`Re = 3.5·10⁵`, off-grid) generates a fresh NeuralFoil reference and
compares it to the prediction — if they match, the model learned to **interpolate across Reynolds
numbers**, not just memorize.

> The takeaway: once the 1-D pipeline is mastered, **adding a dimension is trivial**. That's ML
> maturity.

---

## Files

```
phase1_surrogate/
├── data/naca0012_surrogate_dataset.csv   # 1D: 89 pts, alpha -6..16, Re=4e5
├── data/naca0012_surrogate_2d.csv        # 2D: 315 pts (45 alpha x 7 Re)
├── src/make_dataset.py                    # 1D dataset
├── src/make_dataset_2d.py                 # 2D dataset (alpha x Re grid)
├── src/train_surrogate.py                 # 1D: train/val + early stopping + scheduler
├── src/train_surrogate_2d.py              # 2D: + log(Re) + interpolation demo
├── results/figures/learning_curves.png    # 1D: train vs val
├── results/figures/surrogate_fit.png      # 1D: data vs surrogate
├── results/figures/surrogate_2d_*.png     # 2D: curves, family, interpolation
├── results/surrogate_naca0012.pt          # 1D: weights
├── results/surrogate_2d_naca0012.pt       # 2D: weights
└── requirements.txt
```

## How to run

```bash
cd cfd-projects/piml/phase1_surrogate
pip install -r requirements.txt     # torch + neuralfoil + aerosandbox

python src/make_dataset.py          # 1D dataset
python src/train_surrogate.py       # 1D: train + evaluate + plot

python src/make_dataset_2d.py       # 2D dataset (alpha x Re)
python src/train_surrogate_2d.py    # 2D: train + interpolation demo
```

## Next step

→ **Phase 2: Physics-Informed Neural Networks (PINNs)** — learn solutions from the governing
equations themselves, not from data. See [`../phase2_pinns`](../phase2_pinns).
