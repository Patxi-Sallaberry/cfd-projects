# Phase 2 — Physics-Informed Neural Networks (PINNs)
*In progress · Dec 2026 – Mar 2027 (LiU)*

**The idea.** Instead of learning from data (Phase 1), a PINN learns from the **physics itself**: the
governing equation is put **into the loss**. We differentiate the network's *output* w.r.t. its
*inputs* (via autograd) to build the equation's residual, and we minimize it. With (almost) no data,
the network discovers the solution.

| | Phase 1 (surrogate) | Phase 2 (PINN) |
|---|---|---|
| Learns from | data (input→output) | the equation (PDE/ODE residual) |
| Loss | data error | equation residual + boundary/initial conditions |
| Autograd differentiates | loss w.r.t. **weights** | also output w.r.t. **inputs** (to get u′, u″) |

## Physical quantities

| Symbol | Meaning | Unit |
|---|---|---|
| **t** | time | s |
| **x** | position (1-D space) | m |
| **u** | the field being solved (oscillator: displacement; heat: temperature) | m or K |
| **δ** | damping rate of the oscillator (energy dissipation) | 1/s |
| **ω₀** | natural angular frequency of the oscillator | rad/s |
| **α** | thermal diffusivity (how fast heat spreads), `α = k/(ρ·cₚ)` | m²/s |
| **ν** | kinematic viscosity (Burgers, upcoming) | m²/s |

> ⚠️ **Symbol clash:** in Phase 2, **α is the thermal diffusivity** (m²/s) — *not* the angle of
> attack of Phases 0–1. Same letter, different physics.

---

## 2.1 — First PINN: damped harmonic oscillator ✅

Solve, with **no data**, only the physics (a mass–spring–damper system):

$$u'' + 2\delta\,u' + \omega_0^2\,u = 0,\qquad u(0)=1,\ u'(0)=0\quad(\delta=2\ \mathrm{s^{-1}},\ \omega_0=20\ \mathrm{rad/s})$$

![PINN vs exact](results/figures/pinn_oscillator.png)

The PINN matches the exact solution with **R² = 0.998** — learned purely from the equation residual
plus the two initial conditions. The loss combines:
- `loss_phys = mean(residual²)` at 200 collocation points, where
  `residual = u'' + 2δ·u' + ω₀²·u` (u′, u″ obtained by `torch.autograd.grad` on the input `t`);
- `loss_ic` enforcing `u(0)=1`, `u'(0)=0`.

Run: `python src/pinn_oscillator.py`

> **Key new tool:** `torch.autograd.grad(u, t, create_graph=True)` — differentiating the output
> w.r.t. the input. `create_graph=True` lets us take the *second* derivative and still backprop the
> loss. Everything else (MLP, Adam, training loop) is the Phase 1 machinery.

### Understanding the code step by step

Only the parts that are **new** vs Phase 1 are detailed (the MLP, Adam and the loop are identical).

**① The network represents the solution** (not a data fit)
```python
model = nn.Sequential(nn.Linear(1,32), nn.Tanh(), ... , nn.Linear(32,1))
```
- `t → u`: the network **IS** the function `u(t)` we seek, not an interpolator of known points.
- **`tanh`, definitely not `relu`:** we take the **second derivative** `u''`. A ReLU's second
  derivative is zero everywhere → unable to represent curvature. `tanh` is smooth and infinitely
  differentiable.

**② Collocation points (≠ data)**
```python
t_phys = torch.linspace(0, 1, 200).reshape(-1,1).requires_grad_(True)
```
- 200 points where we **impose the equation** — we don't know `u` there, they are not data.
  `.requires_grad_(True)` because we will differentiate `u` w.r.t. `t`.

**③ The heart of the PINN: differentiate the output w.r.t. the input**
```python
u    = model(t_phys)
u_t  = torch.autograd.grad(u, t_phys, torch.ones_like(u),  create_graph=True)[0]
u_tt = torch.autograd.grad(u_t, t_phys, torch.ones_like(u_t), create_graph=True)[0]
```
- `torch.autograd.grad(u, t_phys, …)` computes `du/dt`. (In Phase 1, `loss.backward()` differentiated
  w.r.t. the **weights**; here we differentiate the **output w.r.t. the input** `t`.)
- **`torch.ones_like(u)`** (`grad_outputs`): since `u` is a vector (200 values), this seed of `1`
  asks, for each point, for `du_i/dt_i` (the network acts pointwise).
- **`create_graph=True`**: keeps the graph of the derivative itself → enables (a) the second
  derivative `u_tt` and (b) backpropagating the loss through these derivatives to train the weights.
- **`[0]`**: `autograd.grad` returns a tuple.

**④ The residual = the equation, put in the loss**
```python
residual  = u_tt + 2*DELTA*u_t + OMEGA0**2 * u     # = 0 if the ODE is satisfied
loss_phys = (residual ** 2).mean()
```
We write the equation literally. Correct solution ⇔ `residual = 0` everywhere → we minimize
`mean(residual²)`.

**⑤ Initial conditions (they select THE right solution)**
```python
u0   = model(t0)                                   # t0 = 0
u0_t = torch.autograd.grad(u0, t0, ...)[0]
loss_ic = (u0 - 1.0)**2 + (u0_t - 0.0)**2          # u(0)=1, u'(0)=0
```
The equation alone admits infinitely many solutions — **including the trivial `u ≡ 0`**, whose
residual is zero. Without ICs, the PINN collapses to that zero (tiny loss but wrong answer). The ICs
**anchor** the physical solution. *(Same idea in flow: these become the boundary conditions.)*

**⑥ Weighting `W_PHYS = 1e-4`**
```python
loss = W_PHYS * loss_phys + loss_ic.squeeze()
```
- `residual ≈ ω0²·u ≈ 400` → `residual² ≈ 1.6×10⁵`, whereas `loss_ic ≈ 1`. Without weighting, physics
  crushes the ICs.
- `W_PHYS=1e-4` brings `loss_phys` to ~O(1), comparable to `loss_ic` → both are respected. Balancing
  the loss terms is a core PINN tuning knob.

**Takeaway.** A low loss does not guarantee the right answer (cf. the trivial solution) → **always
validate** against a reference (here the analytic solution, R²=0.998).

---

## 2.2 — Heat equation `u_t = α u_xx` (a true PDE) ✅

First PDE: a diffusing field `u(x,t)` on x ∈ [0,1], with `u(x,0)=sin(πx)`, ends held at 0, and
thermal diffusivity **α = 0.4 m²/s**. Solved from physics + initial + boundary conditions, **with no
data**. **R² = 0.999** vs the exact solution `sin(πx)·e^{-απ²t}`.

![Heat PINN](results/figures/pinn_heat.png)

The sine bump decays in amplitude over time — that's diffusion. Run: `python src/pinn_heat.py`

**What's new vs the oscillator (2.1):**
- **2 inputs `(x, t)`** → the network learns a *field*, and we take **partial derivatives**: `u_t`
  (grad w.r.t. t) and `u_xx` (grad w.r.t. x, twice). `x` and `t` are kept as separate tensors so we
  can differentiate w.r.t. each.
- **Two kinds of constraint**: an **initial condition** (the t=0 profile) *and* **boundary
  conditions** (the two ends), each its own weighted loss term (`W_IC`, `W_BC`).

Otherwise the recipe is identical: minimize the residual `(u_t − α·u_xx)²` at collocation points.

### Understanding the code step by step

Only the **changes** vs the oscillator (2.1) are detailed — the shared concepts (autograd,
`grad_outputs`, `create_graph`, `tanh`, weighting) are explained in the 2.1 walkthrough.

**① A 2-input network**
```python
model = nn.Sequential(nn.Linear(2,32), nn.Tanh(), ... , nn.Linear(32,1))
# call: model(torch.cat([x, t], dim=1))
```
The first layer takes **2** inputs `(x, t)`. We assemble the columns with `torch.cat([x, t], dim=1)`
→ each row is a pair `(x, t)`. So the network learns a **field** `u(x,t)`.

**② Sampling: x and t kept separate**
```python
xc = torch.rand(N_COL,1, requires_grad=True)            # interior (PDE)
tc = (torch.rand(N_COL,1) * T).requires_grad_(True)
xi = torch.rand(N_IC,1);  ti = torch.zeros(N_IC,1);  ui = torch.sin(pi*xi)   # initial condition
tb = torch.rand(N_BC,1)*T; x0 = torch.zeros(N_BC,1); x1 = torch.ones(N_BC,1) # boundaries
```
- We keep `xc` and `tc` as **separate tensors** (both `requires_grad`) so we can differentiate w.r.t.
  **each** independently.
- Three families of points: **interior** (where we impose the PDE), **t=0** (initial condition, with
  target values `sin(πx)`), **boundaries x=0/x=1** (where `u` must be 0). The IC/BC points do *not*
  need `requires_grad`: they are **value** constraints, no differentiation there.

**③ Partial derivatives**
```python
u    = model(torch.cat([xc, tc], dim=1))
u_t  = torch.autograd.grad(u,   tc, torch.ones_like(u),   create_graph=True)[0]  # ∂u/∂t
u_x  = torch.autograd.grad(u,   xc, torch.ones_like(u),   create_graph=True)[0]  # ∂u/∂x
u_xx = torch.autograd.grad(u_x, xc, torch.ones_like(u_x), create_graph=True)[0]  # ∂²u/∂x²
```
`u` depends on `xc` **and** `tc` (via the `cat`). Differentiating w.r.t. `tc` gives `∂u/∂t`; w.r.t.
`xc`, `∂u/∂x`; re-differentiating `u_x` w.r.t. `xc` gives `∂²u/∂x²`. That's the whole difference vs the
oscillator: **partial** derivatives, one per direction.

**④ Three loss terms**
```python
loss_phys = ((u_t - ALPHA*u_xx)**2).mean()                       # the PDE
loss_ic   = ((model(cat([xi,ti])) - ui)**2).mean()               # u(x,0)=sin(pi x)
loss_bc   = (model(cat([x0,tb]))**2).mean() + (model(cat([x1,tb]))**2).mean()  # u(0,t)=u(1,t)=0
loss = loss_phys + W_IC*loss_ic + W_BC*loss_bc
```
The PDE (residual), the **initial** condition (value match at t=0) and the **boundary** conditions
(zero at the ends) — each its own term, weighted by `W_IC`/`W_BC`. Without IC+BC the solution would
not be unique (cf. the 2.1 lesson).

---

## 2.2b — Inverse problem: recovering an unknown parameter ✅ — *why PINNs matter*

The cases above just re-solve equations whose answer we already know — useful only to **learn and
validate** the method. A PINN earns its keep where a classical solver or an analytic formula **cannot
go**: **inverse problems**.

Here the thermal diffusivity **α (m²/s) is unknown**. We only have **40 noisy "sensor" measurements**
of `u`. The PINN learns the field `u(x,t)` *and* α at once, combining a **data** loss (fit the
measurements) with the **physics** loss (`u_t = α·u_xx`, with α a trainable parameter).

![Inverse PINN](results/figures/pinn_heat_inverse.png)

- **α recovered ≈ 0.44 m²/s** (true 0.4, ~10 % with noisy data) — starting from a deliberately wrong 1.5.
- The **full field is reconstructed (R² = 0.99)** from just 40 scattered points.

Run: `python src/pinn_heat_inverse.py`

### Understanding the code step by step

Only the **changes** vs the forward heat PINN (2.2) are detailed — the partial-derivative machinery
(`u_t`, `u_xx` via autograd) is identical and explained in the 2.2 walkthrough.

**① The measurements — the only "data" here** (sparse and noisy, like sensors)
```python
xm = np.random.rand(N_MEAS,1); tm = np.random.rand(N_MEAS,1)         # 40 random (x,t) locations
um = analytic(xm, tm) + 0.02*np.random.randn(N_MEAS,1)               # values + measurement noise
Xm, Tm, Um = (torch.tensor(a, dtype=torch.float32) for a in (xm,tm,um))
```
The forward problems (2.1, 2.2) used **no data**. The inverse problem **needs** a few measurements:
they are what makes the unknown parameter identifiable. The `0.02*randn` mimics real sensor noise.

**② The unknown parameter, made trainable**
```python
alpha = nn.Parameter(torch.tensor(1.5))      # deliberately wrong start; true value = 0.4
```
`nn.Parameter` turns a plain scalar into a **learnable variable**: it gets a gradient and is updated
just like a network weight. This is the crux of the inverse problem — α is no longer a fixed constant,
it is **discovered** during training.

**③ Optimize the network weights AND α together**
```python
opt = torch.optim.Adam(list(model.parameters()) + [alpha], lr=5e-3)
```
We hand the optimizer **both** the network's weights *and* α. At each step, gradient descent nudges
all of them to reduce the total loss → the field `u(x,t)` and the parameter α are learned jointly.

**④ Two loss terms: data + physics (with the current α)**
```python
loss_data = ((model(torch.cat([Xm,Tm],1)) - Um)**2).mean()           # fit the measurements
u   = model(torch.cat([xc,tc],1))
u_t = grad(u, tc, ...);  u_x = grad(u, xc, ...);  u_xx = grad(u_x, xc, ...)
loss_phys = ((u_t - alpha*u_xx)**2).mean()                           # heat eq with the LEARNED alpha
loss = loss_data + 1e-2*loss_phys
```
- `loss_data` pulls the field through the noisy measurements.
- `loss_phys` forces the field to obey `u_t = α·u_xx` — but with the **trainable** α. This is the
  coupling: the physics ties α to the shape of the field, the data ties the field to reality.
- **Why it works:** data alone → infinitely many fields through 40 noisy points; physics alone → α
  undetermined (and the trivial `u≡0`). **Together** they pin down the single α whose physical field
  matches the data.

**⑤ Watch α converge**
```python
a_hist.append(alpha.item())                  # record alpha every epoch -> the left plot
```
Plotting `a_hist` shows α climbing from 1.5, overshooting, then settling near the true 0.4 — the
visual proof the parameter was recovered.

**Why this is the whole point.** You *cannot* do this with the analytic formula — it requires already
knowing α. This is **data assimilation / parameter inference**: from sparse, noisy measurements +
physics, recover hidden quantities *and* the complete field. In a CFD context: infer an effective
viscosity or an unknown boundary condition, and reconstruct the flow field, from a few pressure taps
or scattered PIV points — something classical forward solvers don't naturally do.

> **Honest takeaway on Phase 2 so far:** 2.1 and 2.2 are *validation* exercises (known answers); 2.2b
> is the *real* use case. You always validate the machinery on a solved problem before trusting it on
> an unsolved one.

---

## 2.2c — Physics enables extrapolation beyond the data ✅ — *another reason PINNs matter*

A second, distinct proof of usefulness. We give noisy measurements of the damped oscillator **only on
the first third of the time** (t ∈ [0, 0.4]) and compare two models — *same architecture* — over the
full [0, 1]:

![Extrapolation](results/figures/pinn_extrapolation.png)

- **A — data only** (red): fits the measured region, then **diverges** in the no-data zone — it has no
  constraint there, so it has no reason to behave.
- **B — data + physics** (green, PINN): the ODE residual forces it to keep obeying the dynamics, so it
  **correctly continues the oscillation** where there is *no data at all*.

In the no-data zone [0.4, 1] the PINN is **~225× more accurate** (RMSE 0.005 vs 1.11).

**Why this matters.** Pure ML cannot extrapolate beyond its data — nothing constrains it there. The
physics acts as a **regularizer** that constrains the solution *everywhere*, even where measurements
are missing. In a CFD context: from sensors in one region you can extend a physically-consistent field
into regions you never measured.

Run: `python src/pinn_oscillator_extrapolation.py`

### What's new in the code
Two models, **same network**, trained differently:
```python
# A: data only
loss = ((mA(Td) - Ud)**2).mean()
# B: data + physics residual over the WHOLE domain (collocation points everywhere, incl. the no-data zone)
loss = ((mB(Td) - Ud)**2).mean() + 1e-4 * ((u_tt + 2*DELTA*u_t + OMEGA0**2*u)**2).mean()
```
The only addition in B is the physics term, evaluated on collocation points covering all of [0, 1] —
including the unmeasured part. That single term is what enables the extrapolation. (No explicit initial
condition is needed: the data near t=0 already anchors the solution.)

---

## Next steps
- **2.3** — flow case: Burgers' equation `u_t + u·u_x = ν·u_xx` (kinematic viscosity **ν**, m²/s), the
  non-linear convective term, then steady flow features toward the NACA 0012 context.

Foundations: see the PyTorch guide [`../../docs/pytorch_guide.md`](../../docs/pytorch_guide.md) (§5 autograd).
