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

---

## 2.1 — First PINN: damped harmonic oscillator ✅

Solve, with **no data**, only the physics:

$$u'' + 2\delta\,u' + \omega_0^2\,u = 0,\qquad u(0)=1,\ u'(0)=0\quad(\delta=2,\ \omega_0=20)$$

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

---

## Next steps
- **2.2** — a true PDE: 1-D heat/diffusion equation `u_t = α u_xx` (2 inputs: x, t).
- **2.3** — flow case (Burgers, then steady flow features) toward the NACA 0012 context.

Foundations: see the PyTorch guide [`../../docs/pytorch_guide.md`](../../docs/pytorch_guide.md) (§5 autograd).
