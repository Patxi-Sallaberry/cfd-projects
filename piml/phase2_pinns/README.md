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

### Comprendre le code pas à pas

Seules les parties **nouvelles** vs la Phase 1 sont détaillées (le MLP, Adam et la boucle sont
identiques).

**① Le réseau représente la solution** (pas un fit de données)
```python
model = nn.Sequential(nn.Linear(1,32), nn.Tanh(), ... , nn.Linear(32,1))
```
- `t → u` : le réseau **EST** la fonction `u(t)` cherchée, pas un interpolateur de points connus.
- **`tanh` et surtout pas `relu`** : on prend la **dérivée seconde** `u''`. Celle d'un ReLU est nulle
  partout → incapable de représenter une courbure. `tanh` est lisse et indéfiniment dérivable.

**② Points de collocation (≠ données)**
```python
t_phys = torch.linspace(0, 1, 200).reshape(-1,1).requires_grad_(True)
```
- 200 points où l'on **impose l'équation** — on ne connaît pas `u` à ces points, ce ne sont pas des
  données. `.requires_grad_(True)` car on va dériver `u` par rapport à `t`.

**③ Le cœur du PINN : dériver la sortie p/r à l'entrée**
```python
u    = model(t_phys)
u_t  = torch.autograd.grad(u, t_phys, torch.ones_like(u),  create_graph=True)[0]
u_tt = torch.autograd.grad(u_t, t_phys, torch.ones_like(u_t), create_graph=True)[0]
```
- `torch.autograd.grad(u, t_phys, …)` = calcule `du/dt`. (En Phase 1, `loss.backward()` dérivait
  p/r aux **poids** ; ici on dérive la **sortie p/r à l'entrée** `t`.)
- **`torch.ones_like(u)`** (`grad_outputs`) : comme `u` est un vecteur (200 valeurs), ce seed à `1`
  demande, pour chaque point, `du_i/dt_i` (le réseau agit point par point).
- **`create_graph=True`** : garde le graphe de la dérivée elle-même → permet (a) la dérivée seconde
  `u_tt` et (b) la rétropropagation de la loss à travers ces dérivées pour entraîner les poids.
- **`[0]`** : `autograd.grad` renvoie un tuple.

**④ Le résidu = l'équation, mise dans la loss**
```python
residual  = u_tt + 2*DELTA*u_t + OMEGA0**2 * u     # = 0 si l'ODE est satisfaite
loss_phys = (residual ** 2).mean()
```
On écrit littéralement l'équation. Solution correcte ⇔ `residual = 0` partout → on minimise
`mean(residual²)`.

**⑤ Conditions initiales (elles sélectionnent LA bonne solution)**
```python
u0   = model(t0)                                   # t0 = 0
u0_t = torch.autograd.grad(u0, t0, ...)[0]
loss_ic = (u0 - 1.0)**2 + (u0_t - 0.0)**2          # u(0)=1, u'(0)=0
```
L'équation seule admet une infinité de solutions — **y compris la solution triviale `u ≡ 0`**, de
résidu nul. Sans CI, le PINN s'effondre vers ce zéro (loss minuscule mais réponse fausse). Les CI
**ancrent** la solution physique. *(Idem en flow : ce seront les conditions aux frontières.)*

**⑥ Pondération `W_PHYS = 1e-4`**
```python
loss = W_PHYS * loss_phys + loss_ic.squeeze()
```
- `residual ≈ ω0²·u ≈ 400` → `residual² ≈ 1.6×10⁵`, alors que `loss_ic ≈ 1`. Sans pondération, la
  physique écrase les CI.
- `W_PHYS=1e-4` ramène `loss_phys` à ~O(1), comparable à `loss_ic` → les deux sont respectés.
  Équilibrer les termes de la loss est un réglage central des PINNs.

**À retenir.** Une loss basse ne garantit pas le bon résultat (cf. la solution triviale) → on
**valide toujours** contre une référence (ici la solution analytique, R²=0.998).

---

## Next steps
- **2.2** — a true PDE: 1-D heat/diffusion equation `u_t = α u_xx` (2 inputs: x, t).
- **2.3** — flow case (Burgers, then steady flow features) toward the NACA 0012 context.

Foundations: see the PyTorch guide [`../../docs/pytorch_guide.md`](../../docs/pytorch_guide.md) (§5 autograd).
