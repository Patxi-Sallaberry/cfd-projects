"""
Phase 2.4 - PINN sur les equations de Navier-Stokes 2D (ecoulement de Kovasznay)
================================================================================
Navier-Stokes incompressible stationnaire 2D (rho = 1) :
   u*u_x + v*u_y = -p_x + nu*(u_xx + u_yy)          (quantite de mouvement x)
   u*v_x + v*v_y = -p_y + nu*(v_xx + v_yy)          (quantite de mouvement y)
   u_x + v_y = 0                                     (continuite / conservation de la masse)

Cas de Kovasznay : il existe une SOLUTION ANALYTIQUE EXACTE -> validation rigoureuse (V&V).
Le reseau a 2 entrees (x,y) et 3 SORTIES (u, v, p). On impose la solution exacte au bord
(Dirichlet) et on minimise les 3 residus a l'interieur.

Sortie : results/figures/pinn_navier_stokes.png
Lancer  : python src/pinn_navier_stokes.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)

RE = 40.0
NU = 1.0 / RE
LAM = RE / 2 - np.sqrt(RE**2 / 4 + 4 * np.pi**2)     # parametre de Kovasznay (~ -0.964)
XMIN, XMAX, YMIN, YMAX = -0.5, 1.0, -0.5, 1.5
N_COL, N_BC, EPOCHS = 2000, 400, 16000

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"


# --- Solution exacte de Kovasznay (numpy) ---
def exact(x, y):
    u = 1 - np.exp(LAM * x) * np.cos(2 * np.pi * y)
    v = LAM / (2 * np.pi) * np.exp(LAM * x) * np.sin(2 * np.pi * y)
    p = 0.5 * (1 - np.exp(2 * LAM * x))
    return u, v, p


def main():
    # --- 1. Reseau : (x,y) -> (u,v,p) ---
    model = nn.Sequential(
        nn.Linear(2, 40), nn.Tanh(), nn.Linear(40, 40), nn.Tanh(),
        nn.Linear(40, 40), nn.Tanh(), nn.Linear(40, 40), nn.Tanh(), nn.Linear(40, 3),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.5)

    # --- 2. Points interieurs (collocation) ---
    xc = (torch.rand(N_COL, 1) * (XMAX - XMIN) + XMIN).requires_grad_(True)
    yc = (torch.rand(N_COL, 1) * (YMAX - YMIN) + YMIN).requires_grad_(True)

    # --- 3. Points de bord : valeurs exactes imposees (u,v,p) ---
    def edge(n, fixed, axis):
        other = np.random.rand(n) * ((YMAX - YMIN) if axis == "x" else (XMAX - XMIN)) \
            + (YMIN if axis == "x" else XMIN)
        if axis == "x":   # x fixe, y varie
            xx, yy = np.full(n, fixed), other
        else:             # y fixe, x varie
            xx, yy = other, np.full(n, fixed)
        return xx, yy
    bx, by = [], []
    for fixed, axis in [(XMIN, "x"), (XMAX, "x"), (YMIN, "y"), (YMAX, "y")]:
        xx, yy = edge(N_BC // 4, fixed, axis); bx.append(xx); by.append(yy)
    bx, by = np.concatenate(bx), np.concatenate(by)
    bu, bv, bp = exact(bx, by)
    Bxy = torch.tensor(np.column_stack([bx, by]), dtype=torch.float32)
    Buvp = torch.tensor(np.column_stack([bu, bv, bp]), dtype=torch.float32)

    def grad(f, x):
        return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]

    for epoch in range(EPOCHS):
        opt.zero_grad()
        out = model(torch.cat([xc, yc], 1))
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        u_x, u_y = grad(u, xc), grad(u, yc)
        v_x, v_y = grad(v, xc), grad(v, yc)
        p_x, p_y = grad(p, xc), grad(p, yc)
        u_xx, u_yy = grad(u_x, xc), grad(u_y, yc)
        v_xx, v_yy = grad(v_x, xc), grad(v_y, yc)
        # les 3 residus de Navier-Stokes
        r_mx = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy)
        r_my = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy)
        r_c = u_x + v_y
        loss_phys = (r_mx**2).mean() + (r_my**2).mean() + (r_c**2).mean()
        # bord : valeurs exactes
        loss_bc = ((model(Bxy) - Buvp) ** 2).mean()
        loss = loss_phys + 10.0 * loss_bc
        loss.backward(); opt.step(); sched.step()
        if epoch % 2000 == 0:
            print(f"epoch {epoch:5d}  loss={loss.item():.3e}  (phys={loss_phys.item():.2e} bc={loss_bc.item():.2e})")

    # --- 4. Validation vs solution exacte sur une grille ---
    xs = np.linspace(XMIN, XMAX, 80); ys = np.linspace(YMIN, YMAX, 80)
    XX, YY = np.meshgrid(xs, ys)
    with torch.no_grad():
        P = model(torch.tensor(np.column_stack([XX.ravel(), YY.ravel()]),
                               dtype=torch.float32)).numpy()
    Up, Vp, Pp = (P[:, i].reshape(80, 80) for i in range(3))
    Ue, Ve, Pe = exact(XX, YY)
    for name, a, b in [("u", Up, Ue), ("v", Vp, Ve), ("p", Pp, Pe)]:
        r2 = 1 - np.sum((a - b) ** 2) / np.sum((b - b.mean()) ** 2)
        print(f"[VALIDATION] {name} : R2 = {r2:.4f}")

    # --- 5. Figure : lignes de courant exactes vs PINN + erreur ---
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    spd_e = np.sqrt(Ue**2 + Ve**2); spd_p = np.sqrt(Up**2 + Vp**2)
    err = np.sqrt((Up - Ue)**2 + (Vp - Ve)**2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].streamplot(XX, YY, Ue, Ve, color=spd_e, cmap="viridis", density=1.1)
    axes[0].set_title("Exact (Kovasznay)")
    axes[1].streamplot(XX, YY, Up, Vp, color=spd_p, cmap="viridis", density=1.1)
    axes[1].set_title("PINN (Navier-Stokes appris)")
    cf = axes[2].contourf(XX, YY, err, levels=20, cmap="magma")
    axes[2].set_title("Erreur |vitesse| PINN vs exact")
    fig.colorbar(cf, ax=axes[2])
    for ax in axes:
        ax.set(xlabel="x", ylabel="y", xlim=(XMIN, XMAX), ylim=(YMIN, YMAX))
    fig.suptitle(f"Ecoulement 2D de Kovasznay (Re={RE:.0f}) — PINN vs solution exacte")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pinn_navier_stokes.png", dpi=150)
    print(f"Figure : {FIG_DIR / 'pinn_navier_stokes.png'}")


if __name__ == "__main__":
    main()
