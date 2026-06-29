"""
Phase 2.3 - PINN sur un PDE NON LINEAIRE : equation de Burgers
==============================================================
   u_t + u*u_x = nu * u_xx     sur x in [-1,1], t in [0,1]
   condition initiale : u(x,0) = -sin(pi x)
   conditions limites : u(-1,t) = u(1,t) = 0
Le terme u*u_x est NON LINEAIRE (terme convectif) : il raidit le profil jusqu'a former
un CHOC vers x=0. Pas de formule analytique simple -> on valide contre un solveur
differences finies (V&V). C'est le benchmark PINN historique (Raissi et al.).

Sortie : results/figures/pinn_burgers.png
Lancer  : python src/pinn_burgers.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(0)
NU = 0.01 / np.pi                  # viscosite (m^2/s) : petite -> choc raide
N_COL, N_IC, N_BC = 2500, 200, 200
EPOCHS = 13000
W_IC, W_BC = 20.0, 20.0

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"


def fd_reference(nx=401, nt=4000):
    """Solveur differences finies (upwind convection + diffusion centree) = reference V&V."""
    x = np.linspace(-1, 1, nx); dx = x[1] - x[0]
    t = np.linspace(0, 1, nt); dt = t[1] - t[0]
    u = -np.sin(np.pi * x)
    U = np.zeros((nt, nx)); U[0] = u
    for n in range(nt - 1):
        un = U[n]
        lap = np.zeros(nx); dudx = np.zeros(nx)
        lap[1:-1] = (un[2:] - 2 * un[1:-1] + un[:-2]) / dx**2
        dudx[1:-1] = np.where(un[1:-1] >= 0,
                              (un[1:-1] - un[:-2]) / dx,        # upwind selon le signe de u
                              (un[2:] - un[1:-1]) / dx)
        new = un + dt * (-un * dudx + NU * lap)
        new[0] = 0.0; new[-1] = 0.0                            # conditions aux limites
        U[n + 1] = new
    return x, t, U


def main():
    # --- 1. Reseau (x,t) -> u (un peu plus profond : le choc est raide) ---
    model = nn.Sequential(
        nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=4000, gamma=0.5)

    # --- 2. Points : interieur (PDE), condition initiale, bords ---
    xc = (torch.rand(N_COL, 1) * 2 - 1).requires_grad_(True)
    tc = torch.rand(N_COL, 1).requires_grad_(True)
    xi = torch.rand(N_IC, 1) * 2 - 1; ti = torch.zeros(N_IC, 1)
    ui = -torch.sin(np.pi * xi)
    tb = torch.rand(N_BC, 1); xb0 = -torch.ones(N_BC, 1); xb1 = torch.ones(N_BC, 1)

    for epoch in range(EPOCHS):
        opt.zero_grad()
        u = model(torch.cat([xc, tc], 1))
        u_t = torch.autograd.grad(u, tc, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, xc, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, xc, torch.ones_like(u_x), create_graph=True)[0]
        # LE residu non lineaire : u*u_x est le terme convectif
        loss_phys = ((u_t + u * u_x - NU * u_xx) ** 2).mean()
        loss_ic = ((model(torch.cat([xi, ti], 1)) - ui) ** 2).mean()
        loss_bc = (model(torch.cat([xb0, tb], 1)) ** 2).mean() \
                + (model(torch.cat([xb1, tb], 1)) ** 2).mean()
        loss = loss_phys + W_IC * loss_ic + W_BC * loss_bc
        loss.backward(); opt.step(); sched.step()
        if epoch % 2000 == 0:
            print(f"epoch {epoch:5d}  loss={loss.item():.3e}  (phys={loss_phys.item():.2e})")

    # --- 3. Reference FD + validation ---
    xr, tr, Ur = fd_reference()
    # compare sur une grille
    xs = np.linspace(-1, 1, 100); ts = np.linspace(0, 1, 100)
    XX, TT = np.meshgrid(xs, ts)
    with torch.no_grad():
        Up = model(torch.tensor(np.column_stack([XX.ravel(), TT.ravel()]),
                                dtype=torch.float32)).numpy().reshape(100, 100)
    # interpole la reference FD sur la meme grille (en x ; t via indices proches)
    Uref = np.array([np.interp(xs, xr, Ur[np.argmin(np.abs(tr - tv))]) for tv in ts])
    r2 = 1 - np.sum((Up - Uref) ** 2) / np.sum((Uref - Uref.mean()) ** 2)
    print(f"\n[VALIDATION PINN vs FD]  R2 = {r2:.4f}")

    # --- 4. Figure : profils (choc) + champ espace-temps ---
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = plt.cm.viridis(np.linspace(0, 0.85, 4))
    for tv, c in zip([0.0, 0.25, 0.5, 0.75], colors):
        xx = np.linspace(-1, 1, 300)
        with torch.no_grad():
            up = model(torch.tensor(np.column_stack([xx, np.full_like(xx, tv)]),
                                    dtype=torch.float32)).numpy().ravel()
        uref = np.interp(xx, xr, Ur[np.argmin(np.abs(tr - tv))])
        ax1.plot(xx, uref, "k--", lw=1.3)
        ax1.plot(xx, up, color=c, lw=1.8, label=f"t = {tv}")
    ax1.plot([], [], "k--", label="reference (FD)")
    ax1.set(xlabel="x", ylabel="u(x,t)", title="Burgers : formation du choc (PINN vs FD)")
    ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8)

    im = ax2.contourf(XX, TT, Up, levels=30, cmap="RdBu_r")
    ax2.set(xlabel="x", ylabel="t", title=f"Champ u(x,t) appris par le PINN (R²={r2:.3f})")
    fig.colorbar(im, ax=ax2, label="u")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pinn_burgers.png", dpi=150)
    print(f"Figure : {FIG_DIR / 'pinn_burgers.png'}")


if __name__ == "__main__":
    main()
