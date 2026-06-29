"""
Phase 2.2 - PINN sur un vrai PDE : equation de la chaleur 1D
============================================================
On resout   u_t = alpha * u_xx   sur x in [0,1], t in [0,T]
  condition initiale : u(x,0) = sin(pi x)
  conditions limites : u(0,t) = u(1,t) = 0
Solution exacte (pour validation) : u(x,t) = sin(pi x) * exp(-alpha pi^2 t).

Nouveautes vs l'oscillateur : 2 entrees (x,t) -> derivees PARTIELLES (u_t et u_xx),
et il faut a la fois la condition initiale ET les conditions aux limites.

Sortie : results/figures/pinn_heat.png
Lancer  : python src/pinn_heat.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(0)

ALPHA, T = 0.4, 1.0
N_COL, N_IC, N_BC = 1500, 200, 200     # points : interieur (PDE), condition initiale, bords
EPOCHS = 12000
W_IC, W_BC = 20.0, 20.0                # poids des contraintes (ancrent la solution)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"


def analytic(x, t):
    return np.sin(np.pi * x) * np.exp(-ALPHA * np.pi**2 * t)


def main():
    # --- 1. Reseau (x,t) -> u, tanh (derivees 2des) ---
    model = nn.Sequential(
        nn.Linear(2, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- 2. Points d'echantillonnage ---
    # interieur : la ou on impose le PDE (x et t separes pour deriver par rapport a chacun)
    xc = torch.rand(N_COL, 1, requires_grad=True)
    tc = (torch.rand(N_COL, 1) * T).requires_grad_(True)
    # condition initiale t=0 : valeurs cibles sin(pi x)
    xi = torch.rand(N_IC, 1); ti = torch.zeros(N_IC, 1)
    ui = torch.sin(np.pi * xi)
    # bords x=0 et x=1 : u doit valoir 0
    tb = torch.rand(N_BC, 1) * T
    x0 = torch.zeros(N_BC, 1); x1 = torch.ones(N_BC, 1)

    hist = []
    for epoch in range(EPOCHS):
        opt.zero_grad()

        # (a) residu du PDE : derivees partielles via autograd
        u = model(torch.cat([xc, tc], dim=1))
        u_t = torch.autograd.grad(u, tc, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, xc, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, xc, torch.ones_like(u_x), create_graph=True)[0]
        loss_phys = ((u_t - ALPHA * u_xx) ** 2).mean()

        # (b) condition initiale : u(x,0) = sin(pi x)
        loss_ic = ((model(torch.cat([xi, ti], dim=1)) - ui) ** 2).mean()

        # (c) conditions aux limites : u(0,t) = u(1,t) = 0
        loss_bc = (model(torch.cat([x0, tb], dim=1)) ** 2).mean() \
                + (model(torch.cat([x1, tb], dim=1)) ** 2).mean()

        loss = loss_phys + W_IC * loss_ic + W_BC * loss_bc
        loss.backward(); opt.step()
        hist.append(loss.item())
        if epoch % 1500 == 0:
            print(f"epoch {epoch:5d}  loss={loss.item():.3e}  "
                  f"(phys={loss_phys.item():.2e} ic={loss_ic.item():.2e} bc={loss_bc.item():.2e})")

    # --- 3. Validation sur une grille (x,t) ---
    xs = np.linspace(0, 1, 100); ts = np.linspace(0, 1, 100)
    XX, TT = np.meshgrid(xs, ts)
    grid = torch.tensor(np.column_stack([XX.ravel(), TT.ravel()]), dtype=torch.float32)
    with torch.no_grad():
        U = model(grid).numpy().ravel()
    Utrue = analytic(XX.ravel(), TT.ravel())
    rmse = np.sqrt(np.mean((U - Utrue) ** 2))
    r2 = 1 - np.sum((U - Utrue) ** 2) / np.sum((Utrue - Utrue.mean()) ** 2)
    print(f"\n[VALIDATION vs analytique]  RMSE = {rmse:.4f}   R2 = {r2:.4f}")

    # --- 4. Figures : profils u(x) a differents instants + convergence ---
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, 4))
    for tv, c in zip([0.0, 0.2, 0.5, 1.0], colors):
        xx = np.linspace(0, 1, 200)
        with torch.no_grad():
            up = model(torch.tensor(np.column_stack([xx, np.full_like(xx, tv)]),
                                    dtype=torch.float32)).numpy().ravel()
        ax1.plot(xx, analytic(xx, tv), "k--", lw=1.4)
        ax1.plot(xx, up, color=c, lw=1.8, label=f"t = {tv}")
    ax1.plot([], [], "k--", label="exact")
    ax1.set(xlabel="x", ylabel="u(x,t)", title="Diffusion de la chaleur : PINN vs exact")
    ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8)

    ax2.plot(hist); ax2.set_yscale("log")
    ax2.set(xlabel="epoch", ylabel="loss (log)", title="Convergence de la loss PINN")
    ax2.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pinn_heat.png", dpi=150)
    print(f"Figure : {FIG_DIR / 'pinn_heat.png'}")


if __name__ == "__main__":
    main()
