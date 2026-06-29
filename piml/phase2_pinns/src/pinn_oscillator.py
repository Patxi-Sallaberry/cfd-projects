"""
Phase 2 - Premier PINN : oscillateur harmonique amorti
======================================================
On resout  u'' + 2*delta*u' + omega0^2 * u = 0,  u(0)=1, u'(0)=0
SANS donnees : le reseau apprend uniquement en rendant nul le RESIDU de l'equation
(+ les conditions initiales). On valide contre la solution analytique exacte.

Nouveaute vs Phase 1 : autograd derive la sortie u par rapport a l'ENTREE t
(pour obtenir u' et u''), et non plus seulement par rapport aux poids.

Sortie : results/figures/pinn_oscillator.png
Lancer  : python src/pinn_oscillator.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(0)

DELTA, OMEGA0 = 2.0, 20.0           # amortissement, pulsation propre (sous-amorti : delta < omega0)
W_PHYS = 1e-4                       # poids du terme physique (equilibre les magnitudes)
EPOCHS = 20000

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"


def analytic(t):
    """Solution exacte de l'oscillateur sous-amorti (pour validation)."""
    wd = np.sqrt(OMEGA0**2 - DELTA**2)
    return np.exp(-DELTA * t) * (np.cos(wd * t) + (DELTA / wd) * np.sin(wd * t))


def main():
    # --- 1. Le reseau : t -> u. tanh car on prend des derivees 2des (besoin de douceur) ---
    model = nn.Sequential(
        nn.Linear(1, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- 2. Points de collocation : la ou on impose l'equation (pas des donnees !) ---
    t_phys = torch.linspace(0, 1, 200).reshape(-1, 1).requires_grad_(True)
    t0 = torch.zeros(1, 1, requires_grad=True)            # pour les conditions initiales

    hist = []
    for epoch in range(EPOCHS):
        opt.zero_grad()

        # (a) RESIDU PHYSIQUE : on derive u par rapport a t via autograd
        u = model(t_phys)
        u_t = torch.autograd.grad(u, t_phys, torch.ones_like(u), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_phys, torch.ones_like(u_t), create_graph=True)[0]
        residual = u_tt + 2 * DELTA * u_t + OMEGA0**2 * u   # = 0 si l'equation est satisfaite
        loss_phys = (residual ** 2).mean()

        # (b) CONDITIONS INITIALES : u(0)=1 et u'(0)=0
        u0 = model(t0)
        u0_t = torch.autograd.grad(u0, t0, torch.ones_like(u0), create_graph=True)[0]
        loss_ic = (u0 - 1.0) ** 2 + (u0_t - 0.0) ** 2

        # (c) loss totale = physique (ponderee) + conditions initiales
        loss = W_PHYS * loss_phys + loss_ic.squeeze()
        loss.backward()
        opt.step()
        hist.append(loss.item())
        if epoch % 2000 == 0:
            print(f"epoch {epoch:5d}  loss={loss.item():.4e}  (phys={loss_phys.item():.3e})")

    # --- 3. Validation contre la solution analytique ---
    tt = np.linspace(0, 1, 400)
    with torch.no_grad():
        u_pred = model(torch.tensor(tt.reshape(-1, 1), dtype=torch.float32)).numpy().ravel()
    u_true = analytic(tt)
    rmse = np.sqrt(np.mean((u_pred - u_true) ** 2))
    r2 = 1 - np.sum((u_pred - u_true) ** 2) / np.sum((u_true - u_true.mean()) ** 2)
    print(f"\n[VALIDATION vs analytique]  RMSE = {rmse:.4f}   R2 = {r2:.4f}")

    # --- 4. Figures ---
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(tt, u_true, "k--", lw=2, label="solution exacte")
    ax1.plot(tt, u_pred, color="tab:orange", lw=1.6, label="PINN (physique seule)")
    ax1.axhline(0, color="grey", lw=0.6)
    ax1.set(xlabel="t [s]", ylabel="u(t)", title="Oscillateur amorti : PINN vs exact")
    ax1.grid(True, alpha=0.3); ax1.legend()

    ax2.plot(hist); ax2.set_yscale("log")
    ax2.set(xlabel="epoch", ylabel="loss (log)", title="Convergence de la loss PINN")
    ax2.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pinn_oscillator.png", dpi=150)
    print(f"Figure : {FIG_DIR / 'pinn_oscillator.png'}")


if __name__ == "__main__":
    main()
