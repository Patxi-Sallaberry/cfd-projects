"""
Phase 2.2b - PINN : probleme INVERSE (retrouver un parametre inconnu)
====================================================================
On ne resout plus une equation deja connue : on a quelques MESURES bruitees
(comme des capteurs epars) et un parametre physique INCONNU (la diffusivite alpha).
Le PINN apprend en meme temps :
  - a coller aux mesures        (loss_data)
  - a respecter l'equation chaleur u_t = alpha*u_xx avec le alpha en cours d'apprentissage (loss_phys)
-> il RETROUVE alpha + reconstruit le champ complet. Impossible avec la seule formule analytique.

Sortie : results/figures/pinn_heat_inverse.png
Lancer  : python src/pinn_heat_inverse.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)

ALPHA_TRUE = 0.4        # valeur "vraie" (le PINN ne la connait pas)
N_MEAS = 40             # nombre de mesures bruitees
EPOCHS = 10000

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"


def analytic(x, t):
    return np.sin(np.pi * x) * np.exp(-ALPHA_TRUE * np.pi**2 * t)


def main():
    # --- 1. "Mesures" : 40 points (x,t) avec leur valeur bruitee (capteurs) ---
    xm = np.random.rand(N_MEAS, 1); tm = np.random.rand(N_MEAS, 1)
    um = analytic(xm, tm) + 0.02 * np.random.randn(N_MEAS, 1)
    Xm, Tm, Um = (torch.tensor(a, dtype=torch.float32) for a in (xm, tm, um))

    # --- 2. Modele u(x,t) + le parametre alpha, INCONNU, qu'on rend entrainable ---
    model = nn.Sequential(
        nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1),
    )
    alpha = nn.Parameter(torch.tensor(1.5))                 # depart volontairement FAUX
    opt = torch.optim.Adam(list(model.parameters()) + [alpha], lr=5e-3)

    xc = torch.rand(800, 1, requires_grad=True)
    tc = torch.rand(800, 1, requires_grad=True)

    a_hist = []
    for epoch in range(EPOCHS):
        opt.zero_grad()
        # (a) coller aux mesures
        loss_data = ((model(torch.cat([Xm, Tm], 1)) - Um) ** 2).mean()
        # (b) respecter la physique avec le alpha appris
        u = model(torch.cat([xc, tc], 1))
        u_t = torch.autograd.grad(u, tc, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, xc, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, xc, torch.ones_like(u_x), create_graph=True)[0]
        loss_phys = ((u_t - alpha * u_xx) ** 2).mean()

        loss = loss_data + 1e-2 * loss_phys
        loss.backward(); opt.step()
        a_hist.append(alpha.item())
        if epoch % 2000 == 0:
            print(f"epoch {epoch:5d}  alpha={alpha.item():.4f}  loss_data={loss_data.item():.2e}")

    err = abs(alpha.item() - ALPHA_TRUE) / ALPHA_TRUE * 100
    print(f"\n>>> alpha RETROUVE = {alpha.item():.4f}  (vrai {ALPHA_TRUE}, erreur {err:.1f}%)")

    # --- 3. Reconstruction du champ complet + validation ---
    xs = np.linspace(0, 1, 80); ts = np.linspace(0, 1, 80)
    XX, TT = np.meshgrid(xs, ts)
    grid = torch.tensor(np.column_stack([XX.ravel(), TT.ravel()]), dtype=torch.float32)
    with torch.no_grad():
        U = model(grid).numpy().reshape(80, 80)
    Utrue = analytic(XX, TT)
    r2 = 1 - np.sum((U - Utrue) ** 2) / np.sum((Utrue - Utrue.mean()) ** 2)
    print(f"    champ reconstruit : R2 = {r2:.4f}")

    # --- 4. Figure : convergence de alpha + champ reconstruit avec les mesures ---
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(a_hist, color="tab:orange", label="alpha appris")
    ax1.axhline(ALPHA_TRUE, color="k", ls="--", label=f"alpha vrai = {ALPHA_TRUE}")
    ax1.set(xlabel="epoch", ylabel="alpha", title="Le parametre inconnu est retrouve")
    ax1.grid(True, alpha=0.3); ax1.legend()

    cf = ax2.contourf(XX, TT, U, levels=20, cmap="viridis")
    ax2.scatter(xm, tm, c="white", s=14, edgecolors="k", linewidths=0.5, label="40 mesures")
    ax2.set(xlabel="x", ylabel="t", title=f"Champ reconstruit (R²={r2:.3f}) depuis les points")
    ax2.legend(loc="upper right", fontsize=8)
    fig.colorbar(cf, ax=ax2, label="u(x,t)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pinn_heat_inverse.png", dpi=150)
    print(f"Figure : {FIG_DIR / 'pinn_heat_inverse.png'}")


if __name__ == "__main__":
    main()
