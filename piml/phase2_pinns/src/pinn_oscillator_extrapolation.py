"""
Phase 2.2C - A quoi sert un PINN : la PHYSIQUE permet d'extrapoler hors des donnees
==================================================================================
On ne fournit des mesures (bruitees) que sur le PREMIER TIERS du temps [0, 0.4].
On compare deux modeles sur tout [0, 1] :
  A) DONNEES SEULES        -> bon sur [0,0.4], puis part en vrille (aucune contrainte au-dela).
  B) DONNEES + PHYSIQUE    -> la physique (oscillateur amorti) le force a continuer correctement.
=> demonstration que la physique sert de "regularisation" et permet l'extrapolation.

Sortie : results/figures/pinn_extrapolation.png
Lancer  : python src/pinn_oscillator_extrapolation.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
DELTA, OMEGA0 = 2.0, 20.0
T_DATA = 0.4                       # les donnees ne couvrent que [0, T_DATA]

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"


def analytic(t):
    wd = np.sqrt(OMEGA0**2 - DELTA**2)
    return np.exp(-DELTA * t) * (np.cos(wd * t) + (DELTA / wd) * np.sin(wd * t))


def make_model():
    torch.manual_seed(0)           # meme initialisation pour une comparaison juste
    return nn.Sequential(
        nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1),
    )


def main():
    # --- Donnees bruitees, UNIQUEMENT sur [0, T_DATA] ---
    td = np.sort(np.random.rand(20, 1) * T_DATA, axis=0)
    ud = analytic(td) + 0.02 * np.random.randn(20, 1)
    Td, Ud = torch.tensor(td, dtype=torch.float32), torch.tensor(ud, dtype=torch.float32)

    # --- Modele A : DONNEES SEULES ---
    mA = make_model(); optA = torch.optim.Adam(mA.parameters(), lr=1e-3)
    for _ in range(8000):
        optA.zero_grad()
        ((mA(Td) - Ud) ** 2).mean().backward()
        optA.step()

    # --- Modele B : DONNEES + PHYSIQUE (residu impose sur tout [0,1]) ---
    mB = make_model(); optB = torch.optim.Adam(mB.parameters(), lr=1e-3)
    tc = torch.linspace(0, 1, 200).reshape(-1, 1).requires_grad_(True)
    for _ in range(15000):
        optB.zero_grad()
        loss_data = ((mB(Td) - Ud) ** 2).mean()
        u = mB(tc)
        u_t = torch.autograd.grad(u, tc, torch.ones_like(u), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, tc, torch.ones_like(u_t), create_graph=True)[0]
        loss_phys = ((u_tt + 2 * DELTA * u_t + OMEGA0**2 * u) ** 2).mean()
        (loss_data + 1e-4 * loss_phys).backward()
        optB.step()

    # --- Evaluation, surtout dans la zone SANS donnees [T_DATA, 1] ---
    tt = np.linspace(0, 1, 400)
    tt_t = torch.tensor(tt.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        uA = mA(tt_t).numpy().ravel(); uB = mB(tt_t).numpy().ravel()
    utrue = analytic(tt)
    ex = tt >= T_DATA
    rmseA = np.sqrt(np.mean((uA[ex] - utrue[ex]) ** 2))
    rmseB = np.sqrt(np.mean((uB[ex] - utrue[ex]) ** 2))
    print(f"RMSE dans la zone SANS donnees [{T_DATA},1] :")
    print(f"  A (donnees seules)     = {rmseA:.3f}")
    print(f"  B (donnees + physique) = {rmseB:.3f}   -> {rmseA/rmseB:.0f}x meilleur")

    # --- Figure ---
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axvspan(T_DATA, 1, color="0.92", label="zone sans donnees")
    ax.plot(tt, utrue, "k--", lw=2, label="solution exacte")
    ax.plot(tt, uA, color="tab:red", lw=1.6, label="A : donnees seules")
    ax.plot(tt, uB, color="tab:green", lw=1.8, label="B : donnees + physique (PINN)")
    ax.scatter(td, ud, c="tab:blue", s=25, zorder=5, label="mesures (sur [0, 0.4])")
    ax.axvline(T_DATA, color="grey", lw=0.8)
    ax.set(xlabel="t [s]", ylabel="u(t)", ylim=(-1.0, 1.2),
           title="La physique permet d'extrapoler hors des donnees")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pinn_extrapolation.png", dpi=150)
    print(f"Figure : {FIG_DIR / 'pinn_extrapolation.png'}")


if __name__ == "__main__":
    main()
