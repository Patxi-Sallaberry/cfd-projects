"""
Phase 2.3 - Figures pedagogiques pour l'equation de Burgers
===========================================================
Genere deux images concretes (sans entrainer de reseau, juste avec le solveur FD) :
  1) burgers_intuition.png  : POURQUOI un choc se forme (chaque point avance a sa propre vitesse u)
  2) burgers_evolution.png  : le profil qui se RAIDIT instant apres instant

Lancer : python src/burgers_illustrate.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

NU = 0.01 / np.pi
FIG_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"


def fd_reference(nx=401, nt=4000):
    x = np.linspace(-1, 1, nx); dx = x[1] - x[0]
    t = np.linspace(0, 1, nt); dt = t[1] - t[0]
    U = np.zeros((nt, nx)); U[0] = -np.sin(np.pi * x)
    for n in range(nt - 1):
        un = U[n]; lap = np.zeros(nx); dudx = np.zeros(nx)
        lap[1:-1] = (un[2:] - 2 * un[1:-1] + un[:-2]) / dx**2
        dudx[1:-1] = np.where(un[1:-1] >= 0, (un[1:-1] - un[:-2]) / dx, (un[2:] - un[1:-1]) / dx)
        new = un + dt * (-un * dudx + NU * lap); new[0] = 0; new[-1] = 0
        U[n + 1] = new
    return x, t, U


def fig_intuition():
    """Pourquoi un choc : chaque point se deplace horizontalement a une vitesse egale a sa hauteur u."""
    x = np.linspace(-1, 1, 400); u0 = -np.sin(np.pi * x)
    xs = np.linspace(-0.85, 0.85, 13); us = -np.sin(np.pi * xs)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, u0, color="tab:blue", lw=2, label="profil initial  u(x,0) = −sin(πx)")
    # fleche horizontale a chaque point : longueur et sens = la valeur u (vitesse de transport)
    ax.quiver(xs, us, us, np.zeros_like(us), angles="xy", scale_units="xy", scale=2.2,
              color="tab:red", width=0.004, label="vitesse de chaque point  (= u)")
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.text(0.02, 0.9, "les points se rejoignent ici\n→ le profil se raidit → CHOC", fontsize=9,
            color="black", va="top")
    ax.text(-0.7, 0.45, "u > 0\n→ va vers la droite", color="tab:red", fontsize=9, ha="center")
    ax.text(0.7, -0.55, "u < 0\n→ va vers la gauche", color="tab:red", fontsize=9, ha="center")
    ax.set(xlabel="x", ylabel="u", title="Pourquoi un choc se forme : chaque point avance à sa propre vitesse u",
           ylim=(-1.2, 1.2))
    ax.grid(True, alpha=0.3); ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "burgers_intuition.png", dpi=150)
    print("burgers_intuition.png")


def fig_evolution():
    """Le profil se raidit instant apres instant (reference FD)."""
    x, t, U = fd_reference()
    times = np.linspace(0, 0.9, 10)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(times)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for tv, c in zip(times, colors):
        ax.plot(x, U[np.argmin(np.abs(t - tv))], color=c, lw=1.8, label=f"t = {tv:.1f}")
    ax.axvline(0, color="grey", ls="--", lw=0.8)
    ax.annotate("le front devient quasi vertical\n(le choc)", xy=(0.0, 0.0), xytext=(0.35, 0.6),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))
    ax.set(xlabel="x", ylabel="u(x,t)", title="Burgers : le profil se raidit dans le temps jusqu'au choc")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIG_DIR / "burgers_evolution.png", dpi=150)
    print("burgers_evolution.png")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_intuition()
    fig_evolution()
    print(f"Figures dans {FIG_DIR}")
