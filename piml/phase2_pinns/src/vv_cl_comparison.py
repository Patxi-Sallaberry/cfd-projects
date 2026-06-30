"""
Validation & Verification : Cl(alpha) du PINN parametrique vs references
========================================================================
Compare trois courbes de portance pour le NACA 4412 inviscide :
  - theorie du profil mince  : Cl = 2*pi*(alpha - alpha_L0)   (reference analytique, approchee)
  - methode des PANNEAUX      : Cl quasi exact (cf. panel_method.py)  <- LA bonne reference
  - PINN parametrique (psi)   : sortie du reseau (x,y,alpha)->psi  (valeurs du run committe)

But : montrer que le PINN capture la bonne PHYSIQUE (courbe lineaire, bon angle de portance nulle)
mais SOUS-ESTIME l'amplitude — et que la methode des panneaux, elle, colle a la theorie. C'est
l'argument de maturite : savoir quel outil est adapte (panneaux pour la portance inviscide exacte,
PINN comme demonstrateur PIML) et valider contre la bonne reference.

Sortie : results/figures/vv_cl_comparison.png
Lancer  : python src/vv_cl_comparison.py
"""

from pathlib import Path as FsPath
import numpy as np
import matplotlib.pyplot as plt
from panel_method import cl_curve

ALPHA_L0_DEG = -4.0
ROOT = FsPath(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures" / "vv_cl_comparison.png"

# Cl(alpha) du PINN parametrique psi (attempt 2), valeurs du run committe (30000 epoques)
ALPHAS = np.linspace(-5, 15, 21)
CL_PINN = np.array([-0.10, -0.05, -0.01, 0.04, 0.09, 0.14, 0.19, 0.24, 0.29, 0.34, 0.40,
                    0.45, 0.50, 0.55, 0.59, 0.64, 0.69, 0.73, 0.77, 0.81, 0.84])


def main():
    cl_panel, _ = cl_curve(ALPHAS)
    cl_theory = 2*np.pi * np.deg2rad(ALPHAS - ALPHA_L0_DEG)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(ALPHAS, cl_theory, "k--", lw=1.5, label="Théorie profil mince  2π(α−α₀)")
    ax.plot(ALPHAS, cl_panel, "o-", color="seagreen", lw=2, ms=4,
            label="Méthode des panneaux (référence quasi exacte)")
    ax.plot(ALPHAS, CL_PINN, "s-", color="crimson", lw=2, ms=4,
            label="PINN paramétrique ψ  (x,y,α)→ψ")
    ax.axhline(0, color="0.7", lw=0.8); ax.axvline(ALPHA_L0_DEG, color="0.7", lw=0.8, ls=":")
    ax.set_xlabel("angle d'attaque α (°)"); ax.set_ylabel("Coefficient de portance Cl")
    ax.set_title("V&V — Cl(α) du NACA 4412 inviscide : PINN vs panneaux vs théorie")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)

    # annotation de l'ecart a 10 deg
    i10 = np.argmin(np.abs(ALPHAS - 10))
    ax.annotate(f"à 10° : panneaux {cl_panel[i10]:.2f}  vs  PINN {CL_PINN[i10]:.2f}\n"
                f"(PINN : comportement linéaire + bon α₀,\n pente réduite → amplitude sous-estimée)",
                xy=(10, CL_PINN[i10]), xytext=(-4.5, 1.65), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="0.4"))

    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"Figure : {FIG}")
    print(f"\n alpha   theorie  panneaux   PINN")
    for a, ct, cpa, cpi in zip(ALPHAS, cl_theory, cl_panel, CL_PINN):
        print(f" {a:6.1f}  {ct:6.2f}   {cpa:6.2f}    {cpi:6.2f}")


if __name__ == "__main__":
    main()
