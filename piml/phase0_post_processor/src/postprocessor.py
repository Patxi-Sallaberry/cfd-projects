"""
Post-processeur CFD - Phase 0 (V&V)
===================================
NACA 0012 : trace la polaire de REFERENCE (NeuralFoil/XFOIL, Re=4e5) et y
superpose, si disponibles, les points de TES simulations Fluent, avec calcul
de l'ecart relatif. Sauvegarde la figure dans outputs/.

Rappel : Cl = L / (q_inf * S_ref), Cd = D / (q_inf * S_ref),
         q_inf = 1/2 * rho * V_inf^2. L, D = forces (Fluent : Reports -> Forces).

Usage : python src/postprocessor.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Constantes de reference (IDENTIQUES aux Reference Values de Fluent) ---
RHO, V_INF, CHORD, SPAN = 1.225, 30.0, 0.20, 0.30
S_REF = CHORD * SPAN                 # aire de reference [m2]
Q_INF = 0.5 * RHO * V_INF**2         # pression dynamique [Pa]

# --- 2. Chemins ---
ROOT       = Path(__file__).resolve().parent.parent
REF_CSV    = ROOT / "data" / "naca0012_reference.csv"   # verite-terrain
FLUENT_CSV = ROOT / "data" / "naca0012_fluent.csv"      # mes runs (optionnel)
OUT_DIR    = ROOT / "outputs"
OUT_PNG    = OUT_DIR / "naca0012_polar.png"


def load_fluent():
    """Renvoie le DataFrame Fluent avec Cl/Cd calcules, ou None si donnees absentes/vides."""
    if not FLUENT_CSV.exists():
        return None
    df = pd.read_csv(FLUENT_CSV, comment="#")
    if df[["lift_N", "drag_N"]].isna().any().any():
        return None                                  # forces pas encore remplies
    df["Cl_cfd"] = df["lift_N"] / (Q_INF * S_REF)
    df["Cd_cfd"] = df["drag_N"] / (Q_INF * S_REF)
    return df


def main():
    # --- 3. Reference (toujours presente) ---
    ref = pd.read_csv(REF_CSV, comment="#")
    print("Polaire de reference :")
    print(ref[["alpha_deg", "Cl_ref", "Cd_ref"]].to_string(index=False))

    fl = load_fluent()

    # --- 4. Figure : Cl(alpha) et Cd(alpha) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(ref["alpha_deg"], ref["Cl_ref"], "-", color="tab:green", label="Reference (NeuralFoil)")
    ax2.plot(ref["alpha_deg"], ref["Cd_ref"], "-", color="tab:green", label="Reference (NeuralFoil)")

    # --- 5. Superposition de mes points Fluent + table d'ecart (si dispo) ---
    if fl is not None:
        ax1.plot(fl["alpha_deg"], fl["Cl_cfd"], "o", color="tab:blue", label="Fluent (moi)")
        ax2.plot(fl["alpha_deg"], fl["Cd_cfd"], "s", color="tab:red",  label="Fluent (moi)")

        cmp = ref.merge(fl[["alpha_deg", "Cl_cfd", "Cd_cfd"]], on="alpha_deg", how="inner")
        cmp["dCl_%"] = 100 * (cmp["Cl_cfd"] - cmp["Cl_ref"]) / cmp["Cl_ref"].replace(0, pd.NA)
        cmp["dCd_%"] = 100 * (cmp["Cd_cfd"] - cmp["Cd_ref"]) / cmp["Cd_ref"]
        print("\nEcart Fluent vs reference :")
        print(cmp[["alpha_deg", "Cl_ref", "Cl_cfd", "dCl_%",
                   "Cd_ref", "Cd_cfd", "dCd_%"]].round(3).to_string(index=False))
    else:
        print("\n[i] Pas de donnees Fluent exploitables (data/naca0012_fluent.csv vide) :")
        print("    on trace uniquement la reference. Remplis lift_N/drag_N puis relance.")

    for ax, ylab, title in ((ax1, "Cl", "Cl vs alpha"), (ax2, "Cd", "Cd vs alpha")):
        ax.set(xlabel="alpha [deg]", ylabel=ylab, title=title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("NACA 0012 - polaire V&V (reference vs Fluent), Re = 4e5")
    fig.tight_layout()

    # --- 6. Sauvegarde ---
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nFigure sauvegardee : {OUT_PNG}")


if __name__ == "__main__":
    main()
