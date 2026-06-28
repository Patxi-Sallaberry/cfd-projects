"""
Post-processeur CFD - Phase 0 (V&V)
===================================
NACA 0012 : compare la polaire de REFERENCE (NeuralFoil/XFOIL, Re=4e5) a MES
simulations Fluent. Trace 3 graphes -- Cl(alpha), Cd(alpha), L/D(alpha) -- avec
mes points superposes a la reference, et imprime l'ecart relatif.

Rappel : Cl = L / (q_inf * S_ref), Cd = D / (q_inf * S_ref), q_inf = 1/2 rho V^2.

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
FLUENT_CSV = ROOT / "data" / "naca0012_fluent.csv"      # mes runs
FIG_DIR    = ROOT / "results" / "figures"               # figures publiees
OUT_PNG    = FIG_DIR / "naca0012_vv_polar.png"


def _filled(df, cols):
    """True si les colonnes 'cols' existent toutes ET sont entierement remplies."""
    return set(cols).issubset(df.columns) and not df[list(cols)].isna().any().any()


def load_fluent():
    """Renvoie le DataFrame Fluent avec colonnes Cl_cfd/Cd_cfd, ou None si rien d'exploitable."""
    if not FLUENT_CSV.exists():
        return None
    df = pd.read_csv(FLUENT_CSV, comment="#")

    # Cas 1 (le mien) : Cl/Cd lus directement dans Fluent.
    if _filled(df, ["Cl_cfd", "Cd_cfd"]):
        return df
    # Cas 2 (secours) : seulement les forces -> on calcule les coefficients.
    if _filled(df, ["lift_N", "drag_N"]):
        df["Cl_cfd"] = df["lift_N"] / (Q_INF * S_REF)
        df["Cd_cfd"] = df["drag_N"] / (Q_INF * S_REF)
        return df
    return None


def main():
    # --- 3. Reference + finesse L/D (toujours presente) ---
    ref = pd.read_csv(REF_CSV, comment="#")
    ref["LD_ref"] = ref["Cl_ref"] / ref["Cd_ref"]
    fl = load_fluent()

    # --- 4. Trois graphes : Cl, Cd, L/D en fonction de alpha ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        (ax1, "Cl_ref", "Cl",  "Cl vs alpha"),
        (ax2, "Cd_ref", "Cd",  "Cd vs alpha"),
        (ax3, "LD_ref", "L/D", "Finesse L/D vs alpha"),
    ]
    for ax, col, ylab, title in panels:
        ax.plot(ref["alpha_deg"], ref[col], "-", color="tab:green", label="Reference (NeuralFoil)")
        ax.set(xlabel="alpha [deg]", ylabel=ylab, title=title)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="grey", lw=0.8)        # repere y=0 (utile pour voir les signes)

    # --- 5. Superposition de mes points Fluent + table d'ecart ---
    if fl is not None:
        fl["LD_cfd"] = fl["Cl_cfd"] / fl["Cd_cfd"]
        ax1.plot(fl["alpha_deg"], fl["Cl_cfd"], "o", color="tab:red", label="Fluent (moi)")
        ax2.plot(fl["alpha_deg"], fl["Cd_cfd"], "o", color="tab:red", label="Fluent (moi)")
        ax3.plot(fl["alpha_deg"], fl["LD_cfd"], "o", color="tab:red", label="Fluent (moi)")

        cmp = ref.merge(fl[["alpha_deg", "Cl_cfd", "Cd_cfd", "LD_cfd"]], on="alpha_deg", how="inner")
        cmp["dCl_%"] = 100 * (cmp["Cl_cfd"] - cmp["Cl_ref"]) / cmp["Cl_ref"].replace(0, pd.NA)
        cmp["dCd_%"] = 100 * (cmp["Cd_cfd"] - cmp["Cd_ref"]) / cmp["Cd_ref"]
        print("Comparaison Fluent vs reference :")
        print(cmp[["alpha_deg", "Cl_ref", "Cl_cfd", "dCl_%",
                   "Cd_ref", "Cd_cfd", "dCd_%", "LD_ref", "LD_cfd"]].round(3).to_string(index=False))
        print("\n/!\\ Ecarts massifs (Cl trop faible, Cd trop eleve) -> run encore non valide (voir README).")
    else:
        print("[i] Pas de donnees Fluent exploitables : on trace uniquement la reference.")

    for ax in (ax1, ax2, ax3):
        ax.legend()
    fig.suptitle("NACA 0012 - V&V : reference vs ma simulation Fluent (Re = 4e5)")
    fig.tight_layout()

    # --- 6. Sauvegarde dans results/figures/ (figure publiee, versionnee) ---
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nFigure sauvegardee : {OUT_PNG}")


if __name__ == "__main__":
    main()
