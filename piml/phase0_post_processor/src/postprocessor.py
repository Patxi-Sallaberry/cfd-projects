"""
Post-processeur CFD - Phase 0
==============================
NACA 0012 : calcule les coefficients aerodynamiques Cl et Cd a partir des
FORCES de portance/trainee exportees de Fluent, puis trace Cl(alpha) et Cd(alpha).

Rappel physique :
    Cl = L / (q_inf * S_ref)        Cd = D / (q_inf * S_ref)
    avec q_inf = 1/2 * rho * V_inf^2 (pression dynamique) et S_ref l'aire de reference.
    L et D sont des FORCES (N), pas des vitesses : elles viennent de l'integrale
    pression + cisaillement sur le profil (Fluent : Reports -> Forces).

Usage :
    python src/postprocessor.py                       # lit data/naca0012_clcd.csv
    python src/postprocessor.py data/naca0012_clcd_EXAMPLE.csv   # test avec l'exemple
"""

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Constantes physiques de reference (IDENTIQUES aux Reference Values de Fluent) ---
RHO    = 1.225              # masse volumique de l'air        [kg/m3]
V_INF  = 30.0              # vitesse amont                    [m/s]
CHORD  = 0.20             # corde du profil                  [m]
SPAN   = 0.30             # envergure                        [m]
S_REF  = CHORD * SPAN      # aire de reference                [m2]
Q_INF  = 0.5 * RHO * V_INF**2   # pression dynamique          [Pa]

# --- 2. Chemins des fichiers, calcules relativement a l'emplacement du script ---
HERE     = Path(__file__).resolve().parent     # .../phase0_post_processor/src
ROOT     = HERE.parent                         # .../phase0_post_processor
DATA_CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "naca0012_clcd.csv"
OUT_DIR  = ROOT / "outputs"
OUT_PNG  = OUT_DIR / "cl_cd_vs_alpha.png"


def main():
    # --- 3. Lecture du CSV (les lignes commencant par '#' sont ignorees) ---
    df = pd.read_csv(DATA_CSV, comment="#")
    print(f"Donnees lues depuis : {DATA_CSV}\n{df}\n")

    # --- 4. Garde-fou : sans les forces, on ne peut pas calculer Cl/Cd ---
    if df[["lift_N", "drag_N"]].isna().any().any():
        print("[!] Colonnes lift_N / drag_N incompletes : impossible de calculer Cl/Cd.")
        print("    -> Renseigne les forces depuis Fluent (Reports -> Forces).")
        print("    -> Pour tester tout de suite : "
              "python src/postprocessor.py data/naca0012_clcd_EXAMPLE.csv")
        return

    # --- 5. Coefficients : on divise la force par (pression dynamique * aire) ---
    df["Cl"] = df["lift_N"] / (Q_INF * S_REF)
    df["Cd"] = df["drag_N"] / (Q_INF * S_REF)
    print("Coefficients calcules :")
    print(df[["alpha_deg", "Cl", "Cd"]], "\n")

    # --- 6. Trace : Cl(alpha) a gauche, Cd(alpha) a droite ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(df["alpha_deg"], df["Cl"], "o-", color="tab:blue")
    ax1.set(xlabel="Angle d'attaque alpha [deg]", ylabel="Cl", title="Cl vs alpha")
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["alpha_deg"], df["Cd"], "s-", color="tab:red")
    ax2.set(xlabel="Angle d'attaque alpha [deg]", ylabel="Cd", title="Cd vs alpha")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("NACA 0012 - post-traitement CFD (Fluent, k-omega SST)")
    fig.tight_layout()

    # --- 7. Sauvegarde de la figure dans outputs/ ---
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Figure sauvegardee : {OUT_PNG}")


if __name__ == "__main__":
    main()
