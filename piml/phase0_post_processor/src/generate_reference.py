"""
Generation de la polaire de REFERENCE (verite-terrain V&V) - NACA 0012
======================================================================
Genere Cl(alpha) et Cd(alpha) avec NeuralFoil (reseau entraine sur XFOIL),
a Re = 400 000, pour servir de reference face a TES simulations Fluent.

(!) Provenance : NeuralFoil est un SUBSTITUT de XFOIL (ni experimental, ni Fluent).
    Donnees REPRODUCTIBLES : il suffit de relancer ce script. Voir data/SOURCES.md.

Re-generer :  python src/generate_reference.py
Dependances :  pip install neuralfoil aerosandbox
"""

from pathlib import Path
import numpy as np
import pandas as pd
import aerosandbox as asb
import neuralfoil as nf

# --- Parametres de generation (a garder coherents avec data/SOURCES.md) ---
RE     = 400_000          # Reynolds (= V_inf * corde / nu), identique a ta sim Fluent
MODEL  = "xlarge"         # taille du modele NeuralFoil (precision croissante)
ALPHAS = np.arange(0, 16, 1.0)   # balayage alpha 0 -> 15 deg

OUT = Path(__file__).resolve().parent.parent / "data" / "naca0012_reference.csv"


def main():
    # --- 1. Profil analytique NACA 0012 + evaluation aero sur tous les alphas ---
    af   = asb.Airfoil("naca0012")
    aero = nf.get_aero_from_airfoil(af, alpha=ALPHAS, Re=RE, model_size=MODEL)

    # --- 2. Mise en table (on garde l'indice de confiance du modele) ---
    df = pd.DataFrame({
        "alpha_deg":  ALPHAS,
        "Cl_ref":     np.round(aero["CL"], 4),
        "Cd_ref":     np.round(aero["CD"], 5),
        "confidence": np.round(aero["analysis_confidence"], 3),
    })

    # --- 3. Ecriture avec un en-tete de provenance (lignes '#' = commentaires) ---
    header = (
        "# NACA 0012 - polaire de REFERENCE (verite-terrain V&V)\n"
        f"# Source : NeuralFoil {nf.__version__} (modele '{MODEL}', entraine sur XFOIL) via AeroSandbox\n"
        f"# Conditions : Re = {RE}, Mach = 0, Ncrit = 9 (defaut). Profil analytique NACA 0012.\n"
        "# /!\\ Substitut de XFOIL : NI experimental NI Fluent. Details : data/SOURCES.md\n"
        "# Regenerer : python src/generate_reference.py\n"
    )
    with open(OUT, "w") as f:
        f.write(header)
    df.to_csv(OUT, mode="a", index=False)

    print(df.to_string(index=False))
    print(f"\nReference ecrite : {OUT}")


if __name__ == "__main__":
    main()
