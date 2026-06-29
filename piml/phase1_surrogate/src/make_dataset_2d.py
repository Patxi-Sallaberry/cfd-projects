"""
Phase 1 (2D) - Dataset d'entrainement du surrogate (alpha, Re) -> (Cl, Cd)
==========================================================================
Genere une GRILLE de polaires NACA 0012 (NeuralFoil) sur plusieurs angles ET
plusieurs Reynolds, pour entrainer un surrogate a 2 entrees.

Sortie : data/naca0012_surrogate_2d.csv
Lancer : python src/make_dataset_2d.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import aerosandbox as asb
import neuralfoil as nf

MODEL  = "xlarge"
ALPHAS = np.arange(-6.0, 16.01, 0.5)                       # 45 angles
RES    = np.array([1e5, 1.5e5, 2.5e5, 4e5, 6e5, 9e5, 1.5e6])  # 7 Reynolds

OUT = Path(__file__).resolve().parent.parent / "data" / "naca0012_surrogate_2d.csv"


def main():
    af = asb.Airfoil("naca0012")
    rows = []
    for Re in RES:
        aero = nf.get_aero_from_airfoil(af, alpha=ALPHAS, Re=float(Re), model_size=MODEL)
        for a, cl, cd in zip(ALPHAS, aero["CL"], aero["CD"]):
            rows.append((round(a, 3), float(Re), round(float(cl), 5), round(float(cd), 6)))

    df = pd.DataFrame(rows, columns=["alpha_deg", "Re", "Cl", "Cd"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} points ({len(ALPHAS)} alpha x {len(RES)} Re)")
    print(df.head())
    print(f"Dataset ecrit : {OUT}")


if __name__ == "__main__":
    main()
