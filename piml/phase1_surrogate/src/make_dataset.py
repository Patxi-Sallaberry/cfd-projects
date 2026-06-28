"""
Phase 1 - Generation du dataset d'entrainement du surrogate
===========================================================
Produit une polaire DENSE NACA 0012 (NeuralFoil, Re=4e5) qui servira a entrainer
le modele surrogate alpha -> (Cl, Cd). Plus de points que la reference Phase 0
(pas par 0.25 deg) pour pouvoir separer train / test proprement.

Sortie : data/naca0012_surrogate_dataset.csv
Regenerer : python src/make_dataset.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import aerosandbox as asb
import neuralfoil as nf

RE     = 400_000
MODEL  = "xlarge"
ALPHAS = np.arange(-6.0, 16.01, 0.25)   # -6 a 16 deg par pas de 0.25 -> 89 points

OUT = Path(__file__).resolve().parent.parent / "data" / "naca0012_surrogate_dataset.csv"


def main():
    af   = asb.Airfoil("naca0012")
    aero = nf.get_aero_from_airfoil(af, alpha=ALPHAS, Re=RE, model_size=MODEL)

    df = pd.DataFrame({
        "alpha_deg": np.round(ALPHAS, 3),
        "Cl":        np.round(aero["CL"], 5),
        "Cd":        np.round(aero["CD"], 6),
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"{len(df)} points generes (alpha {ALPHAS[0]} -> {ALPHAS[-1]} deg, Re={RE})")
    print(df.head())
    print("...")
    print(f"Dataset ecrit : {OUT}")


if __name__ == "__main__":
    main()
