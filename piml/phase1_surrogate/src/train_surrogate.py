"""
Phase 1 - Entrainement du modele surrogate  alpha -> (Cl, Cd)
============================================================
Un petit reseau de neurones (MLP) apprend la polaire NACA 0012 a partir du
dataset genere par make_dataset.py, puis predit Cl/Cd instantanement.

Idee : remplacer une simulation CFD (lente) par une fonction apprise (immediate).

Sorties :
  results/figures/surrogate_fit.png   (prediction vs donnees)
  results/surrogate_naca0012.pt       (poids du modele entraine)
Lancer : python src/train_surrogate.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(0)          # reproductibilite
rng = np.random.default_rng(0)

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data" / "naca0012_surrogate_dataset.csv"
FIG_DIR = ROOT / "results" / "figures"
MODEL_OUT = ROOT / "results" / "surrogate_naca0012.pt"


def main():
    # --- 1. Donnees : entree X = alpha, sorties Y = [Cl, Cd] ---
    df = pd.read_csv(DATA)
    X = df[["alpha_deg"]].values.astype("float32")        # shape (N, 1)
    Y = df[["Cl", "Cd"]].values.astype("float32")         # shape (N, 2)

    # --- 2. Separation train / test (80 % / 20 %), tiree au hasard ---
    N = len(df)
    idx = rng.permutation(N)
    n_test = int(0.2 * N)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    Xtr, Ytr = X[train_idx], Y[train_idx]
    Xte, Yte = X[test_idx],  Y[test_idx]

    # --- 3. Normalisation (moyenne/ecart-type calcules sur le TRAIN seulement) ---
    #     Un reseau apprend bien mieux sur des grandeurs centrees-reduites.
    xm, xs = Xtr.mean(0), Xtr.std(0)
    ym, ys = Ytr.mean(0), Ytr.std(0)
    norm_x = lambda a: (a - xm) / xs
    norm_y = lambda a: (a - ym) / ys
    denorm_y = lambda a: a * ys + ym

    Xtr_t = torch.tensor(norm_x(Xtr));  Ytr_t = torch.tensor(norm_y(Ytr))
    Xte_t = torch.tensor(norm_x(Xte));  Yte_t = torch.tensor(norm_y(Yte))

    # --- 4. Le modele : un MLP 1 -> 64 -> 64 -> 2 (tanh = sorties lisses) ---
    model = nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 2),
    )
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    # --- 5. Boucle d'entrainement ---
    for epoch in range(3000):
        opt.zero_grad()                       # remet les gradients a zero
        pred = model(Xtr_t)                   # prediction sur le train
        loss = loss_fn(pred, Ytr_t)           # erreur quadratique moyenne
        loss.backward()                       # calcule les gradients (retropropagation)
        opt.step()                            # met a jour les poids
        if (epoch + 1) % 500 == 0:
            print(f"epoch {epoch+1:4d}  loss_train = {loss.item():.5f}")

    # --- 6. Evaluation sur le TEST (donnees jamais vues a l'entrainement) ---
    model.eval()
    with torch.no_grad():
        Pte = denorm_y(model(Xte_t).numpy())  # predictions en unites physiques
    for j, name in enumerate(["Cl", "Cd"]):
        true, pred = Yte[:, j], Pte[:, j]
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        r2 = 1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2)
        print(f"[TEST] {name} : RMSE = {rmse:.4f}   R2 = {r2:.4f}")

    # --- 7. Courbe : donnees vs prediction du modele sur tout l'intervalle ---
    a_dense = np.linspace(X.min(), X.max(), 300).astype("float32").reshape(-1, 1)
    with torch.no_grad():
        p_dense = denorm_y(model(torch.tensor(norm_x(a_dense))).numpy())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, j, ylab in ((ax1, 0, "Cl"), (ax2, 1, "Cd")):
        ax.scatter(Xtr, Ytr[:, j], s=18, color="tab:blue", label="train")
        ax.scatter(Xte, Yte[:, j], s=30, color="tab:red", marker="s", label="test")
        ax.plot(a_dense, p_dense[:, j], "-", color="tab:green", lw=2, label="surrogate")
        ax.set(xlabel="alpha [deg]", ylabel=ylab, title=f"{ylab} vs alpha")
        ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle("Phase 1 - surrogate NACA 0012 (MLP)  alpha -> (Cl, Cd)")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "surrogate_fit.png", dpi=150)
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"\nFigure : {FIG_DIR / 'surrogate_fit.png'}\nModele : {MODEL_OUT}")


if __name__ == "__main__":
    main()
