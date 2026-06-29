"""
Phase 1 (2D) - Surrogate  (alpha, Re) -> (Cl, Cd)  pour NACA 0012
=================================================================
Un MLP apprend toute une FAMILLE de polaires (angle x Reynolds), puis sait
INTERPOLER a un Reynolds jamais vu pendant l'entrainement.

Points cles vs la version 1D :
  - 2 entrees : alpha et log10(Re)  (Re passe en log car il varie d'un facteur 10)
  - meme boucle canonique : split train/val + scheduler + early stopping
  - une figure "interpolation" : prediction a un Re hors grille vs reference NeuralFoil

Sorties : results/figures/surrogate_2d_*.png , results/surrogate_2d_naca0012.pt
Lancer  : python src/make_dataset_2d.py  puis  python src/train_surrogate_2d.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
import aerosandbox as asb
import neuralfoil as nf

torch.manual_seed(0)
rng = np.random.default_rng(0)

ROOT      = Path(__file__).resolve().parent.parent
DATA      = ROOT / "data" / "naca0012_surrogate_2d.csv"
FIG_DIR   = ROOT / "results" / "figures"
MODEL_OUT = ROOT / "results" / "surrogate_2d_naca0012.pt"

EPOCHS, PATIENCE = 6000, 800


def main():
    # --- 1. Donnees : entrees = [alpha, log10(Re)], sorties = [Cl, Cd] ---
    df = pd.read_csv(DATA)
    A  = df["alpha_deg"].values.astype("float32")
    Re = df["Re"].values.astype("float32")
    X  = np.column_stack([A, np.log10(Re)]).astype("float32")   # (N, 2)  <- log10(Re) !
    Y  = df[["Cl", "Cd"]].values.astype("float32")              # (N, 2)

    # --- 2. Split train (70%) / validation (30%) ---
    N = len(df); idx = rng.permutation(N); nv = int(0.3 * N)
    vi, ti = idx[:nv], idx[nv:]
    Xtr, Ytr, Xva, Yva = X[ti], Y[ti], X[vi], Y[vi]

    # --- 3. Normalisation par colonne (ajustee sur le train) ---
    xm, xs = Xtr.mean(0), Xtr.std(0)
    ym, ys = Ytr.mean(0), Ytr.std(0)
    nx  = lambda a: (a - xm) / xs
    ny  = lambda a: (a - ym) / ys
    dny = lambda a: a * ys + ym
    Xtr_t, Ytr_t = torch.tensor(nx(Xtr)), torch.tensor(ny(Ytr))
    Xva_t, Yva_t = torch.tensor(nx(Xva)), torch.tensor(ny(Yva))

    # --- 4. Modele 2 -> 64 -> 64 -> 2 + optimiseur + scheduler ---
    model = nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 2),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1500, gamma=0.5)
    loss_fn = nn.MSELoss()

    # --- 5. Boucle canonique : train + validation + early stopping ---
    train_hist, val_hist = [], []
    best_val, best_state, since = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(Xtr_t), Ytr_t); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva_t), Yva_t).item()
        train_hist.append(loss.item()); val_hist.append(vl)
        if vl < best_val - 1e-8:
            best_val, since = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                print(f"Early stopping epoch {epoch}"); break
        if epoch % 1000 == 0:
            print(f"epoch {epoch:4d}  train={loss.item():.6f}  val={vl:.6f}")

    model.load_state_dict(best_state)
    print(f"\nMeilleure val_loss = {best_val:.6f}")

    # --- 6. Metriques de validation (unites physiques) ---
    model.eval()
    with torch.no_grad():
        Pva = dny(model(Xva_t).numpy())
    for j, name in enumerate(["Cl", "Cd"]):
        t, p = Yva[:, j], Pva[:, j]
        r2 = 1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2)
        print(f"[VAL] {name} : RMSE = {np.sqrt(np.mean((p-t)**2)):.5f}   R2 = {r2:.4f}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    def pred(a, re):
        Xq = nx(np.column_stack([a, np.log10(np.full_like(a, re))]).astype("float32"))
        with torch.no_grad():
            return dny(model(torch.tensor(Xq)).numpy())

    # --- 7. Fig 1 : courbes d'apprentissage ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(train_hist, label="train"); ax.plot(val_hist, label="validation")
    ax.set_yscale("log"); ax.set(xlabel="epoch", ylabel="loss MSE (log)", title="Courbes d'apprentissage (2D)")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(); fig.tight_layout()
    fig.savefig(FIG_DIR / "surrogate_2d_learning_curves.png", dpi=150)

    # --- 8. Fig 2 : la famille de polaires (une couleur par Re) ---
    res = sorted(df["Re"].unique())
    colors = cm.viridis(np.linspace(0, 1, len(res)))
    a_dense = np.linspace(A.min(), A.max(), 200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for Re_v, c in zip(res, colors):
        sub = df[df["Re"] == Re_v]
        p = pred(a_dense, Re_v)
        ax1.scatter(sub["alpha_deg"], sub["Cl"], s=10, color=c)
        ax1.plot(a_dense, p[:, 0], color=c, lw=1.5, label=f"Re={Re_v:.0e}")
        ax2.scatter(sub["alpha_deg"], sub["Cd"], s=10, color=c)
        ax2.plot(a_dense, p[:, 1], color=c, lw=1.5)
    ax1.set(xlabel="alpha [deg]", ylabel="Cl", title="Cl(alpha) par Reynolds")
    ax2.set(xlabel="alpha [deg]", ylabel="Cd", title="Cd(alpha) par Reynolds")
    for ax in (ax1, ax2): ax.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, title="points = data, lignes = surrogate")
    fig.suptitle("Surrogate 2D - famille de polaires NACA 0012")
    fig.tight_layout(); fig.savefig(FIG_DIR / "surrogate_2d_family.png", dpi=150)

    # --- 9. Fig 3 : INTERPOLATION a un Re jamais vu (3.5e5, hors grille) ---
    Re_test = 3.5e5
    ref = nf.get_aero_from_airfoil(asb.Airfoil("naca0012"), alpha=a_dense,
                                   Re=Re_test, model_size="xlarge")
    p = pred(a_dense, Re_test)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax1.plot(a_dense, ref["CL"], "k--", lw=2, label="reference NeuralFoil")
    ax1.plot(a_dense, p[:, 0], "tab:red", lw=1.5, label="surrogate (interpolation)")
    ax2.plot(a_dense, ref["CD"], "k--", lw=2, label="reference NeuralFoil")
    ax2.plot(a_dense, p[:, 1], "tab:red", lw=1.5, label="surrogate (interpolation)")
    ax1.set(xlabel="alpha [deg]", ylabel="Cl", title="Cl"); ax2.set(xlabel="alpha [deg]", ylabel="Cd", title="Cd")
    for ax in (ax1, ax2): ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle(f"Interpolation a Re = {Re_test:.1e} (JAMAIS vu a l'entrainement)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "surrogate_2d_interpolation.png", dpi=150)

    # erreur d'interpolation chiffree
    err_cl = np.sqrt(np.mean((p[:, 0] - ref["CL"]) ** 2))
    err_cd = np.sqrt(np.mean((p[:, 1] - ref["CD"]) ** 2))
    print(f"\n[INTERP Re=3.5e5] RMSE Cl = {err_cl:.4f}   RMSE Cd = {err_cd:.5f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Modele : {MODEL_OUT}")


if __name__ == "__main__":
    main()
