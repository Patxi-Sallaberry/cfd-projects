"""
Phase 1 - Entrainement du modele surrogate  alpha -> (Cl, Cd)
============================================================
Version 2 : boucle d'entrainement canonique avec suivi train / validation
par epoch + early stopping (on garde le modele au minimum de la val_loss).

Sorties :
  results/figures/learning_curves.png  (loss train vs validation)
  results/figures/surrogate_fit.png    (prediction vs donnees)
  results/surrogate_naca0012.pt        (meilleurs poids)
Lancer : python src/train_surrogate.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(0)
rng = np.random.default_rng(0)

ROOT      = Path(__file__).resolve().parent.parent
DATA      = ROOT / "data" / "naca0012_surrogate_dataset.csv"
FIG_DIR   = ROOT / "results" / "figures"
MODEL_OUT = ROOT / "results" / "surrogate_naca0012.pt"

EPOCHS   = 5000
PATIENCE = 600        # early stopping : epochs tolerees sans amelioration de la val_loss


def main():
    # --- 1. Donnees : X = alpha, Y = [Cl, Cd] ---
    df = pd.read_csv(DATA)
    X = df[["alpha_deg"]].values.astype("float32")
    Y = df[["Cl", "Cd"]].values.astype("float32")

    # --- 2. Split train (70 %) / validation (30 %) ---
    N = len(df)
    idx = rng.permutation(N)
    n_val = int(0.3 * N)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xva, Yva = X[val_idx], Y[val_idx]

    # --- 3. Normalisation (ajustee sur le TRAIN uniquement) ---
    xm, xs = Xtr.mean(0), Xtr.std(0)
    ym, ys = Ytr.mean(0), Ytr.std(0)
    nx  = lambda a: (a - xm) / xs
    ny  = lambda a: (a - ym) / ys
    dny = lambda a: a * ys + ym
    Xtr_t, Ytr_t = torch.tensor(nx(Xtr)), torch.tensor(ny(Ytr))
    Xva_t, Yva_t = torch.tensor(nx(Xva)), torch.tensor(ny(Yva))

    # --- 4. Modele + optimiseur ---
    model = nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 2),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    # Scheduler : on divise le learning rate par 2 toutes les 1000 epochs.
    # lr eleve au debut (convergence rapide) -> lr faible a la fin (stable, sans pics).
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)
    loss_fn = nn.MSELoss()

    # --- 5. Boucle : entrainement + validation + early stopping ---
    train_hist, val_hist = [], []
    best_val, best_state, since_best = float("inf"), None, 0
    for epoch in range(EPOCHS):
        # (a) entrainement
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t), Ytr_t)
        loss.backward()
        opt.step()
        sched.step()                          # decroit le learning rate au fil des epochs
        # (b) validation (sans gradient)
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xva_t), Yva_t).item()
        train_hist.append(loss.item())
        val_hist.append(vloss)
        # (c) early stopping : on memorise le MEILLEUR modele
        if vloss < best_val - 1e-7:
            best_val, since_best = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= PATIENCE:
                print(f"Early stopping a l'epoch {epoch} (val stagnante depuis {PATIENCE}).")
                break
        if epoch % 500 == 0:
            print(f"epoch {epoch:4d}  train={loss.item():.6f}  val={vloss:.6f}")

    # --- 6. On restaure le meilleur modele (pas le dernier) ---
    model.load_state_dict(best_state)
    print(f"\nMeilleure val_loss = {best_val:.6f}")

    # --- 7. Metriques finales sur la validation (unites physiques) ---
    model.eval()
    with torch.no_grad():
        Pva = dny(model(Xva_t).numpy())
    for j, name in enumerate(["Cl", "Cd"]):
        true, pred = Yva[:, j], Pva[:, j]
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        r2 = 1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2)
        print(f"[VAL] {name} : RMSE = {rmse:.5f}   R2 = {r2:.4f}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- 8. Figure 1 : courbes d'apprentissage (train vs validation) ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(train_hist, label="train")
    ax.plot(val_hist, label="validation")
    ax.set_yscale("log")
    ax.set(xlabel="epoch", ylabel="loss MSE (echelle log)", title="Courbes d'apprentissage")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "learning_curves.png", dpi=150)

    # --- 9. Figure 2 : ajustement de la polaire ---
    a_dense = np.linspace(X.min(), X.max(), 300).astype("float32").reshape(-1, 1)
    with torch.no_grad():
        p_dense = dny(model(torch.tensor(nx(a_dense))).numpy())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, j, ylab in ((ax1, 0, "Cl"), (ax2, 1, "Cd")):
        ax.scatter(Xtr, Ytr[:, j], s=16, color="tab:blue", label="train")
        ax.scatter(Xva, Yva[:, j], s=28, color="tab:red", marker="s", label="validation")
        ax.plot(a_dense, p_dense[:, j], "-", color="tab:green", lw=2, label="surrogate")
        ax.set(xlabel="alpha [deg]", ylabel=ylab, title=f"{ylab} vs alpha")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Phase 1 - surrogate NACA 0012 (MLP)  alpha -> (Cl, Cd)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "surrogate_fit.png", dpi=150)

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"\nFigures : {FIG_DIR}\nModele  : {MODEL_OUT}")


if __name__ == "__main__":
    main()
