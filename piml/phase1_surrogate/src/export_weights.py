"""
Export des modeles surrogate (1D et 2D) vers un JSON utilisable cote navigateur.
===============================================================================
On exporte les POIDS de chaque couche + les CONSTANTES DE NORMALISATION (calculees
a l'identique de l'entrainement, seed fixe), pour pouvoir refaire l'inference en
JavaScript (x@W^T+b + tanh). Un auto-test compare le forward "maison" a PyTorch.

Sortie : web/public/models.json
Lancer  : python src/export_weights.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
WEB_JSON = ROOT.parent.parent / "web" / "public" / "models.json"


def split_norm(X, Y):
    """Rejoue exactement le split train/val (seed 0, 30%) et calcule les normalisations."""
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X)); nv = int(0.3 * len(X))
    ti = idx[nv:]
    Xtr, Ytr = X[ti], Y[ti]
    return (Xtr.mean(0), Xtr.std(0), Ytr.mean(0), Ytr.std(0))


def layers_from_state(path):
    """Extrait [ (W, b, activation) ] d'un MLP nn.Sequential (couches Linear en 0,2,4)."""
    sd = torch.load(path, weights_only=True)
    out = []
    linears = [0, 2, 4]
    for k, i in enumerate(linears):
        W = sd[f"{i}.weight"].numpy()        # (out, in)
        b = sd[f"{i}.bias"].numpy()          # (out,)
        act = "tanh" if k < len(linears) - 1 else "linear"
        out.append({"W": W.tolist(), "b": b.tolist(), "act": act})
    return out


def forward_numpy(model, x_raw):
    """Forward 'maison' (== ce que fera le JS) : features -> normalise -> couches -> denormalise."""
    x = (np.array(x_raw, float) - model["x_mean"]) / model["x_std"]
    for L in model["layers"]:
        x = np.array(L["W"]) @ x + np.array(L["b"])
        if L["act"] == "tanh":
            x = np.tanh(x)
    return x * np.array(model["y_std"]) + np.array(model["y_mean"])


def build_1d():
    df = pd.read_csv(ROOT / "data" / "naca0012_surrogate_dataset.csv")
    X = df[["alpha_deg"]].values.astype("float64")
    Y = df[["Cl", "Cd"]].values.astype("float64")
    xm, xs, ym, ys = split_norm(X, Y)
    return {
        "inputs": ["alpha_deg"], "alpha_range": [-6.0, 16.0],
        "x_mean": xm.tolist(), "x_std": xs.tolist(),
        "y_mean": ym.tolist(), "y_std": ys.tolist(),
        "layers": layers_from_state(ROOT / "results" / "surrogate_naca0012.pt"),
    }


def build_2d():
    df = pd.read_csv(ROOT / "data" / "naca0012_surrogate_2d.csv")
    A = df["alpha_deg"].values.astype("float64")
    Re = df["Re"].values.astype("float64")
    X = np.column_stack([A, np.log10(Re)])
    Y = df[["Cl", "Cd"]].values.astype("float64")
    xm, xs, ym, ys = split_norm(X, Y)
    return {
        "inputs": ["alpha_deg", "log10_Re"],
        "alpha_range": [-6.0, 16.0], "re_range": [1e5, 1.5e6],
        "x_mean": xm.tolist(), "x_std": xs.tolist(),
        "y_mean": ym.tolist(), "y_std": ys.tolist(),
        "layers": layers_from_state(ROOT / "results" / "surrogate_2d_naca0012.pt"),
    }


def main():
    m1, m2 = build_1d(), build_2d()

    # --- auto-test : forward maison vs PyTorch sur quelques points ---
    import torch.nn as nn

    def torch_model(layers, n_in):
        m = nn.Sequential(nn.Linear(n_in, 64), nn.Tanh(),
                          nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))
        sd = {}
        for idx, L in zip([0, 2, 4], layers):
            sd[f"{idx}.weight"] = torch.tensor(L["W"]); sd[f"{idx}.bias"] = torch.tensor(L["b"])
        m.load_state_dict(sd); m.eval(); return m

    # 1D : alpha = 7 deg
    tm = torch_model(m1["layers"], 1)
    xn = (np.array([7.0]) - m1["x_mean"]) / m1["x_std"]
    with torch.no_grad():
        ref = tm(torch.tensor(xn, dtype=torch.float32)).numpy() * m1["y_std"] + m1["y_mean"]
    mine = forward_numpy(m1, [7.0])
    print("1D  alpha=7 :", "maison", np.round(mine, 4), "| torch", np.round(ref, 4))
    assert np.allclose(mine, ref, atol=1e-4)

    # 2D : alpha = 7, Re = 3.5e5
    tm = torch_model(m2["layers"], 2)
    feat = np.array([7.0, np.log10(3.5e5)])
    xn = (feat - m2["x_mean"]) / m2["x_std"]
    with torch.no_grad():
        ref = tm(torch.tensor(xn, dtype=torch.float32)).numpy() * m2["y_std"] + m2["y_mean"]
    mine = forward_numpy(m2, [7.0, np.log10(3.5e5)])
    print("2D  a=7,Re=3.5e5 :", "maison", np.round(mine, 4), "| torch", np.round(ref, 4))
    assert np.allclose(mine, ref, atol=1e-4)

    WEB_JSON.parent.mkdir(parents=True, exist_ok=True)
    WEB_JSON.write_text(json.dumps({"model1d": m1, "model2d": m2}))
    print(f"\nOK - auto-test passe. Ecrit : {WEB_JSON}  ({WEB_JSON.stat().st_size//1024} ko)")


if __name__ == "__main__":
    main()
