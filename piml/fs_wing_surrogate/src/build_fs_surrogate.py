"""
FS Aero Explorer - construction du surrogate aileron (NACA 4412)
================================================================
Genere un dataset (angle x Reynolds) pour le NACA 4412, entraine un surrogate
2D (angle, Re) -> (Cl, Cd), le valide, et EXPORTE les poids pour le navigateur
(web/public/fs_wing_model.json) avec les constantes physiques d'un aileron FS.

Monte inverse, ce profil cambre genere de la DOWNFORCE (downforce ~ Cl * q * S).

Lancer : python src/build_fs_surrogate.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import aerosandbox as asb
import neuralfoil as nf

torch.manual_seed(0); rng = np.random.default_rng(0)

# --- Constantes physiques d'un element d'aileron FS (representatif) ---
CHORD, SPAN = 0.30, 1.20          # m  (corde, envergure) -> aire S = 0.36 m2
RHO, MU = 1.225, 1.789e-5          # air : densite (kg/m3), viscosite (Pa.s)
V_MIN, V_MAX = 5.0, 35.0           # m/s (vitesse voiture : ~18 a 126 km/h)
ALPHAS = np.arange(-4.0, 16.01, 0.5)
# Reynolds correspondants a la plage de vitesse : Re = rho*V*c/mu
RES = np.round(np.linspace(RHO * V_MIN * CHORD / MU, RHO * V_MAX * CHORD / MU, 7), -3)

ROOT = Path(__file__).resolve().parent.parent
WEB_JSON = ROOT.parent.parent / "web" / "public" / "fs_wing_model.json"
FIG = ROOT / "results" / "fs_wing_validation.png"


def make_dataset():
    af = asb.Airfoil("naca4412")
    rows = []
    for Re in RES:
        aero = nf.get_aero_from_airfoil(af, alpha=ALPHAS, Re=float(Re), model_size="xlarge")
        for a, cl, cd in zip(ALPHAS, aero["CL"], aero["CD"]):
            rows.append((a, float(Re), float(cl), float(cd)))
    return pd.DataFrame(rows, columns=["alpha", "Re", "Cl", "Cd"])


def main():
    df = make_dataset()
    X = np.column_stack([df["alpha"].values, np.log10(df["Re"].values)]).astype("float32")
    Y = df[["Cl", "Cd"]].values.astype("float32")

    # split train/val + normalisation (memes recettes que Phase 1)
    N = len(df); idx = rng.permutation(N); nv = int(0.3 * N)
    vi, ti = idx[:nv], idx[nv:]
    Xtr, Ytr, Xva, Yva = X[ti], Y[ti], X[vi], Y[vi]
    xm, xs = Xtr.mean(0), Xtr.std(0); ym, ys = Ytr.mean(0), Ytr.std(0)
    nx = lambda a: (a - xm) / xs; ny = lambda a: (a - ym) / ys; dny = lambda a: a * ys + ym
    Xtr_t, Ytr_t = torch.tensor(nx(Xtr)), torch.tensor(ny(Ytr))
    Xva_t, Yva_t = torch.tensor(nx(Xva)), torch.tensor(ny(Yva))

    model = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1500, gamma=0.5)
    loss_fn = nn.MSELoss()
    best, best_state, since = 1e9, None, 0
    for epoch in range(6000):
        model.train(); opt.zero_grad()
        loss_fn(model(Xtr_t), Ytr_t).backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva_t), Yva_t).item()
        if vl < best - 1e-8:
            best, since = vl, 0; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= 800: break
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        Pva = dny(model(Xva_t).numpy())
    for j, name in enumerate(["Cl", "Cd"]):
        t, p = Yva[:, j], Pva[:, j]
        r2 = 1 - np.sum((p - t) ** 2) / np.sum((t - t.mean()) ** 2)
        print(f"[VAL] {name} : R2 = {r2:.4f}")

    # --- export JSON pour le navigateur (memes champs que web/src/lib/model.ts) ---
    sd = model.state_dict()
    layers = []
    for k, i in enumerate([0, 2, 4]):
        layers.append({"W": sd[f"{i}.weight"].numpy().tolist(),
                       "b": sd[f"{i}.bias"].numpy().tolist(),
                       "act": "tanh" if k < 2 else "linear"})
    bundle = {
        "inputs": ["alpha_deg", "log10_Re"],
        "alpha_range": [float(ALPHAS[0]), float(ALPHAS[-1])],
        "re_range": [float(RES[0]), float(RES[-1])],
        "v_range": [V_MIN, V_MAX],
        "physics": {"chord": CHORD, "span": SPAN, "rho": RHO, "mu": MU},
        "x_mean": xm.tolist(), "x_std": xs.tolist(),
        "y_mean": ym.tolist(), "y_std": ys.tolist(),
        "layers": layers,
    }
    WEB_JSON.parent.mkdir(parents=True, exist_ok=True)
    WEB_JSON.write_text(json.dumps(bundle))
    print(f"Export : {WEB_JSON} ({WEB_JSON.stat().st_size // 1024} ko)")

    # --- figure de validation : Cl & Cd vs angle a 2 vitesses ---
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for V, c in [(15, "tab:blue"), (30, "tab:red")]:
        Re = RHO * V * CHORD / MU
        sub = df[np.isclose(df["Re"], RES[np.argmin(np.abs(RES - Re))])]
        a_dense = np.linspace(-4, 16, 200)
        Xq = nx(np.column_stack([a_dense, np.log10(np.full_like(a_dense, RES[np.argmin(np.abs(RES - Re))]))]).astype("float32"))
        with torch.no_grad():
            P = dny(model(torch.tensor(Xq)).numpy())
        ax1.scatter(sub["alpha"], sub["Cl"], s=10, color=c)
        ax1.plot(a_dense, P[:, 0], color=c, label=f"V≈{V} m/s")
        ax2.scatter(sub["alpha"], sub["Cd"], s=10, color=c)
        ax2.plot(a_dense, P[:, 1], color=c)
    ax1.set(xlabel="angle [deg]", ylabel="Cl (→ downforce)", title="Portance NACA 4412")
    ax2.set(xlabel="angle [deg]", ylabel="Cd (→ traînée)", title="Traînée NACA 4412")
    for ax in (ax1, ax2): ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("FS wing surrogate (NACA 4412) — points = NeuralFoil, lignes = surrogate")
    fig.tight_layout(); fig.savefig(FIG, dpi=150)
    print(f"Figure : {FIG}")


if __name__ == "__main__":
    main()
