"""
Phase 2.5 - PINN : ecoulement INVISCIDE autour d'un NACA 4412 (regime aileron rapide, Re eleve)
===============================================================================================
A haut Reynolds (aileron FS, Re ~ 2e5), l'ecoulement VISIBLE est inviscide (couche limite
infiniment fine). On resout donc l'ecoulement potentiel, en formulation VITESSE (single-valued) :
   continuite        : u_x + v_y = 0
   irrotationnalite  : v_x - u_y = 0
   non-penetration   : u.n = 0 sur la surface de l'aile
   amont             : (u,v) -> (1,0) au loin
   Kutta             : (u,v) = 0 au bord de fuite  (-> circulation/portance correcte)
Le champ inviscide est INDEPENDANT du Re -> valable a 2e5 et au-dela. La couche limite / la
trainee visqueuse, elles, exigent une CFD classique (cf. front-wing-CFD).

Sortie : results/figures/pinn_flow_airfoil_inviscid.png
Lancer  : python src/pinn_flow_airfoil_inviscid.py
"""

from pathlib import Path as FsPath
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)

ALPHA_DEG, RE_LABEL = 10.0, 200000
XMIN, XMAX, YMIN, YMAX = -1.5, 3.5, -2.0, 2.0
EPOCHS = 20000

ROOT = FsPath(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures" / "pinn_flow_airfoil_inviscid.png"


def naca4412(n=180, alpha_deg=ALPHA_DEG):
    m, p, t = 0.04, 0.4, 0.12
    beta = np.linspace(0, np.pi, n)
    x = (1 - np.cos(beta)) / 2
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    yc = np.where(x < p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    px = np.concatenate([xu[::-1], xl[1:]]); py = np.concatenate([yu[::-1], yl[1:]])
    a = -np.deg2rad(alpha_deg); cx = 0.25
    rx = cx + (px - cx)*np.cos(a) - py*np.sin(a)
    ry = (px - cx)*np.sin(a) + py*np.cos(a)
    return np.column_stack([rx, ry])


def outward_normals(poly):
    c = poly.mean(0)
    nrm = np.zeros_like(poly)
    for i in range(len(poly)):
        tang = poly[(i+1) % len(poly)] - poly[i-1]
        nx, ny = tang[1], -tang[0]
        n = np.array([nx, ny]); n /= (np.linalg.norm(n) + 1e-9)
        if np.dot(n, poly[i] - c) < 0:
            n = -n
        nrm[i] = n
    return nrm


def main():
    poly = naca4412()
    airfoil = MplPath(poly)
    normals = outward_normals(poly)
    a = -np.deg2rad(ALPHA_DEG)
    te = np.array([[0.25 + 0.75*np.cos(a), 0.75*np.sin(a)]])     # bord de fuite (rotation de (1,0))

    def sample_fluid(n, box):
        x0, x1, y0, y1 = box
        pts = np.random.rand(int(n*1.4), 2) * [x1-x0, y1-y0] + [x0, y0]
        return pts[~airfoil.contains_points(pts)][:n]
    col = np.vstack([sample_fluid(2500, (XMIN, XMAX, YMIN, YMAX)),
                     sample_fluid(1500, (-0.6, 1.9, -0.9, 0.9))])
    xc = torch.tensor(col[:, :1], dtype=torch.float32, requires_grad=True)
    yc = torch.tensor(col[:, 1:], dtype=torch.float32, requires_grad=True)

    surf = torch.tensor(poly, dtype=torch.float32)
    nrm = torch.tensor(normals, dtype=torch.float32)
    # amont impose sur entree (gauche) + haut + bas SEULEMENT.
    # La sortie (droite) reste LIBRE -> le downwash de l'aile portante peut la traverser
    # (sinon clamper v=0 en sortie etoufferait la circulation et la portance).
    n_ff = 140
    ff = np.vstack([np.column_stack([np.full(n_ff, XMIN), np.linspace(YMIN, YMAX, n_ff)]),
                    np.column_stack([np.linspace(XMIN, XMAX, n_ff), np.full(n_ff, YMAX)]),
                    np.column_stack([np.linspace(XMIN, XMAX, n_ff), np.full(n_ff, YMIN)])])
    ff = torch.tensor(ff, dtype=torch.float32)
    te_t = torch.tensor(te, dtype=torch.float32)

    model = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                          nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 2))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=8000, gamma=0.5)

    def grad(f, x):
        return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]

    for epoch in range(EPOCHS):
        opt.zero_grad()
        o = model(torch.cat([xc, yc], 1)); u, v = o[:, 0:1], o[:, 1:2]
        u_x, u_y, v_x, v_y = grad(u, xc), grad(u, yc), grad(v, xc), grad(v, yc)
        loss_phys = ((u_x + v_y)**2).mean() + ((v_x - u_y)**2).mean()      # continuite + irrotationnalite
        ob = model(surf); loss_body = ((ob[:, 0:1]*nrm[:, 0:1] + ob[:, 1:2]*nrm[:, 1:2])**2).mean()
        fb = model(ff); loss_ff = ((fb[:, 0:1]-1)**2 + fb[:, 1:2]**2).mean()
        loss_kutta = (model(te_t)**2).mean()
        loss = loss_phys + 10*loss_body + 10*loss_ff + 10*loss_kutta
        loss.backward(); opt.step(); sched.step()
        if epoch % 1500 == 0:
            print(f"epoch {epoch:5d}  loss={loss.item():.3e}  "
                  f"(phys={loss_phys.item():.2e} body={loss_body.item():.2e} kutta={loss_kutta.item():.2e})")

    # --- Cl par integration de la pression (Bernoulli) : verification ---
    with torch.no_grad():
        sv = model(surf).numpy()
    cp = 1 - (sv[:, 0]**2 + sv[:, 1]**2)
    ds = np.linalg.norm(np.diff(poly, axis=0, append=poly[:1]), axis=1)
    cl = -np.sum(cp * normals[:, 1] * ds)        # composante y de la force de pression
    print(f"\n[Cl inviscide ~ {cl:.2f}]  (NACA 4412 a {ALPHA_DEG:.0f}deg : attendu ~1.5)")

    # --- Rendu : lignes de courant + champ de pression ---
    nx, ny = 360, 240
    xs = np.linspace(XMIN, XMAX, nx); ys = np.linspace(YMIN, YMAX, ny)
    XX, YY = np.meshgrid(xs, ys); grid = np.column_stack([XX.ravel(), YY.ravel()])
    with torch.no_grad():
        P = model(torch.tensor(grid, dtype=torch.float32)).numpy()
    U = P[:, 0].reshape(ny, nx); V = P[:, 1].reshape(ny, nx)
    inside = airfoil.contains_points(grid).reshape(ny, nx)
    U[inside] = np.nan; V[inside] = np.nan
    spd = np.sqrt(U**2 + V**2); Cp = 1 - (U**2 + V**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2))
    ax1.streamplot(XX, YY, U, V, color=spd, cmap="turbo", density=2.4, linewidth=0.7, arrowsize=0.6)
    ax1.fill(poly[:, 0], poly[:, 1], color="0.1", zorder=5)
    ax1.set_title("Lignes de courant (couleur = vitesse)")
    im = ax2.contourf(XX, YY, Cp, levels=40, cmap="RdBu")
    ax2.fill(poly[:, 0], poly[:, 1], color="0.1", zorder=5)
    ax2.set_title("Coefficient de pression Cp (bleu = aspiration)")
    fig.colorbar(im, ax=ax2, label="Cp")
    for ax in (ax1, ax2):
        ax.set(xlim=(XMIN+0.4, XMAX-0.6), ylim=(YMIN+0.5, YMAX-0.5), xlabel="x", ylabel="y")
        ax.set_aspect("equal")
    fig.suptitle(f"PINN inviscide — écoulement autour d'un NACA 4412 (α={ALPHA_DEG:.0f}°, régime Re≈{RE_LABEL:,}, Cl≈{cl:.2f})")
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"Figure : {FIG}")


if __name__ == "__main__":
    main()
