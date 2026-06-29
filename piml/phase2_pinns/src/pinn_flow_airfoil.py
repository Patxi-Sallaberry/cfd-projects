"""
Phase 2.5 - PINN : ecoulement de Navier-Stokes VISQUEUX autour d'un profil NACA 4412
====================================================================================
Extension de 2.4 (Kovasznay) a une VRAIE GEOMETRIE d'aile :
   u*u_x + v*u_y = -p_x + nu*(u_xx+u_yy)
   u*v_x + v*v_y = -p_y + nu*(v_xx+v_yy)
   u_x + v_y = 0
avec : NO-SLIP (u=v=0) sur la surface de l'aile, ecoulement amont (u=1, v=0) au loin,
pression nulle en sortie. Reynolds MODERE (laminaire) pour converger -> illustratif.

Sortie : results/figures/pinn_flow_airfoil.png
Lancer  : python src/pinn_flow_airfoil.py
"""

from pathlib import Path as FsPath
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)

ALPHA_DEG = 10.0
RE = 100.0
NU = 1.0 / RE
XMIN, XMAX, YMIN, YMAX = -1.5, 3.5, -2.0, 2.0
EPOCHS = 14000

ROOT = FsPath(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures" / "pinn_flow_airfoil.png"


def naca4412(n=160, alpha_deg=ALPHA_DEG):
    """Contour ferme du NACA 4412, tourne de l'angle d'attaque (autour du quart de corde)."""
    m, p, t = 0.04, 0.4, 0.12
    beta = np.linspace(0, np.pi, n)
    x = (1 - np.cos(beta)) / 2                       # raffine bord d'attaque/fuite
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    yc = np.where(x < p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    px = np.concatenate([xu[::-1], xl[1:]])
    py = np.concatenate([yu[::-1], yl[1:]])
    # rotation -alpha autour du quart de corde -> angle d'attaque, ecoulement horizontal
    a = -np.deg2rad(alpha_deg); cx = 0.25
    rx = cx + (px - cx)*np.cos(a) - py*np.sin(a)
    ry = (px - cx)*np.sin(a) + py*np.cos(a)
    return np.column_stack([rx, ry])


def main():
    poly = naca4412()
    airfoil = MplPath(poly)

    def sample_fluid(n, box):
        x0, x1, y0, y1 = box
        pts = np.random.rand(int(n*1.4), 2) * [x1-x0, y1-y0] + [x0, y0]
        pts = pts[~airfoil.contains_points(pts)]      # on enleve l'interieur de l'aile
        return pts[:n]

    col = np.vstack([sample_fluid(2600, (XMIN, XMAX, YMIN, YMAX)),
                     sample_fluid(1200, (-0.6, 1.9, -0.9, 0.9))])   # densifie pres de l'aile
    xc = torch.tensor(col[:, :1], dtype=torch.float32, requires_grad=True)
    yc = torch.tensor(col[:, 1:], dtype=torch.float32, requires_grad=True)

    # surface (no-slip), amont (inlet/haut/bas), sortie (p=0)
    surf = torch.tensor(poly, dtype=torch.float32)
    n_ff = 120
    ff = np.vstack([np.column_stack([np.full(n_ff, XMIN), np.linspace(YMIN, YMAX, n_ff)]),
                    np.column_stack([np.linspace(XMIN, XMAX, n_ff), np.full(n_ff, YMAX)]),
                    np.column_stack([np.linspace(XMIN, XMAX, n_ff), np.full(n_ff, YMIN)])])
    ff = torch.tensor(ff, dtype=torch.float32)
    out = torch.tensor(np.column_stack([np.full(n_ff, XMAX), np.linspace(YMIN, YMAX, n_ff)]),
                       dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 3),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=4500, gamma=0.5)

    def grad(f, x):
        return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]

    for epoch in range(EPOCHS):
        opt.zero_grad()
        o = model(torch.cat([xc, yc], 1)); u, v, p = o[:, 0:1], o[:, 1:2], o[:, 2:3]
        u_x, u_y = grad(u, xc), grad(u, yc)
        v_x, v_y = grad(v, xc), grad(v, yc)
        p_x, p_y = grad(p, xc), grad(p, yc)
        u_xx, u_yy = grad(u_x, xc), grad(u_y, yc)
        v_xx, v_yy = grad(v_x, xc), grad(v_y, yc)
        r_mx = u*u_x + v*u_y + p_x - NU*(u_xx + u_yy)
        r_my = u*v_x + v*v_y + p_y - NU*(v_xx + v_yy)
        r_c = u_x + v_y
        loss_phys = (r_mx**2).mean() + (r_my**2).mean() + (r_c**2).mean()
        sb = model(surf);  loss_wall = (sb[:, 0:2]**2).mean()                 # no-slip
        fb = model(ff);     loss_ff = ((fb[:, 0:1]-1)**2 + fb[:, 1:2]**2).mean()  # amont u=1,v=0
        loss_out = (model(out)[:, 2:3]**2).mean()                            # p=0 en sortie
        loss = loss_phys + 10*loss_wall + 10*loss_ff + loss_out
        loss.backward(); opt.step(); sched.step()
        if epoch % 2000 == 0:
            print(f"epoch {epoch:5d}  loss={loss.item():.3e}  "
                  f"(phys={loss_phys.item():.2e} wall={loss_wall.item():.2e} ff={loss_ff.item():.2e})")

    # --- Rendu : lignes de courant colorees par la vitesse ---
    nx, ny = 320, 220
    xs = np.linspace(XMIN, XMAX, nx); ys = np.linspace(YMIN, YMAX, ny)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    with torch.no_grad():
        P = model(torch.tensor(grid, dtype=torch.float32)).numpy()
    U = P[:, 0].reshape(ny, nx); V = P[:, 1].reshape(ny, nx)
    inside = airfoil.contains_points(grid).reshape(ny, nx)
    U[inside] = np.nan; V[inside] = np.nan
    spd = np.sqrt(U**2 + V**2)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.streamplot(XX, YY, U, V, color=spd, cmap="turbo", density=2.2, linewidth=0.7, arrowsize=0.6)
    ax.fill(poly[:, 0], poly[:, 1], color="0.1", zorder=5)
    ax.set(xlim=(XMIN + 0.3, XMAX - 0.3), ylim=(YMIN + 0.4, YMAX - 0.4), xlabel="x", ylabel="y",
           title=f"PINN — écoulement Navier-Stokes autour d'un NACA 4412 (α={ALPHA_DEG:.0f}°, Re={RE:.0f})")
    ax.set_aspect("equal")
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"Figure : {FIG}")


if __name__ == "__main__":
    main()
