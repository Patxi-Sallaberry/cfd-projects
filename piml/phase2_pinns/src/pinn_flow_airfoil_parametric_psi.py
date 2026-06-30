"""
Phase 2.6 - PINN PARAMETRIQUE (formulation FONCTION DE COURANT psi)
===================================================================
Un seul reseau  (x, y, alpha) -> psi  pour TOUTE la plage d'angles. La vitesse en derive :
       u = d(psi)/dy ,   v = -d(psi)/dx
=> la continuite (u_x + v_y = 0) est satisfaite EXACTEMENT, par construction.

Pourquoi psi plutot que (u,v) directement :
   La 1re tentative (formulation vitesse, cf. pinn_flow_airfoil_parametric.py) convergeait mais
   produisait ~4x trop peu de portance : la circulation (grandeur GLOBALE qui cree la portance)
   restait sous-developpee, le reseau tombant sur l'optimum facile a circulation ~nulle.
   En formulation psi, imposer psi=0 sur TOUTE la paroi epingle fermement la ligne de courant du
   corps -> la condition de Kutta selectionne enfin la bonne circulation.

ASTUCE DE REPERE (inchangee) : aile FIXE (alpha=0), c'est l'ECOULEMENT AMONT qui tourne.

Pertes :
   Laplace      : psi_xx + psi_yy = 0        (irrotationnel + incompressible -> potentiel)
   paroi        : psi = 0                     sur la surface (l'aile est une ligne de courant)
   amont        : (psi_y, -psi_x) -> (cos a, sin a)   au loin   <- alpha entre ICI
                  (!) on impose la VITESSE, pas la valeur de psi (sinon on etouffe la circulation)
   Kutta        : (psi_y, -psi_x) = 0         au bord de fuite (1, 0)  -> fixe la circulation

Sortie : results/figures/pinn_flow_airfoil_parametric_psi.png
Lancer  : python src/pinn_flow_airfoil_parametric_psi.py
"""

from pathlib import Path as FsPath
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)

ALPHA_MIN, ALPHA_MAX = -5.0, 15.0
RE_LABEL = 200000
XMIN, XMAX, YMIN, YMAX = -1.5, 3.5, -2.0, 2.0
EPOCHS = 30000
ALPHA_L0_DEG = -4.0                       # angle de portance nulle du NACA 4412 (reference theorique)

ROOT = FsPath(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures" / "pinn_flow_airfoil_parametric_psi.png"


def naca4412(n=180):
    """NACA 4412 a alpha=0 (aile FIXE ; c'est l'amont qui tourne)."""
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
    return np.column_stack([px, py])


def outward_normals(poly):
    c = poly.mean(0)
    nrm = np.zeros_like(poly)
    for i in range(len(poly)):
        tang = poly[(i+1) % len(poly)] - poly[i-1]
        n = np.array([tang[1], -tang[0]]); n /= (np.linalg.norm(n) + 1e-9)
        if np.dot(n, poly[i] - c) < 0:
            n = -n
        nrm[i] = n
    return nrm


A_MID = 0.5 * (ALPHA_MIN + ALPHA_MAX)
A_HALF = 0.5 * (ALPHA_MAX - ALPHA_MIN)
def norm_alpha(a_deg):
    return (a_deg - A_MID) / A_HALF


def main():
    poly = naca4412()
    airfoil = MplPath(poly)
    normals = outward_normals(poly)
    te = np.array([[1.0, 0.0]])              # bord de fuite FIXE

    def sample_fluid(n, box):
        x0, x1, y0, y1 = box
        pts = np.random.rand(int(n*1.4), 2) * [x1-x0, y1-y0] + [x0, y0]
        return pts[~airfoil.contains_points(pts)][:n]
    col = np.vstack([sample_fluid(3000, (XMIN, XMAX, YMIN, YMAX)),
                     sample_fluid(2000, (-0.6, 1.9, -0.9, 0.9))])
    N = len(col)
    xc = torch.tensor(col[:, :1], dtype=torch.float32, requires_grad=True)
    yc = torch.tensor(col[:, 1:], dtype=torch.float32, requires_grad=True)

    # paroi : valeur de psi imposee (=0) -> pas besoin de gradient
    surf = torch.tensor(poly, dtype=torch.float32)
    n_surf = len(poly)

    # far-field (entree+haut+bas, sortie libre) : il faut la VITESSE -> requires_grad
    n_ff = 160
    ff_np = np.vstack([np.column_stack([np.full(n_ff, XMIN), np.linspace(YMIN, YMAX, n_ff)]),
                       np.column_stack([np.linspace(XMIN, XMAX, n_ff), np.full(n_ff, YMAX)]),
                       np.column_stack([np.linspace(XMIN, XMAX, n_ff), np.full(n_ff, YMIN)])])
    xf = torch.tensor(ff_np[:, :1], dtype=torch.float32, requires_grad=True)
    yf = torch.tensor(ff_np[:, 1:], dtype=torch.float32, requires_grad=True)
    n_ff_tot = len(ff_np)

    # bord de fuite : vitesse nulle -> requires_grad
    n_te = 64
    xt = torch.tensor(np.full((n_te, 1), te[0, 0]), dtype=torch.float32, requires_grad=True)
    yt = torch.tensor(np.full((n_te, 1), te[0, 1]), dtype=torch.float32, requires_grad=True)

    model = nn.Sequential(nn.Linear(3, 96), nn.Tanh(), nn.Linear(96, 96), nn.Tanh(),
                          nn.Linear(96, 96), nn.Tanh(), nn.Linear(96, 96), nn.Tanh(),
                          nn.Linear(96, 1))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10000, gamma=0.5)

    def grad(f, x):
        return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]

    def velocity(x, y, a_norm):
        """psi -> (u, v) = (psi_y, -psi_x). Renvoie aussi psi (pour la paroi)."""
        psi = model(torch.cat([x, y, a_norm], 1))
        u = grad(psi, y)
        v = -grad(psi, x)
        return u, v, psi

    def rand_alpha_norm(n):
        a_deg = np.random.uniform(ALPHA_MIN, ALPHA_MAX, size=(n, 1)).astype(np.float32)
        return torch.tensor(a_deg), torch.tensor(norm_alpha(a_deg))

    for epoch in range(EPOCHS):
        opt.zero_grad()

        # --- Laplace dans le fluide ---
        _, ac = rand_alpha_norm(N)
        u, v, _ = velocity(xc, yc, ac)
        psi_xx = grad(v.mul(-1), xc)          # v = -psi_x  -> -v = psi_x ; d/dx -> psi_xx
        psi_yy = grad(u, yc)                  # u =  psi_y         ; d/dy -> psi_yy
        loss_lap = ((psi_xx + psi_yy)**2).mean()

        # --- paroi : psi = 0 (ligne de courant du corps) ---
        _, as_ = rand_alpha_norm(n_surf)
        psi_b = model(torch.cat([surf, as_], 1))
        loss_body = (psi_b**2).mean()

        # --- amont : vitesse -> (cos a, sin a) ---
        ad_ff, an_ff = rand_alpha_norm(n_ff_tot)
        ar = torch.deg2rad(ad_ff)
        uf, vf, _ = velocity(xf, yf, an_ff)
        loss_ff = ((uf - torch.cos(ar))**2 + (vf - torch.sin(ar))**2).mean()

        # --- Kutta : vitesse nulle au bord de fuite ---
        _, an_te = rand_alpha_norm(n_te)
        ut, vt, _ = velocity(xt, yt, an_te)
        loss_kutta = (ut**2 + vt**2).mean()

        loss = loss_lap + 20*loss_body + 10*loss_ff + 10*loss_kutta
        loss.backward(); opt.step(); sched.step()
        if epoch % 2000 == 0:
            print(f"epoch {epoch:5d}  loss={loss.item():.3e}  "
                  f"(lap={loss_lap.item():.2e} body={loss_body.item():.2e} "
                  f"ff={loss_ff.item():.2e} kutta={loss_kutta.item():.2e})")

    # ============================ VALIDATION : Cl(alpha) ============================
    xs_surf = torch.tensor(poly[:, :1], dtype=torch.float32, requires_grad=True)
    ys_surf = torch.tensor(poly[:, 1:], dtype=torch.float32, requires_grad=True)
    ds = np.linalg.norm(np.diff(poly, axis=0, append=poly[:1]), axis=1)

    def cl_at(alpha_deg):
        an = torch.full((n_surf, 1), float(norm_alpha(alpha_deg)))
        u, v, _ = velocity(xs_surf, ys_surf, an)
        u = u.detach().numpy().ravel(); v = v.detach().numpy().ravel()
        cp = 1 - (u**2 + v**2)
        fx = -np.sum(cp * normals[:, 0] * ds)
        fy = -np.sum(cp * normals[:, 1] * ds)
        a = np.deg2rad(alpha_deg)
        return -fx*np.sin(a) + fy*np.cos(a)         # projection perpendiculaire a l'amont

    alphas = np.linspace(ALPHA_MIN, ALPHA_MAX, 21)
    cl_pinn = np.array([cl_at(a) for a in alphas])
    cl_theory = 2*np.pi * np.deg2rad(alphas - ALPHA_L0_DEG)
    print("\n alpha(deg)  Cl_PINN  Cl_theorie")
    for a, cp_, ct in zip(alphas, cl_pinn, cl_theory):
        print(f"  {a:6.1f}    {cp_:6.2f}    {ct:6.2f}")

    # ============================ FIGURE ============================
    nx, ny = 320, 220
    xs = np.linspace(XMIN, XMAX, nx); ys = np.linspace(YMIN, YMAX, ny)
    XX, YY = np.meshgrid(xs, ys); grid = np.column_stack([XX.ravel(), YY.ravel()])
    inside = airfoil.contains_points(grid).reshape(ny, nx)
    gx = torch.tensor(grid[:, :1], dtype=torch.float32, requires_grad=True)
    gy = torch.tensor(grid[:, 1:], dtype=torch.float32, requires_grad=True)

    def field_at(alpha_deg):
        an = torch.full((grid.shape[0], 1), float(norm_alpha(alpha_deg)))
        u, v, _ = velocity(gx, gy, an)
        U = u.detach().numpy().reshape(ny, nx); V = v.detach().numpy().reshape(ny, nx)
        U[inside] = np.nan; V[inside] = np.nan
        return U, V

    show = [0.0, 5.0, 10.0, 15.0]
    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 3)
    positions = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]
    for adeg, pos in zip(show, positions):
        ax = fig.add_subplot(pos)
        U, V = field_at(adeg)
        spd = np.sqrt(U**2 + V**2)
        ax.streamplot(XX, YY, U, V, color=spd, cmap="turbo", density=1.8, linewidth=0.6, arrowsize=0.5)
        ax.fill(poly[:, 0], poly[:, 1], color="0.1", zorder=5)
        ax.set_title(f"α = {adeg:.0f}°  (Cl≈{cl_at(adeg):.2f})", fontsize=11)
        ax.set(xlim=(XMIN+0.4, XMAX-0.6), ylim=(YMIN+0.6, YMAX-0.6))
        ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")

    axc = fig.add_subplot(gs[:, 2])
    axc.plot(alphas, cl_theory, "k--", label="Profil mince : 2π(α−α₀)")
    axc.plot(alphas, cl_pinn, "o-", color="crimson", label="PINN paramétrique (ψ)")
    axc.axhline(0, color="0.7", lw=0.8); axc.axvline(ALPHA_L0_DEG, color="0.7", lw=0.8, ls=":")
    axc.set_xlabel("angle d'attaque α (°)"); axc.set_ylabel("Coefficient de portance Cl")
    axc.set_title("Validation : Cl(α) — un seul réseau pour tous les angles")
    axc.legend(); axc.grid(alpha=0.3)

    fig.suptitle(f"PINN paramétrique ψ : (x, y, α)→ψ → (u, v) — NACA 4412 inviscide, "
                 f"α∈[{ALPHA_MIN:.0f}°,{ALPHA_MAX:.0f}°], Re≈{RE_LABEL:,}", fontsize=13)
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"\nFigure : {FIG}")


if __name__ == "__main__":
    main()
