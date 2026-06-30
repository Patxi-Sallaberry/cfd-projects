"""
Phase 2.6 - PINN PARAMETRIQUE : ecoulement inviscide autour d'un NACA 4412 pour TOUTE une
plage d'angles d'attaque, avec UN SEUL reseau  (x, y, alpha) -> (u, v)
=========================================================================================
Au lieu d'entrainer un PINN par angle, on ajoute alpha comme 3e entree du reseau et on
l'entraine sur une plage [ALPHA_MIN, ALPHA_MAX] d'un coup. Ensuite, evaluer le champ a
n'importe quel angle est gratuit (un simple forward pass) -> ideal pour un slider interactif.

ASTUCE DE REPERE (le point cle) :
   Plutot que de faire tourner l'aile quand alpha change (ce qui deplacerait la geometrie,
   les normales, les points de collocation...), on garde l'AILE FIXE (a alpha=0) et on fait
   tourner l'ECOULEMENT AMONT. C'est physiquement equivalent pour l'inviscide. Du coup la
   geometrie est calculee UNE SEULE FOIS ; seules les conditions aux limites portent alpha.

Equations (formulation vitesse, ecoulement potentiel) :
   continuite        : u_x + v_y = 0
   irrotationnalite  : v_x - u_y = 0
   non-penetration   : u.n = 0           sur la surface de l'aile (fixe)
   amont             : (u,v) -> (cos a, sin a)   au loin   <- alpha entre ICI
   Kutta             : (u,v) = 0          au bord de fuite (1, 0)

Sortie : results/figures/pinn_flow_airfoil_parametric.png
Lancer  : python src/pinn_flow_airfoil_parametric.py
"""

from pathlib import Path as FsPath
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)

# Plage d'angles couverte par UN seul reseau (en degres)
ALPHA_MIN, ALPHA_MAX = -5.0, 15.0
RE_LABEL = 200000
XMIN, XMAX, YMIN, YMAX = -1.5, 3.5, -2.0, 2.0
EPOCHS = 30000

# Angle de portance nulle du NACA 4412 (cambrure 4 %) -> sert de reference theorique
ALPHA_L0_DEG = -4.0

ROOT = FsPath(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures" / "pinn_flow_airfoil_parametric.png"


def naca4412(n=180):
    """Geometrie du NACA 4412 a alpha=0 (l'aile reste FIXE ; c'est l'amont qui tournera)."""
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
    """Normales unitaires sortantes en chaque point de la surface."""
    c = poly.mean(0)
    nrm = np.zeros_like(poly)
    for i in range(len(poly)):
        tang = poly[(i+1) % len(poly)] - poly[i-1]
        n = np.array([tang[1], -tang[0]]); n /= (np.linalg.norm(n) + 1e-9)
        if np.dot(n, poly[i] - c) < 0:
            n = -n
        nrm[i] = n
    return nrm


# --- normalisation de alpha (degres) -> [-1, 1] pour que le tanh le "voie" bien ---
A_MID = 0.5 * (ALPHA_MIN + ALPHA_MAX)
A_HALF = 0.5 * (ALPHA_MAX - ALPHA_MIN)
def norm_alpha(a_deg):
    return (a_deg - A_MID) / A_HALF


def main():
    poly = naca4412()                       # aile fixe
    airfoil = MplPath(poly)
    normals = outward_normals(poly)
    te = np.array([[1.0, 0.0]])             # bord de fuite FIXE (le profil n'est pas tourne)

    # ----- points de collocation (fluide), calcules UNE seule fois -----
    def sample_fluid(n, box):
        x0, x1, y0, y1 = box
        pts = np.random.rand(int(n*1.4), 2) * [x1-x0, y1-y0] + [x0, y0]
        return pts[~airfoil.contains_points(pts)][:n]
    col = np.vstack([sample_fluid(3000, (XMIN, XMAX, YMIN, YMAX)),
                     sample_fluid(2000, (-0.6, 1.9, -0.9, 0.9))])
    N = len(col)
    xc = torch.tensor(col[:, :1], dtype=torch.float32, requires_grad=True)
    yc = torch.tensor(col[:, 1:], dtype=torch.float32, requires_grad=True)

    surf = torch.tensor(poly, dtype=torch.float32)
    nrm = torch.tensor(normals, dtype=torch.float32)
    n_surf = len(poly)

    # frontiere lointaine : entree (gauche) + haut + bas. La SORTIE (droite) reste libre
    # pour laisser passer le downwash de l'aile portante (sinon on etouffe la circulation).
    n_ff = 160
    ff = np.vstack([np.column_stack([np.full(n_ff, XMIN), np.linspace(YMIN, YMAX, n_ff)]),
                    np.column_stack([np.linspace(XMIN, XMAX, n_ff), np.full(n_ff, YMAX)]),
                    np.column_stack([np.linspace(XMIN, XMAX, n_ff), np.full(n_ff, YMIN)])])
    ff = torch.tensor(ff, dtype=torch.float32)
    n_ff_tot = len(ff)
    te_t = torch.tensor(te, dtype=torch.float32)

    # reseau : 3 entrees (x, y, alpha_norm) -> 2 sorties (u, v). Un peu plus large car le
    # probleme est plus dur (toute une famille d'ecoulements a representer).
    model = nn.Sequential(nn.Linear(3, 96), nn.Tanh(), nn.Linear(96, 96), nn.Tanh(),
                          nn.Linear(96, 96), nn.Tanh(), nn.Linear(96, 96), nn.Tanh(),
                          nn.Linear(96, 2))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10000, gamma=0.5)

    def grad(f, x):
        return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]

    def rand_alpha(n):
        """tire n angles (deg) uniformement dans la plage, + la version normalisee pour le reseau."""
        a_deg = np.random.uniform(ALPHA_MIN, ALPHA_MAX, size=(n, 1)).astype(np.float32)
        return (torch.tensor(a_deg),
                torch.tensor(norm_alpha(a_deg)))

    for epoch in range(EPOCHS):
        opt.zero_grad()

        # chaque point de collocation recoit un alpha tire au hasard dans la plage
        _, ac = rand_alpha(N)
        o = model(torch.cat([xc, yc, ac], 1)); u, v = o[:, 0:1], o[:, 1:2]
        u_x, u_y, v_x, v_y = grad(u, xc), grad(u, yc), grad(v, xc), grad(v, yc)
        loss_phys = ((u_x + v_y)**2).mean() + ((v_x - u_y)**2).mean()

        # non-penetration sur l'aile (alpha aleatoire par point)
        _, as_ = rand_alpha(n_surf)
        ob = model(torch.cat([surf, as_], 1))
        loss_body = ((ob[:, 0:1]*nrm[:, 0:1] + ob[:, 1:2]*nrm[:, 1:2])**2).mean()

        # amont : (u,v) -> (cos a, sin a)  <- c'est ICI qu'alpha entre dans la physique
        ad_ff, an_ff = rand_alpha(n_ff_tot)
        ar = torch.deg2rad(ad_ff)
        fb = model(torch.cat([ff, an_ff], 1))
        loss_ff = ((fb[:, 0:1] - torch.cos(ar))**2 + (fb[:, 1:2] - torch.sin(ar))**2).mean()

        # Kutta : vitesse nulle au bord de fuite, pour tout alpha
        ad_te, an_te = rand_alpha(64)
        kb = model(torch.cat([te_t.repeat(64, 1), an_te], 1))
        loss_kutta = (kb**2).mean()

        loss = loss_phys + 10*loss_body + 10*loss_ff + 10*loss_kutta
        loss.backward(); opt.step(); sched.step()
        if epoch % 2000 == 0:
            print(f"epoch {epoch:5d}  loss={loss.item():.3e}  "
                  f"(phys={loss_phys.item():.2e} body={loss_body.item():.2e} "
                  f"ff={loss_ff.item():.2e} kutta={loss_kutta.item():.2e})")

    # ============================ VALIDATION : Cl(alpha) ============================
    def cl_at(alpha_deg):
        """Cl par integration de la pression (Bernoulli), force projetee perpendiculairement
        a l'amont (cos a, sin a). Repere aile-fixe -> direction de portance = (-sin a, cos a)."""
        an = torch.full((n_surf, 1), float(norm_alpha(alpha_deg)))
        with torch.no_grad():
            sv = model(torch.cat([surf, an], 1)).numpy()
        cp = 1 - (sv[:, 0]**2 + sv[:, 1]**2)                  # |amont| = 1 -> Cp = 1 - V^2
        ds = np.linalg.norm(np.diff(poly, axis=0, append=poly[:1]), axis=1)
        fx = -np.sum(cp * normals[:, 0] * ds)
        fy = -np.sum(cp * normals[:, 1] * ds)
        a = np.deg2rad(alpha_deg)
        return -fx*np.sin(a) + fy*np.cos(a)                   # projection sur la portance

    alphas = np.linspace(ALPHA_MIN, ALPHA_MAX, 21)
    cl_pinn = np.array([cl_at(a) for a in alphas])
    cl_theory = 2*np.pi * np.deg2rad(alphas - ALPHA_L0_DEG)   # profil mince : 2*pi*(a - a_L0)
    print("\n alpha(deg)  Cl_PINN  Cl_theorie")
    for a, cp_, ct in zip(alphas, cl_pinn, cl_theory):
        print(f"  {a:6.1f}    {cp_:6.2f}    {ct:6.2f}")

    # ============================ FIGURE ============================
    nx, ny = 320, 220
    xs = np.linspace(XMIN, XMAX, nx); ys = np.linspace(YMIN, YMAX, ny)
    XX, YY = np.meshgrid(xs, ys); grid = np.column_stack([XX.ravel(), YY.ravel()])
    inside = airfoil.contains_points(grid).reshape(ny, nx)

    def field_at(alpha_deg):
        an = np.full((grid.shape[0], 1), norm_alpha(alpha_deg), dtype=np.float32)
        with torch.no_grad():
            P = model(torch.tensor(np.column_stack([grid, an]), dtype=torch.float32)).numpy()
        U = P[:, 0].reshape(ny, nx); V = P[:, 1].reshape(ny, nx)
        U[inside] = np.nan; V[inside] = np.nan
        return U, V

    show = [0.0, 5.0, 10.0, 15.0]
    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 3)
    # 4 panneaux d'ecoulement : haut-gauche, haut-milieu, bas-gauche, bas-milieu
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

    # panneau de validation Cl(alpha) sur toute la colonne de droite
    axc = fig.add_subplot(gs[:, 2])
    axc.plot(alphas, cl_theory, "k--", label="Profil mince : 2π(α−α₀)")
    axc.plot(alphas, cl_pinn, "o-", color="crimson", label="PINN paramétrique")
    axc.axhline(0, color="0.7", lw=0.8); axc.axvline(ALPHA_L0_DEG, color="0.7", lw=0.8, ls=":")
    axc.set_xlabel("angle d'attaque α (°)"); axc.set_ylabel("Coefficient de portance Cl")
    axc.set_title("Validation : Cl(α) — un seul réseau pour tous les angles")
    axc.legend(); axc.grid(alpha=0.3)

    fig.suptitle(f"PINN paramétrique (x, y, α)→(u, v) — NACA 4412 inviscide, "
                 f"α∈[{ALPHA_MIN:.0f}°,{ALPHA_MAX:.0f}°], Re≈{RE_LABEL:,}", fontsize=13)
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"\nFigure : {FIG}")


if __name__ == "__main__":
    main()
