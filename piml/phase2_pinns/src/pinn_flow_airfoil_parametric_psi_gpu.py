"""
Phase 2.7 - PINN PARAMETRIQUE psi, version GPU-READY (device-agnostique)
========================================================================
Meme modele physique que pinn_flow_airfoil_parametric_psi.py (formulation fonction de courant
psi, un seul reseau (x, y, alpha) -> psi pour toute la plage d'angles), mais REECRIT pour tirer
parti d'un GPU CUDA quand il y en a un -- sans changer une ligne pour l'utilisateur :

    DEVICE = cuda si disponible, sinon cpu

Le meme fichier tourne donc sur CPU (machine actuelle) ET, tel quel, sur une vraie carte NVIDIA
(machine de labo, Google Colab GPU gratuit, ...) ou il est nettement plus rapide.

Optimisations GPU (la ou ca compte vraiment) :
  1. TOUS les tenseurs (collocation, paroi, far-field, bord de fuite) vivent sur le DEVICE :
     zero transfert CPU<->GPU dans la boucle d'entrainement.
  2. L'echantillonnage des angles se fait DIRECTEMENT sur le device avec torch.rand (pas de
     numpy -> tensor -> .to(device) a chaque epoque, qui forcerait une synchronisation hote/GPU).
     Astuce : alpha normalise dans [-1,1] = simplement torch.rand()*2 - 1 (uniforme).
  3. TF32 active (matmuls plus rapides sur GPU Ampere+), et on ne lit la loss (.item(), qui
     synchronise) que toutes les 2000 epoques.

NB. CUDA exige un GPU NVIDIA + un build torch CUDA (pip install torch --index-url
.../cu121). Un GPU Intel/AMD (DirectX12) n'expose PAS CUDA et ne supporte pas de facon fiable
l'autograd de 2nd ordre dont le PINN a besoin -> rester sur CPU, ou utiliser une carte NVIDIA.

Sortie : results/figures/pinn_flow_airfoil_parametric_psi_gpu.png
Lancer  : python src/pinn_flow_airfoil_parametric_psi_gpu.py
"""

from pathlib import Path as FsPath
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)

# ---------- selection du device : c'est tout ce qui change pour passer sur GPU ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True       # matmuls TF32 (Ampere+) -> plus rapide
    torch.backends.cudnn.allow_tf32 = True

ALPHA_MIN, ALPHA_MAX = -5.0, 15.0
RE_LABEL = 200000
XMIN, XMAX, YMIN, YMAX = -1.5, 3.5, -2.0, 2.0
EPOCHS = 30000
ALPHA_L0_DEG = -4.0
A_MID = 0.5 * (ALPHA_MIN + ALPHA_MAX)
A_HALF = 0.5 * (ALPHA_MAX - ALPHA_MIN)
DEG2RAD = np.pi / 180.0

ROOT = FsPath(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures" / "pinn_flow_airfoil_parametric_psi_gpu.png"


def naca4412(n=180):
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


def norm_alpha(a_deg):
    return (a_deg - A_MID) / A_HALF


def main():
    print(f"[device] entrainement sur : {DEVICE}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if DEVICE.type == 'cuda' else ''}")
    poly = naca4412()
    airfoil = MplPath(poly)
    normals = outward_normals(poly)
    te = np.array([[1.0, 0.0]])

    def sample_fluid(n, box):
        x0, x1, y0, y1 = box
        pts = np.random.rand(int(n*1.4), 2) * [x1-x0, y1-y0] + [x0, y0]
        return pts[~airfoil.contains_points(pts)][:n]
    col = np.vstack([sample_fluid(3000, (XMIN, XMAX, YMIN, YMAX)),
                     sample_fluid(2000, (-0.6, 1.9, -0.9, 0.9))])
    N = len(col)

    def dev_grad(arr):                                 # tenseur feuille sur le DEVICE, requires_grad
        return torch.tensor(arr, dtype=torch.float32, device=DEVICE, requires_grad=True)

    # ---- TOUS les tenseurs persistants sont crees une fois, directement sur le DEVICE ----
    xc, yc = dev_grad(col[:, :1]), dev_grad(col[:, 1:])
    surf = torch.tensor(poly, dtype=torch.float32, device=DEVICE)       # paroi (valeur -> no grad)
    n_surf = len(poly)

    n_ff = 160
    ff_np = np.vstack([np.column_stack([np.full(n_ff, XMIN), np.linspace(YMIN, YMAX, n_ff)]),
                       np.column_stack([np.linspace(XMIN, XMAX, n_ff), np.full(n_ff, YMAX)]),
                       np.column_stack([np.linspace(XMIN, XMAX, n_ff), np.full(n_ff, YMIN)])])
    xf, yf = dev_grad(ff_np[:, :1]), dev_grad(ff_np[:, 1:])
    n_ff_tot = len(ff_np)

    n_te = 64
    xt = dev_grad(np.full((n_te, 1), te[0, 0]))
    yt = dev_grad(np.full((n_te, 1), te[0, 1]))

    model = nn.Sequential(nn.Linear(3, 96), nn.Tanh(), nn.Linear(96, 96), nn.Tanh(),
                          nn.Linear(96, 96), nn.Tanh(), nn.Linear(96, 96), nn.Tanh(),
                          nn.Linear(96, 1)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10000, gamma=0.5)

    def grad(f, x):
        return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]

    def velocity(x, y, a_norm):
        psi = model(torch.cat([x, y, a_norm], 1))
        return grad(psi, y), -grad(psi, x), psi        # u = psi_y, v = -psi_x

    def sample_alpha(n):
        """alpha echantillonne DIRECTEMENT sur le device. a_norm uniforme dans [-1,1]."""
        a_norm = torch.rand(n, 1, device=DEVICE) * 2.0 - 1.0
        a_rad = (a_norm * A_HALF + A_MID) * DEG2RAD
        return a_rad, a_norm

    for epoch in range(EPOCHS):
        opt.zero_grad()

        _, ac = sample_alpha(N)                                          # Laplace
        u, v, _ = velocity(xc, yc, ac)
        loss_lap = ((grad(v.mul(-1), xc) + grad(u, yc))**2).mean()

        _, as_ = sample_alpha(n_surf)                                    # paroi : psi = 0
        loss_body = (model(torch.cat([surf, as_], 1))**2).mean()

        ar, an = sample_alpha(n_ff_tot)                                  # amont : (cos a, sin a)
        uf, vf, _ = velocity(xf, yf, an)
        loss_ff = ((uf - torch.cos(ar))**2 + (vf - torch.sin(ar))**2).mean()

        _, an_te = sample_alpha(n_te)                                    # Kutta
        ut, vt, _ = velocity(xt, yt, an_te)
        loss_kutta = (ut**2 + vt**2).mean()

        loss = loss_lap + 20*loss_body + 10*loss_ff + 10*loss_kutta
        loss.backward(); opt.step(); sched.step()
        if epoch % 2000 == 0:                                            # .item() synchronise -> rare
            print(f"epoch {epoch:5d}  loss={loss.item():.3e}  "
                  f"(lap={loss_lap.item():.2e} body={loss_body.item():.2e} "
                  f"ff={loss_ff.item():.2e} kutta={loss_kutta.item():.2e})")

    # ============================ VALIDATION : Cl(alpha) ============================
    xs = dev_grad(poly[:, :1]); ys = dev_grad(poly[:, 1:])
    ds = np.linalg.norm(np.diff(poly, axis=0, append=poly[:1]), axis=1)

    def cl_at(alpha_deg):
        an = torch.full((n_surf, 1), float(norm_alpha(alpha_deg)), device=DEVICE)
        u, v, _ = velocity(xs, ys, an)
        u = u.detach().cpu().numpy().ravel(); v = v.detach().cpu().numpy().ravel()
        cp = 1 - (u**2 + v**2)
        fx = -np.sum(cp * normals[:, 0] * ds); fy = -np.sum(cp * normals[:, 1] * ds)
        a = np.deg2rad(alpha_deg)
        return -fx*np.sin(a) + fy*np.cos(a)

    alphas = np.linspace(ALPHA_MIN, ALPHA_MAX, 21)
    cl_pinn = np.array([cl_at(a) for a in alphas])
    cl_theory = 2*np.pi * np.deg2rad(alphas - ALPHA_L0_DEG)
    print("\n alpha(deg)  Cl_PINN  Cl_theorie")
    for a, cp_, ct in zip(alphas, cl_pinn, cl_theory):
        print(f"  {a:6.1f}    {cp_:6.2f}    {ct:6.2f}")

    # ============================ FIGURE ============================
    nx, ny = 320, 220
    xs_g = np.linspace(XMIN, XMAX, nx); ys_g = np.linspace(YMIN, YMAX, ny)
    XX, YY = np.meshgrid(xs_g, ys_g); grid = np.column_stack([XX.ravel(), YY.ravel()])
    inside = airfoil.contains_points(grid).reshape(ny, nx)
    gx = dev_grad(grid[:, :1]); gy = dev_grad(grid[:, 1:])

    def field_at(alpha_deg):
        an = torch.full((grid.shape[0], 1), float(norm_alpha(alpha_deg)), device=DEVICE)
        u, v, _ = velocity(gx, gy, an)
        U = u.detach().cpu().numpy().reshape(ny, nx); V = v.detach().cpu().numpy().reshape(ny, nx)
        U[inside] = np.nan; V[inside] = np.nan
        return U, V

    show = [0.0, 5.0, 10.0, 15.0]
    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 3)
    for adeg, pos in zip(show, [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]):
        ax = fig.add_subplot(pos)
        U, V = field_at(adeg)
        ax.streamplot(XX, YY, U, V, color=np.sqrt(U**2+V**2), cmap="turbo",
                      density=1.8, linewidth=0.6, arrowsize=0.5)
        ax.fill(poly[:, 0], poly[:, 1], color="0.1", zorder=5)
        ax.set_title(f"α = {adeg:.0f}°  (Cl≈{cl_at(adeg):.2f})", fontsize=11)
        ax.set(xlim=(XMIN+0.4, XMAX-0.6), ylim=(YMIN+0.6, YMAX-0.6)); ax.set_aspect("equal")
        ax.set_xlabel("x"); ax.set_ylabel("y")

    axc = fig.add_subplot(gs[:, 2])
    axc.plot(alphas, cl_theory, "k--", label="Profil mince : 2π(α−α₀)")
    axc.plot(alphas, cl_pinn, "o-", color="crimson", label="PINN paramétrique ψ (GPU-ready)")
    axc.axhline(0, color="0.7", lw=0.8); axc.axvline(ALPHA_L0_DEG, color="0.7", lw=0.8, ls=":")
    axc.set_xlabel("angle d'attaque α (°)"); axc.set_ylabel("Cl")
    axc.set_title("Validation : Cl(α)"); axc.legend(); axc.grid(alpha=0.3)

    fig.suptitle(f"PINN paramétrique ψ (GPU-ready, device={DEVICE.type}) — NACA 4412 inviscide, "
                 f"α∈[{ALPHA_MIN:.0f}°,{ALPHA_MAX:.0f}°]", fontsize=13)
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"\nFigure : {FIG}")


if __name__ == "__main__":
    main()
