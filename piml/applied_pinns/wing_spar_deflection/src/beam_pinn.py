"""
Applied PINN - Fleche d'un LONGERON D'AILE sous la portance (poutre d'Euler-Bernoulli)
======================================================================================
Application concrete (structures / aerospatial) : de combien flechit le longeron encastre
d'une demi-aile sous une charge de portance repartie q(x) ? On resout la poutre d'Euler-
Bernoulli avec un PINN, et on en tire d'un seul reseau la fleche, la pente, le MOMENT
flechissant et l'EFFORT TRANCHANT.

Equation (4e ordre) :            EI * w''''(x) = q(x)          x in [0, L]
Conditions aux limites (encastree-libre / cantilever) :
   w(0)  = 0   (pas de deplacement a l'emplanture)     w'(0)  = 0   (pas de pente : encastrement)
   w''(L)= 0   (moment nul au saumon libre)            w'''(L)= 0   (effort tranchant nul au saumon)

CHOIX 1 - ADIMENSIONNEMENT (leçon cle des PINN : normaliser). Avec xi = x/L et W = w / w_ref
ou w_ref = q0 L^4 / (EI), l'equation d'une charge UNIFORME q0 devient simplement :
        W''''(xi) = 1     sur xi in [0, 1],   memes CL.
=> entrees O(1), sortie O(0.1) : entrainement beaucoup plus stable qu'avec des E~1e11, I~1e-5.
Solution analytique exacte (pour la V&V) :  W(xi) = xi^2 (xi^2 - 4 xi + 6) / 24  ->  fleche au bout = 1/8.

CHOIX 2 - charge UNIFORME (q0 constant) : donne une solution analytique propre pour VALIDER.
Le code est ecrit pour qu'on puisse remplacer le second membre par n'importe quel q(xi)
(ex. distribution elliptique realiste) — on perd juste la validation analytique simple.

Sortie : results/beam_pinn.png     |     Lancer : python src/beam_pinn.py   (~1 min CPU)
"""

from pathlib import Path as FsPath
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)

# ---- parametres DIMENSIONNELS (uniquement pour raconter/afficher des chiffres physiques) ----
L_SPAN = 5.0            # demi-envergure du longeron [m]
E_MOD  = 70e9           # module d'Young (aluminium) [Pa]
I_SEC  = 3.0e-5         # moment quadratique de la section [m^4]
Q0     = 1200.0         # charge de portance repartie, supposee uniforme [N/m]
EI     = E_MOD * I_SEC
W_REF  = Q0 * L_SPAN**4 / EI          # fleche de reference [m]  (echelle d'adimensionnement)

EPOCHS = 5000                          # Adam (le probleme est petit -> rapide en CPU)
LBFGS_STEPS = 30                       # polissage L-BFGS (2nd ordre) : ecrase le residu
W_BC = 100.0                           # poids des conditions aux limites (cruciales au 4e ordre)

ROOT = FsPath(__file__).resolve().parent.parent
FIG = ROOT / "results" / "beam_pinn.png"


def W_exact(xi):
    """Solution analytique adimensionnee (cantilever + charge uniforme) — reference V&V."""
    return xi**2 * (xi**2 - 4*xi + 6) / 24.0


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {dev}")

    # reseau : xi -> W. tanh (indispensable : on derive 4 fois, ReLU'' = 0).
    model = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                          nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1)).to(dev)

    def grad(f, x):
        return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]

    def derivs(xi):
        """Renvoie W, W', W'', W''', W'''' — l'autograd imbrique 4 fois (nouveaute vs Phase 2)."""
        W = model(xi)
        W1 = grad(W, xi); W2 = grad(W1, xi); W3 = grad(W2, xi); W4 = grad(W3, xi)
        return W, W1, W2, W3, W4

    # points de collocation (ou l'on impose l'EDO) + points de bord (xi=0 et xi=1)
    xic = torch.linspace(0, 1, 200, device=dev).reshape(-1, 1).requires_grad_(True)
    xi0 = torch.zeros(1, 1, device=dev, requires_grad=True)
    xiL = torch.ones(1, 1, device=dev, requires_grad=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def losses():
        _, _, _, _, W4 = derivs(xic)
        loss_phys = ((W4 - 1.0)**2).mean()                       # EDO : W'''' = 1
        W0, W0_1, _, _, _ = derivs(xi0)                          # encastrement : W(0)=0, W'(0)=0
        _, _, WL_2, WL_3, _ = derivs(xiL)                        # saumon libre : W''(1)=0, W'''(1)=0
        loss_bc = W0**2 + W0_1**2 + WL_2**2 + WL_3**2
        return loss_phys, loss_bc.squeeze()

    for epoch in range(EPOCHS):
        opt.zero_grad()
        lp, lb = losses()
        loss = lp + W_BC * lb
        loss.backward(); opt.step()
        if epoch % 1000 == 0:
            print(f"epoch {epoch:5d}  loss={loss.item():.3e}  (phys={lp.item():.2e} bc={lb.item():.2e})")

    # --- polissage L-BFGS : ici la collocation est FIXE -> loss deterministe -> L-BFGS ideal
    #     (contraste avec le PINN parametrique de l'aile ou le lot changeait a chaque epoque) ---
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                              history_size=50, line_search_fn="strong_wolfe")
    def closure():
        lbfgs.zero_grad()
        lp, lb = losses()
        loss = lp + W_BC * lb
        loss.backward(); return loss
    for _ in range(LBFGS_STEPS):
        lbfgs.step(closure)
    lp, lb = losses()
    print(f"[apres L-BFGS] phys={lp.item():.2e}  bc={lb.item():.2e}")

    # ============================ VALIDATION vs solution analytique ============================
    xi = torch.linspace(0, 1, 400, device=dev).reshape(-1, 1).requires_grad_(True)
    W, W1, W2, W3, _ = derivs(xi)
    xn = xi.detach().cpu().numpy().ravel()
    Wn = W.detach().cpu().numpy().ravel()
    We = W_exact(xn)
    r2 = 1 - np.sum((Wn - We)**2) / np.sum((We - We.mean())**2)
    tip_pinn, tip_exact = Wn[-1], 1/8
    print(f"\nR2(W) = {r2:.5f}   |   fleche au bout : PINN {tip_pinn:.4f}  vs exact {tip_exact:.4f} (adim)")

    # --- retour aux grandeurs DIMENSIONNELLES pour raconter la physique ---
    tip_m = tip_pinn * W_REF                       # fleche au saumon [m]
    M_root = Q0 * L_SPAN**2 / 2                     # moment flechissant a l'emplanture [N.m] (= q0 L^2/2)
    print(f"Fleche au saumon ~ {tip_m*100:.1f} cm  ({tip_m/L_SPAN*100:.2f} % de la demi-envergure)")
    print(f"Moment a l'emplanture ~ {M_root/1e3:.1f} kN.m")

    # ============================ FIGURE ============================
    # grandeurs dimensionnelles derivees DU MEME reseau :
    x_m = xn * L_SPAN
    w_cm = Wn * W_REF * 100                                   # fleche [cm]
    M = Q0 * L_SPAN**2 * W2.detach().cpu().numpy().ravel()    # moment  M(x) = q0 L^2 W''(xi)
    V = Q0 * L_SPAN * W3.detach().cpu().numpy().ravel()       # tranchant V(x) = q0 L W'''(xi)
    M_ex = Q0 * L_SPAN**2 * ((xn - 1)**2 / 2)                 # exact : W'' = (xi-1)^2/2
    V_ex = Q0 * L_SPAN * (xn - 1)                             # exact : W''' = (xi-1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))

    # (1) forme flechie de l'aile + charge
    ax1.plot(x_m, np.zeros_like(x_m), "--", color="0.6", lw=1, label="aile non chargée")
    ax1.plot(x_m, -w_cm, color="crimson", lw=2.5, label="longeron fléchi (PINN)")
    ax1.plot(x_m, -W_exact(xn)*W_REF*100, "k:", lw=1.2, label="exact")
    for xa in np.linspace(0.15, 1, 10)*L_SPAN:                # fleches de charge (portance vers le haut)
        ax1.annotate("", xy=(xa, 0.6), xytext=(xa, 0.0),
                     arrowprops=dict(arrowstyle="->", color="steelblue", lw=1))
    ax1.scatter([0], [0], s=80, marker="s", color="0.2", zorder=5, label="encastrement")
    ax1.set_title("Longeron d'aile fléchi sous la portance"); ax1.set_xlabel("envergure x [m]")
    ax1.set_ylabel("flèche vers le bas [cm]"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # (2) W(xi) PINN vs exact
    ax2.plot(xn, We, "k-", lw=2, label="exact")
    ax2.plot(xn, Wn, "--", color="crimson", lw=1.8, label="PINN")
    ax2.set_title(f"Flèche adimensionnée W(ξ)   (R²={r2:.4f})")
    ax2.set_xlabel("ξ = x/L"); ax2.set_ylabel("W"); ax2.legend(); ax2.grid(alpha=0.3)

    # (3) moment & tranchant, derives du MEME reseau
    ax3.plot(x_m, M/1e3, color="darkorange", lw=2, label="Moment M(x) [kN·m] — PINN")
    ax3.plot(x_m, M_ex/1e3, "k:", lw=1)
    ax3.plot(x_m, V/1e3, color="teal", lw=2, label="Tranchant V(x) [kN] — PINN")
    ax3.plot(x_m, V_ex/1e3, "k:", lw=1, label="exact")
    ax3.axhline(0, color="0.7", lw=0.8)
    ax3.set_title("Moment fléchissant & effort tranchant"); ax3.set_xlabel("envergure x [m]")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    fig.suptitle("PINN — flèche d'un longeron d'aile (poutre d'Euler-Bernoulli, charge de portance)", fontsize=13)
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"\nFigure : {FIG}")


if __name__ == "__main__":
    main()
