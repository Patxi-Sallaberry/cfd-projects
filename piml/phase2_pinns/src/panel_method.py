"""
Methode des PANNEAUX (vortex panel method de Kuethe & Chow) - reference V&V pour le PINN
========================================================================================
L'outil CLASSIQUE pour la portance inviscide 2D : on discretise la surface de l'aile en
panneaux portant une distribution de tourbillons (vortex) d'intensite lineaire, et on impose
   - tangence (pas de flux normal) a chaque point de controle,
   - la condition de Kutta au bord de fuite.
=> un petit systeme lineaire (M+1 equations) resolu en quelques millisecondes, donnant un
Cl(alpha) quasi exact (niveau XFOIL inviscide). C'est la BONNE reference pour valider le PINN
parametrique (bien plus fiable que la seule theorie du profil mince).

Repere identique au PINN : aile fixe a alpha=0, l'amont arrive sous l'angle alpha.
Reference : Kuethe & Chow, "Foundations of Aerodynamics" ; Anderson, "Fundamentals of Aerodynamics".

Lancer : python src/panel_method.py
"""

import numpy as np


def naca4412_nodes(n=100):
    """Noeuds de la surface, ordonnes dans le SENS HORAIRE depuis le bord de fuite
    (TE -> extrados -> BA -> intrados -> TE), comme l'exige la methode des panneaux."""
    m, p, t = 0.04, 0.4, 0.12
    beta = np.linspace(0, np.pi, n)
    x = (1 - np.cos(beta)) / 2
    yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    yc = np.where(x < p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1-2*p) + 2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p - x), 2*m/(1-p)**2*(p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    # ordre HORAIRE requis par la methode : TE -> intrados -> BA -> extrados -> TE
    Xlo, Ylo = xl[::-1], yl[::-1]                 # de TE a BA (intrados)
    Xup, Yup = xu[1:], yu[1:]                     # de BA a TE (extrados), sans dupliquer le BA
    XB = np.concatenate([Xlo, Xup])
    YB = np.concatenate([Ylo, Yup])
    return XB, YB


def solve_panels(XB, YB, alpha_deg):
    """Resout la distribution de tourbillons et renvoie (Cp, Cl, points de controle, Cl_par_circulation)."""
    al = np.deg2rad(alpha_deg)
    M = len(XB) - 1                                   # nombre de panneaux
    # geometrie des panneaux
    X = 0.5*(XB[:-1] + XB[1:])                         # points de controle (milieux)
    Y = 0.5*(YB[:-1] + YB[1:])
    S = np.hypot(XB[1:]-XB[:-1], YB[1:]-YB[:-1])       # longueurs
    TH = np.arctan2(YB[1:]-YB[:-1], XB[1:]-XB[:-1])    # angles des panneaux (Phi)

    # coefficients d'influence (formules de Kuethe & Chow, vortex d'intensite lineaire)
    CN1 = np.zeros((M, M)); CN2 = np.zeros((M, M))
    CT1 = np.zeros((M, M)); CT2 = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if i == j:
                CN1[i, j] = -1.0; CN2[i, j] = 1.0
                CT1[i, j] = 0.5*np.pi; CT2[i, j] = 0.5*np.pi
            else:
                A = -(X[i]-XB[j])*np.cos(TH[j]) - (Y[i]-YB[j])*np.sin(TH[j])
                B = (X[i]-XB[j])**2 + (Y[i]-YB[j])**2
                C = np.sin(TH[i]-TH[j]); D = np.cos(TH[i]-TH[j])
                E = (X[i]-XB[j])*np.sin(TH[j]) - (Y[i]-YB[j])*np.cos(TH[j])
                F = np.log(1.0 + (S[j]**2 + 2*A*S[j])/B)
                G = np.arctan2(E*S[j], B + A*S[j])
                P = (X[i]-XB[j])*np.sin(TH[i]-2*TH[j]) + (Y[i]-YB[j])*np.cos(TH[i]-2*TH[j])
                Q = (X[i]-XB[j])*np.cos(TH[i]-2*TH[j]) - (Y[i]-YB[j])*np.sin(TH[i]-2*TH[j])
                CN2[i, j] = D + 0.5*Q*F/S[j] - (A*C + D*E)*G/S[j]
                CN1[i, j] = 0.5*D*F + C*G - CN2[i, j]
                CT2[i, j] = C + 0.5*P*F/S[j] + (A*D - C*E)*G/S[j]
                CT1[i, j] = 0.5*C*F - D*G - CT2[i, j]

    # systeme normal (M+1 inconnues : intensites nodales gamma') + condition de Kutta
    An = np.zeros((M+1, M+1)); At = np.zeros((M, M+1))
    for i in range(M):
        An[i, 0] = CN1[i, 0]; An[i, M] = CN2[i, M-1]
        At[i, 0] = CT1[i, 0]; At[i, M] = CT2[i, M-1]
        for j in range(1, M):
            An[i, j] = CN1[i, j] + CN2[i, j-1]
            At[i, j] = CT1[i, j] + CT2[i, j-1]
    RHS = np.append(np.sin(TH - al), 0.0)             # tangence : flux normal nul
    An[M, 0] = 1.0; An[M, M] = 1.0                     # Kutta : gamma'(TE_haut) + gamma'(TE_bas) = 0

    gamma = np.linalg.solve(An, RHS)                  # intensites nodales (normalisees par 2pi V)

    Vt = np.cos(TH - al) + At @ gamma                 # vitesse tangentielle aux points de controle
    Cp = 1.0 - Vt**2
    # Cl par integration de pression, projete perpendiculairement a l'amont
    nx = -np.sin(TH); ny = np.cos(TH)                  # normales sortantes (ordre horaire)
    fx = -np.sum(Cp * nx * S); fy = -np.sum(Cp * ny * S)
    Cl = -fx*np.sin(al) + fy*np.cos(al)
    # Cl par la circulation totale (Kutta-Joukowski) : Cl = 2*Gamma, Gamma = 2pi * sum(gamma_avg * S)
    Cl_circ = 4.0*np.pi * np.sum(0.5*(gamma[:-1] + gamma[1:]) * S)
    return Cp, Cl, (X, Y), Cl_circ


def cl_curve(alphas, n=120):
    XB, YB = naca4412_nodes(n)
    cl, clc = [], []
    for a in alphas:
        _, c, _, cc = solve_panels(XB, YB, a)
        cl.append(c); clc.append(cc)
    return np.array(cl), np.array(clc)


if __name__ == "__main__":
    alphas = np.linspace(-5, 15, 21)
    cl, clc = cl_curve(alphas)
    th = 2*np.pi*np.deg2rad(alphas - (-4.0))
    print(" alpha   Cl_panneaux  Cl_circulation  Cl_theorie(mince)")
    for a, c, cc, t in zip(alphas, cl, clc, th):
        print(f" {a:6.1f}    {c:7.3f}      {cc:7.3f}        {t:7.3f}")
    # pente et angle de portance nulle
    slope = np.polyfit(alphas, cl, 1)
    print(f"\npente dCl/dalpha = {slope[0]:.4f} /deg  (theorie ~0.110) ;  "
          f"alpha_L0 = {-slope[1]/slope[0]:.2f} deg  (attendu ~ -4)")
