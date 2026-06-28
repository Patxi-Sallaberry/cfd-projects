# Mémo ANSYS — Domaine + maillage couche limite (NACA 0012)

Objectif : corriger les **2 causes restantes** des résultats non-physiques (cf. README Phase 0) —
le **blocage de domaine** (Cl 5-8× trop faible) et le **maillage sans couche limite** (Cd ~20× trop
élevé). Une fois ce mémo appliqué, tes Cl/Cd doivent tomber sur la polaire de référence.

Cas : NACA 0012, corde `c = 0.20 m`, V∞ = 30 m/s, ρ = 1.225, μ = 1.7894e-5 → **Re_c ≈ 4.1×10⁵**.

---

## 1. Le domaine fluide (corrige le blocage)

Ton domaine actuel `y ∈ [−0.3, 0.3]` = **±1.5 corde** : beaucoup trop petit, l'aile étouffe le canal.
Règle aéro : les frontières doivent être à **10–20 cordes** du profil pour ne pas perturber l'écoulement.

**Dimensions cibles** (origine au bord d'attaque, corde le long de +x) :

| Frontière | Distance | Coordonnée (c = 0.2 m) |
|---|---|---|
| Entrée (amont) | 12.5 c | x = **−2.5 m** |
| Sortie (aval, sillage) | 20 c | x = **+4.0 m** |
| Haut / bas (farfield) | 15 c | y = **±3.0 m** |
| Envergure (z) | inchangé | z ∈ [0, 0.3] m (avec symmetry) |

> Le sillage a besoin de plus de place que l'amont → on met la sortie **plus loin** (20 c) que l'entrée.
> Un domaine en **C** ou en **O** est l'idéal pour un profil, mais une boîte de cette taille suffit.

### Conditions aux limites
| Surface | Type | Réglage |
|---|---|---|
| Entrée | `velocity-inlet` | Components : Ux = 30·cos α, Uy = 30·sin α |
| Haut + bas | `velocity-inlet` | mêmes composantes (farfield uniforme) — ou `symmetry` si ≥ 15 c |
| Sortie | `pressure-outlet` | 0 Pa gauge |
| Aile | `wall` | no-slip |
| Faces latérales (z) | `symmetry` | quasi-2D |

> ⚠️ Mets **toute l'inclinaison** dans l'inlet (Ux/Uy) et garde le profil horizontal — comme tu fais déjà.
> Et n'oublie pas les **Force Vectors** corrigés par angle : voir `docs/fluent_reports_NACA0012.md`.

---

## 2. La couche limite — première maille pour y⁺ ≈ 1 (corrige le Cd)

Le modèle **k-ω SST** ne donne un bon frottement (donc un bon Cd) que si la **première maille au mur**
est dans la sous-couche visqueuse : **y⁺ ≈ 1**. Voici le calcul pour ton cas (à savoir refaire) :

```
Cf  = 0.026 / Re_c^(1/7)        = 0.026 / (4.1e5)^0.143 ≈ 0.0041     (plaque plane turbulente)
τ_w = ½ · ρ · V∞² · Cf          = 0.5·1.225·30²·0.0041   ≈ 2.26 Pa
u_τ = √(τ_w / ρ)                = √(2.26/1.225)           ≈ 1.36 m/s  (vitesse de frottement)
y₁  = y⁺ · μ / (ρ · u_τ)        = 1·1.7894e-5/(1.225·1.36) ≈ 1.1e-5 m
```

➡️ **Hauteur de première maille ≈ 1.1×10⁻⁵ m (≈ 0.011 mm).**

### Réglages d'inflation (ANSYS Meshing → Inflation sur le wall de l'aile)
| Paramètre | Valeur |
|---|---|
| First Layer Height | **1.1e-5 m** |
| Nombre de couches | **20 à 25** |
| Growth Rate | **1.2** |
| Option | *First Layer Thickness* |

> Vérif : 25 couches à GR 1.2 partant de 1.1e-5 m couvrent ~5–6 mm, soit l'épaisseur de couche limite
> attendue (δ ≈ 0.37·c/Re_c^(1/5) ≈ 5.6 mm). C'est cohérent.

---

## 3. Maillage surfacique + raffinage

| Zone | Taille de maille |
|---|---|
| Surface de l'aile (Face/Edge Sizing) | ~1 mm |
| Bord d'attaque + bord de fuite | **0.2–0.5 mm** (forts gradients) |
| Body of Influence (boîte autour de l'aile + sillage) | ~5–10 mm |
| Champ lointain | croissance douce, Growth Rate ≤ 1.2 |

**Cible : 200 000 – 500 000 cellules** (vs tes 56k). Type : tétra/poly + prismes d'inflation au mur.

---

## 4. Qualité de maillage à viser (Mesh Metrics)

| Métrique | Cible |
|---|---|
| Orthogonal Quality (min) | **> 0.2** (idéal > 0.1 partout) |
| Skewness (max) | **< 0.85** |
| Aspect Ratio dans l'inflation | élevé est NORMAL (mailles fines tangentiellement) |

---

## 5. Setup solveur (rappels)

- **Reference Values** (compute from `inlet`) : Area = 0.06 m², Length = 0.2 m, Velocity = 30, Density = 1.225.
- **Methods** : schéma **Second Order Upwind** (momentum + turbulence), gradient Least-Squares.
- **Convergence** : ne te fie PAS à « converged » en 42 itérations. Lance **500–1000 itérations**, exige :
  - résidus vitesses/k/ω < **1e-5**, continuité < **1e-4** (tu étais à 9e-4),
  - **plateau** stable de Cl ET Cd (Report Plots).

---

## 6. Vérification APRÈS calcul (le réflexe V&V)

1. **Contrôle y⁺** : `Results → Contours → Turbulence → Wall Yplus` sur l'aile. Doit être **≈ 1** (max < 5).
   Si y⁺ ≫ 1 → réduis `First Layer Height` et recommence.
2. **Confronte à la référence** : remplis `data/naca0012_fluent.csv` et lance `python src/postprocessor.py`.
   Attendu cette fois : Cl ≈ 0 / 0.6 / 1.0 / 1.2 et Cd ≈ 0.008 / 0.011 / 0.021 / 0.05 → **tes points
   sur la courbe verte**.

---

## Checklist express

- [ ] Domaine agrandi : entrée −2.5 m, sortie +4 m, haut/bas ±3 m
- [ ] Inflation : first layer **1.1e-5 m**, 20–25 couches, GR 1.2
- [ ] Surface aile ~1 mm, LE/TE 0.2–0.5 mm, 200k–500k cellules
- [ ] Orthogonal Quality > 0.2, Skewness < 0.85
- [ ] Force Vectors corrigés par angle (autre mémo)
- [ ] 500–1000 itérations, continuité < 1e-4, Cl/Cd en plateau
- [ ] y⁺ ≈ 1 vérifié après calcul
- [ ] Comparaison à la référence OK
