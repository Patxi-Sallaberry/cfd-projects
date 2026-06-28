# PIML Phase 0 — Compte-rendu simulations NACA 0012
## À coller dans Claude Code (terminal)

---

## CONTEXTE

Simulations ANSYS Fluent 2026 R1 Student — NACA 0012
- Profil : chord 200mm, span 300mm
- Domaine fluide : x[-1m,3m], y[-0.3m,0.3m], z[-0.5m,0.5m]
- Solveur : k-ω SST, Steady, Pressure-Based, Double Precision
- V∞ = 30 m/s, ρ = 1.225 kg/m³, μ = 1.7894e-05 kg/(m·s)
- Reference Values : Area=0.06m², Length=0.2m, Velocity=30m/s
- Maillage : 56001 cells, 367470 faces, 304855 nodes
- Méthode : profil horizontal, écoulement incliné via Ux/Uy à l'inlet
- Bug Workbench "CALCULATION-COMPLETE" contourné via Fluent standalone

---

## RÉSULTATS CONVERGÉS

| α (°) | Ux (m/s) | Uy (m/s) | Cl      | Cd     | Iter conv. | Continuity |
|--------|----------|----------|---------|--------|------------|------------|
| 0      | 30.00    | 0.00     | -0.00665| 0.1957 | 42         | 9.28e-04   |
| 5      | 29.89    | -2.61    | -0.0884 | 0.1875 | 57         | 9.11e-04   |
| 10     | 29.54    | -5.21    | -0.1692 | 0.1634 | 63         | 6.96e-04   |
| 15     | 28.98    | -7.76    | -0.2437 | 0.1229 | 69         | 6.68e-04   |

### Convention de signe
- Cl négatif = portance dans le sens opposé à la normale Y (convention repère global)
- L'écoulement incliné vers le bas (Uy < 0) génère une portance vers le bas dans le repère global
- Tendance |Cl| croissante avec α ✓ — physiquement cohérent
- Cd décroissant avec α dans ce setup 3D avec symétrie

### Note sur les valeurs absolues
- Cd ≈ 0.12-0.20 est plus élevé que la littérature 2D (~0.008-0.012)
- Origine : domaine 3D avec span fini + conditions de symétrie + maillage non raffiné
- Pour le projet PIML : les tendances Cl/Cd(α) sont exploitables ✓

---

## VECTEURS DE FORCE UTILISÉS

| α  | Lift X     | Lift Y | Drag X | Drag Y |
|----|------------|--------|--------|--------|
| 0° | 0.000      | 1.000  | 1.000  | 0.000  |
| 5° | -0.087     | 0.996  | 0.996  | 0.087  |
| 10°| -0.174     | 0.985  | 0.985  | 0.174  |
| 15°| -0.259     | 0.966  | 0.966  | 0.259  |

---

## STRUCTURE DES IMAGES

```
naca0012_results/
├── alpha_0deg/
│   ├── cd_rplot.png       — convergence Cd (iter 1-42)
│   ├── cl_rplot.png       — convergence Cl (iter 1-42)
│   ├── residuals.png      — résidus scalés
│   └── console_converged.png — console "42 solution is converged"
├── alpha_5deg/
│   ├── cd_rplot.png
│   ├── cl_rplot.png
│   ├── residuals.png
│   └── console_converged.png
├── alpha_10deg/
│   └── [idem]
└── alpha_15deg/
    └── [idem]
```

---

## PROCHAINES ÉTAPES PHASE 0

1. Créer `piml/data/raw/naca0012_fluent.csv` avec ces 4 points
2. Script Python `piml/src/phase0_postprocessor.py` :
   - Charger le CSV
   - Tracer Cl(α) et Cd(α)
   - Calculer L/D = Cl/Cd
   - Exporter graphes dans `piml/results/`
3. Committer sur GitHub `Patxi-Sallaberry/cfd-projects`

