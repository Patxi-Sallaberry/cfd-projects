# CFD & CAO Projects — Patxi Sallaberry

Ingénieur en formation (INSA Toulouse — Ingénierie Systèmes), ce repo regroupe mes projets personnels de simulation numérique et de modélisation 3D, réalisés en dehors du cursus académique pour développer des compétences en aérodynamique et mécanique des fluides computationnelle.

---

## 🛠️ Outils

| Domaine | Outils |
|---|---|
| CAO | Fusion 360, CATIA V5 |
| CFD | ANSYS Fluent 2026 R1, OpenFOAM *(à venir)* |
| Maillage | ANSYS Mechanical Meshing, ANSYS DesignModeler |
| Post-processing | ANSYS CFD-Post, ParaView *(à venir)* |
| Scripting | Python (génération de profils, automatisation) |

---

## 📁 Projets

### 🛩️ NACA 0012 — Simulation aérodynamique
**Dossier :** [`naca0012-airfoil-CFD/`](./naca0012-airfoil-CFD)

Simulation CFD d'un profil symétrique NACA 0012 à différents angles d'attaque.

- **Solveur :** ANSYS Fluent 2026 R1
- **Modèle de turbulence :** k-ω SST
- **Régime :** stationnaire, incompressible
- **Re ≈ 400 000** — V = 30 m/s, corde = 200 mm

| α | Statut |
|---|---|
| 0° | À venir |
| 5° | À venir |
| 10° | À venir |
| 15° | ✅ Simulé |

---

### 🏎️ Formula Student Front Wing — Aileron bi-élément
**Dossier :** [`front-wing-CFD/`](./front-wing-CFD)

Simulation CFD d'un aileron avant Formula Student deux éléments (main plane + flap), profil NACA 4412 inversé. Géométrie générée par script Python dans Fusion 360.

- **Solveur :** ANSYS Fluent 2026 R1
- **Modèle de turbulence :** k-ω SST
- **Régime :** stationnaire, incompressible, α = 15°
- **Re ≈ 500 000** — V = 30 m/s, corde main plane = 250 mm

| Paramètre | Résultat |
|---|---|
| Cl | **−0.52** (downforce ✅) |
| Cd | **0.163** |
| Cl/Cd | **3.19** |
| Convergence | ~145 itérations |

---

## 🗺️ Roadmap

- [x] NACA 0012 — α = 15°
- [ ] NACA 0012 — polaire complète (α = 0°, 5°, 10°, 15°)
- [x] Formula Student front wing — α = 15°
- [ ] Formula Student front wing — sweep angle de flap (0°, 15°, 30°)
- [ ] Effet de sol — simulation en ground effect
- [ ] Comparaison NACA 4412 vs 2412
- [ ] Introduction OpenFOAM

---

*Patxi Sallaberry — INSA Toulouse · Erasmus @ Linköping University (LiU) · Fall 2026*  
*Contact : [LinkedIn](https://linkedin.com/in/patxi-sallaberry) · [GitHub](https://github.com/Patxi-Sallaberry)*
