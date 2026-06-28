# Provenance des données — Phase 0 (NACA 0012)

Ce dossier suit une démarche **V&V (Verification & Validation)** : on sépare clairement
une **référence** (vérité-terrain documentée) des **simulations Fluent** personnelles, et on
quantifie l'écart entre les deux. Aucune donnée n'est inventée : chaque source est traçable.

---

## 1. `naca0012_reference.csv` — RÉFÉRENCE (vérité-terrain)

| Champ | Valeur |
|---|---|
| Outil | **NeuralFoil 0.3.2** (modèle `xlarge`), via AeroSandbox 4.2.9 |
| Nature | Réseau de neurones **entraîné sur XFOIL** — substitut de XFOIL |
| Conditions | Profil NACA 0012 analytique · **Re = 400 000** · Mach = 0 · Ncrit = 9 (défaut) |
| Reproductible | Oui — `python src/generate_reference.py` régénère le fichier à l'identique |
| Généré le | 2026-06-28 |

⚠️ **Honnêteté de provenance.** NeuralFoil **n'est pas** de l'expérimental ni du CFD Fluent :
c'est un modèle rapide calibré sur XFOIL (méthode panneaux + couche limite intégrale). Il sert ici
de **référence de comparaison**, pas de mesure absolue. Pour une validation plus stricte, voir §3.

## 2. `naca0012_fluent.csv` — MES SIMULATIONS (à remplir)

Mes propres runs ANSYS Fluent 2026 R1 (NACA 0012, k-ω SST, Re ≈ 400k), pour α = 0°, 5°, 10°, 15°.
Colonnes `lift_N` / `drag_N` à renseigner depuis les *Force Reports* Fluent
(voir `docs/fluent_reports_NACA0012.md` pour les bonnes directions de Force Vector).
Ces données sont **les miennes**, maillage grossier assumé — elles servent à mesurer
**mon écart à la référence**, pas à être présentées comme une vérité.

## 3. Sources de validation plus strictes (si besoin plus tard)

Pour aller au-delà de NeuralFoil, par ordre de rigueur :

- **XFOIL** à Re = 400k, Ncrit = 9 (Mark Drela) — référence panneaux de facto.
  Polaires en ligne : Airfoil Tools (`airfoiltools.com`, NACA 0012, Re standard 200k/500k).
- **Données expérimentales** NACA 0012 : Ladson (NASA TM 4074, 1988), Abbott & von Doenhoff (1959) —
  ⚠️ à Re élevé (3–9×10⁶), donc **comportement de décrochage différent** de ton Re = 400k.
- **Études CFD publiées** (k-ω SST) couvrant Re = 1.2–4.8×10⁵ — comparables à ta config.

> Règle : toute donnée ajoutée ici doit citer sa source (auteur/outil, année, Re, méthode) dans ce fichier.
