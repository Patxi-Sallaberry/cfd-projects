# Mémo Fluent 2026 R1 — Reports Cl/Cd avec le bon Force Vector (NACA 0012)

Objectif : extraire **Cl et Cd** (ou les forces `lift_N` / `drag_N`) pour α = 0°, 5°, 10°, 15°,
afin de remplir `piml/phase0_post_processor/data/naca0012_clcd.csv` et tracer la polaire.

> ⚠️ Le piège qui a faussé ton α=15° (Cl négatif, Cd ~0.12) = **mauvaise orientation du Force Vector**.
> Ce mémo le corrige.

---

## 1. Pourquoi la direction dépend de l'angle

Tu simules l'angle d'attaque en **inclinant la vitesse d'entrée**, pas l'aile :
`Ux = 30·cos(α)`, `Uy = −30·sin(α)`. Le vecteur vitesse amont **V∞** pointe donc dans la
direction `(cos α, −sin α)`.

Or, par définition :
- **Traînée (drag)** = composante de la force **parallèle** à V∞
- **Portance (lift)** = composante **perpendiculaire** à V∞

Si tu laisses les directions par défaut `(1,0,0)` et `(0,1,0)`, tu mesures Fx/Fy dans le repère
du **maillage**, pas le vrai lift/drag → erreur qui grossit avec α (négligeable à 0°, grosse à 15°).

**Directions à renseigner** (perpendiculaire de V∞ orientée vers le haut) :
- Drag direction = `( cos α, −sin α, 0 )`
- Lift direction = `( sin α,  cos α, 0 )`

---

## 2. Valeurs exactes à taper (X, Y, Z)

| α | **Drag** Direction Vector (X, Y, Z) | **Lift** Direction Vector (X, Y, Z) |
|---|---|---|
| 0°  | `1.0000, 0.0000, 0` | `0.0000, 1.0000, 0` |
| 5°  | `0.9962, −0.0872, 0` | `0.0872, 0.9962, 0` |
| 10° | `0.9848, −0.1736, 0` | `0.1736, 0.9848, 0` |
| 15° | `0.9659, −0.2588, 0` | `0.2588, 0.9659, 0` |

> Z = 0 car l'écoulement est dans le plan XY (les faces latérales sont en `symmetry`).

---

## 3. Étape préalable — Reference Values (sinon Cl/Cd faux)

`Setup → Reference Values`, compute from `inlet`, puis vérifie/force :

| Champ | Valeur | Pourquoi |
|---|---|---|
| Area | **0.06** m² | aire de référence = corde × span = 0.20 × 0.30 |
| Density | **1.225** kg/m³ | ρ air |
| Velocity | **30** m/s | V∞ (norme, identique à tous les angles) |
| Length | **0.2** m | corde (pour les moments, pas critique ici) |
| Depth | 0.3 m | span |

> Ces valeurs doivent être **identiques** aux constantes de `postprocessor.py` (`S_REF`, `RHO`, `V_INF`).
> Si ton cas est 2D un jour, Area = corde × 1 m = 0.2 → il faudra ajuster le script.

---

## 4. Créer les Report Definitions (Cl et Cd)

Fais-le **une fois**, puis tu ne changeras que les Direction Vectors à chaque angle.

### Cd (coefficient de traînée)
1. `Solution → Report Definitions → New → Force Report → Drag…`
2. Name : `Cd`
3. **Wall Zones** : sélectionne `wall_naca`
4. **Force Vector** : entre la **Drag** direction du tableau §2 (selon α)
5. Coche **Per Zone** = non, **Average Over** = 1
6. Active **Report File** + **Report Plot** + **Print to Console** → tu suis la convergence
7. OK

### Cl (coefficient de portance)
Idem mais `… → Lift…`, Name `Cl`, Force Vector = la **Lift** direction du tableau §2.

> Astuce : un Force Report « Lift/Drag » sort directement le **coefficient** (il utilise les
> Reference Values). Si tu veux aussi la **force en Newtons**, va dans
> `Results → Reports → Forces…`, sélectionne `wall_naca`, mets la même Direction Vector,
> et lis « Total Force ». Tu peux me donner soit les Cl/Cd, soit les forces — le script gère les deux.

---

## 5. Procédure pour chaque angle

Pour α = 0°, puis 5°, 10°, 15° :

1. `Boundary Conditions → inlet` → mets les composantes `Ux`, `Uy` de l'angle
   (méthode *Components* : Ux=30cosα, Uy=−30sinα).
2. **Édite les deux Report Definitions** `Cl` et `Cd` → mets les Direction Vectors de l'angle (§2).
3. `Initialize` (Hybrid) puis `Run Calculation` (~200 itérations, jusqu'à plateau de Cl/Cd).
4. Note les valeurs **convergées** de `Cl` et `Cd` (et/ou les forces en N).
5. Recommence à l'angle suivant.

---

## 6. Vérifications de bon sens (avant de me les donner)

Ordres de grandeur attendus pour un NACA 0012 à Re ≈ 4·10⁵ :

| α | Cl attendu | Cd attendu |
|---|---|---|
| 0°  | ≈ 0 (profil symétrique) | ~0.008 |
| 5°  | ~0.5 – 0.6 | ~0.01 |
| 10° | ~1.0 – 1.1 | ~0.02 |
| 15° | ~1.1 – 1.5 (proche décrochage) | ~0.03 – 0.06 |

🚩 Si tu obtiens encore un **Cl négatif** ou un **Cd > 0.1**, c'est que la direction du Force Vector
ou l'orientation de la géométrie est encore inversée → on debug avant de tracer.
(Le maillage grossier à 56k cellules creusera un écart sur les valeurs, mais **les signes et
ordres de grandeur doivent être bons**.)

---

## 7. Ce que tu me donnes ensuite

Pour chaque angle, soit :
- **(a)** `Cl` et `Cd` lus dans Fluent, soit
- **(b)** les forces `lift_N` et `drag_N` (Newtons).

Je remplis le CSV, on relance `python src/postprocessor.py`, et on trace la **vraie polaire**.
On comparera aussi à XFOIL / tables NACA pour valider.
