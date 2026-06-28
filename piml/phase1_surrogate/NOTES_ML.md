# Fiche de révision — Comprendre le modèle surrogate (Phase 1)

Tout ce qu'il faut comprendre sur le ML de la Phase 1 : ce qu'on fait, où ça tourne, et
chaque ligne de `src/train_surrogate.py` expliquée. Document autonome — lisible seul.

## Sommaire
1. [L'idée en une phrase](#1-lidée-en-une-phrase)
2. [Où tourne le ML (GitHub vs ta machine)](#2-où-tourne-le-ml)
3. [Le pipeline complet](#3-le-pipeline-complet)
4. [Le code ligne par ligne](#4-le-code-ligne-par-ligne)
5. [Glossaire](#5-glossaire)
6. [Comment lancer](#6-comment-lancer)

---

## 1. L'idée en une phrase

On apprend une **fonction** `f : α → (Cl, Cd)` à partir d'exemples, pour pouvoir prédire les
coefficients aérodynamiques **instantanément**, sans relancer une simulation CFD (lente).

C'est de l'**apprentissage supervisé** (on a les bonnes réponses pour s'entraîner) en
**régression** (la sortie est un nombre continu, pas une catégorie).

```
Aujourd'hui sans ML :   α  →  [CFD : des heures]   →  Cl, Cd
Avec le surrogate :     α  →  [réseau : millisecondes]  →  Cl, Cd
```

---

## 2. Où tourne le ML

Confusion fréquente à lever :

| | Rôle |
|---|---|
| **GitHub** | Un **placard** : stocke le code et les résultats, garde l'historique, partage. **N'exécute rien.** |
| **Ta machine** | C'est là que le ML **tourne** : sur le **processeur (CPU)** de ton PC, via Python. |

Le cycle réel :
```
1. on écrit le code          (sur la machine)
2. on le LANCE : python ...  (ça calcule sur la machine : CPU, ou GPU si disponible)
3. ça produit résultats + figures
4. on push sur GitHub        (sauvegarde / partage)
```

Pas besoin d'endroit spécial : un laptop suffit ici (entraînement en quelques secondes).
On irait sur **Google Colab** (GPU gratuit, dans le navigateur), Kaggle ou un cluster
seulement pour de **gros** modèles.

---

## 3. Le pipeline complet

```
   données (89 paires α→Cl,Cd)
        │
        ▼
   séparation train / test   ──►  20 % mis de côté pour vérifier la généralisation
        │
        ▼
   normalisation (centrer-réduire, ajustée sur le train)
        │
        ▼
   modèle MLP  1→64→64→2   (~4400 paramètres à régler)
        │
        ▼
   boucle d'entraînement (3000 epochs) :
        forward → erreur (MSE) → rétropropagation → mise à jour (Adam)
        │
        ▼
   évaluation sur le test : RMSE, R²
        │
        ▼
   inférence : donner un α, obtenir (Cl, Cd) instantanément
```

---

## 4. Le code ligne par ligne

### Bloc 1 — Les données (entrée → sorties)
```python
X = df[["alpha_deg"]].values.astype("float32")   # entree  (N, 1)
Y = df[["Cl", "Cd"]].values.astype("float32")    # sorties (N, 2)
```
- `X` = la **feature** (entrée) : l'angle α. Forme `(N, 1)` = N exemples, 1 variable.
- `Y` = les **targets** (cibles) : Cl et Cd. Forme `(N, 2)`.
- `float32` = flottants 32 bits, le format standard des réseaux (rapide, assez précis).
- Espace d'entrée de dimension **1**, sortie de dimension **2**.

### Bloc 2 — Séparation train / test
```python
idx = rng.permutation(N)
n_test = int(0.2 * N)
test_idx, train_idx = idx[:n_test], idx[n_test:]
```
- On mélange les indices et on réserve **20 %** des points pour le **test**.
- **Pourquoi c'est essentiel** : si on notait le modèle sur ce qu'il a appris, il pourrait
  avoir appris **par cœur** sans généraliser (= **surapprentissage / overfitting**). Le test,
  *jamais vu pendant l'entraînement*, mesure la **vraie** capacité de prédiction.

### Bloc 3 — Normalisation
```python
xm, xs = Xtr.mean(0), Xtr.std(0)        # moyenne / ecart-type
ym, ys = Ytr.mean(0), Ytr.std(0)
norm_x = lambda a: (a - xm) / xs        # centrer-reduire
```
- On ramène chaque variable à **moyenne 0, écart-type 1** (standardisation).
- **Pourquoi** : α va de −6 à 16, Cd de 0.006 à 0.07 — des échelles très différentes. La
  descente de gradient converge **bien mieux** quand tout est à la même échelle.
- **Important** : moyenne/écart-type calculés sur le **train uniquement**, jamais sur le test.
  Sinon de l'information du test "fuiterait" dans l'entraînement (= **data leakage**), et la
  performance mesurée serait faussement optimiste.

### Bloc 4 — Le modèle (un MLP)
```python
model = nn.Sequential(
    nn.Linear(1, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, 2),
)
```
- `nn.Linear(1, 64)` = une **couche** de 64 **neurones**. Chaque neurone calcule
  `y = w·x + b` (somme pondérée + biais). Ici 1 entrée → 64 sorties.
- `nn.Tanh()` = **fonction d'activation** non-linéaire. **Sans elle**, empiler des couches
  linéaires resterait globalement linéaire → incapable d'apprendre une courbe. C'est la
  non-linéarité qui permet de modéliser la polaire (et son décrochage).
- Architecture : `1 → 64 → 64 → 2`. La dernière couche sort 2 nombres : (Cl, Cd).
- **Nombre de paramètres** : `1·64+64` + `64·64+64` + `64·2+2` = 128 + 4160 + 130 ≈ **4 400**.
  Entraîner = ajuster ces ~4400 nombres.
- **Théorème d'approximation universelle** : un MLP avec assez de neurones peut approximer
  *n'importe quelle fonction continue*. D'où sa capacité à apprendre la polaire.

### Bloc 4bis — Erreur et optimiseur
```python
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
```
- `MSELoss` = **erreur quadratique moyenne** : `moyenne((prédit − vrai)²)`. C'est **le nombre
  à minimiser** ; plus il est petit, mieux le modèle colle aux données.
- `Adam` = l'**optimiseur**, l'algorithme qui modifie les poids pour faire baisser l'erreur.
- `lr=1e-2` = **learning rate** (taille du pas de mise à jour). Trop grand → instable ; trop
  petit → lent. Adam adapte le pas automatiquement.

### Bloc 5 — La boucle d'entraînement (le cœur du ML)
```python
for epoch in range(3000):
    opt.zero_grad()             # 1. remet les gradients a zero
    pred = model(Xtr_t)         # 2. forward : predit Cl, Cd
    loss = loss_fn(pred, Ytr_t) # 3. mesure l'erreur
    loss.backward()             # 4. backward : calcule les gradients
    opt.step()                  # 5. met a jour les poids
```
Ces 5 lignes **sont** l'apprentissage. Une **epoch** = un passage complet. Répété 3000 fois :
1. `zero_grad` : efface les gradients de l'itération précédente.
2. **Forward pass** : on pousse α dans le réseau → il sort une prédiction (Cl, Cd).
3. **Loss** : on compare au vrai → un nombre d'erreur.
4. **Backward = rétropropagation** : PyTorch calcule `∂loss/∂poids` pour chaque poids
   (dérivées en chaîne). Le réseau "sait" dans quel sens bouger chaque poids.
5. `step` : Adam déplace chaque poids un peu, dans le sens qui réduit l'erreur.

La `loss` qui descend vers 0 = **preuve que le modèle apprend**.

### Bloc 6 — Évaluation sur le test
```python
model.eval()
with torch.no_grad():
    Pte = denorm_y(model(Xte_t).numpy())
...
rmse = np.sqrt(np.mean((pred - true) ** 2))
r2   = 1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2)
```
- `torch.no_grad()` : en évaluation, pas besoin de gradients → plus rapide.
- `denorm_y` : on **dé-normalise** pour revenir aux vraies unités (Cl, Cd physiques).
- **RMSE** = erreur typique, en unités physiques (racine de l'erreur quadratique moyenne).
- **R²** (coefficient de détermination) : **1 = parfait**, **0 = pas mieux que prédire la
  moyenne**, négatif = pire que la moyenne. Calculé sur le **test** → mesure la généralisation.

> ⚠️ Ici R² = 1.000 parce que les données sont **lisses et sans bruit** (une polaire propre).
> C'est un **jalon d'apprentissage**, pas un problème difficile. La vraie difficulté arrive avec
> des données **bruitées, éparses** ou des entrées **plus nombreuses** (ex. ajouter le Reynolds).

### Bloc 7 — Inférence + figure
```python
a_dense = np.linspace(X.min(), X.max(), 300)...
p_dense = denorm_y(model(torch.tensor(norm_x(a_dense))).numpy())
```
- On crée 300 angles fins, et on demande au modèle de **prédire** = **inférence**. C'est ça,
  *utiliser* le modèle entraîné : un α en entrée → (Cl, Cd) en sortie, **immédiatement**.
- On sauvegarde la figure (`results/figures/surrogate_fit.png`) et les **poids entraînés**
  (`results/surrogate_naca0012.pt`) pour réutiliser le modèle sans le ré-entraîner.

---

## 5. Glossaire

| Terme | Définition courte |
|---|---|
| **Apprentissage supervisé** | On apprend à partir d'exemples *étiquetés* (entrée + bonne réponse). |
| **Régression** | Prédire une valeur *continue* (vs classification = une catégorie). |
| **Feature / target** | Variable d'entrée / valeur à prédire. |
| **Train / test** | Données d'apprentissage / données réservées pour vérifier la généralisation. |
| **Overfitting** | Le modèle apprend "par cœur" le train mais généralise mal. |
| **Normalisation** | Mettre les variables à la même échelle (moyenne 0, écart-type 1). |
| **Data leakage** | Fuite d'info du test vers l'entraînement → performance faussée. |
| **MLP** | Perceptron multicouche : un réseau de couches de neurones. |
| **Neurone** | Calcule `w·x + b` (somme pondérée + biais). |
| **Activation (tanh)** | Non-linéarité qui permet d'apprendre des courbes. |
| **Paramètres / poids** | Les nombres (`w`, `b`) que l'entraînement ajuste. |
| **Loss (MSE)** | Mesure d'erreur à minimiser : moyenne des écarts au carré. |
| **Epoch** | Un passage complet sur les données d'entraînement. |
| **Forward pass** | Calcul entrée → sortie du réseau. |
| **Rétropropagation** | Calcul des gradients de la loss par rapport aux poids. |
| **Gradient descent / Adam** | Algorithme qui met à jour les poids pour baisser la loss. |
| **Learning rate** | Taille du pas de mise à jour des poids. |
| **Inférence** | Utiliser le modèle entraîné pour prédire sur de nouvelles entrées. |
| **RMSE / R²** | Métriques de qualité : erreur typique / qualité d'ajustement (1 = parfait). |

---

## 6. Comment lancer

```bash
cd cfd-projects/piml/phase1_surrogate
pip install -r requirements.txt     # torch + neuralfoil + aerosandbox
python src/make_dataset.py          # (re)génère le dataset
python src/train_surrogate.py       # entraîne + évalue + trace
```
