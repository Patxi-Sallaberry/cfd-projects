# Phase 1 — Surrogate Model (α → Cl, Cd)
*PIML Roadmap · Sep–Nov 2026 · LiU*

A first **surrogate model**: a small neural network (PyTorch MLP) that learns the NACA 0012
polar `α → (Cl, Cd)` from data and predicts it **instantly** — replacing an expensive CFD run
with a learned function. This README is also a **self-contained study sheet**: it explains every
concept and every line of `src/train_surrogate.py`.

> New to PyTorch itself? Read the companion guide [`docs/pytorch_guide.md`](../../docs/pytorch_guide.md)
> (zero → expert), then come back here for the applied walkthrough.

---

## Result

![Surrogate fit — Cl and Cd vs alpha](results/figures/surrogate_fit.png)

| Output | Test RMSE | Test R² |
|---|---|---|
| Cl | 0.0008 | 1.000 |
| Cd | 0.00002 | 1.000 |

The surrogate reproduces the polar on **held-out test points** it never saw during training.

> **Why so perfect?** The data is a smooth, noiseless function (a NeuralFoil/XFOIL polar), so a
> tiny MLP fits it almost exactly. This is a *learning milestone*, not a hard ML problem — the real
> difficulty comes with **noisy/sparse** data and **higher-dimensional** inputs (see Next step).

> ⚠️ **Provenance.** Training data is an XFOIL-class prediction, *not* my own CFD (my Phase 0 Fluent
> run was non-physical — see `../phase0_post_processor`). A clean, cited ground truth is the right
> choice for learning the ML pipeline.

---

# Comprendre le modèle (fiche de révision)

## 1. L'idée en une phrase

On apprend une **fonction** `f : α → (Cl, Cd)` à partir d'exemples, pour prédire les coefficients
aérodynamiques **instantanément**, sans relancer une simulation CFD (lente).

C'est de l'**apprentissage supervisé** (on a les bonnes réponses pour s'entraîner) en
**régression** (la sortie est un nombre continu, pas une catégorie).

```
Sans ML :   α  →  [CFD : des heures]        →  Cl, Cd
Surrogate : α  →  [réseau : millisecondes]  →  Cl, Cd
```

## 2. Où tourne le ML

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
Un laptop suffit ici. On irait sur **Google Colab** (GPU gratuit, navigateur) ou un cluster
seulement pour de **gros** modèles.

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

## 4. Le code ligne par ligne

### Bloc 1 — Les données (entrée → sorties)
```python
X = df[["alpha_deg"]].values.astype("float32")   # entree  (N, 1)
Y = df[["Cl", "Cd"]].values.astype("float32")    # sorties (N, 2)
```
- `X` = la **feature** (entrée) : l'angle α. Forme `(N, 1)` = N exemples, 1 variable.
- `Y` = les **targets** (cibles) : Cl et Cd. Forme `(N, 2)`.
- `float32` = flottants 32 bits, le format standard des réseaux (rapide, assez précis).

### Bloc 2 — Séparation train / test
```python
idx = rng.permutation(N)
n_test = int(0.2 * N)
test_idx, train_idx = idx[:n_test], idx[n_test:]
```
- On mélange les indices et on réserve **20 %** des points pour le **test**.
- **Pourquoi c'est essentiel** : si on notait le modèle sur ce qu'il a appris, il pourrait avoir
  appris **par cœur** sans généraliser (= **surapprentissage / overfitting**). Le test, *jamais vu
  pendant l'entraînement*, mesure la **vraie** capacité de prédiction.

### Bloc 3 — Normalisation
```python
xm, xs = Xtr.mean(0), Xtr.std(0)        # moyenne / ecart-type
ym, ys = Ytr.mean(0), Ytr.std(0)
norm_x = lambda a: (a - xm) / xs        # centrer-reduire
```
- On ramène chaque variable à **moyenne 0, écart-type 1** (standardisation).
- **Pourquoi** : α va de −6 à 16, Cd de 0.006 à 0.07 — des échelles très différentes. La descente
  de gradient converge **bien mieux** quand tout est à la même échelle.
- **Important** : moyenne/écart-type calculés sur le **train uniquement**. Sinon de l'information du
  test "fuiterait" dans l'entraînement (= **data leakage**) et la performance serait faussée.

### Bloc 4 — Le modèle (un MLP)
```python
model = nn.Sequential(
    nn.Linear(1, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, 2),
)
```
- `nn.Linear(1, 64)` = une **couche** de 64 **neurones**. Chaque neurone calcule `y = w·x + b`
  (somme pondérée + biais). Ici 1 entrée → 64 sorties.
- `nn.Tanh()` = **activation** non-linéaire. **Sans elle**, empiler des couches linéaires resterait
  linéaire → incapable d'apprendre une courbe. La non-linéarité permet de modéliser la polaire.
- Architecture `1 → 64 → 64 → 2` ; la dernière couche sort (Cl, Cd).
- **Paramètres** : `1·64+64` + `64·64+64` + `64·2+2` = 128 + 4160 + 130 ≈ **4 400** nombres à régler.
- **Théorème d'approximation universelle** : un MLP avec assez de neurones approxime n'importe quelle
  fonction continue → d'où sa capacité à apprendre la polaire.

### Bloc 4bis — Erreur et optimiseur
```python
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
```
- `MSELoss` = **erreur quadratique moyenne** `moyenne((prédit − vrai)²)`, **le nombre à minimiser**.
- `Adam` = l'**optimiseur** qui modifie les poids. `lr=1e-2` = **learning rate** (taille du pas).

### Bloc 5 — La boucle d'entraînement (le cœur)
```python
for epoch in range(3000):
    opt.zero_grad()             # 1. remet les gradients a zero
    pred = model(Xtr_t)         # 2. forward : predit Cl, Cd
    loss = loss_fn(pred, Ytr_t) # 3. mesure l'erreur
    loss.backward()             # 4. backward : calcule les gradients
    opt.step()                  # 5. met a jour les poids
```
Une **epoch** = un passage. Répété 3000 fois :
1. `zero_grad` : efface les gradients de l'itération précédente.
2. **Forward pass** : on pousse α dans le réseau → prédiction (Cl, Cd).
3. **Loss** : comparaison au vrai → un nombre d'erreur.
4. **Backward = rétropropagation** : PyTorch calcule `∂loss/∂poids` pour chaque poids.
5. `step` : Adam déplace chaque poids dans le sens qui réduit l'erreur.

La `loss` qui descend vers 0 = **preuve que le modèle apprend**.

### Bloc 6 — Évaluation sur le test
```python
model.eval()
with torch.no_grad():
    Pte = denorm_y(model(Xte_t).numpy())
rmse = np.sqrt(np.mean((pred - true) ** 2))
r2   = 1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2)
```
- `torch.no_grad()` : pas de gradients en évaluation → plus rapide.
- `denorm_y` : retour aux vraies unités (Cl, Cd physiques).
- **RMSE** = erreur typique en unités physiques ; **R²** = qualité d'ajustement (**1 = parfait**,
  0 = pas mieux que la moyenne), calculé sur le **test** → mesure la généralisation.

### Bloc 7 — Inférence + figure
```python
a_dense = np.linspace(X.min(), X.max(), 300)...
p_dense = denorm_y(model(torch.tensor(norm_x(a_dense))).numpy())
```
- On demande au modèle de **prédire** sur 300 angles fins = **inférence** : un α → (Cl, Cd)
  immédiatement. On sauvegarde la figure et les **poids** (`results/surrogate_naca0012.pt`).

## 5. Glossaire

| Terme | Définition courte |
|---|---|
| **Apprentissage supervisé** | Apprendre à partir d'exemples *étiquetés* (entrée + bonne réponse). |
| **Régression** | Prédire une valeur *continue* (vs classification = catégorie). |
| **Feature / target** | Variable d'entrée / valeur à prédire. |
| **Train / test** | Données d'apprentissage / réservées pour vérifier la généralisation. |
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
| **RMSE / R²** | Métriques : erreur typique / qualité d'ajustement (1 = parfait). |

---

## Files

```
phase1_surrogate/
├── data/naca0012_surrogate_dataset.csv   # 89 pts, alpha -6..16, Re=4e5 (NeuralFoil)
├── src/make_dataset.py                    # regenerates the dataset
├── src/train_surrogate.py                 # train + evaluate + plot
├── results/figures/surrogate_fit.png      # data vs surrogate
├── results/surrogate_naca0012.pt          # trained weights
└── requirements.txt
```

## How to run

```bash
cd cfd-projects/piml/phase1_surrogate
pip install -r requirements.txt     # torch + neuralfoil + aerosandbox
python src/make_dataset.py          # (re)generate the dataset
python src/train_surrogate.py       # train + evaluate + plot
```

## Next step

- Add **Reynolds** as a second input → 2-D surrogate `(α, Re) → (Cl, Cd)`.
- Then **Phase 2**: physics-informed networks (PINNs) under PDE constraints.
