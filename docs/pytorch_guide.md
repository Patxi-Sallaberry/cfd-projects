# Guide PyTorch — de zéro à expert

Guide autonome pour maîtriser **PyTorch**, la bibliothèque de deep learning utilisée dans tout le
parcours PIML (Phase 1 surrogate, Phase 2 PINNs). Aucune base requise. À lire dans l'ordre la
première fois, puis à utiliser comme référence.

## Sommaire
1. [C'est quoi PyTorch + installation](#1-cest-quoi-pytorch)
2. [Les tenseurs (la brique de base)](#2-les-tenseurs)
3. [Opérations, shapes et broadcasting](#3-opérations-shapes-et-broadcasting)
4. [CPU / GPU (device)](#4-cpu--gpu)
5. [Autograd : la dérivation automatique](#5-autograd)
6. [Tout à la main : une régression linéaire](#6-tout-à-la-main)
6b. [La descente de gradient en profondeur](#6b-la-descente-de-gradient-en-profondeur)
7. [Construire un modèle : nn.Module](#7-construire-un-modèle)
8. [Catalogue de couches](#8-catalogue-de-couches)
9. [Fonctions d'activation](#9-fonctions-dactivation)
10. [Fonctions de perte (loss)](#10-fonctions-de-perte)
11. [Optimiseurs](#11-optimiseurs)
12. [La boucle d'entraînement canonique](#12-la-boucle-dentraînement)
13. [Les données : Dataset & DataLoader](#13-les-données)
14. [Sur/sous-apprentissage et régularisation](#14-régularisation)
15. [Sauvegarder et recharger un modèle](#15-sauvegarder-recharger)
16. [Entraîner sur GPU](#16-gpu)
17. [Pièges classiques et debugging](#17-pièges-classiques)
18. [Exemple complet de bout en bout](#18-exemple-complet)
19. [Pour aller plus loin](#19-pour-aller-plus-loin)
20. [PINNs — dériver par rapport aux entrées](#20-pinns--dériver-par-rapport-aux-entrées)

---

## 1. C'est quoi PyTorch

PyTorch fait **trois choses** :
1. **Calcul sur tenseurs** (comme NumPy) — mais accéléré sur **GPU**.
2. **Dérivation automatique** (*autograd*) — il calcule tout seul les gradients, indispensable pour
   entraîner des réseaux.
3. **Briques de deep learning** (`torch.nn`) — couches, fonctions de perte, optimiseurs.

Installation (version CPU) :
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
```python
import torch
print(torch.__version__)              # ex. 2.12.1+cpu
print(torch.cuda.is_available())      # True si un GPU NVIDIA est utilisable
```

---

## 2. Les tenseurs

Un **tenseur** = un tableau multidimensionnel de nombres (scalaire, vecteur, matrice, ou plus). C'est
l'unique structure de données de PyTorch.

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])          # vecteur (3,)
b = torch.zeros(2, 3)                       # matrice 2x3 de zeros
c = torch.ones(2, 3)
d = torch.rand(2, 3)                        # uniformes dans [0,1)
e = torch.randn(2, 3)                       # gaussiennes (moyenne 0, ecart-type 1)
f = torch.arange(0, 10, 2)                  # [0,2,4,6,8]
g = torch.linspace(0, 1, 5)                # 5 valeurs de 0 a 1
```

Attributs essentiels :
```python
x = torch.randn(2, 3)
x.shape       # torch.Size([2, 3])  -> la forme
x.dtype       # torch.float32       -> le type
x.device      # cpu                 -> ou il vit (cpu / cuda)
x.ndim        # 2                   -> nombre de dimensions
```

Conversion avec NumPy (très courant) :
```python
import numpy as np
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)          # numpy  -> tensor (memoire partagee)
back = t.numpy()                   # tensor -> numpy
```

> ⚠️ Le `dtype` par défaut des réseaux est `float32`. Si tu pars d'un array NumPy en `float64`,
> convertis : `torch.tensor(arr, dtype=torch.float32)`.

---

## 3. Opérations, shapes et broadcasting

```python
x = torch.randn(3)
y = torch.randn(3)
x + y, x * y, x ** 2, x.exp(), x.mean(), x.sum()   # element-par-element + reductions

A = torch.randn(2, 3)
B = torch.randn(3, 4)
A @ B            # produit matriciel -> (2, 4)
A.T              # transposee
```

**Manipuler les shapes** (compétence n°1 du debugging) :
```python
x = torch.arange(12)          # (12,)
x.reshape(3, 4)               # (3, 4)
x.view(3, 4)                  # idem (memoire contigue)
x.reshape(3, 4).flatten()     # (12,)
t = torch.randn(5)
t.unsqueeze(0)                # (1, 5)  -> ajoute une dimension
t.unsqueeze(1)                # (5, 1)
torch.randn(1, 5).squeeze()   # (5,)    -> enleve les dimensions de taille 1
```

**Broadcasting** : PyTorch aligne automatiquement des shapes compatibles.
```python
A = torch.randn(4, 3)
v = torch.randn(3)            # traite comme (1, 3)
A + v                        # (4, 3) : v ajoute a chaque ligne
```

Convention : pour les réseaux, la **1ʳᵉ dimension est presque toujours le batch** (le nombre
d'exemples). Une entrée de forme `(N, d)` = N exemples de dimension d.

---

## 4. CPU / GPU

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.randn(1000, 1000, device=device)   # cree directement sur le device
y = torch.randn(1000, 1000).to(device)       # deplace un tensor existant
```
Règle d'or : **modèle et données doivent être sur le même device**, sinon erreur. Sur CPU (ton cas
actuel) tout marche pareil, juste plus lentement pour les gros modèles.

---

## 5. Autograd

C'est **la** raison d'être de PyTorch. Si un tenseur a `requires_grad=True`, PyTorch enregistre
toutes les opérations dans un **graphe de calcul**, et `.backward()` calcule les dérivées.

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1          # y = (x+1)^2
y.backward()                    # calcule dy/dx
print(x.grad)                   # 2x + 2 = 8.0
```

Avec un vecteur de paramètres :
```python
w = torch.randn(3, requires_grad=True)
x = torch.tensor([1.0, 2.0, 3.0])
loss = ((w * x).sum() - 10) ** 2
loss.backward()
print(w.grad)                   # d(loss)/dw, un vecteur (3,)
```

Désactiver autograd (en inférence/évaluation → plus rapide, moins de mémoire) :
```python
with torch.no_grad():
    pred = model(x)             # pas de graphe construit
```

### Le graphe de calcul (`grad_fn`)

Quand tu calcules sur un tenseur `requires_grad=True`, PyTorch enregistre chaque opération dans un
**graphe de calcul**. Chaque résultat porte un `grad_fn` (la fonction qui sait le dériver).

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x + 1
print(y.grad_fn)        # <AddBackward0> : derniere op = addition
y.backward()
print(x.grad)           # 2x+3 = 7.0
```
Graphe :
```
x ──► x²  ──┐
x ──► 3·x ──┼──► somme ──► y
       1  ──┘
```
`backward()` parcourt ce graphe **à l'envers** en multipliant les dérivées locales (règle de la
chaîne) et dépose le résultat dans le `.grad` des feuilles. Autograd = la dérivation analytique,
automatisée.

### L'accumulation des gradients (la raison de `zero_grad`)

Chaque `backward()` **additionne** dans `.grad`, il ne remplace pas :
```python
w = torch.tensor(1.0, requires_grad=True)
for i in range(3):
    (3 * w).backward()
    print(w.grad)        # 3.0, puis 6.0, puis 9.0  -> ca s'accumule !
w.grad.zero_()           # -> remis a 0
```
D'où l'obligation de remettre à zéro à **chaque** itération d'entraînement (`w.grad.zero_()` ou
`opt.zero_grad()`). L'accumulation est **voulue** (elle permet de cumuler des gradients sur
plusieurs passes), mais c'est à toi de réinitialiser.

> 🐞 Oublier `zero_grad()` = piège n°1 : les gradients des tours précédents polluent la mise à jour.

### Feuille (leaf) vs non-feuille

```python
a = torch.tensor(2.0, requires_grad=True)   # feuille
b = a * 5                                    # non-feuille (resultat d'op)
print(a.is_leaf, b.is_leaf)                  # True False
```
- **Feuille** : un tenseur créé directement (tes **paramètres**). C'est là que `.grad` est stocké.
- **Non-feuille** : un résultat d'opération (une activation). Possède un `grad_fn` ; son `.grad`
  n'est pas conservé par défaut (économie de mémoire).

### `no_grad` vs `detach`

Les deux coupent le gradient, mais différemment :
| | Portée | Usage |
|---|---|---|
| `with torch.no_grad():` | **contexte** (tout le bloc) | inférence/éval, mise à jour manuelle des poids |
| `x.detach()` | **un tenseur** | stopper le gradient sur un chemin, logger une valeur |

```python
with torch.no_grad():
    y = model(x)            # aucun graphe construit -> rapide, peu de memoire
val = loss.detach().item()  # recuperer une valeur sans trainer le graphe
```

---

## 6. Tout à la main

Pour démystifier : un entraînement complet **sans** `nn` ni optimiseur, juste autograd.

```python
import torch

# donnees : y = 2x + 1 (+ bruit)
X = torch.linspace(-1, 1, 100).unsqueeze(1)
Y = 2 * X + 1 + 0.1 * torch.randn_like(X)

w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.1

for epoch in range(200):
    pred = X * w + b                      # forward
    loss = ((pred - Y) ** 2).mean()       # MSE
    loss.backward()                       # gradients dans w.grad, b.grad
    with torch.no_grad():                 # mise a jour manuelle
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_(); b.grad.zero_()    # remise a zero

print(w.item(), b.item())                 # ~2.0, ~1.0
```
**Tout le deep learning est là.** Les couches `nn` et les optimiseurs ne font qu'automatiser et
généraliser ces 6 lignes.

---

## 6b. La descente de gradient en profondeur

Cette section décortique le "pourquoi" mathématique de la boucle ci-dessus. C'est le noyau à
maîtriser : tout réseau, même à des millions de paramètres, fonctionne **exactement** comme ça.

### Les maths : d'où viennent les gradients

La loss MSE, en fonction des paramètres `w` et `b` :

$$L(w,b) = \frac{1}{N}\sum_{i=1}^{N} \big(\underbrace{w x_i + b}_{\text{pred}_i} - y_i\big)^2$$

En dérivant (règle de la chaîne) on obtient les gradients :

$$\frac{\partial L}{\partial w} = \frac{1}{N}\sum_i 2\,(\text{pred}_i - y_i)\,x_i
\qquad
\frac{\partial L}{\partial b} = \frac{1}{N}\sum_i 2\,(\text{pred}_i - y_i)$$

C'est **exactement** ce que `loss.backward()` calcule et range dans `w.grad` et `b.grad`. Autograd
n'est rien d'autre que cette dérivation, faite automatiquement en parcourant le graphe de calcul à
l'envers (`mean ← carré ← soustraction ← (×, +) ← w, b`).

**Interprétation** : le gradient est la **pente** de l'erreur. S'il est positif, augmenter le
paramètre augmente l'erreur → il faut le **diminuer**. D'où la mise à jour :

$$w \leftarrow w - \text{lr}\cdot\frac{\partial L}{\partial w}$$

Le signe `−` = on va dans le sens **opposé** à la pente = on descend le bol. C'est *descendre une
colline en suivant la plus grande pente*.

### Lire une convergence

Trajectoire typique (données `y = 2x + 1` + bruit 0.1, `lr = 0.1`) :

| epoch | loss | w | b | dL/dw |
|---|---|---|---|---|
| 0 | 1.1736 | 1.323 | 0.000 | −0.46 |
| 20 | 0.0199 | 1.834 | 0.992 | −0.11 |
| 60 | 0.0105 | 1.990 | 1.004 | −0.007 |
| 200 | 0.0105 | 2.000 | 1.004 | ≈ 0 |

À lire : les paramètres glissent vers la vraie solution (2, 1), la loss chute, et les **gradients
tendent vers 0**. Gradient nul = on est au fond du bol = plus de mise à jour = **convergence**.

### Le plancher de bruit (noise floor)

La loss se stabilise à **0.0105**, pas à 0. Pourquoi ? On a ajouté un bruit d'écart-type 0.1, et
`0.1² = 0.01 ≈ 0.0105`. **La loss ne peut pas descendre sous la variance du bruit** : le modèle a
parfaitement appris la loi `2x+1`, le reste est du bruit irréductible.

> Conséquence : **un modèle ne bat jamais le bruit de ses données**. Si tes données sont *sans*
> bruit (ex. une polaire XFOIL propre), la loss peut tendre vers 0 et R² → 1. Avec de vraies mesures
> bruitées, il reste toujours une loss résiduelle — et c'est normal, pas un défaut du modèle.

### L'effet du learning rate (`lr`)

Le `lr` est l'hyperparamètre **n°1**. Même problème, 200 epochs, trois valeurs :

| lr | comportement | après 200 epochs |
|---|---|---|
| 0.001 | pas trop petits → **rampe** | w=1.41, b=0.33, loss=0.58 *(pas convergé)* |
| 0.1 | bien dosé → **converge** | w=2.00, b=1.00, loss=0.0105 ✓ |
| 1.5 | pas trop grands → **diverge** | loss : 1e6 → 1e30 → `inf` → `nan` 💥 |

```
trop petit  ──────  IDÉAL  ──────  trop grand
(rampe)            (converge)       (diverge → nan)
```

- **Trop grand** : on saute par-dessus le minimum et on rebondit de plus en plus haut → explosion.
  Une fois un `nan` apparu, il contamine tout (`nan + x = nan`).
- **Trop petit** : on avance, mais beaucoup trop lentement.
- **Régler en pratique** : pars de `1e-3`. La loss explose → divise le `lr` par 10 ; elle bouge à
  peine → multiplie par 10. (Adam est plus tolérant, mais le principe tient.)

> 🐞 **Réflexe debugging** : loss = `nan` → suspecte d'abord un **`lr` trop grand**.

### Le lien avec les vrais réseaux

Ces 6 lignes faites main = ce que `nn` + optimiseur automatisent :

| À la main (section 6) | Avec `nn` + optimiseur |
|---|---|
| `w`, `b` déclarés à la main | `model.parameters()` (des milliers d'un coup) |
| `pred = X*w + b` | `pred = model(X)` (plusieurs couches) |
| `w -= lr * w.grad` | `opt.step()` (+ Adam adapte le pas) |
| `w.grad.zero_()` | `opt.zero_grad()` |

`loss.backward()` est **identique** dans les deux cas : autograd gère 2 ou 2 millions de paramètres
de la même manière.

---

## 7. Construire un modèle

Deux façons.

**A. `nn.Sequential`** — pour empiler simplement :
```python
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(1, 64), nn.Tanh(),
    nn.Linear(64, 64), nn.Tanh(),
    nn.Linear(64, 2),
)
```

**B. `nn.Module`** — pour tout contrôler (la vraie façon "pro") :
```python
class Surrogate(nn.Module):
    def __init__(self, n_in=1, n_hidden=64, n_out=2):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_out)
        self.act = nn.Tanh()

    def forward(self, x):                 # definit le passage entree -> sortie
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)

model = Surrogate()
```
- `__init__` déclare les couches (qui contiennent les **paramètres**).
- `forward` décrit le **flux de calcul**. On appelle `model(x)` (jamais `model.forward(x)`
  directement).
- `model.parameters()` donne tous les poids ; `sum(p.numel() for p in model.parameters())` les compte.

### Une couche n'est que `w` et `b`

`nn.Linear` ne fait rien de plus que la formule de la section 6 — elle range `w` (`.weight`) et
`b` (`.bias`) pour toi :
```python
lin = nn.Linear(1, 1)
x = torch.tensor([[2.0]])
assert torch.allclose(lin(x), x @ lin.weight.T + lin.bias)   # vrai !
```
Ces poids sont des **`nn.Parameter`** : des **tenseurs-feuilles** (section 5) avec
`requires_grad=True`, qui en plus **s'enregistrent automatiquement** dès qu'on les assigne comme
attribut d'un `nn.Module`. C'est ce qui permet à `model.parameters()` de tous les retrouver, et donc
à `Adam(model.parameters())` de tous les optimiser.

> 📐 `nn.Linear(in, out)` stocke `weight` en forme **`(out, in)`** et calcule `x @ Wᵀ + b`.

### Inspecter un modèle

```python
sum(p.numel() for p in model.parameters())     # nombre total de parametres
for name, p in model.named_parameters():        # liste detaillee
    print(name, tuple(p.shape), p.requires_grad)
# fc1.weight (8,1) | fc1.bias (8,) | fc2.weight (2,8) | fc2.bias (2,)
```
Après un `loss.backward()`, **chaque** paramètre a reçu son `.grad` → prêt pour `opt.step()`.

### `model(x)` vs `model.forward(x)`

Appelle **toujours** `model(x)`. Le `__call__` exécute `forward` **et** d'éventuels *hooks* (utilisés
par certaines fonctionnalités). Appeler `forward` directement les court-circuite.

### `Sequential` ou `Module` : lequel ?

| | Quand l'utiliser |
|---|---|
| `nn.Sequential` | Empilement **linéaire** simple (un MLP). Code court. |
| `nn.Module` (classe) | Dès qu'il faut **plus** : plusieurs entrées, branches, connexions résiduelles, logique conditionnelle. Indispensable pour les **PINNs** (Phase 2). |

Les deux produisent le même modèle à architecture égale (mêmes paramètres). Le surrogate de la
Phase 1 (`1→64→64→2`) est exactement cette mécanique, en plus large — aucune notion nouvelle par
rapport aux sections 5–6.

---

## 8. Catalogue de couches

| Couche | Rôle |
|---|---|
| `nn.Linear(in, out)` | Couche dense (fully-connected) : `y = xW^T + b`. La base des MLP. |
| `nn.Conv2d(...)` | Convolution 2D : images, champs spatiaux (utile en Phase 2). |
| `nn.BatchNorm1d/2d` | Normalise les activations entre couches → entraînement plus stable. |
| `nn.Dropout(p)` | Éteint aléatoirement une fraction `p` de neurones → régularisation. |
| `nn.LSTM / nn.GRU` | Séquences temporelles. |
| `nn.Embedding` | Représentation vectorielle de catégories (NLP). |

Pour un surrogate aéro, on reste surtout sur des `nn.Linear`.

---

## 9. Fonctions d'activation

La non-linéarité qui rend le réseau capable d'apprendre des courbes.

| Activation | Quand l'utiliser |
|---|---|
| `nn.ReLU()` | Le défaut moderne pour les couches cachées (rapide, peu de saturation). |
| `nn.Tanh()` | Sorties lisses bornées dans (−1, 1) ; bien pour des fonctions régulières (polaires, PINNs). |
| `nn.Sigmoid()` | Sortie dans (0, 1) ; probabilités binaires. |
| `nn.GELU() / SiLU()` | Variantes douces de ReLU, fréquentes dans les gros modèles. |
| *(aucune)* en sortie | Pour une **régression**, la dernière couche est **linéaire** (pas d'activation). |

> En PIML, `Tanh` est souvent préféré car ses dérivées sont lisses (les PINNs dérivent le réseau).

---

## 10. Fonctions de perte (loss)

La quantité à minimiser. Choisie selon la tâche.

| Loss | Tâche |
|---|---|
| `nn.MSELoss()` | Régression (erreur quadratique moyenne) — notre cas. |
| `nn.L1Loss()` | Régression robuste aux valeurs aberrantes (erreur absolue). |
| `nn.SmoothL1Loss()` | Compromis MSE/L1 (Huber). |
| `nn.CrossEntropyLoss()` | Classification multi-classes. |
| `nn.BCEWithLogitsLoss()` | Classification binaire. |

```python
loss_fn = nn.MSELoss()
loss = loss_fn(pred, target)     # un scalaire
```

---

## 11. Optimiseurs

L'algorithme qui met à jour les poids à partir des gradients.

| Optimiseur | Note |
|---|---|
| `torch.optim.SGD` | Descente de gradient (option `momentum=0.9`). Simple, robuste. |
| `torch.optim.Adam` | Pas adaptatif par paramètre. **Le défaut** pour démarrer. |
| `torch.optim.AdamW` | Adam avec *weight decay* correct (régularisation). Recommandé. |

```python
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
```
- `lr` (learning rate) = l'hyperparamètre le plus important. Commence à `1e-3`. Trop grand → la loss
  explose/oscille ; trop petit → apprentissage très lent.
- **Scheduler** (optionnel) : baisser le `lr` au fil du temps.
  ```python
  sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)
  # ... appeler sched.step() a chaque epoch
  ```

---

## 12. La boucle d'entraînement

Le squelette **canonique**, à connaître par cœur :

```python
model.train()                          # mode entrainement
for epoch in range(n_epochs):
    opt.zero_grad()                    # 1. remet les gradients a zero
    pred = model(X)                    # 2. forward
    loss = loss_fn(pred, Y)            # 3. erreur
    loss.backward()                    # 4. backward (gradients)
    opt.step()                         # 5. mise a jour des poids
```

Les 5 étapes, toujours dans cet ordre :
1. `zero_grad` — sinon les gradients s'**accumulent** d'une itération à l'autre.
2. `forward` — calcule les prédictions.
3. `loss` — mesure l'écart aux cibles.
4. `backward` — autograd calcule `∂loss/∂poids` partout.
5. `step` — l'optimiseur déplace les poids.

> Oublier `zero_grad()` est **l'erreur n°1** des débutants → gradients faux, entraînement cassé.

### La vraie boucle : avec validation

En pratique on ajoute un **deuxième temps** par epoch : la **validation**, pour surveiller la
généralisation pendant l'entraînement.

```python
train_hist, val_hist = [], []
for epoch in range(n_epochs):
    # --- 1) entrainement ---
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(Xtr), Ytr)
    loss.backward()
    opt.step()
    # --- 2) validation (pas de gradient, pas de mise a jour) ---
    model.eval()
    with torch.no_grad():
        val = loss_fn(model(Xva), Yva)
    train_hist.append(loss.item()); val_hist.append(val.item())
```

### Pourquoi `model.train()` / `model.eval()`

Ils **changent le comportement** de certaines couches : `Dropout` et `BatchNorm` sont **actives en
`train()`**, **neutralisées en `eval()`**. Oublier `eval()` pour valider/prédire → métriques
faussées. (Sur un MLP sans dropout/batchnorm, ça ne change rien, mais c'est une habitude à prendre.)

### La signature de l'overfitting

On surveille **toujours** la `val_loss`, jamais la `train_loss` seule. Exemple réel (gros modèle,
14 points bruités, bruit d'écart-type 0.3) :

| epoch | train_loss | val_loss |
|---|---|---|
| 0 | 0.94 | 4.46 |
| 2000 | 0.0000 | 0.10 |
| 4000 | 0.0000 | 0.10 |

- `train_loss → 0` : le modèle **mémorise** les points d'entraînement, **y compris leur bruit**
  (une loss nulle sur des données bruitées est impossible si on a vraiment appris la tendance).
- `val_loss` bloquée à ≈ 0.10 ≈ `0.3²` : sur des points neufs, il ne bat pas le **noise floor**.
- **L'écart énorme train ≪ val = overfitting.** Un bon ajustement donne `train ≈ val`.

Diagnostic rapide :
```
train ↓  et  val ↓               -> sain
train ↓  et  val plat/↑ (gros ecart) -> OVERFITTING
```
Remèdes (voir §14) : modèle plus petit, plus de données, `weight_decay`, `dropout`, **early stopping**.

---

## 13. Les données

Pour de petits jeux, on passe tout le tableau d'un coup (*full-batch*, comme en Phase 1). Pour de
gros jeux, on découpe en **mini-batches** avec `Dataset` + `DataLoader`.

```python
from torch.utils.data import TensorDataset, DataLoader

ds = TensorDataset(X, Y)                         # associe entrees et cibles
loader = DataLoader(ds, batch_size=32, shuffle=True)

for epoch in range(n_epochs):
    for xb, yb in loader:                        # un mini-batch a la fois
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
```
- **Mini-batch** = un sous-ensemble d'exemples. On met à jour les poids après chaque batch → c'est la
  *descente de gradient stochastique* (SGD). Plus rapide et meilleure généralisation que le full-batch
  sur de grosses données.
- `batch_size` typique : 16–256. `shuffle=True` mélange à chaque epoch.

---

## 14. Régularisation

**Surapprentissage (overfitting)** : le modèle excelle sur le train mais échoue sur le test. On le
détecte en suivant les deux losses : si la train baisse mais la test remonte → overfitting.

Remèdes :
| Technique | Comment |
|---|---|
| **Plus de données** | Le remède le plus efficace. |
| **Weight decay** | `Adam(..., weight_decay=1e-4)` : pénalise les gros poids. |
| **Dropout** | `nn.Dropout(0.1)` entre les couches. Actif en `train()`, ignoré en `eval()`. |
| **Early stopping** | Arrêter quand la loss de validation cesse de baisser. |
| **Modèle plus petit** | Moins de paramètres = moins de capacité à mémoriser. |

> D'où l'importance de `model.train()` / `model.eval()` : ils activent/désactivent dropout et
> batchnorm. **Toujours** passer en `eval()` pour évaluer ou prédire.

### L'idée centrale

Régulariser = **sacrifier un peu d'ajustement sur le train pour gagner en généralisation**. Sur le
cas overfitté du §12 (gros modèle, 14 points, bruit 0.3, noise floor = 0.09) :

| Config | train | val |
|---|---|---|
| Baseline (gros, aucune reg) | **0.0000** | 0.097 |
| + `weight_decay=1e-2` | 0.0569 | **0.0915** |
| + `dropout=0.2` | 0.0291 | 0.128 |
| petit modèle (1-8-1) | 0.0033 | 0.143 |

Le `weight_decay` fait **remonter** la train_loss (il empêche de mémoriser le bruit) et **descendre**
la val_loss au noise floor : l'écart train/val se ferme → modèle sain.

> ⚠️ **La régularisation n'est pas magique** : ici dropout et "petit modèle" ont *empiré* la val
> (dropout perturbe trop un petit réseau ; 8 neurones sous-apprennent). **Le ML est empirique** : on
> essaie, on **valide**, on garde ce qui marche.

### Sous-apprentissage vs sur-apprentissage

```
modèle trop petit          juste            modèle trop gros
(underfit : val haute)   (val minimale)   (overfit : ecart train/val)
```
But : trouver la **capacité** qui minimise la val_loss.

### Early stopping (le réflexe à coder)

```python
best_val, best_state = float("inf"), None
for epoch in range(n_epochs):
    ...                                   # 1) entrainement (train + backward + step)
    val = evaluate()                      # 2) validation
    if val < best_val:                    # 3) on memorise le MEILLEUR modele
        best_val, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)         # 4) on restaure le meilleur (pas le dernier)
```
On garde le modèle au **minimum de la val_loss**, pas celui de la dernière epoch (qui peut déjà
réoverfitter).

---

## 15. Sauvegarder / recharger

La bonne pratique : sauver le **`state_dict`** (les poids), pas l'objet entier.

```python
# sauvegarde
torch.save(model.state_dict(), "model.pt")

# rechargement (il faut recreer la meme architecture)
model = Surrogate()
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()
```

> 🔒 **Sécurité** : utilise **toujours** `weights_only=True`. Par défaut (sur les anciennes versions),
> `torch.load` "dépickle" des objets Python arbitraires → un fichier `.pt` malveillant peut **exécuter
> du code** à l'ouverture. `weights_only=True` n'autorise que des tenseurs/structures simples. Ne charge
> jamais un `.pt` d'origine non fiable sans cette option.

---

## 16. GPU

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)   # donnees sur le meme device que le modele
    ...
```
Rien d'autre ne change. Sur CPU (ton cas), tout fonctionne — c'est juste plus lent sur les gros
modèles. Pour un GPU gratuit : Google Colab.

---

## 17. Pièges classiques

| Symptôme | Cause fréquente |
|---|---|
| Loss qui ne baisse pas | `zero_grad()` oublié ; `lr` trop grand/petit ; données non normalisées. |
| Loss = `NaN` | `lr` trop grand ; division par 0 ; gradients qui explosent. |
| `shape mismatch` | Mauvaise forme d'entrée — affiche `x.shape` partout. |
| `expected ... cpu/cuda` | Modèle et données sur des devices différents. |
| Résultats non reproductibles | Pas de seed → `torch.manual_seed(0)`. |
| Test bien meilleur que prévu | **Data leakage** (normalisation calculée sur tout le jeu). |
| Prédictions étranges en éval | Oubli de `model.eval()` (dropout/batchnorm encore actifs). |

---

## 18. Exemple complet de bout en bout

Tout ce qui précède, condensé :

```python
import torch, torch.nn as nn

torch.manual_seed(0)

# 1. donnees factices : y = sin(x)
X = torch.linspace(-3, 3, 200).unsqueeze(1)
Y = torch.sin(X)

# 2. split train/test
n = len(X); idx = torch.randperm(n)
tr, te = idx[:160], idx[160:]
Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]

# 3. modele
model = nn.Sequential(nn.Linear(1, 64), nn.Tanh(),
                      nn.Linear(64, 64), nn.Tanh(),
                      nn.Linear(64, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

# 4. entrainement
model.train()
for epoch in range(2000):
    opt.zero_grad()
    loss = loss_fn(model(Xtr), Ytr)
    loss.backward()
    opt.step()

# 5. evaluation
model.eval()
with torch.no_grad():
    test_loss = loss_fn(model(Xte), Yte).item()
print("test MSE:", test_loss)

# 6. inference
with torch.no_grad():
    print(model(torch.tensor([[1.5]])).item(), "vs", torch.sin(torch.tensor(1.5)).item())
```

---

## 19. Pour aller plus loin

- **`torchvision` / `torchaudio`** : datasets et modèles prêts (images, audio).
- **CNN** (`Conv2d`) : pour des champs spatiaux — utile quand on passera aux champs de pression/vitesse.
- **PINNs** (Phase 2) : on dérive le réseau par rapport à ses **entrées** (x, y, t) via autograd pour
  injecter les équations de Navier-Stokes dans la loss. Tout ce guide s'y applique directement.
- **Écosystème** : PyTorch Lightning (structurer l'entraînement), Weights & Biases (suivi),
  `torch.compile` (accélération).
- **Doc officielle** : https://pytorch.org/docs et les tutoriels https://pytorch.org/tutorials

> Lien avec le projet : la Phase 1 (`piml/phase1_surrogate/`) applique les sections 7, 10, 11, 12 et
> 15. La Phase 2 (PINNs) est détaillée dans la **section 20** ci-dessous.

---

## 20. PINNs — dériver par rapport aux entrées

Tout ce qui précède différencie la loss **par rapport aux poids** (pour entraîner). Les **PINNs**
(Physics-Informed Neural Networks) ajoutent une idée : différencier aussi la **sortie du réseau par
rapport à ses entrées** (x, t…) pour écrire une équation différentielle, et minimiser son résidu.
Cette section concentre la boîte à outils PINN (utilisée dans `piml/phase2_pinns/`).

### 20.1 Le changement de point de vue

| | Entraînement classique | PINN |
|---|---|---|
| On dérive | la **loss** p/r aux **poids** | aussi la **sortie** p/r aux **entrées** |
| Outil | `loss.backward()` | `torch.autograd.grad(u, x, …)` |
| But | apprendre des données | satisfaire une équation (résidu nul) |

### 20.2 Dériver la sortie par rapport à l'entrée

```python
x = torch.linspace(0, 1, 200).reshape(-1, 1).requires_grad_(True)   # l'entree doit suivre le graphe
u = model(x)
u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
```
- **`.requires_grad_(True)`** sur l'**entrée** : sans ça, pas de dérivée p/r à x.
- **`grad_outputs=torch.ones_like(u)`** : comme `u` est un vecteur, ce seed à 1 donne, pour chaque
  point, `du_i/dx_i` (le réseau agit point par point).
- **`create_graph=True`** : garde le graphe de la dérivée → permet (a) les dérivées d'ordre supérieur,
  (b) la rétropropagation de la loss à travers ces dérivées pour entraîner les poids.
- **`[0]`** : `autograd.grad` renvoie un tuple.

### 20.3 Dérivées d'ordre supérieur et partielles

```python
u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]   # derivee seconde
```
Pour **plusieurs entrées**, on les garde en **tenseurs séparés** et on dérive p/r à chacun :
```python
u   = model(torch.cat([x, t], dim=1))
u_t = torch.autograd.grad(u,   t, torch.ones_like(u),   create_graph=True)[0]   # ∂u/∂t
u_x = torch.autograd.grad(u,   x, torch.ones_like(u),   create_graph=True)[0]   # ∂u/∂x
u_xx= torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]   # ∂²u/∂x²
```

> ⚠️ **`tanh`, pas `relu`** : on prend des dérivées d'ordre 2 ; la dérivée seconde d'un ReLU est nulle
> partout → incapable de représenter une courbure. Toujours une activation **lisse** pour un PINN.

### 20.4 Résidu d'une équation + loss physique

On écrit l'équation littéralement et on minimise son carré moyen, sur des **points de collocation**
(des endroits où l'on impose l'équation — **pas** des données) :
```python
residual  = u_t - alpha * u_xx          # ex. equation de la chaleur : = 0 si satisfaite
loss_phys = (residual ** 2).mean()
```
> **Termes non linéaires** : ils s'écrivent directement. Ex. Burgers :
> `residual = u_t + u*u_x - nu*u_xx` — le produit `u*u_x` suffit, autograd gère les dérivées. Quand
> l'équation **n'a pas de solution analytique**, on valide le PINN contre un **solveur numérique**
> (différences finies). Exemple : `piml/phase2_pinns/src/pinn_burgers.py`.

### 20.5 Conditions initiales / aux limites

L'équation seule a une infinité de solutions (dont la triviale `u ≡ 0`, de résidu nul !). Les
**CI/CL** sélectionnent la bonne, comme des contraintes de **valeur** (pas de dérivée) :
```python
loss_ic = ((model(cat([xi, ti])) - u_target) ** 2).mean()      # u(x,0) impose
loss_bc = (model(cat([x0, tb])) ** 2).mean()                   # u(0,t) = 0
```

### 20.6 Pondérer les termes de loss

Les termes ont souvent des **échelles très différentes** (un résidu en `ω²·u` peut valoir ~10⁴ vs une
CI ~1). On pondère pour les équilibrer, sinon un terme écrase les autres :
```python
loss = W_PHYS * loss_phys + W_IC * loss_ic + W_BC * loss_bc
```
C'est un **réglage central** des PINNs (équilibrage des termes).

### 20.7 Problème inverse : apprendre un paramètre physique

Un paramètre inconnu (viscosité, diffusivité…) devient une **variable entraînable** via `nn.Parameter`,
optimisée **en même temps** que les poids, avec un terme de **données** :
```python
alpha = nn.Parameter(torch.tensor(1.5))                          # inconnu a retrouver
opt = torch.optim.Adam(list(model.parameters()) + [alpha], lr=5e-3)
...
loss = loss_data + 1e-2 * loss_phys      # data tire le champ vers la realite ; physique fixe alpha
```
→ depuis quelques mesures + la physique, on **retrouve** le paramètre et on reconstruit le champ.
C'est *le* vrai usage des PINNs (assimilation de données / inférence de paramètre).

### 20.8 Recette minimale (PINN complet)

```python
model = nn.Sequential(nn.Linear(1,32), nn.Tanh(), nn.Linear(32,32), nn.Tanh(), nn.Linear(32,1))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
t = torch.linspace(0, 1, 200).reshape(-1, 1).requires_grad_(True)
t0 = torch.zeros(1, 1, requires_grad=True)

for epoch in range(20000):
    opt.zero_grad()
    u   = model(t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_tt= torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    loss_phys = ((u_tt + 2*DELTA*u_t + OMEGA0**2 * u) ** 2).mean()   # oscillateur amorti
    u0  = model(t0)
    u0_t= torch.autograd.grad(u0, t0, torch.ones_like(u0), create_graph=True)[0]
    loss_ic = (u0 - 1)**2 + (u0_t - 0)**2
    loss = 1e-4 * loss_phys + loss_ic.squeeze()
    loss.backward(); opt.step()
```

### 20.9 La physique comme régularisation (extrapolation)

Le terme physique ne sert pas qu'à résoudre une équation : il **contraint la solution partout**, même
là où il n'y a **aucune donnée**. Un modèle data-only ne sait pas extrapoler hors de ses données ; en
ajoutant le résidu de l'équation sur des points de collocation couvrant **tout** le domaine, le réseau
**prolonge** une solution physiquement valide dans les zones non mesurées.
```python
# A : donnees seules -> diverge hors des donnees
loss = ((model(Td) - Ud) ** 2).mean()
# B : donnees + physique (collocation sur TOUT le domaine) -> extrapole correctement
loss = ((model(Td) - Ud) ** 2).mean() + W_PHYS * (residual_on(tc) ** 2).mean()
```
Exemple : `piml/phase2_pinns/src/pinn_oscillator_extrapolation.py` (la physique extrapole ~225× mieux
qu'un fit data-only dans la zone sans données).

### 20.10 Systèmes d'équations (plusieurs sorties)

Pour un **système** (ex. Navier-Stokes 2D : 3 équations, sorties `u, v, p`), le réseau a **plusieurs
sorties** qu'on découpe, et la loss **somme un résidu par équation** :
```python
out = model(torch.cat([x, y], 1));  u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
loss_phys = (r_momentum_x ** 2).mean() + (r_momentum_y ** 2).mean() + (r_continuite ** 2).mean()
```
Astuce : imposer une variable au **bord** (ex. la pression) **fixe sa constante** quand l'équation ne
la détermine qu'à une constante près. Exemple complet : `piml/phase2_pinns/src/pinn_navier_stokes.py`
(écoulement de Kovasznay, validé contre la solution exacte de Navier-Stokes).

### Pièges spécifiques aux PINNs

| Symptôme | Cause |
|---|---|
| `grad` renvoie `None` / erreur | entrée sans `requires_grad_(True)`, ou `create_graph` oublié |
| Impossible d'avoir `u_xx` | `create_graph=True` manquant sur la 1ʳᵉ dérivée |
| Courbure non apprise | activation `relu` (dérivée 2de nulle) → utiliser `tanh` |
| Solution triviale `u≡0` | conditions initiales/limites absentes ou trop faibles |
| Un terme domine | loss mal pondérée (ajuster `W_PHYS`, `W_IC`…) |
| Loss basse mais résultat faux | toujours **valider** contre une solution connue |

> Exemples complets dans le repo : `piml/phase2_pinns/` (oscillateur, équation de la chaleur,
> problème inverse), chacun documenté pas à pas dans son README.
