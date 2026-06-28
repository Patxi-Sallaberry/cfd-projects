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

Points clés :
- Les gradients **s'accumulent** → il faut les remettre à zéro à chaque étape (`zero_grad`).
- `.detach()` sort un tenseur du graphe (utile pour logger une valeur sans gradient).

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
> 15. La Phase 2 (PINNs) ajoutera surtout la section 5 (autograd sur les entrées).
