# MNIST

Entraînement d'un CNN sur le dataset MNIST (PyTorch, CPU) + interface web pour exploiter le modèle.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Entraînement

```bash
.venv/bin/python train.py --epochs 5 --batch-size 128 --lr 1e-3
```

- Télécharge MNIST dans `data/` (60 000 train / 10 000 test).
- Augmentation légère (translations/rotation aléatoires) + normalisation.
- Sauvegarde le meilleur checkpoint dans `model/mnist_cnn.pt` et l'historique dans `model/mnist_cnn.json`.

## Évaluation

```bash
.venv/bin/python evaluate.py --checkpoint model/mnist_cnn.pt
```

Recharge le checkpoint, recalcule l'exactitude sur le jeu de test et affiche la matrice de confusion + le recall par classe.

## Résultat (5 époques, CPU, 4 threads)

| Époque | train_loss | test_loss | test_acc |
|--------|-----------|-----------|----------|
| 1 | 0.5465 | 0.0497 | 98.40 % |
| 2 | 0.2129 | 0.0337 | 98.84 % |
| 3 | 0.1710 | 0.0297 | 98.98 % |
| 4 | 0.1507 | 0.0288 | 99.03 % |
| 5 | 0.1447 | 0.0257 | **99.08 %** |

**Exactitude finale sur le jeu de test : 99.08 %** (9 908 / 10 000), ~7.5 min sur CPU.

## Modèle

`MNISTNet` : 2 blocs convolutifs (32 → 64 filtres, 3×3, MaxPool, Dropout) + classifieur FC (64·7·7 → 128 → 10). ~431 k paramètres, Adam, lr 1e-3 avec decay par palier.

## Interface web

```bash
.venv/bin/python app.py --host 0.0.0.0 --port 5000
```

Puis ouvre `http://localhost:5000` (dans un Codespace : onglet *Ports* → forward du port 5000).

Fonctionnalités :

- **zone de dessin** à la souris/tactile (pinceau réglable, annuler, effacer, prédiction automatique à la fin du trait) ;
- **prédiction** avec chiffre prédit, confiance et barres de probabilités des 10 classes ;
- **aperçu de l'entrée modèle** (28×28 après recentrage/normalisation côté serveur) ;
- **24 échantillons du jeu de test** cliquables, avec comparaison au vrai label ;
- **historique** des prédictions ;
- détection automatique de la polarité (trait blanc sur fond noir *ou* inverse) et recentrage du chiffre dans un cadre 20×20 comme MNIST.

API :

| Méthode | Route | Description |
|---|---|---|
| GET | `/` | interface de dessin |
| GET | `/api/info` | métadonnées du modèle (époque, exactitude, appareil) |
| GET | `/api/samples` | 24 chiffres du jeu de test (image + vrai label) |
| POST | `/api/predict` | `{"image": "data:image/png;base64,..."}` → `{pred, confidence, probs[10], top3, preview}` |

Codes d'erreur : `400` si image absente, illisible ou sans trait détecté.

Tests de bout en bout (le serveur doit tourner) :

```bash
.venv/bin/python test_api.py --base http://127.0.0.1:5000
```

## Utilisation dans un autre script

```python
import torch
from PIL import Image
from torchvision import transforms

from evaluate import load_model
from train import MNISTNet

model, ckpt = load_model("model/mnist_cnn.pt", torch.device("cpu"))
tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((ckpt["input_mean"],), (ckpt["input_std"],)),
])
img = tf(Image.open("mon_chiffre.png")).unsqueeze(0)
classe = model(img).argmax(dim=1).item()
```

Note : l'entrée doit être un chiffre manuscrit 28×28 en niveaux de gris, fond sombre / trait clair (convention MNIST).
