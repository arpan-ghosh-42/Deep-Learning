# Project 1 · Handwritten Character Recognition

Two PyTorch CNNs for handwritten input, run on CPU:

| Notebook | Dataset | Model | Saved result |
|---|---|---|---|
| `01_emnist_balanced_cnn.ipynb` | EMNIST Balanced: 47 classes (digits + upper/lower-case letters), 112,800 train / 18,800 test | 2 × (Conv → ReLU → MaxPool) → FC 128 → 47; random-rotation augmentation; Adam, 5 epochs | Test accuracy **86.97%**, plus a 47-class confusion matrix |
| `02_hasyv2_math_symbols_cnn.ipynb` | HASYv2 fold 1, filtered to 19 symbols: `0–9 + - x y z \times \div \sqrt{} \pi` (4,665 train / 519 val / 583 test) | 2 × (Conv → ReLU → MaxPool) → FC 128 → 19; rotation + translation augmentation; Adam, 10 epochs | Test accuracy **95.20%**, plus a confusion matrix and sample predictions |

`HASYv2_math_CNN.pth` holds the trained weights saved by notebook 02.

## Data (not committed)

| Path | How to get it |
|---|---|
| `data/EMNIST/` | Downloaded automatically by `torchvision.datasets.EMNIST(download=True)` in notebook 01. |
| `HASYv2.tar.bz2` → `HASYv2/` | HASYv2 from Zenodo ([10.5281/zenodo.259444](https://doi.org/10.5281/zenodo.259444)). Put the archive in this folder; notebook 02 extracts it if `HASYv2/` is missing. |

The `data/` folder also contains a copy of CIFAR-10 that neither notebook uses.
