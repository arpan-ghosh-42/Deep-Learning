# 03 · Convolutional Neural Networks: CIFAR-10

Both notebooks train the same PyTorch CNN on CIFAR-10 (10 classes, 32×32 RGB):

```
Conv(3→32) → ReLU → MaxPool → Conv(32→64) → ReLU → MaxPool → Conv(64→128) → ReLU → MaxPool
→ Flatten → Linear(2048→256) → ReLU → Linear(256→10)
```

| Notebook | Status |
|---|---|
| `01_cnn_cifar10.ipynb` | The most recent run was interrupted during training (`KeyboardInterrupt`). An earlier completed 10-epoch run reached **75.4% test accuracy**; it is preserved in [`archive/recovered_checkpoints/cnn_cifar10_completed_run.ipynb`](../archive/recovered_checkpoints/). |
| `02_cnn_cifar10_reimplementation.ipynb` | Same model, rewritten from scratch. Only epoch 1 of 10 was logged (loss 1.365); no evaluation yet. |

**Data:** `01_cnn_cifar10.ipynb` downloads CIFAR-10 to `data/` on first run. `02` expects it to be there already, so run `01` first. `data/` is not committed.
