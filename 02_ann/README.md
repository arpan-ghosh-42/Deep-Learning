# 02 · Artificial Neural Networks (MLPs)

Feed-forward networks in PyTorch trained on tabular data with a scaled 80/20 train/test split (`random_state=42`).

## `classification/`: Date Fruit varieties (7 classes)

| Notebook | Model | Saved result |
|---|---|---|
| `01_ann_classification_date_fruit.ipynb` | 34 → 64 → 64 → 7, ReLU, Adam, CrossEntropy, 100 epochs | Test accuracy **95.0%** (180 samples) |
| `02_ann_classification_date_fruit_reimplementation.ipynb` | Same architecture, rewritten from scratch | Trains to final loss 0.025; no evaluation cell yet |

Data: `DateFruit_Dataset.csv` (898 rows, 34 shape, colour and texture features, `Class` label).

## `regression/`: power-plant energy output

| Notebook | Model | Saved result |
|---|---|---|
| `01_ann_regression_power_plant.ipynb` | 4 → 6 → 6 → 1, ReLU, Adam, MSE, 100 epochs; saves the best validation model | Test MSE **18.87**, R² **0.934** |
| `02_ann_regression_power_plant_reimplementation.ipynb` | Same task, rewritten from scratch | Test MSE **19.57**, R² **0.932** |

Data: `powerplant_data.csv` (9,568 rows; `AT` temperature, `V` vacuum, `AP` pressure, `RH` humidity → `PE` energy output).
`best_model.pt` is the checkpoint both regression notebooks write and then reload. Each run overwrites it.

Dataset sources are discussed in the [root README](../README.md#datasets-and-licensing).
