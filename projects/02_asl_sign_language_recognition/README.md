# Project 2 · ASL Sign Language Recognition

Classifies static American Sign Language hand signs from images: 36 classes (digits `0–9` and letters `a–z`), 2,515 images (70 per class; 65 for `t`). Built with TensorFlow / Keras on CPU.

| Notebook | Approach | Saved result |
|---|---|---|
| `01_asl_cnn_baseline.ipynb` | `ImageDataGenerator` with an 80/20 split; 2 × (Conv → MaxPool) → Dense 128 → Dropout → softmax; early stopping on validation loss. Includes training curves, a classification report, a confusion matrix and an `ipywidgets` upload-and-predict cell. | Validation accuracy **87.28%** (503 images). Model saved as `asl_cnn_phase1.keras`. |
| `02_asl_cnn_augmented.ipynb` | `tf.data` pipeline with a stratified 70/15/15 train/val/test split; Keras augmentation layers (rotation, zoom, contrast, brightness); 3 × (Conv → BatchNorm → ReLU → MaxPool) → Dense 256; `ModelCheckpoint` + `ReduceLROnPlateau`; a held-out test evaluation; and a helper to predict on your own photo. | **Outputs not saved.** `sign_cnn_phase1.keras` is the checkpoint this notebook writes. Re-run the notebook to reproduce its metrics. |

The second notebook's notes cover known limits: the images are tightly cropped hands on plain backgrounds, `j` and `z` are motion signs, and similar shapes (`m/n/s`, `a/e`, `1/d`, `2/v`) are commonly confused.

## Data (not committed)

Both notebooks read `dataset/asl_dataset/<class>/*.jpeg`, with one sub-folder per class.
**Source and licence are not recorded in this repository.** The file names (e.g. `hand1_b_bot_seg_1_cropped.jpeg`) suggest the Massey University 2D static hand gesture dataset (Barczak et al., 2011). Confirm the source before redistributing.

## Environment

TensorFlow does not support every recent Python release. Notebook 02's header says TensorFlow was installed under Python 3.11, and an earlier autosave of it recorded TensorFlow 2.21 / Keras 3.15 on CPU.
