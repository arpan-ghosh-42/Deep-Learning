# 04 · RNNs and NLP: IMDB sentiment

| Notebook | What it does | Status |
|---|---|---|
| `01_imdb_text_preprocessing.ipynb` | Cleans 50k IMDB reviews: de-duplication (49,582 rows remain), lowercasing, URL/punctuation/HTML removal, NLTK stop-word removal, Porter stemming and label encoding. | Executed (preprocessing only; no model) |
| `02_imdb_sentiment_rnn_lstm_gru_distilbert.ipynb` | Five-phase sequence-modelling notebook: vocabulary + padding + embeddings; a vanilla RNN vs LSTM vs GRU comparison; BiLSTM; BiLSTM with additive attention; DistilBERT fine-tuning; and a benchmark against TF-IDF + logistic regression. | **Not executed yet.** Set `FAST_RUN = True` for a quick smoke test. A GPU is recommended for the DistilBERT phase. |

**Data:** put `IMDB Dataset.csv` (columns `review`, `sentiment`) in this folder. It is not committed (~66 MB).

**Note on notebook 01:** the stop-word step uses `str.replace` on the whole review, so it also deletes those letters *inside* other words. The saved output shows `"wonderful"` → `"wderful"`. Notebook 02 deliberately keeps stop words and does not stem.

An earlier TF-IDF → RNN version of notebook 02 is kept for reference in [`archive/recovered_checkpoints/`](../archive/recovered_checkpoints/).
