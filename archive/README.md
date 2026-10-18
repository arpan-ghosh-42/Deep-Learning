# Archive

Material kept for completeness that is not part of the main learning path.

## Scratch notebooks

| Notebook | Contents |
|---|---|
| `python_string_module_basics.ipynb` | Python `string` module constants and a list-comprehension exercise. |
| `openai_client_setup.ipynb` | `pip install openai` and client import. The API key now comes from the `OPENAI_API_KEY` environment variable. |

## `recovered_checkpoints/`

Jupyter autosave copies (`.ipynb_checkpoints`) that held **outputs or code the current notebooks no longer have**. They were renamed and kept here when the checkpoint folders were cleaned up. They are records of earlier runs and are not meant to be re-run from this folder; relative data paths will not resolve here.

| Notebook | What it preserves | Current notebook |
|---|---|---|
| `cnn_cifar10_completed_run.ipynb` | A completed 10-epoch CIFAR-10 training run with **75.4% test accuracy**. The current notebook's last run was interrupted. | `03_cnn/01_cnn_cifar10.ipynb` |
| `imdb_tfidf_rnn_earlier_version.ipynb` | An earlier approach: TF-IDF features fed to a vanilla RNN (later replaced by the sequence-based notebook). Its saved outputs show file paths from a different machine, so its accuracy figure is not reported as a result of this repository. | `04_rnn_nlp/02_imdb_sentiment_rnn_lstm_gru_distilbert.ipynb` |
| `rag_pipeline_llm_integration_executed.ipynb` | A full run of the RAG pipeline including OpenAI and Groq answers. The current notebook's run stopped at an import error. | `05_rag_llm/02_rag_pipeline_llm_integration.ipynb` |
| `river_flood_sense_ai_live_run.ipynb` | A run in which the live-weather fetch succeeded and produced a full flood assessment for Ghatal. The current notebook's last run failed on a network error. | `projects/03_river_flood_sense_ai/River_Flood_Sense_AI.ipynb` |
