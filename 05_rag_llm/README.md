# 05 · RAG and LLM APIs

| Notebook | What it does | Status |
|---|---|---|
| `01_rag_pipeline.ipynb` | Ingestion and retrieval: `TextLoader` / `PyPDFLoader` → `RecursiveCharacterTextSplitter` (500 / 50) → `all-MiniLM-L6-v2` embeddings (384-d) → persistent ChromaDB collection → top-k similarity retrieval. Ends with Groq (`ChatGroq`) setup. | Executed through retrieval: 2 PDFs / 32 pages → 320 chunks indexed. The final LLM cell was not run. |
| `02_rag_pipeline_llm_integration.ipynb` | The same pipeline plus answer generation with OpenAI (`ChatOpenAI`) and Groq (`ChatGroq`) through a `generate_output()` helper. | The saved run stopped at `ModuleNotFoundError: langchain_community`. A complete earlier run is kept in [`archive/recovered_checkpoints/rag_pipeline_llm_integration_executed.ipynb`](../archive/recovered_checkpoints/). |
| `03_llm_apis_groq_openai.ipynb` | LLM API experiments: Groq through the OpenAI-compatible endpoint, a vision prompt, image generation through Pollinations, a small `PersonalAssistant` class (Q&A, image requests, email summarisation), and OpenAI image-API cells. | Executed (Groq and Pollinations outputs saved) |

## Before running

1. **Install:** the `05_rag_llm` section of the root [`requirements.txt`](../requirements.txt).
2. **API keys:** set them as environment variables. No keys are stored in these notebooks.
   - `GROQ_API_KEY`: notebooks 01 and 03
   - `POLLINATIONS_API_KEY`: notebook 03
   - Notebook 02 uses `"paste-your-api-key-here"` placeholders; replace them locally and do not commit the result.
3. **Source documents:** these are **not in the repository**. Create:
   ```
   05_rag_llm/data/Python.txt     # any plain-text file
   05_rag_llm/data/pdfs/*.pdf     # PDFs to index (the saved run indexed 2 PDFs, including "Attention Is All You Need")
   ```
   The Chroma store is written to `data/vector_store/` (ignored by Git).
