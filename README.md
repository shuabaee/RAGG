# RAGG for Research

A Streamlit RAG chatbot that answers questions **only** from the research PDFs you upload, with a `[Filename, p. X]` citation after each claim and the exact sentence from the PDF as proof. The front end follows the "Ragg for Research" mock-up. The chatbot is in the **Ragg for Research Model** section near the bottom of the page.


## What RAGG does (scope)
| Ask for | What you get |
|---|---|
| A question about a paper | A direct answer with `[File, p. X]` citations, plus an Evidence panel |
| "Show the evidence" | **Answer → Source → Actual Text from PDF → Explanation** |
| Summary | Objective, Methodology, Findings, Limitations, Conclusion |
| Methodology / dataset | Research design, data/participants, algorithms/instruments, procedure, evaluation measures |
| Compare | A table (objective, method, data, key result, limitation) plus labelled interpretation |
| Gaps / limitations | The authors' stated limitations, then clearly labelled *Potential research gap* ideas |
| RRL help | Findings grouped into themes and a suggested outline (not the full RRL text) |
| Follow-ups ("what are its limitations?") | Resolved to the paper discussed in the previous answer |

**Vague or out-of-scope input is handled before any AI call:**
- Greetings, "help", "what can you do" → a scope guide that lists the papers in scope and example questions.
- A question with no topic ("explain", "tell me something") → asks what you want to know, and lists the papers.
- "This paper" when several papers are selected → asks which one (reply "paper 2", a title, or an author).
- A topic not in the papers → `I could not find sufficient information in the uploaded documents to answer this question.`
- Writing a thesis, inventing data or citations, browsing the web, or prompt-injection attempts → a polite refusal plus the scope guide.

## How it works
Upload PDFs → text extracted page by page (PyMuPDF, falls back to pypdf) → split into ~900-character chunks that never cross pages → BM25 keyword index, plus embeddings (Ollama or OpenAI) stored in a local **ChromaDB** (`./chroma_db`, reused across restarts) → keyword and vector results merged with reciprocal rank fusion → the top 20 are **reranked by a FlashRank cross-encoder** (`ms-marco-MiniLM-L-12-v2`, toggle in ⚙ Model settings or `RAGG_RERANK=0`) and the best 6 kept → top passages are sent to the model as numbered sources `S1…Sn` → the model cites `[S#]` tags, which the **app** converts to real filenames and page numbers → every excerpt is checked against the extracted page text. A **✓ Verified verbatim** badge means the exact words were found on that page. If they were not, the app shows the closest real sentence from the page, labelled as such. With PyMuPDF, you can open the page image with the excerpt highlighted.

**Knowledge Spaces:** create notebooks, move papers between them, and click paper cards to select or deselect them. Only selected papers in the active notebook are searched.


## Limits (v1)
- No OCR, so scanned or image-only PDFs are flagged and not read.
- Papers and chat live in the browser session only. They are cleared when the page is refreshed.
- The demo papers are invented samples for testing, not real publications.


## Run the app locally

### Prerequisites
- **Python 3.10 or newer** (the dev container uses 3.11). Check with `python --version`.
- **Git** (optional, only needed to clone the repo).
- **Ollama** (optional, for free local answers) or an **OpenAI API key** (optional, for ChatGPT). With neither, RAGG still runs in extractive mode and shows verbatim passages only.

### 1. Get the code
```bash
git clone https://github.com/shuabaee/RAGG.git
cd RAGG
```
If you downloaded the ZIP instead, unzip it and `cd` into the folder (it is named `RAGG-main`).

### 2. Create a virtual environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
python -m pip install -r requirements.txt
```
This installs Streamlit, PyMuPDF, pypdf, NumPy, requests, ChromaDB and FlashRank.

### 4. Start the app
```bash
python -m streamlit run app.py
```
Streamlit opens <http://localhost:8501> in your browser. If it does not, open that address yourself. To use a different port, add `--server.port 8502`.

### 5. Choose a model provider
Open **Model settings** in the **Ragg for Research Model** section and pick a provider. You do not need to edit any config file or store an API key on disk.

**Option A: local Ollama (free, private)**
1. Install Ollama from <https://ollama.com> and make sure it is running.
2. Pull a chat model and an embedding model:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```
   `nomic-embed-text` is supported: RAGG detects installed embedding models automatically and uses it for vector search. Without an embedding model, RAGG falls back to keyword (BM25) search only.
3. In **Model settings**, choose **Ollama (local)**. The host defaults to `http://localhost:11434`, and installed chat models appear in the **Model** dropdown.

**Option B: ChatGPT via OpenAI**
In **Model settings**, choose **ChatGPT (OpenAI API key)** and paste your key into the **OpenAI API key** box. Leave the model blank to use `gpt-4o-mini`. The key stays in your browser session only and is not saved to disk.

**Option C: no model (extractive mode)**
In **Model settings**, choose **None — extractive mode**. RAGG returns the most relevant verbatim passages with citations and makes no AI calls. If Ollama is not running, RAGG falls back to this mode automatically.

Click **Save settings** after changing anything.

### 6. Try it out
1. Scroll to the **Ragg for Research Model** section near the bottom of the page.
2. Upload your own PDFs (up to 25 MB each), or use the two demo papers in `sample_papers/`.
3. Click the paper cards to select the papers you want searched.
4. Ask a question, for example *"What is the methodology of this paper?"* or *"Compare the selected papers."*

### Optional settings
Everything above can be done from the UI. If you want to change the startup defaults, set these as environment variables or in `.streamlit/secrets.toml` (copy `.streamlit/secrets.toml.example`; the file is git-ignored).

| Setting | Default | What it does |
|---|---|---|
| `RAGG_PROVIDER` | `ollama` | `ollama`, `openai` or `none`. |
| `OLLAMA_HOST` | `http://localhost:11434` | Address of your Ollama server. |
| `OPENAI_BASE_URL` | OpenAI | Base URL for an OpenAI-compatible API (Groq, OpenRouter and so on). |
| `RAGG_MODEL` | first installed Ollama chat model, or `gpt-4o-mini` for OpenAI | Chat model name. |
| `RAGG_EMBED_MODEL` | first installed embedding model | Embedding model name. Without one, RAGG uses keyword search only. |
| `RAGG_RERANK` | `1` | Set to `0` to turn off FlashRank reranking. |
| `RAGG_THINK` | `1` | Ollama only: set to `0` to hide the reasoning of thinking models. |
| `RAGG_CHROMA_DIR` | `./chroma_db` | Where the local vector database is stored. |
| `RAGG_RERANK_CACHE` | `~/.cache/ragg_flashrank` | Where the reranker model is cached. |

Environment variable example:
```bash
# macOS / Linux
RAGG_PROVIDER=none python -m streamlit run app.py

# Windows (PowerShell)
$env:RAGG_PROVIDER = "none"; python -m streamlit run app.py
```

### Run the tests
```bash
python -m pip install pytest
python -m pytest
```
Tests marked `slow` download and run the real FlashRank model. Skip them with `python -m pytest -m "not slow"`.

### Troubleshooting
| Problem | Fix |
|---|---|
| `No module named streamlit` | Activate the virtual environment (step 2) and re-run step 3. |
| Answers are only verbatim passages | RAGG is in extractive mode. Start Ollama, or add an OpenAI key in **Model settings**. |
| Ollama models do not show up | Confirm Ollama is running (`ollama list`) and that the **Ollama Host URL** is correct. |
| First question is slow | FlashRank downloads its reranker model on first use, and Ollama loads the model into memory. Later questions are faster. Turn off **Rerank results** in **Model settings** to skip reranking. |
| A PDF is flagged and not read | It is probably scanned or image-only. RAGG has no OCR, so use a text-based PDF. |
| Stale search results after changing embedding models | Delete the `chroma_db/` folder and re-upload your papers. |
| Port 8501 already in use | Run with `--server.port 8502`, or stop the other Streamlit app. |

### Run in a dev container or Codespace
The repo includes a `.devcontainer/` config (Python 3.11). Opening the project in GitHub Codespaces or VS Code Dev Containers installs the requirements and starts the app on port 8501 automatically.
