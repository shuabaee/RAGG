# RAGG for Research

A Streamlit RAG chatbot that answers questions **only** from the research PDFs you upload, with a `[Filename, p. X]` citation after each claim and the exact sentence from the PDF as proof. The front end follows the "Ragg for Research" mock-up. The chatbot is in the **Ragg for Research Model** section near the bottom of the page.

## Deploy on Streamlit Community Cloud
1. Unzip this folder and upload **all of its contents** to a new GitHub repository. Keep the folder structure (`app.py`, `core/`, `static/`, `sample_papers/`, `.streamlit/`, `requirements.txt`). Note: `.streamlit` starts with a dot, so make sure it gets uploaded.
2. Go to share.streamlit.io → **Create app** → pick the repo → main file: `app.py` → **Deploy**.
3. Optional but recommended: open the app's **Settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
   You can get a free key at aistudio.google.com/apikey. See `.streamlit/secrets.toml.example` for OpenAI or Groq.

With no key, RAGG still works in **extractive mode**: it shows the most relevant verbatim passages with page numbers, but it does not write prose answers.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

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
Upload PDFs → text extracted page by page (PyMuPDF, falls back to pypdf) → split into ~900-character chunks that never cross pages → BM25 keyword index, plus Gemini/OpenAI embeddings when a key is set (merged with reciprocal rank fusion) → top passages are sent to the model as numbered sources `S1…Sn` → the model cites `[S#]` tags, which the **app** converts to real filenames and page numbers → every excerpt is checked against the extracted page text. A **✓ Verified verbatim** badge means the exact words were found on that page. If they were not, the app shows the closest real sentence from the page, labelled as such. With PyMuPDF, you can open the page image with the excerpt highlighted.

**Knowledge Spaces:** create notebooks, move papers between them, and click paper cards to select or deselect them. Only selected papers in the active notebook are searched.

## Files
```
app.py                 Streamlit page: landing sections, library panel, chat
core/pdf_utils.py      PDF text + metadata extraction, page rendering
core/retrieval.py      chunking, section detection, BM25, hybrid search
core/router.py         intent detection, vague/out-of-scope handling, follow-ups
core/engine.py         retrieval -> generation -> citation + quote verification
core/llm.py            Gemini / OpenAI-compatible chat + embeddings (plain HTTPS)
core/prompts.py        system prompt from the proposal + task templates
core/ui.py             mock-up HTML/CSS
static/                mock-up images
sample_papers/         two FICTIONAL demo PDFs ("Try Interactive Demo")
```

## Limits (v1)
- No OCR, so scanned or image-only PDFs are flagged and not read.
- Papers and chat live in the browser session only. They are cleared when the page is refreshed.
- The demo papers are invented samples for testing, not real publications.
