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
Upload PDFs → text extracted page by page (PyMuPDF, falls back to pypdf) → split into ~900-character chunks that never cross pages → BM25 keyword index, plus embeddings (Ollama/Gemini/OpenAI) stored in a local **ChromaDB** (`./chroma_db`, reused across restarts) → keyword and vector results merged with reciprocal rank fusion → the top 20 are **reranked by a FlashRank cross-encoder** (`ms-marco-MiniLM-L-12-v2`, toggle in ⚙ Model settings or `RAGG_RERANK=0`) and the best 6 kept → top passages are sent to the model as numbered sources `S1…Sn` → the model cites `[S#]` tags, which the **app** converts to real filenames and page numbers → every excerpt is checked against the extracted page text. A **✓ Verified verbatim** badge means the exact words were found on that page. If they were not, the app shows the closest real sentence from the page, labelled as such. With PyMuPDF, you can open the page image with the excerpt highlighted.

**Knowledge Spaces:** create notebooks, move papers between them, and click paper cards to select or deselect them. Only selected papers in the active notebook are searched.


## Limits (v1)
- No OCR, so scanned or image-only PDFs are flagged and not read.
- Papers and chat live in the browser session only. They are cleared when the page is refreshed.
- The demo papers are invented samples for testing, not real publications.
