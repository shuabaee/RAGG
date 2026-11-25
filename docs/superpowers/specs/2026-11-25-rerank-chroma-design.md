# RAGG: ChromaDB vector store + FlashRank reranking — design

## Why
Improve answer quality and speed without giving up what makes RAGG better than a stock
LangChain pipeline (routing, sufficiency check, citations, quote verification, extractive fallback).
- **Reranking** fixes the main weakness of fast retrieval: BM25 and bi-encoder embeddings never read
  the question and the chunk together, so the top 6 often contains near-misses.
- **ChromaDB** persists embeddings on disk, so a PDF already embedded is never re-embedded
  (big win with local Ollama), and gives the project a real vector database to describe.

## Goals
- Single-question (`qa`-style) retrieval becomes: BM25 (in-memory) + Chroma dense query → RRF fusion
  → top 20 → FlashRank rerank → best 6 → LLM (or extractive mode).
- Embeddings persist across restarts in `./chroma_db`.
- Reranking can be switched on/off in ⚙ Model settings, so with/without can be compared.

## Non-goals
- No LangChain.
- Summary / compare / RRL intents keep using `_overview_chunks` unchanged.
- Sufficiency-check thresholds unchanged (still BM25 overlap + dense ≥ 0.62).
- No vector cleanup when a paper is removed from the UI — vectors stay as a cache so re-uploading is instant.

## Anti-goals (would be a failure even if it "works")
- The app breaking or erroring when `chromadb` or `flashrank` is missing, or the model download fails.
  Both must degrade silently to today's behaviour (with a one-line notice).
- Answers getting noticeably slower (rerank budget: well under 1 s for 20 chunks on CPU).

## Approaches considered
- **Reranker** — FlashRank (ONNX, no torch) **[picked]** vs sentence-transformers CrossEncoder
  (better quality, but ~700 MB torch; bad fit for Streamlit Cloud) vs using the chat LLM as reranker
  (no dep, but an extra slow/costly call and nothing in no-model mode).
- **Chroma role** — vector store *and* dense retriever **[picked]** vs persistence-only cache with numpy
  search (smaller change, but Chroma does no real work).

## Design
**`core/vectorstore.py` (new)**
- Lazy `chromadb.PersistentClient(path=RAGG_CHROMA_DIR or "./chroma_db")`; import failure → `available() == False`.
- One collection per embedding key (`provider:model`), name = `ragg_` + short hash of the key, cosine space.
- Records: `id=chunk_id`, embedding, `document=chunk text`, metadata `{paper_id, page, section, text_hash}`.
- `has_paper(key, paper_id, chunks)`: true when stored ids and text hashes match the current chunks
  (paper_id is already a content hash; text_hash catches chunker changes). Mismatch → delete that paper's records and re-add.
- `add(key, chunks, vectors)`, `query(key, qvec, paper_ids, n) -> {chunk_id: cosine_similarity}`.

**`core/reranker.py` (new)**
- Lazy FlashRank `Ranker(model_name="ms-marco-MiniLM-L-12-v2")` (~34 MB, downloaded once, cached).
- `rerank(query, hits, k) -> (hits, notice)`; any failure returns the original order + notice.

**`core/retrieval.py`** — `search()` additionally accepts `dense_scores: dict[chunk_id, float]`
(Chroma results). Chunks missing from it get no dense RRF term. Existing numpy `embeddings/query_vec` path kept as fallback.

**`core/models.py`** — `Hit.rerank: float | None = None`; `LLMConfig.rerank: bool = True` (in `llm.py`, env `RAGG_RERANK`).

**`core/pipeline.py`**
- `ensure_embeddings`: if Chroma available → embed only papers not already stored, write to Chroma;
  else today's in-memory `ix.emb`.
- `qa` path: `search(k=20, ...)` → `rerank(..., k=6)` if enabled → rest of pipeline unchanged.

**`app.py`** — toggle "Rerank results (FlashRank)" in Model settings, preserved on Save;
caption showing vector store in use (Chroma / in-memory).

**Housekeeping** — `requirements.txt` += `chromadb`, `flashrank`; `.gitignore` += `chroma_db/`.
Verified both resolve on local Python 3.14 (`chromadb 1.5.9`, `FlashRank 0.2.10`).

## Testing
- Small pytest tests with fake embeddings (no Ollama needed): Chroma round-trip + re-use, stale-hash re-add,
  `search()` with `dense_scores`, reranker fallback when FlashRank is unavailable.
- Manual run on `sample_papers/` with rerank on vs off.

## Left to the builder
Exact candidate count for the Chroma query (≈40), notice wording, test layout.
