# ChromaDB + FlashRank Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist chunk embeddings in ChromaDB and use it as the dense retriever, then rerank the fused top 20 with FlashRank down to 6.

**Architecture:** Two new self-contained modules (`core/vectorstore.py`, `core/reranker.py`) that degrade to "unavailable" instead of raising. `retrieval.search()` gains a `dense_scores` input; `pipeline.py` wires store → search → rerank for the `qa` path only. `app.py` gets a toggle.

**Tech Stack:** Python 3.11 (deploy) / 3.14 (local), Streamlit, numpy, chromadb 1.5.x, FlashRank 0.2.x, pytest.

**Spec:** `docs/superpowers/specs/2026-11-25-rerank-chroma-design.md`

## Global Constraints

- No LangChain.
- Missing/broken `chromadb` or `flashrank` (or failed model download) must never raise to the UI: fall back to today's behaviour plus a one-line notice.
- Chroma path: env `RAGG_CHROMA_DIR`, default `./chroma_db`; cosine space; one collection per embed key `provider:model`, named `ragg_` + first 16 hex of sha1(key).
- Reranker model: `ms-marco-MiniLM-L-12-v2`. Candidates 20 → keep 6.
- Summary / compare / RRL intents unchanged. Sufficiency thresholds unchanged (overlap ≥ 1 or dense ≥ 0.62).
- `LLMConfig.rerank: bool`, default from env `RAGG_RERANK` (default on; "0"/"false"/"off" disable).

## Review Focus

1. Chroma has the paper but chunk text changed (chunker edit) → stale vectors must be replaced, not mixed. Test: `test_stale_text_hash_readds` (Task 1).
2. Query scoped to papers in scope only → other papers' chunks must never appear. Test: `test_query_filters_by_paper` (Task 1).
3. FlashRank import or model load fails → original order returned, notice set, no exception. Test: `test_rerank_fallback_when_unavailable` (Task 2).
4. Chroma returns chunk ids not in the current chunk list (paper removed from scope / stale) → ignored by `search()`. Test: `test_dense_scores_unknown_ids_ignored` (Task 3).
5. Switching provider/embedding model → different collection, no dimension clash. Test: `test_separate_collections_per_key` (Task 1).

---

### Task 1: Chroma vector store

**Files:**
- Create: `core/vectorstore.py`, `tests/test_vectorstore.py`, `tests/conftest.py`
- Modify: `requirements.txt` (+ `chromadb>=1.0`, `flashrank>=0.2`, `pytest` stays dev-only, not added), `.gitignore` (+ `chroma_db/`)

**Interfaces:**
- Produces:
  - `available() -> bool`
  - `has_paper(key: str, paper_id: str, chunks: list[Chunk]) -> bool` — True iff stored ids == chunk ids and every stored `text_hash` matches; on mismatch deletes that paper's records and returns False.
  - `add(key: str, chunks: list[Chunk], vectors: np.ndarray) -> None` (upsert)
  - `query(key: str, qvec: np.ndarray, paper_ids: list[str], n: int = 40) -> dict[str, float]` — chunk_id → cosine similarity (`1 - distance`); `{}` on any error or empty collection.
  - `reset_client_for_tests(path: str) -> None`
- `tests/conftest.py`: fixture `store` that calls `reset_client_for_tests(tmp_path)`; helper `make_chunks(paper_id, texts) -> list[Chunk]`.

- [ ] **Step 1: Write failing tests** — `test_roundtrip_and_reuse` (add 3 chunks with unit vectors, `has_paper` True, `query` with vector equal to chunk 2 returns chunk 2 with score ≈ 1.0), `test_stale_text_hash_readds` (change one chunk's text → `has_paper` False and paper's records gone), `test_query_filters_by_paper` (two papers, query scoped to one → only its ids), `test_separate_collections_per_key` (dim 3 under key A, dim 5 under key B, both query fine).
- [ ] **Step 2: Run** `python -m pytest tests/test_vectorstore.py -v` — expect FAIL (module missing).
- [ ] **Step 3: Implement `core/vectorstore.py`** — lazy `chromadb.PersistentClient`, `get_or_create_collection(name, metadata={"hnsw:space": "cosine"})`; `text_hash` = sha1(text)[:16]; `where={"paper_id": {"$in": ids}}` (single id → `{"paper_id": id}`); `n_results=min(n, collection.count())`. Wrap every Chroma call in try/except → safe default.
- [ ] **Step 4: Run** tests — expect PASS.
- [ ] **Step 5: Commit** `feat(retrieval): add persistent chroma vector store`

### Task 2: FlashRank reranker

**Files:**
- Create: `core/reranker.py`, `tests/test_reranker.py`
- Modify: `core/models.py` (`Hit.rerank: float | None = None`)

**Interfaces:**
- Produces: `rerank(query: str, hits: list[Hit], k: int = 6) -> tuple[list[Hit], str]` — returns top-k hits sorted by rerank score (each `hit.rerank` set) and `""`; on unavailability returns `hits[:k]` unchanged and a notice `"Reranker unavailable (<reason>). Using standard ranking."`. Module-level lazy `_ranker`; `_load_ranker()` is the patch point for tests.

- [ ] **Step 1: Write failing tests** — `test_rerank_reorders` (monkeypatch `_load_ranker` with a fake whose `rerank(req)` scores passages by containing "sample size"; 3 hits, relevant one last → comes first, `rerank` set, notice ""), `test_rerank_fallback_when_unavailable` (`_load_ranker` raises ImportError → original first k, notice startswith "Reranker unavailable"), `test_rerank_real_model` (marked `@pytest.mark.slow`, skips if flashrank missing; real model ranks the limitation sentence above an unrelated one).
- [ ] **Step 2: Run** — expect FAIL.
- [ ] **Step 3: Implement** using `flashrank.Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=<user cache>)` and `RerankRequest(query, passages=[{"id": i, "text": h.chunk.text}])`. Cache a load failure so it is not retried every question.
- [ ] **Step 4: Run** — expect PASS.
- [ ] **Step 5: Commit** `feat(retrieval): add flashrank reranker with safe fallback`

### Task 3: `search()` accepts Chroma scores

**Files:**
- Modify: `core/retrieval.py:78-129`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Produces: `search(chunks, query, k=6, embeddings=None, query_vec=None, section_boost=None, dense_scores: dict[str, float] | None = None) -> list[Hit]`. When `dense_scores` is given it overrides the numpy path: dense rank is among chunks present in the dict; absent chunks get no dense term and `Hit.dense=None`.

- [ ] **Step 1: Write failing tests** — `test_dense_scores_promote_chunk` (chunk with zero keyword overlap but dense 0.9 appears in results with `dense == 0.9`), `test_dense_scores_unknown_ids_ignored` (dict containing an id not in chunks → no error, unaffected results), `test_search_without_dense_unchanged` (same output as before for keyword-only).
- [ ] **Step 2: Run** — expect FAIL on the first.
- [ ] **Step 3: Implement** — build `dense` array only for present ids; rank among them; RRF term `1/(60+rank)` for present ids only.
- [ ] **Step 4: Run** all tests — PASS.
- [ ] **Step 5: Commit** `feat(retrieval): fuse chroma dense scores in hybrid search`

### Task 4: Wire into pipeline + settings toggle

**Files:**
- Modify: `core/llm.py` (`LLMConfig.rerank`), `core/pipeline.py:56-75, 139-179`, `app.py:64-70, 355-360` (toggle + caption, keep `rerank` on Save), `README.md` (short pipeline section)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 1–3 functions.
- `ensure_embeddings`: if `vectorstore.available()` → for each paper not `has_paper(...)`, embed and `add(...)`; set `ix.emb_key = key` and leave `ix.emb=None`. Else: today's code.
- `qa` path: if store available and query vector exists → `dense_scores = vectorstore.query(key, qv, [p.paper_id ...], 40)`; `search(..., k=20 if cfg.rerank else 6, dense_scores=...)`; if `cfg.rerank` → `hits, msg = rerank(r.search_query, hits, 6)`; append msg to `notice`. Sufficiency check runs on the final 6.

- [ ] **Step 1: Write failing tests** — `test_answer_uses_reranker_in_extractive_mode` (provider "none", monkeypatch `reranker._load_ranker` with fake; answer passage order follows fake scores), `test_answer_rerank_off` (`cfg.rerank=False` → fake ranker never called), `test_ensure_embeddings_skips_stored_paper` (monkeypatch `llm.embed` counter + temp store; second call embeds 0 times).
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement** as in Interfaces; app toggle `st.toggle("Rerank results (FlashRank)", value=cfg.rerank, key="set_rerank")`, caption `Vector store: ChromaDB (./chroma_db)` or `Vector store: in-memory`.
- [ ] **Step 4: Run** `python -m pytest -v` — PASS.
- [ ] **Step 5: Commit** `feat(pipeline): use chroma + flashrank in qa retrieval`

### Task 5: Manual check

- [ ] Install deps, run `streamlit run app.py`, upload `sample_papers/*.pdf`, ask a limitations question with rerank on and off; restart app and confirm no re-embedding (Ollama) / no errors (no-model mode).
