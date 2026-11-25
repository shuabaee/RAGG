"""RAGG for Research — Streamlit app.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import html
import os
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from core import engine, vectorstore
from core.llm import LLMConfig, ollama_chat_models
from core.pdf_utils import load_pdf, render_page_png
from core.ui import CSS, landing_bottom, landing_top

APP_DIR = Path(__file__).parent
SAMPLE_DIR = APP_DIR / "sample_papers"
MAX_MB = 25

st.set_page_config(page_title="Ragg for Research", page_icon="📚", layout="wide", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- state
def _secret(name: str, default: str = "") -> str:
    try:
        v = st.secrets.get(name, None)
        if v:
            return str(v)
    except Exception:
        pass
    return os.environ.get(name, default)


def init_state():
    s = st.session_state
    s.setdefault("papers", {})  # pid -> Paper
    s.setdefault("indexes", {})  # pid -> PaperIndex
    s.setdefault("selected", [])  # pids highlighted (in scope)
    s.setdefault("notebooks", ["General"])
    s.setdefault("active_nb", "All papers")
    s.setdefault("messages", [])
    s.setdefault("focus_ids", [])
    s.setdefault("uploader_n", 0)
    s.setdefault("pending_q", None)
    s.setdefault("page_cache", {})
    if "llm" not in s:
        oai = _secret("OPENAI_API_KEY")
        prov_env = _secret("RAGG_PROVIDER").lower()
        if prov_env in ("ollama", "openai", "none"):
            provider = prov_env
        elif oai:
            provider = "openai"
        else:
            provider = "ollama"  # default: local Ollama; without it running, RAGG falls back to extractive mode
        key = oai if provider == "openai" else ""
        s.llm = LLMConfig(
            provider=provider,
            api_key=key,
            model=_secret("RAGG_MODEL"),
            base_url=_secret("OLLAMA_HOST") if provider in ("ollama", "llama") else _secret("OPENAI_BASE_URL"),
            embed_model=_secret("RAGG_EMBED_MODEL"),
            rerank=_secret("RAGG_RERANK", "1").strip().lower() not in ("0", "false", "off", "no"),
            think=_secret("RAGG_THINK", "0").strip().lower() in ("1", "true", "on", "yes"),
        )


init_state()
S = st.session_state


# ----------------------------------------------------------------------------- paper ops
def add_pdf(name: str, data: bytes, notebook: str) -> tuple[bool, str]:
    if len(data) > MAX_MB * 1024 * 1024:
        return False, f"{name}: larger than {MAX_MB} MB."
    paper = load_pdf(name, data, notebook)
    if paper.paper_id in S.papers:
        return False, f"{name} is already in your Knowledge Space."
    if paper.status == "error":
        return False, f"{name}: {paper.message}"
    S.papers[paper.paper_id] = paper
    S.indexes[paper.paper_id] = engine.build_index(paper)
    if paper.status == "ready" and paper.paper_id not in S.selected:
        S.selected.append(paper.paper_id)
    if paper.status == "empty":
        return False, f"{name}: {paper.message}"
    n = len(S.indexes[paper.paper_id].chunks)
    return True, f"Added {name} — {paper.n_pages} pages, {n} passages indexed."


def load_demo():
    nb = "Demo"
    if nb not in S.notebooks:
        S.notebooks.append(nb)
    for f in sorted(SAMPLE_DIR.glob("*.pdf")):
        add_pdf(f.name, f.read_bytes(), nb)
    S.active_nb = nb


if S.get("_demo_done") is None and st.query_params.get("demo") == "1":
    load_demo()
    S._demo_done = True
    st.query_params.clear()


def papers_in_nb(nb: str) -> list:
    ps = list(S.papers.values())
    return ps if nb == "All papers" else [p for p in ps if p.notebook == nb]


def scope_papers() -> tuple[list, str]:
    nb_papers = [p for p in papers_in_nb(S.active_nb) if p.status == "ready"]
    chosen = [p for p in nb_papers if p.paper_id in S.selected]
    if chosen:
        return chosen, f"Searching <b>{len(chosen)}</b> selected paper(s) in <b>{html.escape(S.active_nb)}</b>"
    if nb_papers:
        return nb_papers, f"No paper selected — searching all <b>{len(nb_papers)}</b> in <b>{html.escape(S.active_nb)}</b>"
    return [], "No papers in scope yet — add a PDF to start"


def uploader(prefix: str):
    files = st.file_uploader(
        "Research PDFs (text-based)", type=["pdf"], accept_multiple_files=True, key=f"{prefix}_{S.uploader_n}"
    )
    nb_default = S.active_nb if S.active_nb in S.notebooks else S.notebooks[0]
    nb = st.selectbox("Add to notebook", S.notebooks, index=S.notebooks.index(nb_default), key=f"{prefix}_nb_{S.uploader_n}")
    if st.button("Add to Knowledge Space", key=f"{prefix}_go_{S.uploader_n}", type="primary", disabled=not files):
        msgs = []
        with st.spinner("Reading and indexing…"):
            for f in files:
                ok, msg = add_pdf(f.name, f.getvalue(), nb)
                msgs.append(("✅ " if ok else "⚠️ ") + msg)
        S.upload_report = msgs
        S.uploader_n += 1
        st.rerun()


# ----------------------------------------------------------------------------- left: library
def render_library():
    with st.container(key="library"):
        st.markdown("<h3>Ragg for Research</h3><p class='lib-sub'>Select a knowledge source</p>", unsafe_allow_html=True)

        # notebooks
        st.markdown("<div class='lib-label'>Notebooks</div>", unsafe_allow_html=True)
        with st.popover("New notebook", icon=":material/add:"):
            name = st.text_input("Notebook name", key="nb_name", placeholder="e.g. Thesis – CNN studies")
            if st.button("Create", key="nb_create", type="primary"):
                nm = (name or "").strip()
                if nm and nm not in S.notebooks and nm != "All papers":
                    S.notebooks.append(nm)
                    S.active_nb = nm
                    st.rerun()
        names = ["All papers"] + S.notebooks
        nb_box = st.container(height=200, border=False, key="nb_list") if len(names) > 6 else st.container(key="nb_list")
        with nb_box:
            for ni, nb in enumerate(names):
                count = len(papers_in_nb(nb))
                on = nb == S.active_nb
                if st.button(
                    f"{nb} · {count}",
                    key=f"{'nbrowon' if on else 'nbrow'}_{ni}",
                    icon=":material/folder_open:" if on else ":material/folder:",
                    help=nb if len(nb) > 22 else None,
                    width="stretch",
                ):
                    S.active_nb = nb
                    st.rerun()

        # papers
        st.markdown("<div class='lib-label'>Research Papers</div>", unsafe_allow_html=True)
        with st.popover("New paper", icon=":material/add:"):
            uploader("upA")
        if S.get("upload_report"):
            for m in S.upload_report:
                st.caption(m)
            S.upload_report = None

        st.text_input("Search papers", key="paper_search", placeholder="Search by title or author…",
                      label_visibility="collapsed", icon=":material/search:")
        query = S.get("paper_search", "") or ""
        listed = [p for p in papers_in_nb(S.active_nb) if query.lower() in (p.display_title + " " + p.authors + " " + p.filename).lower()]
        with st.container(height=330, border=False, key="paper_list"):
            if not listed:
                st.caption("No papers here yet. Use **New paper** to upload PDFs, or try the demo." if not query else "No title matches your search.")
                if not S.papers:
                    if st.button("Load 2 demo papers", key="demo_btn"):
                        load_demo()
                        st.rerun()
            for p in listed:
                on = p.paper_id in S.selected
                with st.container(key=f"pcard_{'on_' if on else ''}{p.paper_id}"):
                    c1, c2 = st.columns([0.86, 0.14], vertical_alignment="top")
                    with c1:
                        if st.button(
                            p.display_title, key=f"pick_{p.paper_id}",
                            help=f"{p.display_title}\n\nClick to {'remove this paper from' if on else 'add this paper to'} the search scope.",
                        ):
                            if on:
                                S.selected.remove(p.paper_id)
                            else:
                                S.selected.append(p.paper_id)
                            st.rerun()
                        meta = html.escape(p.short_label)
                        warn = f"<div class='pc-warn'>⚠ {html.escape(p.message)}</div>" if p.message else ""
                        st.markdown(f"<div class='pc-meta'>{meta}</div>{warn}", unsafe_allow_html=True)
                    with c2:
                        with st.popover("⋮"):
                            ix = S.indexes.get(p.paper_id)
                            st.markdown(f"**{html.escape(p.display_title)}**")
                            st.caption(f"{p.filename} · {p.n_pages} pages · {len(ix.chunks) if ix else 0} passages · notebook: {p.notebook}")
                            if p.status == "ready" and st.button("Summarize this paper", key=f"sum_{p.paper_id}"):
                                S.selected = [p.paper_id]
                                S.pending_q = f"Summarize the paper “{p.display_title}”."
                                st.rerun()
                            target = st.selectbox("Move to notebook", S.notebooks, index=S.notebooks.index(p.notebook) if p.notebook in S.notebooks else 0, key=f"mv_{p.paper_id}")
                            if target != p.notebook and st.button("Move", key=f"mvgo_{p.paper_id}"):
                                p.notebook = target
                                st.rerun()
                            if st.button("Remove paper", key=f"rm_{p.paper_id}"):
                                S.papers.pop(p.paper_id, None)
                                S.indexes.pop(p.paper_id, None)
                                if p.paper_id in S.selected:
                                    S.selected.remove(p.paper_id)
                                st.rerun()


# ----------------------------------------------------------------------------- right: chat
BADGE = {
    "verified": "<span class='b-ok'>✓ Verified verbatim on this page</span>",
    "closest": "<span class='b-near'>Closest matching sentence on this page</span>",
    "retrieved": "<span class='b-ret'>Retrieved passage (verbatim)</span>",
}
SEC = engine._SEC_NAMES
AVATAR_USER = ":material/person:"
AVATAR_BOT = ":material/auto_stories:"


def _page_png(pid: str, page: int, quote: str):
    key = (pid, page, quote[:80])
    if key not in S.page_cache:
        p = S.papers.get(pid)
        S.page_cache[key] = render_page_png(p.data, page, quote) if p else None
    return S.page_cache[key]


def render_evidence(items: list[dict], mi: int, label: str, kind: str):
    if not items:
        return
    with st.expander(label):
        for ei, e in enumerate(items):
            st.markdown(
                f"<div class='ev-card'><div class='ev-head'>📄 {html.escape(e['filename'])} · p. {e['page']}"
                f" <span class='sec-tag'>{SEC.get(e['section'], e['section'])}</span> {BADGE.get(e['status'], '')}</div>"
                f"<div class='ev-q'>“{html.escape(e['quote'])}”</div></div>",
                unsafe_allow_html=True,
            )
            if e["paper_id"] in S.papers:
                if st.toggle(f"View page {e['page']} of the PDF", key=f"pg_{kind}_{mi}_{ei}"):
                    png = _page_png(e["paper_id"], e["page"], e["quote"])
                    if png:
                        st.image(png, caption=f"{e['filename']} — page {e['page']} (excerpt highlighted where found)")
                    else:
                        st.caption("Page preview unavailable for this file.")


def render_message(mi: int, m: dict):
    avatar = AVATAR_USER if m["role"] == "user" else AVATAR_BOT
    with st.chat_message(m["role"], avatar=avatar):
        if m["role"] == "user":
            st.markdown(m["content"])
            return
        if m.get("reasoning"):  # collapsed by default; click to read how the model got there
            with st.expander("Thought process", icon=":material/psychology:"):
                st.markdown(f"<div class='reasoning'>{html.escape(m['reasoning'])}</div>", unsafe_allow_html=True)
        st.markdown(m["content"], unsafe_allow_html=True)
        if m.get("notice"):
            st.caption("⚠️ " + m["notice"])
        ev = m.get("evidence") or []
        render_evidence(ev, mi, f"Evidence · {len(ev)} excerpt(s) from your PDFs", "ev")
        ps = m.get("passages") or []
        render_evidence(ps, mi, "Top 3 retrieved passages", "ps")


def run_question(q: str):
    q = q.strip()[:2000]
    if not q:
        return
    papers, _ = scope_papers()
    history = [{"role": x["role"], "content": x.get("plain") or x["content"]} for x in S.messages]
    S.messages.append({"role": "user", "content": q, "plain": q})
    try:
        rep = engine.answer(q, papers, S.indexes, history, S.focus_ids, S.llm)
    except Exception as e:  # never crash the page
        rep = engine.Reply(
            content=f"Something went wrong while answering ({type(e).__name__}). Please try again or rephrase.",
            plain="Error.", mode="refuse",
        )
    if rep.focus_ids:
        S.focus_ids = rep.focus_ids
    S.messages.append(
        {
            "role": "assistant",
            "content": rep.content,
            "plain": rep.plain,
            "evidence": [asdict(e) for e in rep.evidence],
            "passages": [asdict(e) for e in rep.passages],
            "notice": rep.notice,
            "mode": rep.mode,
            "reasoning": rep.reasoning,
        }
    )


def model_settings():
    cfg: LLMConfig = S.llm
    with st.popover("Model settings", icon=":material/tune:"):
        st.caption("Keys entered here stay in this browser session only. For a deployed app, put the key in Streamlit **Secrets**.")
        opts = {
            "Ollama (local)": "ollama",
            "ChatGPT (OpenAI API key)": "openai",
            "None — extractive mode": "none",
        }
        cur = next((k for k, v in opts.items() if v in (cfg.provider, "ollama" if cfg.provider == "llama" else "")), "Ollama (local)")
        choice = st.selectbox("Provider", list(opts), index=list(opts).index(cur), key="set_provider")
        prov = opts[choice]
        if prov == "ollama":
            base = st.text_input("Ollama Host URL", value=cfg.base_url or "http://localhost:11434", key="set_base_ollama")
            installed = ollama_chat_models(base)  # cached; embedding-only models are left out
            if installed:
                cur_m = cfg.model if prov in (cfg.provider, "llama") and cfg.model in installed else installed[0]
                idx = installed.index(cur_m) if cur_m in installed else 0
                model = st.selectbox("Model", installed, index=idx, key="set_model_ollama")
            else:
                st.warning("⚠️ No models detected on this Ollama host. Make sure Ollama is running and a model is downloaded (e.g. `ollama run llama3`), or enter a model name manually.")
                model = st.text_input(
                    "Model name",
                    value=cfg.model if prov in (cfg.provider, "llama") else "",
                    key="set_model_ollama",
                    placeholder="No model detected (enter name or leave blank)",
                )
            key = ""
        elif prov == "openai":  # ChatGPT
            key = st.text_input("OpenAI API key", value=cfg.api_key if prov == cfg.provider else "", type="password", key="set_key",
                                placeholder="sk-...")
            model = st.text_input("ChatGPT model (blank = default)", value=cfg.model if prov == cfg.provider else "", key="set_model",
                                  placeholder="gpt-4o-mini")
            base = cfg.base_url if prov == cfg.provider else ""  # OPENAI_BASE_URL secret still honoured
        else:
            key, model, base = "", "", ""

        think = st.toggle(
            "Let the model think first",
            value=cfg.think,
            key="set_think",
            disabled=prov == "none",
            help="Reasoning models (e.g. qwen3, deepseek-r1) reason before answering. Slower; "
                 "their reasoning appears under “Thought process”. Ignored by models that can’t do it.",
        )
        rerank = st.toggle(
            "Rerank results (FlashRank)",
            value=cfg.rerank,
            key="set_rerank",
            help="Fetch 20 candidate passages, then let a small cross-encoder pick the 6 most relevant.",
        )
        st.caption(
            f"Vector store: **ChromaDB** (`{vectorstore.location()}`) — embeddings are saved and reused."
            if vectorstore.available()
            else "Vector store: **in-memory** (ChromaDB not available) — embeddings are rebuilt each session."
        )

        if st.button("Save settings", key="set_save", type="primary"):
            S.llm = LLMConfig(provider=prov, api_key=key.strip(), model=model.strip(), base_url=base.strip(), embed_model=cfg.embed_model, rerank=rerank, think=think)
            for ix in S.indexes.values():  # embeddings must be rebuilt for a new provider
                ix.emb, ix.emb_key, ix.in_store = None, "", False
            st.rerun()


def render_chat():
    with st.container(key="chat_panel"):
        with st.container(key="chat_head"):
            h1, h2, h3 = st.columns([0.6, 0.22, 0.18], vertical_alignment="center")
            with h1:
                dot = "" if S.llm.enabled else " off"
                st.markdown(f"<div class='model-chip'><span class='dot{dot}'></span>{html.escape(S.llm.label())}</div>", unsafe_allow_html=True)
            with h2:
                model_settings()
            with h3:
                if st.button("Clear chat", key="clear_chat", disabled=not S.messages, icon=":material/delete_sweep:"):
                    S.messages, S.focus_ids = [], []
                    st.rerun()

        papers, scope_txt = scope_papers()
        box = st.container(height=540, border=False, key="chat_box")
        with box:
            if not S.messages and not S.pending_q:
                st.markdown(
                    "<div class='empty'><div class='big'>Hi there, I'm Ragg!</div>"
                    "<div class='small'>Ask me to cross-reference papers.</div></div>",
                    unsafe_allow_html=True,
                )
                with st.container(key="chips"):
                    sugg = engine.suggested_prompts(papers)
                    cols = st.columns(2)
                    for i, sq in enumerate(sugg):
                        with cols[i % 2]:
                            if st.button(sq, key=f"chip_{i}", width="stretch"):
                                S.pending_q = sq
                                st.rerun()
            for mi, m in enumerate(S.messages):
                render_message(mi, m)
            if S.pending_q:
                _q, S.pending_q = S.pending_q, None
                with st.chat_message("user", avatar=AVATAR_USER):
                    st.markdown(_q)
                with st.chat_message("assistant", avatar=AVATAR_BOT):
                    with st.spinner("Ragg is reading your sources…"):
                        run_question(_q)
                st.rerun()

        st.markdown(f"<div class='scope-chip'>{scope_txt}</div>", unsafe_allow_html=True)
        with st.container(key="ask_bar"):
            with st.form("ask_form", clear_on_submit=True, border=False):
                c1, c2 = st.columns([0.8, 0.2], vertical_alignment="center")
                with c1:
                    q = st.text_input("Question", placeholder="Type a question...", label_visibility="collapsed")
                with c2:
                    sent = st.form_submit_button("Ask Ragg", type="primary")
        if sent and q and q.strip():
            S.pending_q = q
            st.rerun()


# ----------------------------------------------------------------------------- page
st.markdown(landing_top(), unsafe_allow_html=True)
with st.container(key="model_wrap"):
    left, right = st.columns([1, 2.35], gap="large")
    with left:
        render_library()
    with right:
        render_chat()
st.markdown(landing_bottom(), unsafe_allow_html=True)