"""RAGG answer pipeline: route -> retrieve -> generate (or extract) -> verify -> package."""
from __future__ import annotations

import difflib
import html
import re
from dataclasses import dataclass, field

import numpy as np

from . import llm as llm_mod
from .pdf_utils import Paper, squash
from .prompts import EVIDENCE_MODE, INSUFFICIENT, SYSTEM_PROMPT, TASKS
from .retrieval import GENERIC, Chunk, Hit, best_sentences, chunk_paper, search, split_sentences, tokenize
from .router import Route, route

INTENT_WORDS = set(
    tokenize(
        """method methodology methods research design approach algorithm dataset datasets data participants respondents
        sample instrument metric metrics evaluation measures summary summarize overview compare comparison versus differ
        difference differences similarity similarities contrast gap gaps limitation limitations future work further
        recommendation recommendations weakness weaknesses finding findings result results conclusion conclusions outcome
        rrl review related literature theme themes outline synthesis used conducted measured collected trained evaluated""",
        set(),
    )
)

INTENT_QUERY = {
    "method": "methodology method research design dataset data participants algorithm model procedure evaluation metric",
    "gaps": "limitation limitations future work further research recommend weakness",
    "findings": "results findings conclusion show demonstrate outperform significant",
    "summary": "objective aim purpose method results findings conclusion limitation",
    "compare": "objective method dataset results conclusion limitation",
    "rrl": "findings results conclusion contribution",
}
SECTION_BOOST = {
    "method": {"method": 1.7, "data": 1.6},
    "gaps": {"limitations": 2.0, "future": 2.0, "conclusion": 1.4, "discussion": 1.3},
    "findings": {"results": 1.6, "conclusion": 1.5, "abstract": 1.4, "discussion": 1.3},
}


# ----------------------------------------------------------------------------- index
@dataclass
class PaperIndex:
    chunks: list[Chunk]
    emb: np.ndarray | None = None
    emb_key: str = ""  # provider:model used for emb


def build_index(paper: Paper) -> PaperIndex:
    return PaperIndex(chunks=chunk_paper(paper))


def ensure_embeddings(cfg: llm_mod.LLMConfig, indexes: dict[str, PaperIndex], ids: list[str]) -> str | None:
    """Embed chunks for the given papers if the provider supports it. Returns a warning or None."""
    key = f"{cfg.provider}:{cfg.resolved_embed()}"
    if not cfg.enabled or not cfg.resolved_embed():
        return None
    for pid in ids:
        ix = indexes.get(pid)
        if ix is None or (ix.emb is not None and ix.emb_key == key) or not ix.chunks:
            continue
        try:
            ix.emb = llm_mod.embed(cfg, [c.text for c in ix.chunks])
            ix.emb_key = key
        except Exception as e:  # keep working with keyword search
            return f"Semantic embeddings unavailable ({e}). Using keyword search only."
    return None


# ----------------------------------------------------------------------------- result
@dataclass
class Evidence:
    paper_id: str
    filename: str
    page: int
    section: str
    quote: str
    status: str  # verified | closest | retrieved


@dataclass
class Reply:
    content: str  # markdown/html for display
    plain: str  # text kept in chat history for the model
    evidence: list[Evidence] = field(default_factory=list)
    passages: list[Evidence] = field(default_factory=list)
    focus_ids: list[str] = field(default_factory=list)
    mode: str = ""  # scope | ai | extractive | refuse
    notice: str = ""


# ----------------------------------------------------------------------------- scope texts
def _paper_list(papers: list[Paper], limit: int = 8) -> str:
    rows = []
    for i, p in enumerate(papers[:limit], 1):
        rows.append(f"{i}. **{html.escape(p.display_title)}** — {html.escape(p.short_label)} · {p.n_pages} pages")
    if len(papers) > limit:
        rows.append(f"…and {len(papers) - limit} more.")
    return "\n".join(rows)


def suggested_prompts(papers: list[Paper]) -> list[str]:
    if not papers:
        return [
            "What can you do?",
            "What kind of PDFs can I upload?",
        ]
    p = papers[0]
    t = p.display_title
    t = t if len(t) < 60 else t[:57] + "…"
    out = [f"Summarize “{t}”", "What methodology and dataset were used?", "What is the major finding? Show the evidence."]
    if len(papers) > 1:
        out.append("Compare the methods and results of the selected papers")
        out.append("Group these papers into themes for my RRL")
    else:
        out.append("What limitations and research gaps are mentioned?")
    return out


SCOPE_CAN = """**What I can do with your papers**
- Answer questions, with a **[File, p. X]** citation after each claim
- Summarize a paper by objective, method, findings, limitations, and conclusion
- Pull out the research design, datasets, algorithms, and evaluation measures
- Compare two or more papers in a table
- List the limitations the authors state, and suggest *possible* research gaps (clearly labelled)
- Group findings into themes and draft an RRL outline
- Show the **exact text and page** that supports an answer (add “show the evidence”)

**What I won’t do**
- Search the internet or use sources you did not upload
- Invent citations, page numbers, data, results, or quotes
- Write your thesis, paper, or chapter for you to submit
- Read scanned or image-only PDFs (no OCR in this version)"""


def scope_reply(papers: list[Paper], lead: str) -> Reply:
    if papers:
        body = f"{lead}\n\n**Papers in scope ({len(papers)})**\n{_paper_list(papers)}\n\n{SCOPE_CAN}\n\n**Try asking**\n" + "\n".join(
            f"- {s}" for s in suggested_prompts(papers)
        )
    else:
        body = (
            f"{lead}\n\nThere are **no papers in scope yet**. Add a text-based research PDF with **＋ New Paper** "
            f"(or the **＋** button beside the search box), then ask me about it.\n\n{SCOPE_CAN}"
        )
    return Reply(content=body, plain=re.sub(r"\*\*", "", body), mode="scope")


def no_papers_reply() -> Reply:
    return scope_reply(
        [],
        "I answer only from research PDFs you upload, and I don’t have any to search right now.",
    )


# ----------------------------------------------------------------------------- helpers
def _cite_html(filename: str, page: int) -> str:
    return f'<span class="cite">{html.escape(filename)}, p. {page}</span>'


def _cite_plain(filename: str, page: int) -> str:
    return f"[{filename}, p. {page}]"


def verify_quote(quote: str, chunk: Chunk, page_text: str) -> tuple[str, str]:
    """Return (text_to_show, status). Only text that really exists in the PDF page is shown."""
    q = squash(quote).strip(" \"'“”‘’")
    if len(q) < 12:
        return "", "rejected"
    page_sq = squash(page_text)
    if q in page_sq or q.lower() in page_sq.lower():
        # return the original-case text from the page
        i = page_sq.lower().find(q.lower())
        return page_sq[i : i + len(q)], "verified"
    # closest real sentence on that chunk
    best, best_r = "", 0.0
    for s in split_sentences(chunk.text):
        r = difflib.SequenceMatcher(None, q.lower(), s.lower()).ratio()
        if r > best_r:
            best, best_r = s, r
    if best_r >= 0.6:
        return best, "closest"
    return "", "rejected"


def _ev_from_hit(h: Hit, query: str, papers_by_id: dict[str, Paper], n: int = 2) -> Evidence:
    sents = best_sentences(h.chunk.text, query, n=n)
    return Evidence(h.chunk.paper_id, h.chunk.filename, h.chunk.page, h.chunk.section, " ".join(sents), "retrieved")


def _overview_chunks(chunks: list[Chunk], query: str, n: int) -> list[Chunk]:
    """Pick section-diverse chunks for a paper overview (abstract, method, results, conclusion, limitations)."""
    picked: list[Chunk] = []
    if not chunks:
        return picked
    want = ["abstract", "intro", "method", "data", "results", "discussion", "conclusion", "limitations", "future"]
    by_sec: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_sec.setdefault(c.section, []).append(c)
    # first chunk of the paper usually holds title + abstract
    first = [c for c in chunks if c.page == chunks[0].page][:1]
    picked.extend(first)
    for sec in want:
        if sec in by_sec and len(picked) < n:
            hits = search(by_sec[sec], query, k=1)
            c = hits[0].chunk if hits else by_sec[sec][0]
            if c not in picked:
                picked.append(c)
    if len(picked) < n:
        for h in search(chunks, query, k=n):
            if h.chunk not in picked and h.chunk.section != "references":
                picked.append(h.chunk)
            if len(picked) >= n:
                break
    return picked[:n]


# ----------------------------------------------------------------------------- main entry
def answer(
    question: str,
    scope_papers: list[Paper],
    indexes: dict[str, PaperIndex],
    history: list[dict],
    focus_ids: list[str],
    cfg: llm_mod.LLMConfig,
) -> Reply:
    papers_by_id = {p.paper_id: p for p in scope_papers}
    r: Route = route(question, scope_papers, history, [f for f in focus_ids if f in papers_by_id])

    # ---------- non-retrieval replies
    if r.kind == "greeting":
        return scope_reply(scope_papers, "Hi! I’m **RAGG**, a research assistant that answers only from the PDFs in your Knowledge Space.") if scope_papers else no_papers_reply()
    if r.kind == "thanks":
        return Reply("You’re welcome. Ask me anything else about your papers.", "You're welcome.", mode="scope")
    if r.kind == "help":
        return scope_reply(scope_papers, "I’m **RAGG**, a research assistant for the PDFs you upload. Every answer comes from your papers and shows its page.")
    if r.kind == "out_of_scope":
        rep = scope_reply(scope_papers, f"I can’t {r.reason}. That is outside what RAGG does.")
        rep.mode = "refuse"
        return rep
    if not scope_papers:
        return no_papers_reply()
    if r.kind == "list":
        body = f"**Papers in scope ({len(scope_papers)})**\n{_paper_list(scope_papers, 20)}\n\nSelect or deselect papers on the left to change what I search."
        return Reply(body, re.sub(r"\*\*", "", body), mode="scope")
    if r.kind == "vague":
        return scope_reply(
            scope_papers,
            "That question is too broad for me to search. I can only answer from the papers in scope, so please tell me **what** you want to know about them.",
        )
    if r.kind == "clarify":
        body = (
            f"You have **{len(scope_papers)} papers** in scope. Which one do you mean?\n\n{_paper_list(scope_papers)}\n\n"
            "Reply with the number (for example, “paper 2”), part of the title, or the author’s surname. "
            "You can also click one paper on the left so only it is selected."
        )
        return Reply(body, re.sub(r"\*\*", "", body), mode="scope")

    # ---------- retrieval
    if r.question:
        question = r.question
    target_papers = [papers_by_id[i] for i in r.target_ids if i in papers_by_id] or scope_papers
    intent = r.intent
    if intent == "compare" and len(target_papers) < 2:
        if len(scope_papers) >= 2:
            target_papers = scope_papers
        else:
            body = (
                "A comparison needs **at least two papers** in scope, and I only have one:\n\n"
                f"{_paper_list(scope_papers)}\n\nUpload or select another paper, then ask again. "
                "Meanwhile I can summarize this one or list its methods and limitations."
            )
            return Reply(body, re.sub(r"\*\*", "", body), mode="scope")

    notice = ensure_embeddings(cfg, indexes, [p.paper_id for p in target_papers]) or ""
    q_search = r.search_query + " " + INTENT_QUERY.get(intent, "")

    def _qvec():
        if not cfg.enabled or not cfg.resolved_embed():
            return None
        try:
            v = llm_mod.embed(cfg, [r.search_query], is_query=True)
            return v[0] if v is not None else None
        except Exception:
            return None

    selected: list[Chunk] = []
    hits: list[Hit] = []
    if intent in ("summary", "compare", "rrl"):
        cap = {"summary": 4, "compare": 5, "rrl": 8}[intent]
        per = {"summary": 7, "compare": 4, "rrl": 3}[intent]
        if intent == "summary" and len(target_papers) > cap:
            per = 3
        for p in target_papers[:cap]:
            ix = indexes.get(p.paper_id)
            if ix:
                selected.extend(_overview_chunks(ix.chunks, q_search, per))
        hits = [Hit(c, 1.0, 1.0, None, 1) for c in selected]
    else:
        all_chunks: list[Chunk] = []
        all_emb = []
        use_dense = True
        for p in target_papers:
            ix = indexes.get(p.paper_id)
            if not ix:
                continue
            all_chunks.extend(ix.chunks)
            if ix.emb is not None and len(ix.emb) == len(ix.chunks):
                all_emb.append(ix.emb)
            else:
                use_dense = False
        emb = np.vstack(all_emb) if (use_dense and all_emb) else None
        qv = _qvec() if emb is not None else None
        hits = search(all_chunks, q_search, k=6, embeddings=emb, query_vec=qv, section_boost=SECTION_BOOST.get(intent))
        selected = [h.chunk for h in hits]

        # ---------- sufficiency check (missing-evidence handling)
        topical = set(tokenize(question, GENERIC | INTENT_WORDS))
        if r.search_query != question:  # follow-up: topic may come from the previous turn
            topical |= set(tokenize(r.search_query, GENERIC | INTENT_WORDS))
        best_overlap = max((h.overlap for h in hits), default=0)
        best_dense = max((h.dense for h in hits if h.dense is not None), default=None)
        lexical_ok = (not topical) or best_overlap >= 1
        dense_ok = best_dense is not None and best_dense >= 0.62
        if not hits or (not lexical_ok and not dense_ok):
            body = (
                f"{INSUFFICIENT}\n\n"
                f"I searched **{len(target_papers)} paper(s)**: "
                + "; ".join(f"*{html.escape(p.display_title)}*" for p in target_papers[:5])
                + ".\n\nTry using words the paper itself uses, ask about a specific part (method, results, limitations), "
                "or check that the right papers are selected on the left."
            )
            return Reply(body, INSUFFICIENT, mode="refuse")

    if not selected:
        return Reply(INSUFFICIENT, INSUFFICIENT, mode="refuse")

    passages = [_ev_from_hit(h, r.search_query, papers_by_id) for h in hits[:3]] if intent not in ("summary", "compare", "rrl") else []

    if cfg.enabled:
        try:
            rep = _generate(question, r, selected, scope_papers, papers_by_id, history, cfg)
            rep.passages = passages
            rep.notice = notice
            return rep
        except llm_mod.LLMError as e:
            notice = f"The AI model could not answer ({e}). Showing extractive results instead."
        except Exception as e:  # network etc.
            notice = f"The AI model could not be reached ({type(e).__name__}). Showing extractive results instead."

    rep = _extractive(question, r, selected, hits, target_papers, papers_by_id, indexes)
    rep.passages = passages
    rep.notice = notice
    return rep


# ----------------------------------------------------------------------------- generation
def _generate(question, r: Route, chunks: list[Chunk], scope_papers, papers_by_id, history, cfg) -> Reply:
    src_lines = []
    for i, c in enumerate(chunks, 1):
        p = papers_by_id.get(c.paper_id)
        title = p.display_title if p else c.filename
        safe = c.text.replace("</source>", "</ source>")
        src_lines.append(
            f'<source id="S{i}" file="{c.filename}" page="{c.page}" section="{c.section}" paper_title="{title}">\n{safe}\n</source>'
        )
    papers_note = "\n".join(f"- {p.display_title} ({p.filename}; {p.short_label})" for p in scope_papers[:12])
    task = TASKS.get(r.intent, TASKS["qa"])
    if r.wants_evidence:
        task += "\n" + EVIDENCE_MODE
    user_msg = (
        f"Papers currently in scope:\n{papers_note}\n\n"
        f"<sources>\n" + "\n\n".join(src_lines) + "\n</sources>\n\n"
        f"Task: {task}\n\nQuestion: {question}"
    )
    hist = [{"role": m["role"], "content": m["content"][:2500]} for m in history[-6:]]
    raw = llm_mod.chat(cfg, SYSTEM_PROMPT, hist, user_msg)

    part1, _, part2 = raw.partition("### EVIDENCE")
    part1 = part1.strip()

    if INSUFFICIENT[:40].lower() in part1.lower() and len(part1) < len(INSUFFICIENT) + 80:
        return Reply(INSUFFICIENT, INSUFFICIENT, mode="refuse")

    # ---- verify evidence lines
    evidence: list[Evidence] = []
    for line in part2.strip().splitlines():
        m = re.match(r"\s*[-*]?\s*\[?S(\d+)\]?\s*(?:::|:|—|-)\s*(.+)$", line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if not (0 <= idx < len(chunks)):
            continue
        c = chunks[idx]
        p = papers_by_id.get(c.paper_id)
        page_text = p.pages[c.page - 1] if p else c.text
        shown, status = verify_quote(m.group(2), c, page_text)
        if status in ("verified", "closest") and not any(e.quote == shown for e in evidence):
            evidence.append(Evidence(c.paper_id, c.filename, c.page, c.section, shown, status))
        if len(evidence) >= 3:
            break

    # ---- convert [S#] tags into real citations
    cited: list[int] = []

    def _tag(m):
        ids = [int(x) for x in re.findall(r"S(\d+)", m.group(0))]
        out_h, seen = [], set()
        for i in ids:
            if 1 <= i <= len(chunks):
                c = chunks[i - 1]
                key = (c.filename, c.page)
                if key in seen:
                    continue
                seen.add(key)
                cited.append(i)
                out_h.append(_cite_html(c.filename, c.page))
        return " ".join(out_h)

    # ---- flag quotation marks in the answer that are not real text from the sources (before adding HTML)
    all_src = squash(" ".join(c.text for c in chunks)).lower()

    def _chk(m):
        inner = m.group(2)
        if len(inner.split()) < 5 or squash(inner).lower() in all_src:
            return m.group(0)
        return m.group(0) + " \u2063UNVERIFIED\u2063"

    part1 = re.sub(r"([\"“])([^\"“”\n<>]{20,400})([\"”])", _chk, part1)

    tag_re = re.compile(r"\[\s*S\d+(?:\s*[,;]\s*S?\d+)*\s*\](?:\s*\[\s*S\d+(?:\s*[,;]\s*S?\d+)*\s*\])*|\(S\d+(?:\s*,\s*S\d+)*\)")
    content = tag_re.sub(_tag, part1).replace("\u2063UNVERIFIED\u2063", '<span class="unver">unverified quote</span>')
    plain = tag_re.sub(
        lambda m: " ".join(
            _cite_plain(chunks[int(i) - 1].filename, chunks[int(i) - 1].page)
            for i in re.findall(r"S?(\d+)", m.group(0))
            if 1 <= int(i) <= len(chunks)
        ),
        part1,
    ).replace(" \u2063UNVERIFIED\u2063", "")

    # ---- evidence-mode layout: Answer / Source / Actual Text from PDF / Explanation
    if r.wants_evidence:
        if evidence:
            e = evidence[0]
            block = (
                f"\n\n**Source** — {_cite_html(e.filename, e.page)}\n\n"
                f"**Actual Text from PDF** — <span class='quote'>“{html.escape(e.quote)}”</span>"
                + ("" if e.status == "verified" else " <span class='unver'>closest matching sentence on the page</span>")
                + "\n\n"
            )
        else:
            block = "\n\n**Source** — <span class='unver'>page verification unavailable: no excerpt could be matched to the PDF text</span>\n\n"
        parts = re.split(r"(\*\*Explanation\*\*)", content, maxsplit=1)
        content = parts[0].rstrip() + block + "".join(parts[1:]) if len(parts) > 1 else content + block

    focus = []
    for i in cited:
        pid = chunks[i - 1].paper_id
        if pid not in focus:
            focus.append(pid)
    if not focus:
        focus = list(dict.fromkeys(c.paper_id for c in chunks))[:1]

    if not evidence:  # always give the user real text to check
        for i in (cited or [1])[:2]:
            c = chunks[i - 1]
            evidence.append(Evidence(c.paper_id, c.filename, c.page, c.section, " ".join(best_sentences(c.text, question, 2)), "retrieved"))

    return Reply(content=content, plain=plain, evidence=evidence, focus_ids=focus, mode="ai")


# ----------------------------------------------------------------------------- extractive fallback
_SEC_NAMES = {
    "abstract": "Abstract", "intro": "Introduction", "related": "Related work", "method": "Methodology", "data": "Data",
    "results": "Results", "discussion": "Discussion", "conclusion": "Conclusion", "limitations": "Limitations",
    "future": "Future work", "front": "Front matter", "references": "References",
}


def _extractive(question, r: Route, chunks, hits, target_papers, papers_by_id, indexes) -> Reply:
    lines: list[str] = []
    plain: list[str] = []
    evidence: list[Evidence] = []
    head = "*Extractive mode: no AI model is connected, so I am showing the most relevant verbatim text from your papers. Add an API key in ⚙ Model settings for written answers.*\n\n"

    if r.intent in ("summary", "compare", "rrl"):
        heads = {
            "Objective": "objective aim purpose goal this study propose investigate",
            "Methodology": "method methodology design dataset participants algorithm approach",
            "Findings": "results findings show demonstrate achieved significant accuracy",
            "Limitations": "limitation limited future work further research",
            "Conclusion": "conclusion conclude in summary overall",
        }
        for p in target_papers[:5]:
            pcs = [c for c in chunks if c.paper_id == p.paper_id]
            if not pcs:
                continue
            lines.append(f"**{html.escape(p.display_title)}** · {html.escape(p.short_label)}")
            for h, q in heads.items():
                found = search(pcs, q, k=1)
                if found and found[0].overlap > 0:
                    c = found[0].chunk
                    s = best_sentences(c.text, q, 1)[0]
                    lines.append(f"- **{h}:** “{html.escape(s)}” {_cite_html(c.filename, c.page)}")
                    plain.append(f"{h}: {s} {_cite_plain(c.filename, c.page)}")
                    if len(evidence) < 3:
                        evidence.append(Evidence(c.paper_id, c.filename, c.page, c.section, s, "verified"))
                else:
                    lines.append(f"- **{h}:** not found in the retrieved text")
            lines.append("")
        body = head + "\n".join(lines)
        return Reply(body, "\n".join(plain), evidence=evidence, focus_ids=[p.paper_id for p in target_papers[:1]], mode="extractive")

    for h in hits[:3]:
        c = h.chunk
        sents = best_sentences(c.text, r.search_query, 2)
        s = " ".join(sents)
        lines.append(f"- “{html.escape(s)}” {_cite_html(c.filename, c.page)} <span class='sec-tag'>{_SEC_NAMES.get(c.section, c.section)}</span>")
        plain.append(f"{s} {_cite_plain(c.filename, c.page)}")
        evidence.append(Evidence(c.paper_id, c.filename, c.page, c.section, s, "verified"))
    body = head + "**Most relevant passages**\n" + "\n".join(lines)
    focus = list(dict.fromkeys(h.chunk.paper_id for h in hits[:1]))
    return Reply(body, "\n".join(plain), evidence=evidence, focus_ids=focus, mode="extractive")
