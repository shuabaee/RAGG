"""Context builder, scope guidelines, and prompt assembly for RAGG."""
from __future__ import annotations

import html
import re

from .chunker import best_sentences
from .models import Chunk, Evidence, Hit, Reply
from .pdf_utils import Paper
from .retrieval import search, tokenize

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

_SEC_NAMES = {
    "abstract": "Abstract", "intro": "Introduction", "related": "Related work", "method": "Methodology", "data": "Data",
    "results": "Results", "discussion": "Discussion", "conclusion": "Conclusion", "limitations": "Limitations",
    "future": "Future work", "front": "Front matter", "references": "References",
}


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
