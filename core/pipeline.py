"""Complete RAG Pipeline Orchestrator for RAGG (Trove-aligned).
Coordinates query routing, hybrid retrieval, sufficiency check, generation, and quote verification.
"""
from __future__ import annotations

import html
import re

import numpy as np

from . import llm as llm_mod
from .chunker import best_sentences, chunk_paper
from .context_builder import (
    INTENT_QUERY,
    INTENT_WORDS,
    SECTION_BOOST,
    _SEC_NAMES,
    _ev_from_hit,
    _overview_chunks,
    _paper_list,
    no_papers_reply,
    scope_reply,
)
from .models import Chunk, Evidence, Hit, PaperIndex, Reply
from .pdf_utils import Paper, squash
from .prompts import EVIDENCE_MODE, INSUFFICIENT, SYSTEM_PROMPT, TASKS
from .retrieval import GENERIC, search, tokenize
from .router import Route, route
from .verifier import _cite_html, _cite_plain, verify_quote


class RAGPipeline:
    """Central orchestrator for indexing, retrieval, and answering in RAGG."""

    def __init__(self, cfg: llm_mod.LLMConfig | None = None):
        self.cfg = cfg

    def build_index(self, paper: Paper) -> PaperIndex:
        """Build chunks and lexical index representation for a paper."""
        return PaperIndex(chunks=chunk_paper(paper))

    def ensure_embeddings(
        self,
        cfg: llm_mod.EmbedConfig,
        indexes: dict[str, PaperIndex],
        ids: list[str],
    ) -> str | None:
        """Embed chunks for the given papers if the provider supports it. Returns a warning or None."""
        key = f"{cfg.provider}:{cfg.resolved_model()}"
        if not cfg.enabled or not cfg.resolved_model():
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

    def answer(
        self,
        question: str,
        scope_papers: list[Paper],
        indexes: dict[str, PaperIndex],
        history: list[dict],
        focus_ids: list[str],
        cfg: llm_mod.LLMConfig | None = None,
        embed_cfg: llm_mod.EmbedConfig | None = None,
    ) -> Reply:
        """Main RAGG answer pipeline: route -> retrieve -> sufficiency check -> generate (or extract) -> verify."""
        active_cfg = cfg or self.cfg
        if active_cfg is None:
            active_cfg = llm_mod.LLMConfig()

        active_embed_cfg = embed_cfg or getattr(self, "embed_cfg", None)
        if active_embed_cfg is None:
            active_embed_cfg = llm_mod.EmbedConfig()

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

        notice = self.ensure_embeddings(active_embed_cfg, indexes, [p.paper_id for p in target_papers]) or ""
        q_search = r.search_query + " " + INTENT_QUERY.get(intent, "")

        def _qvec():
            if not active_embed_cfg.enabled or not active_embed_cfg.resolved_model():
                return None
            try:
                v = llm_mod.embed(active_embed_cfg, [r.search_query], is_query=True)
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

        if active_cfg.enabled:
            try:
                rep = self._generate(question, r, selected, scope_papers, papers_by_id, history, active_cfg)
                rep.passages = passages
                rep.notice = notice
                return rep
            except llm_mod.LLMError as e:
                notice = f"The AI model could not answer ({e}). Showing extractive results instead."
            except Exception as e:  # network etc.
                notice = f"The AI model could not be reached ({type(e).__name__}). Showing extractive results instead."

        rep = self._extractive(question, r, selected, hits, target_papers, papers_by_id, indexes)
        rep.passages = passages
        rep.notice = notice
        return rep

    def _generate(
        self,
        question: str,
        r: Route,
        chunks: list[Chunk],
        scope_papers: list[Paper],
        papers_by_id: dict[str, Paper],
        history: list[dict],
        cfg: llm_mod.LLMConfig,
    ) -> Reply:
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

    def _extractive(
        self,
        question: str,
        r: Route,
        chunks: list[Chunk],
        hits: list[Hit],
        target_papers: list[Paper],
        papers_by_id: dict[str, Paper],
        indexes: dict[str, PaperIndex],
    ) -> Reply:
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


# Global default pipeline instance for procedural calls
_default_pipeline = RAGPipeline()


def build_index(paper: Paper) -> PaperIndex:
    return _default_pipeline.build_index(paper)


def ensure_embeddings(cfg: llm_mod.EmbedConfig, indexes: dict[str, PaperIndex], ids: list[str]) -> str | None:
    return _default_pipeline.ensure_embeddings(cfg, indexes, ids)


def answer(
    question: str,
    scope_papers: list[Paper],
    indexes: dict[str, PaperIndex],
    history: list[dict],
    focus_ids: list[str],
    cfg: llm_mod.LLMConfig,
) -> Reply:
    return _default_pipeline.answer(question, scope_papers, indexes, history, focus_ids, cfg)
