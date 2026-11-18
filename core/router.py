"""Decide what kind of request a message is, before any retrieval or model call.

This is what keeps RAGG specific when a user types something vague:
greetings, "help", "what can you do", one-word prompts, or out-of-scope asks
get a clear scope reply instead of a guessed answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .pdf_utils import Paper
from .retrieval import GENERIC, tokenize

INTENTS = [
    ("compare", r"\bcompar|\bversus\b|\bvs\.?\b|\bdiffer|\bsimilarit|\bcontrast|\bboth (papers|studies)|\bacross (the )?(papers|studies)"),
    ("rrl", r"\brrl\b|review of related lit|literature review|related literature|\bthem(e|es|atic)\b|\boutline\b|synthes"),
    ("gaps", r"\bgaps?\b|\blimitation|\bfuture (work|research|stud)|further (stud|research)|\bweakness|\bshortcoming|\brecommendation"),
    ("method", r"\bmethod|\bresearch design|\bapproach\b|\balgorithm|\bdata ?sets?\b|\bdata collection|\bparticipants|\brespondents|\bsample size|\binstrument|\bmetric|\bevaluation measure|\bframework used|\bmodel used|\bhow (was|were|did) .* (done|conducted|measured|collected|trained|evaluated)"),
    ("summary", r"\btell me (about|something about) (this|the|that|these)\b|\bdescribe (this|the|that) (paper|study|article)|\bsummar|\boverview\b|\btl;?dr\b|\bgist\b|\bwhat('?s| is| are) (this|the|these) (paper|study|studies|papers|article)s? about|\bmain (idea|point)s?\b|\babstract\b|\bbreak ?down\b"),
    ("findings", r"\bfinding|\bresults?\b|\bconclu|\boutcome|\bdiscover|\bshow(ed|s)? that|\bperform(ance|ed)\b|\baccuracy\b|\beffect\b"),
]

EVIDENCE_RE = re.compile(r"\b(show|give|provide|with|include)\b.*\b(evidence|proof|source|quote|excerpt|exact text|actual text)|\bverbatim\b|\bwhere (does|did) it say|\bwhich page\b|\bcite\b", re.I)

GREETING_RE = re.compile(r"^\s*(hi+|hello+|hey+|yo|hiya|good (morning|afternoon|evening|day)|greetings|kumusta|musta|sup|what'?s up)\b[\s!.?,]*(ragg|there|po)?[\s!.?]*$", re.I)
THANKS_RE = re.compile(r"^\s*(thanks?|thank you|ty|salamat|ok(ay)?|cool|nice|great|got it|noted)[\s!.?]*(ragg|po)?[\s!.]*$", re.I)
HELP_RE = re.compile(r"\b(what can you do|what do you do|how (do|can) i use|how does (this|ragg|it) work|who are you|what are you|what is ragg|help( me)?$|^help\b|your (features|capabilit|scope)|what (kind|type)s? of questions|example (questions|prompts)|what should i ask|guide me)\b", re.I)
LIST_RE = re.compile(r"\b(what|which|list|show)\b.*\b(papers|documents|files|pdfs|sources)\b.*\b(have|uploaded|loaded|there|available|selected|in scope)\b|^\s*(list|show) (my )?(papers|documents|files|pdfs)\s*$", re.I)

OUT_OF_SCOPE = [
    (re.compile(r"\b(write|make|create|generate|do)\b.{0,25}\b(my|a|an|the|entire|whole|full)\b.{0,25}\b(thesis|research paper|term paper|capstone|essay|chapter \d|manuscript|dissertation|whole paper)\b", re.I),
     "write a complete paper, thesis, or chapter for you to submit as your own work"),
    (re.compile(r"\b(fabricat|make up|invent|fake|falsif)\w*\b.{0,30}\b(data|results?|citations?|references?|sources?|findings|experiment|quotes?)\b", re.I),
     "invent data, results, citations, or quotations"),
    (re.compile(r"\b(search|browse|look up|google)\b.{0,20}\b(the )?(web|internet|online|google|google scholar)\b|\blatest (news|papers|research) (on|about)\b|\bfind (me )?(more|other|new) (papers|studies|sources)\b", re.I),
     "browse the internet or find papers outside your uploads (this version only reads the PDFs you add)"),
    (re.compile(r"\b(ignore|forget|disregard)\b.{0,20}\b(previous|prior|above|your)\b.{0,15}\b(instructions?|rules?|prompt)\b|\b(reveal|show|print)\b.{0,15}\b(system prompt|hidden instructions|api key)\b", re.I),
     "change my rules or reveal hidden instructions or keys"),
    (re.compile(r"\b(weather|stock price|bitcoin|recipe|lyrics|horoscope|lotto|nba|movie times?)\b", re.I),
     "answer general questions that are not about your uploaded papers"),
]

REFER_RE = re.compile(r"\b(it|its|it's|they|their|them|these|those|the (paper|study|authors?|same (paper|study))|this (paper|study|one)|that (paper|study|one|part|finding|method)|above|previous|earlier|you (said|mentioned))\b", re.I)
CONTINUE_RE = re.compile(r"^\s*(and|also|so|then|what about|how about|why|how so|more|elaborate|explain (further|more|that)|continue|go on|what else|tell me more|expand)\b", re.I)
THIS_PAPER_RE = re.compile(r"\b(this|that|the|current|selected) (paper|study|article|document|pdf|research)\b|\bits\b|\bthe authors?\b", re.I)
ORDINAL = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3, "fifth": 4, "5th": 4, "last": -1}


@dataclass
class Route:
    kind: str  # greeting | thanks | help | list | out_of_scope | vague | clarify | rag
    intent: str = "qa"  # qa | summary | method | compare | gaps | rrl | findings
    wants_evidence: bool = False
    target_ids: list[str] = field(default_factory=list)  # papers to restrict to (empty = all in scope)
    search_query: str = ""
    reason: str = ""
    question: str = ""  # the question to answer (may be rebuilt after a clarification)


def detect_intent(q: str) -> str:
    for name, pat in INTENTS:
        if re.search(pat, q, re.I):
            return name
    return "qa"


def match_papers(q: str, papers: list[Paper]) -> list[str]:
    ql = q.lower()
    hits = []
    for i, p in enumerate(papers):
        title_toks = set(tokenize(p.display_title, GENERIC))
        q_toks = set(tokenize(q, GENERIC))
        if title_toks and len(title_toks & q_toks) >= max(2, int(0.5 * len(title_toks))):
            hits.append(p.paper_id)
            continue
        stem = re.sub(r"\.pdf$", "", p.filename, flags=re.I).lower()
        if len(stem) > 3 and stem in ql:
            hits.append(p.paper_id)
            continue
        if p.authors:
            surname = re.split(r",| and |;|&", p.authors)[0].strip().split()
            if surname:
                s = surname[-1].lower().strip(".")
                if len(s) > 2 and re.search(rf"\b{re.escape(s)}\b", ql):
                    hits.append(p.paper_id)
                    continue
    m = re.search(r"\b(paper|study|document|pdf|article)\s*#?\s*(\d)\b", ql)
    if m:
        idx = int(m.group(2)) - 1
        if 0 <= idx < len(papers):
            hits.append(papers[idx].paper_id)
    for word, idx in ORDINAL.items():
        if re.search(rf"\b{word} (paper|study|document|pdf|article|one)\b", ql) and papers:
            hits.append(papers[idx].paper_id)
    return list(dict.fromkeys(hits))


def route(q: str, papers: list[Paper], history: list[dict], focus_ids: list[str]) -> Route:
    q = q.strip()
    if GREETING_RE.match(q):
        return Route("greeting")
    if THANKS_RE.match(q):
        return Route("thanks")
    if HELP_RE.search(q):
        return Route("help")
    for rx, what in OUT_OF_SCOPE:
        if rx.search(q):
            return Route("out_of_scope", reason=what)
    if LIST_RE.search(q):
        return Route("list")

    targets = match_papers(q, papers)
    # reply to "Which one do you mean?" -> answer the earlier question for the chosen paper
    last_bot = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
    if targets and "Which one do you mean?" in last_bot and len(tokenize(q, GENERIC)) <= 4:
        users = [m["content"] for m in history if m["role"] == "user"]
        if users:
            prev_q = users[-1]
            return Route("rag", intent=detect_intent(prev_q), wants_evidence=bool(EVIDENCE_RE.search(prev_q)),
                         target_ids=targets, search_query=prev_q, question=prev_q)

    intent = detect_intent(q)
    wants_evidence = bool(EVIDENCE_RE.search(q))
    topical = tokenize(q, GENERIC)
    has_history = any(m["role"] == "user" for m in history)
    refers_back = has_history and bool(REFER_RE.search(q))
    is_followup = has_history and (refers_back or bool(CONTINUE_RE.search(q)))

    # build the retrieval query (follow-ups borrow the previous question's topic)
    search_q = q
    if is_followup and not targets:
        prev = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        search_q = f"{prev} {q}"
        if focus_ids and intent != "compare":
            targets = list(focus_ids)

    # "this paper" with several papers loaded and nothing to resolve it to -> ask
    if not targets and THIS_PAPER_RE.search(q) and len(papers) > 1 and intent != "compare":
        return Route("clarify", intent=intent, wants_evidence=wants_evidence, reason="which_paper")

    # vague: nothing topical and no recognisable task
    if intent == "qa" and len(topical) == 0 and not targets and not is_followup:
        return Route("vague", reason="no_topic")
    if intent == "qa" and len(topical) == 1 and len(q.split()) <= 2 and not is_followup:
        # a single keyword like "CNN" -> treat as "what do the papers say about CNN?"
        search_q = q

    return Route("rag", intent=intent, wants_evidence=wants_evidence, target_ids=targets, search_query=search_q, question=q)
