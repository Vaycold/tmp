"""
Paper Retrieval Agent (arXiv direct).
- No negative_keywords usage
- Acronym/synonym expansion
- Retry with query relaxation when 0 papers
"""

from __future__ import annotations

import re
import time
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

from rank_bm25 import BM25Okapi

from models import AgentState, Paper
from utils import tokenize
from config import config


_ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _quote(term: str) -> str:
    term = _norm(term)
    if not term:
        return ""
    return f'"{term}"' if " " in term else term


# -----------------------------
# 1) Acronym / synonym expansion
# -----------------------------
_EXPANSION_MAP = {
    # PHM / RUL / SHM
    "shm": ["structural health monitoring", "health monitoring"],
    "structural health monitoring": ["SHM", "health monitoring"],
    "rul": ["remaining useful life", "useful life prediction"],
    "remaining useful life": ["RUL", "useful life prediction"],
    "phm": ["prognostics and health management", "prognostics"],
    "prognostics and health management": ["PHM", "prognostics"],

    # Fault / diagnosis / prognosis
    "fault diagnosis": ["diagnosis", "fault detection"],
    "fault detection": ["anomaly detection", "fault diagnosis"],
    "fault prognosis": ["prognosis", "rul prediction"],

    # PINN
    "pinn": ["physics-informed neural network", "physics informed neural network"],
    "physics-informed neural network": ["PINN", "physics informed neural network"],
    "physics informed neural network": ["PINN", "physics-informed neural network"],

    # Common domain terms
    "condition monitoring": ["health monitoring", "prognostics"],
    "mechanical systems": ["rotating machinery", "mechanical engineering"],
    "rotating machinery": ["bearings", "gearbox", "turbomachinery"],
}


def expand_keywords(keywords: list[str], max_terms: int = 10) -> list[str]:
    """
    Expand keywords with acronyms/synonyms.
    - Keeps order preference: original terms first, then expansions.
    - Caps final size to prevent overly strict queries.
    """
    base = [_norm(k) for k in (keywords or []) if _norm(k)]
    out: list[str] = []
    seen = set()

    def add(t: str):
        t = _norm(t)
        if not t:
            return
        key = t.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(t)

    # add originals
    for k in base:
        add(k)

    # add expansions
    for k in base:
        key = k.lower()
        for e in _EXPANSION_MAP.get(key, []):
            add(e)
        # case-insensitive lookup for keys not normalized in map
        # try a few heuristic variants
        if key not in _EXPANSION_MAP:
            for e in _EXPANSION_MAP.get(key.replace("-", " "), []):
                add(e)

    return out[:max_terms]


# ------------------------------------
# 2) Build multi-level arXiv query tries
# ------------------------------------
def build_arxiv_query_candidates(refined_query: str, keywords: list[str]) -> list[dict]:
    """
    Much more relaxed candidate queries for arXiv.
    Strategy:
      - OR-first (recall), minimal AND
      - Prefer all: over ti/abs for early tries
      - Avoid long phrase quoting
    """
    import re

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    def tokenish(s: str) -> str:
        """avoid quoting long phrases; keep at most 2 words as phrase."""
        s = norm(s)
        if not s:
            return ""
        words = s.split()
        if len(words) <= 2:
            # keep phrase quoting only for <=2 words
            return f'"{s}"' if len(words) == 2 else s
        # for long phrase, take first 2 words to avoid over-restriction
        return f'"{" ".join(words[:2])}"'

    rq = norm(refined_query)
    exp = expand_keywords(keywords, max_terms=12)

    # define core intent terms (PINN/SHM/RUL/PHM/fault...) via expanded list
    # pick top few, but DON'T force AND between them
    core = [e for e in exp[:8] if e]

    # also use a shortened refined query
    rq_short = " ".join(rq.split()[:4]) if rq else ""

    def allf(t: str) -> str:
        return f"all:{tokenish(t)}" if t else ""

    def tiabs(t: str) -> str:
        # use ti/abs only in later "precision" attempts
        return f"(ti:{tokenish(t)} OR abs:{tokenish(t)})" if t else ""

    # OR groups
    core_all_or = " OR ".join([allf(t) for t in core if t])
    core_tiabs_or = " OR ".join([tiabs(t) for t in core if t])

    candidates: list[dict] = []

    # L1: broad recall — (all:core OR ...) AND all:rq_short (optional)
    q1 = f"({core_all_or})" if core_all_or else ""
    if rq_short:
        q1 = f"{q1} AND {allf(rq_short)}" if q1 else allf(rq_short)
    candidates.append({"level": 1, "search_query": q1 or 'all:"research"', "desc": "broad all: OR + optional all:rq_short"})

    # L2: even broader — all:rq_short only
    q2 = allf(rq_short) if rq_short else ""
    candidates.append({"level": 2, "search_query": q2 or 'all:"research"', "desc": "all:rq_short only"})

    # L3: core OR only (all:)
    q3 = f"({core_all_or})" if core_all_or else 'all:"research"'
    candidates.append({"level": 3, "search_query": q3, "desc": "all: core OR only"})

    # L4: mild precision — (ti/abs core OR ...) AND all:rq_short
    q4 = f"({core_tiabs_or})" if core_tiabs_or else ""
    if rq_short:
        q4 = f"{q4} AND {allf(rq_short)}" if q4 else allf(rq_short)
    candidates.append({"level": 4, "search_query": q4 or (allf(rq_short) if rq_short else 'all:"research"'),
                       "desc": "ti/abs core OR + optional all:rq_short"})

    # L5: last-resort single-term fallbacks (very broad)
    # if you have PINN-ish terms, fall back to them
    fallback_terms = []
    for t in exp:
        tl = t.lower()
        if "physics" in tl or "pinn" in tl:
            fallback_terms.append(t)
        if len(fallback_terms) >= 2:
            break

    if fallback_terms:
        q5 = " OR ".join([allf(t) for t in fallback_terms])
        candidates.append({"level": 5, "search_query": q5, "desc": "fallback physics/pinn OR"})
    else:
        candidates.append({"level": 5, "search_query": 'all:"physics-informed" OR all:"PINN"', "desc": "fallback hardcoded"})

    # de-dup
    seen = set()
    uniq = []
    for c in candidates:
        sq = c["search_query"]
        if not sq or sq in seen:
            continue
        seen.add(sq)
        uniq.append(c)
    return uniq


# -----------------------------
# 3) arXiv API fetch + parse
# -----------------------------
def _arxiv_url(search_query: str, start: int, max_results: int,
               sortBy: str = "relevance", sortOrder: str = "descending") -> str:
    params = {
        "search_query": search_query,
        "start": int(start),
        "max_results": int(max_results),
        "sortBy": sortBy,
        "sortOrder": sortOrder,
    }
    return f"{_ARXIV_API}?{urlencode(params)}"


def _parse_atom(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    entries = root.findall("atom:entry", _ATOM_NS)

    out: list[Paper] = []
    for e in entries:
        arxiv_id_url = (e.findtext("atom:id", default="", namespaces=_ATOM_NS) or "").strip()
        arxiv_id = arxiv_id_url.replace("http://arxiv.org/abs/", "").replace("https://arxiv.org/abs/", "").strip()
        if not arxiv_id:
            continue

        title = _norm(e.findtext("atom:title", default="", namespaces=_ATOM_NS))
        abstract = _norm(e.findtext("atom:summary", default="", namespaces=_ATOM_NS))
        if not title or not abstract:
            continue

        authors = []
        for a in e.findall("atom:author", _ATOM_NS):
            name = (a.findtext("atom:name", default="", namespaces=_ATOM_NS) or "").strip()
            if name:
                authors.append(name)

        published = (e.findtext("atom:published", default="", namespaces=_ATOM_NS) or "").strip()
        year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else 0

        url = arxiv_id_url
        for link in e.findall("atom:link", _ATOM_NS):
            if link.attrib.get("rel") == "alternate" and link.attrib.get("href"):
                url = link.attrib["href"]
                break

        out.append(Paper(
            paper_id=f"arxiv:{arxiv_id}",
            title=title,
            abstract=abstract,
            url=url,
            year=year,
            authors=authors
        ))

    return out


def _fetch_arxiv(search_query: str, max_total: int, page_size: int, max_pages: int, errors: list[str]) -> list[Paper]:
    raw: list[Paper] = []

    for page in range(max_pages):
        start = page * page_size
        if start >= max_total:
            break
        url = _arxiv_url(search_query=search_query, start=start, max_results=min(page_size, max_total - start))

        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            batch = _parse_atom(r.text)
            raw.extend(batch)
            if not batch:
                break
            time.sleep(0.3)
        except Exception as e:
            errors.append(f"[arXiv] API call failed: {e}")
            break

    # dedup by paper_id
    uniq = {}
    for p in raw:
        uniq[p.paper_id] = p
    return list(uniq.values())


# -----------------------------
# 4) BM25 selection
# -----------------------------
def _bm25_select(papers: list[Paper], query_text: str, top_k: int) -> tuple[list[Paper], float]:
    if not papers:
        return [], 0.0
    corpus = [tokenize(p.abstract) for p in papers]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query_text))
    pairs = list(zip(papers, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:top_k]
    for p, s in top:
        p.score_bm25 = float(s)
    selected = [p for p, _ in top]
    avg = sum(p.score_bm25 for p in selected) / len(selected) if selected else 0.0
    return selected, avg


# -----------------------------
# Main node
# -----------------------------
def paper_retrieval_node(state: AgentState) -> AgentState:
    """
    arXiv direct retrieval with retry & expansion.
    - Does NOT use negative_keywords
    - Retries with relaxed queries if 0 hits
    """
    if state["iteration"] > 0:
        print(f"\n🔄 Re-running Retrieval (iteration {state['iteration']})")
    print(f"\n📚 Paper Retrieval Node (arXiv direct)")

    if "trace" not in state or state["trace"] is None:
        state["trace"] = {}
    if "errors" not in state or state["errors"] is None:
        state["errors"] = []

    refined_query = state.get("refined_query", "")
    keywords = state.get("keywords", [])

    # trace init
    state["trace"].setdefault("retrieval", {})
    state["trace"]["retrieval"]["arxiv"] = {
        "tries": [],
        "used_level": None,
        "used_search_query": None,
        "raw_count": 0,
        "selected_count": 0,
        "avg_bm25": 0.0,
        "expanded_keywords": expand_keywords(keywords, max_terms=10),
    }

    # pagination params
    max_total = int(getattr(config, "ARXIV_MAX_RESULTS", 60))
    page_size = int(getattr(config, "ARXIV_PAGE_SIZE", 30))
    max_pages = int(getattr(config, "ARXIV_MAX_PAGES", 3))

    # BM25 query text: refined_query + expanded keywords 일부
    exp_kws = state["trace"]["retrieval"]["arxiv"]["expanded_keywords"]
    bm25_query_text = _norm(" ".join([refined_query] + exp_kws[:6]))

    # 1) 후보 쿼리 생성 (strict -> relaxed)
    candidates = build_arxiv_query_candidates(refined_query, exp_kws)

    # 2) 0이면 재시도
    raw: list[Paper] = []
    used = None

    for cand in candidates:
        level = cand["level"]
        sq = cand["search_query"]
        desc = cand["desc"]

        raw = _fetch_arxiv(sq, max_total=max_total, page_size=page_size, max_pages=max_pages, errors=state["errors"])

        state["trace"]["retrieval"]["arxiv"]["tries"].append({
            "level": level,
            "desc": desc,
            "search_query": sq,
            "raw_count": len(raw),
        })

        if raw:
            used = cand
            break

    # 기록
    if used:
        state["trace"]["retrieval"]["arxiv"]["used_level"] = used["level"]
        state["trace"]["retrieval"]["arxiv"]["used_search_query"] = used["search_query"]
    else:
        state["trace"]["retrieval"]["arxiv"]["used_level"] = None
        state["trace"]["retrieval"]["arxiv"]["used_search_query"] = None

    state["trace"]["retrieval"]["arxiv"]["raw_count"] = len(raw)

    if not raw:
        print("  ⚠️ No papers found after retries")
        state["papers"] = []
        state["trace"]["papers_retrieved"] = 0
        state["trace"]["avg_bm25"] = 0.0
        return state

    print(f"  ✓ Retrieved {len(raw)} papers (after retry)")

    # 3) 평가 기반 선별(BM25)
    top_k = int(getattr(config, "TOP_K_PAPERS", 10))
    selected, avg_bm25 = _bm25_select(raw, query_text=bm25_query_text or _norm(refined_query), top_k=top_k)

    state["papers"] = selected
    state["trace"]["papers_retrieved"] = len(selected)
    state["trace"]["avg_bm25"] = avg_bm25

    state["trace"]["retrieval"]["arxiv"]["selected_count"] = len(selected)
    state["trace"]["retrieval"]["arxiv"]["avg_bm25"] = avg_bm25

    print(f"  ✓ Selected top {len(selected)} by BM25 | avg_bm25={avg_bm25:.2f}")
    return state