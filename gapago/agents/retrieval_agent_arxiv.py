"""
Retrieval Agent (arXiv direct).
- No domain-limited expansion map
- LLM generates expansions and relaxed query candidates
- Retries when 0-hit (or low-hit) using candidates
"""

from __future__ import annotations

import json
import re
import time
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from models import AgentState, Paper
from utils import tokenize
from config import config
from llm import llm_chat, parse_json


_ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _safe_phrase(term: str, max_words: int = 2) -> str:
    """
    arXiv에서 과도한 phrase quoting으로 0-hit이 쉽게 나므로,
    3단어 이상은 앞 2단어까지만 사용.
    """
    term = _norm(term)
    if not term:
        return ""
    words = term.split()
    if len(words) <= 1:
        return term
    return f"\"{' '.join(words[:max_words])}\""


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


def _parse_atom(xml_text: str) -> List[Paper]:
    root = ET.fromstring(xml_text)
    entries = root.findall("atom:entry", _ATOM_NS)

    out: List[Paper] = []
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


def _fetch_arxiv(search_query: str, max_total: int, page_size: int, max_pages: int,
                errors: List[str]) -> List[Paper]:
    raw: List[Paper] = []
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
    uniq = {p.paper_id: p for p in raw}
    return list(uniq.values())


def _bm25_select(papers: List[Paper], query_text: str, top_k: int) -> Tuple[List[Paper], float]:
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


def _llm_expand_and_build_candidates(
    refined_query: str,
    keywords: List[str],
    trace_memory: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    LLM이 약어/동의어/관련 키워드 확장 및 arXiv 검색 후보 쿼리(완화 단계)를 생성.
    하드코딩 사전 없음.
    """
    rq = _norm(refined_query)
    kws = [_norm(k) for k in (keywords or []) if _norm(k)]
    mem = trace_memory or {}

    prompt = f"""You are an information retrieval engineer for arXiv.

INPUT:
- refined_query: {rq}
- keywords: {kws}
- user_memory (optional): {json.dumps(mem, ensure_ascii=False)[:1500]}

GOAL:
1) Expand terms with acronyms/synonyms/related terms (domain-agnostic).
2) Produce relaxed arXiv search_query candidates (from broad->narrow).
   - Prefer OR to preserve recall.
   - Avoid long quoted phrases (<=2 words per quote).
   - Use all: predominantly; ti:/abs: only in later candidates.
3) Provide a BM25 query text (plain) for ranking.

OUTPUT JSON ONLY with keys:
{{
  "expanded_terms": ["<=12 terms, include acronyms"],
  "bm25_query_text": "plain text for BM25",
  "candidates": [
    {{"level": 1, "search_query": "arXiv search_query", "desc": "broadest"}},
    ...
  ]
}}
"""

    messages = [
        {"role": "system", "content": "You generate robust arXiv search queries with high recall."},
        {"role": "user", "content": prompt},
    ]
    resp = llm_chat(messages)
    data = parse_json(resp)

    # defensive normalization
    expanded = data.get("expanded_terms", []) or []
    expanded = [_norm(x) for x in expanded if isinstance(x, str) and _norm(x)]
    expanded = expanded[:12]

    bm25_text = _norm(data.get("bm25_query_text", "")) or _norm(" ".join([rq] + expanded[:6]))

    candidates = data.get("candidates", []) or []
    cleaned = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        lvl = c.get("level")
        sq = _norm(c.get("search_query", ""))
        desc = _norm(c.get("desc", ""))
        if sq:
            cleaned.append({"level": lvl, "search_query": sq, "desc": desc})

    # fallback if LLM returns empty
    if not cleaned:
        # very broad fallback, OR-first
        core = expanded[:6] if expanded else [rq] if rq else ["research"]
        or_group = " OR ".join([f'all:{_safe_phrase(t)}' for t in core if _safe_phrase(t)])
        cleaned = [
            {"level": 1, "search_query": f"({or_group})" if or_group else 'all:"research"', "desc": "fallback broad OR"},
            {"level": 2, "search_query": f'all:{_safe_phrase(rq)}' if rq else 'all:"research"', "desc": "fallback all:rq"},
        ]

    return {"expanded_terms": expanded, "bm25_query_text": bm25_text, "candidates": cleaned}


def paper_retrieval_node(state: AgentState) -> AgentState:
    """
    arXiv retrieval with:
    - LLM-based expansion & candidate generation (no hardcoded map)
    - retry when 0-hit (or too few hits)
    """
    if state.get("iteration", 0) > 0:
        print(f"\n🔄 Re-running Retrieval (iteration {state['iteration']})")
    print("\n📚 Paper Retrieval Node (arXiv direct)")

    if "trace" not in state or state["trace"] is None:
        state["trace"] = {}
    if "errors" not in state or state["errors"] is None:
        state["errors"] = []

    state["trace"].setdefault("retrieval", {})
    state["trace"]["retrieval"]["arxiv"] = {
        "tries": [],
        "used_level": None,
        "used_search_query": None,
        "raw_count": 0,
        "selected_count": 0,
        "avg_bm25": 0.0,
        "expanded_terms": [],
        "bm25_query_text": "",
    }

    refined_query = state.get("refined_query", "")
    keywords = state.get("keywords", [])

    # LLM expansion + candidates
    mem = (state.get("trace", {}).get("memory", {}) or {})
    try:
        pack = _llm_expand_and_build_candidates(refined_query, keywords, trace_memory=mem)
    except Exception as e:
        state["errors"].append(f"[arXiv] LLM expansion failed: {e}")
        pack = {"expanded_terms": [], "bm25_query_text": _norm(refined_query), "candidates": []}

    state["trace"]["retrieval"]["arxiv"]["expanded_terms"] = pack.get("expanded_terms", [])
    state["trace"]["retrieval"]["arxiv"]["bm25_query_text"] = pack.get("bm25_query_text", "")

    candidates = pack.get("candidates", []) or []
    if not candidates:
        candidates = [{"level": 1, "search_query": 'all:"research"', "desc": "fallback"}]

    # pagination parameters
    max_total = int(getattr(config, "ARXIV_MAX_RESULTS", 80))
    page_size = int(getattr(config, "ARXIV_PAGE_SIZE", 40))
    max_pages = int(getattr(config, "ARXIV_MAX_PAGES", 3))

    # retry condition: 0-hit or too-few hit
    min_raw_required = int(getattr(config, "ARXIV_MIN_RAW", 8))

    raw_best: List[Paper] = []
    used = None

    for cand in candidates:
        lvl = cand.get("level")
        sq = cand.get("search_query", "")
        desc = cand.get("desc", "")

        raw = _fetch_arxiv(sq, max_total=max_total, page_size=page_size, max_pages=max_pages, errors=state["errors"])

        state["trace"]["retrieval"]["arxiv"]["tries"].append({
            "level": lvl,
            "desc": desc,
            "search_query": sq,
            "raw_count": len(raw),
        })

        # keep best for fallback
        if len(raw) > len(raw_best):
            raw_best = raw
            used = cand

        if len(raw) >= min_raw_required:
            used = cand
            raw_best = raw
            break

    # finalize used query
    if used:
        state["trace"]["retrieval"]["arxiv"]["used_level"] = used.get("level")
        state["trace"]["retrieval"]["arxiv"]["used_search_query"] = used.get("search_query")

    state["trace"]["retrieval"]["arxiv"]["raw_count"] = len(raw_best)

    if not raw_best:
        print("  ⚠️ No papers found after retries")
        state["papers"] = []
        state["trace"]["papers_retrieved"] = 0
        state["trace"]["avg_bm25"] = 0.0
        return state

    # BM25 selection
    bm25_query_text = state["trace"]["retrieval"]["arxiv"]["bm25_query_text"] or _norm(refined_query)
    top_k = int(getattr(config, "TOP_K_PAPERS", 10))
    selected, avg_bm25 = _bm25_select(raw_best, query_text=bm25_query_text, top_k=top_k)

    state["papers"] = selected
    state["trace"]["papers_retrieved"] = len(selected)
    state["trace"]["avg_bm25"] = avg_bm25

    state["trace"]["retrieval"]["arxiv"]["selected_count"] = len(selected)
    state["trace"]["retrieval"]["arxiv"]["avg_bm25"] = avg_bm25

    print(f"  ✓ raw={len(raw_best)} selected={len(selected)} avg_bm25={avg_bm25:.2f}")
    return state