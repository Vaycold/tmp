from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm import get_llm
from states import AgentState


llm = get_llm()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _safe_json_loads(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}


def meaning_expand_node(state: AgentState) -> AgentState:
    refined_query = state.get("refined_query", "")
    keywords = state.get("keywords", [])

    trace = state.get("trace") or {}
    errors = state.get("errors") or []

    trace.setdefault("retrieval", {})
    trace["retrieval"].setdefault("arxiv", {})

    mem = (trace.get("memory", {}) or {})
    rq = _norm(refined_query)
    kws = [_norm(k) for k in (keywords or []) if _norm(k)]

    prompt = f"""You are an information retrieval engineer.

INPUT:
- refined_query: {rq}
- keywords: {kws}
- user_memory (optional): {json.dumps(mem, ensure_ascii=False)[:1500]}

GOAL:
1) Expand terms with acronyms/synonyms/related terms (domain-agnostic).
2) Provide a BM25 query text (plain) for ranking.

OUTPUT JSON ONLY with keys:
{
    "expanded_terms": ["<=12 terms, include acronyms"],
    "bm25_query_text": "plain text for BM25"
}
"""

    messages = [
        SystemMessage(content="You generate keyword expansions with high recall."),
        HumanMessage(content=prompt),
    ]

    try:
        resp = llm.invoke(messages)
        data = _safe_json_loads(getattr(resp, "content", "") or "")
    except Exception as e:
        errors.append(f"[meaning_expand] LLM expansion failed: {e}")
        data = {}

    expanded = data.get("expanded_terms", []) or []
    expanded = [_norm(x) for x in expanded if isinstance(x, str) and _norm(x)]
    expanded = expanded[:12]

    bm25_text = _norm(data.get("bm25_query_text", "")) or _norm(" ".join([rq] + expanded[:6]))

    trace["retrieval"]["arxiv"]["expanded_terms"] = expanded
    trace["retrieval"]["arxiv"]["bm25_query_text"] = bm25_text

    payload = {
        "expanded_terms": expanded,
        "bm25_query_text": bm25_text,
    }
    last = AIMessage(content=json.dumps(payload, ensure_ascii=False), name="meaning_expand")

    return {
        "messages": [last],
        "sender": "meaning_expand",
        "trace": trace,
        "errors": errors,
    }