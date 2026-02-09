"""
LangGraph skeleton for a Research GAP Analysis Agent
- LLM: ChatGPT API (OpenAI)
- Paper search: arXiv API only
- Goal: Make state in/out structure explicit (not perfect logic)
- Nodes: N0, N1, N2, N4(arXiv), N6, N7, N8, N9, N10, N11, N13
  (ScienceON/MDPI nodes omitted)
"""

from __future__ import annotations

import os
import time
import hashlib
import re
from typing import Any, Dict, List, Optional, TypedDict, Literal, Tuple

from urllib.parse import urlencode
import feedparser  # pip install feedparser
from langgraph.graph import StateGraph, END


# =========================================================
# 0) State (AgentState) — uses your chosen key names
# =========================================================
SourceType = Literal["scienceon", "arxiv", "mdpi"]  # keep union; we will use "arxiv" only


class AgentState(TypedDict, total=False):
    # A) Query
    user_question: str
    normalized_query: str
    query_variants: List[str]

    # B) Constraints (typo kept as user decided: "constrains")
    query_constrains: Dict[str, Any]

    # C) Sub-queries
    sub_query: Dict[SourceType, Dict[str, Any]]
    query_arxiv: Dict[str, Any]

    # D) Raw retrieval outputs
    raw_arxiv_papers: List[Dict[str, Any]]
    retrieval_stats_arxiv: Dict[str, Any]

    # E) Normalized / Dedup
    normalized_papers: List[Dict[str, Any]]
    papers: List[Dict[str, Any]]
    dedup_map: Dict[str, List[str]]

    # F) Ranking
    ranked_papers: List[Dict[str, Any]]
    paper_score: Dict[str, float]

    # G) Evidence
    evidence: List[Dict[str, Any]]
    evidence_stats: Dict[str, Any]

    # H) GAP
    gap_clusters: Dict[str, Any]
    gap_candidates: List[Dict[str, Any]]

    # I) Critic / Retry control
    critic: Dict[str, Any]
    retry_action: List[str]
    max_retries: int
    retry_count: int

    # J) Ops
    errors: List[str]
    trace: List[Dict[str, Any]]
    cost: Dict[str, Any]

    # K) Output
    final_report: str


# =========================================================
# 1) Utilities — trace/cost helpers (state-visible)
# =========================================================
def _now_ms() -> int:
    return int(time.time() * 1000)


def add_trace(state: AgentState, node: str, msg: str, meta: Optional[Dict[str, Any]] = None) -> None:
    """Append a trace event into state['trace'] (in-place)."""
    if "trace" not in state:
        state["trace"] = []
    state["trace"].append({"node": node, "ts_ms": _now_ms(), "message": msg, "meta": meta or {}})


def add_error(state: AgentState, node: str, err: str) -> None:
    """Append an error into state['errors'] (in-place)."""
    if "errors" not in state:
        state["errors"] = []
    state["errors"].append(f"[{node}] {err}")


def inc_cost(state: AgentState, key: str, amount: int = 1) -> None:
    """Increment a counter inside state['cost']."""
    if "cost" not in state:
        state["cost"] = {}
    state["cost"][key] = int(state["cost"].get(key, 0)) + amount


def stable_hash(text: str) -> str:
    """Hash helper for dedup/keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# =========================================================
# 2) LLM call (ChatGPT API) — minimal wrapper
#    NOTE: This is a runnable example if you configure OPENAI_API_KEY.
# =========================================================
def call_chatgpt(system: str, user: str, model: str = "gpt-4o-mini") -> str:
    """
    Minimal ChatGPT API call.
    - Uses official OpenAI Python SDK v1 style.
    - You must `pip install openai` and set OPENAI_API_KEY.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("OpenAI SDK not installed. `pip install openai`") from e

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not client.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


# =========================================================
# 3) Node stubs with visible State in/out
# =========================================================

# -------------------------
# N0 Initialize
# -------------------------
def N0_initialize(state: AgentState) -> AgentState:
    node = "N0_initialize"
    add_trace(state, node, "Initializing required state keys")

    # Only set defaults if missing; preserve upstream provided keys.
    out: AgentState = {}

    out["query_constrains"] = state.get("query_constrains", {
        "year_range": None,      # e.g., (2019, 2026)
        "top_k": 30,             # retrieval top-k
        "final_k": 10,           # final papers to present
        "max_results_per_source": 200,
    })

    out["sub_query"] = state.get("sub_query", {})
    out["raw_arxiv_papers"] = state.get("raw_arxiv_papers", [])
    out["retrieval_stats_arxiv"] = state.get("retrieval_stats_arxiv", {})

    out["normalized_papers"] = state.get("normalized_papers", [])
    out["papers"] = state.get("papers", [])
    out["dedup_map"] = state.get("dedup_map", {})

    out["ranked_papers"] = state.get("ranked_papers", [])
    out["paper_score"] = state.get("paper_score", {})

    out["evidence"] = state.get("evidence", [])
    out["evidence_stats"] = state.get("evidence_stats", {})

    out["gap_clusters"] = state.get("gap_clusters", {})
    out["gap_candidates"] = state.get("gap_candidates", [])

    out["critic"] = state.get("critic", {})
    out["retry_action"] = state.get("retry_action", [])
    out["max_retries"] = state.get("max_retries", 0)  # N12 미구현 → 0 권장
    out["retry_count"] = state.get("retry_count", 0)

    out["errors"] = state.get("errors", [])
    out["trace"] = state.get("trace", [])
    out["cost"] = state.get("cost", {})

    out["final_report"] = state.get("final_report", "")

    return out


# -------------------------
# N1 QueryAnalyze
# -------------------------
def N1_query_analyze(state: AgentState) -> AgentState:
    """
    Role:
    - user_question -> normalized_query
    - optionally create query_variants and update query_constrains
    Implementation detail:
    - Here we do simple normalization without LLM to keep it robust.
    - You can switch to LLM-based query understanding later.
    """
    node = "N1_query_analyze"
    q = (state.get("user_question") or "").strip()
    add_trace(state, node, "Analyzing user question", {"user_question": q})

    if not q:
        add_error(state, node, "Empty user_question")
        return {"normalized_query": "", "query_variants": [], "query_constrains": state.get("query_constrains", {})}

    # minimal "normalization"
    normalized = re.sub(r"\s+", " ", q)

    # very simple variants (placeholder)
    variants = [normalized]

    return {
        "normalized_query": normalized,
        "query_variants": variants,
        "query_constrains": state.get("query_constrains", {}),
    }


# -------------------------
# N2 SubqueryBuilder (arXiv only)
# -------------------------
def N2_subquery_builder(state: AgentState) -> AgentState:
    """
    Role:
    - Build arXiv query spec from normalized_query + constrains
    Output keys (as you defined):
    - query_arxiv
    - sub_query["arxiv"] (kept in sync)
    """
    node = "N2_subquery_builder"
    nq = state.get("normalized_query", "")
    c = state.get("query_constrains", {})
    add_trace(state, node, "Building arXiv subquery", {"normalized_query": nq, "constrains": c})

    # arXiv API uses a query string; we will keep it simple:
    # Example: search_query=all:"wind field" AND all:"UAV"
    # NOTE: In real implementation, escape quotes and build fielded query properly.
    arxiv_q = f'all:"{nq}"'
    top_k = int(c.get("top_k", 30))

    query_arxiv = {
        "search_query": arxiv_q,
        "start": 0,
        "max_results": min(top_k, 100),  # arXiv feed can return many; keep small for demo
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    # Keep sub_query synchronized
    sub_query = dict(state.get("sub_query", {}))
    sub_query["arxiv"] = query_arxiv

    return {
        "query_arxiv": query_arxiv,
        "sub_query": sub_query,
    }


# -------------------------
# N4 Retrieve_arXiv
# -------------------------
def N4_retrieve_arxiv(state: AgentState) -> AgentState:
    """
    Role:
    - Call arXiv API and store raw results
    Output keys:
    - raw_arxiv_papers
    - retrieval_stats_arxiv
    """
    node = "N4_retrieve_arxiv"
    spec = state.get("query_arxiv") or state.get("sub_query", {}).get("arxiv")
    if not spec:
        add_error(state, node, "Missing query_arxiv / sub_query['arxiv']")
        return {"raw_arxiv_papers": [], "retrieval_stats_arxiv": {"status": "missing_query"}}

    add_trace(state, node, "Calling arXiv API", {"query_arxiv": spec})

    # arXiv query endpoint (ATOM)
    # Example: http://export.arxiv.org/api/query?search_query=...&start=0&max_results=10
    base = "http://export.arxiv.org/api/query"


    params = {
        "search_query": spec["search_query"],
        "start": spec.get("start", 0),
        "max_results": spec.get("max_results", 30),
        "sortBy": spec.get("sortBy", "relevance"),
        "sortOrder": spec.get("sortOrder", "descending"),
    }
    url = f"http://export.arxiv.org/api/query?{urlencode(params)}"

    t0 = _now_ms()
    try:
        feed = feedparser.parse(url)
        inc_cost(state, "arxiv_api_calls", 1)

        raw: List[Dict[str, Any]] = []
        for entry in feed.entries:
            raw.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", ""),
                "authors": [a.get("name", "") for a in entry.get("authors", [])],
                "published": entry.get("published", ""),
                "updated": entry.get("updated", ""),
                "id": entry.get("id", ""),
                "links": entry.get("links", []),
                "tags": [t.get("term", "") for t in entry.get("tags", [])] if "tags" in entry else [],
            })

        dt = _now_ms() - t0
        stats = {
            "status": "ok",
            "count": len(raw),
            "latency_ms": dt,
            "query_url": url,
        }
        return {"raw_arxiv_papers": raw, "retrieval_stats_arxiv": stats}

    except Exception as e:
        dt = _now_ms() - t0
        add_error(state, node, f"arXiv retrieval failed: {e}")
        return {"raw_arxiv_papers": [], "retrieval_stats_arxiv": {"status": "error", "latency_ms": dt}}


# -------------------------
# N6 Normalize
# -------------------------
def N6_normalize(state: AgentState) -> AgentState:
    """
    Role:
    - Convert raw_arxiv_papers -> normalized_papers (your normalized schema)
    Output key:
    - normalized_papers
    """
    node = "N6_normalize"
    raw = state.get("raw_arxiv_papers", [])
    add_trace(state, node, "Normalizing raw arXiv papers", {"raw_count": len(raw)})

    normalized: List[Dict[str, Any]] = []
    for r in raw:
        title = (r.get("title") or "").strip()
        abstract = (r.get("summary") or "").strip()
        arxiv_id = (r.get("id") or "").strip()

        # year parse from published (YYYY-MM-DD...)
        year = None
        pub = r.get("published") or ""
        if len(pub) >= 4 and pub[:4].isdigit():
            year = int(pub[:4])

        paper_id = stable_hash(arxiv_id or title)

        normalized.append({
            "paper_id": paper_id,
            "source": "arxiv",
            "identifier": arxiv_id,
            "doi": None,
            "title": title,
            "abstract": abstract,
            "authors": r.get("authors", []),
            "year": year or 0,
            "url": arxiv_id,
            "keywords": r.get("tags", []),
            "venue": "arXiv",
            "metadata_raw": r,
        })

    return {"normalized_papers": normalized}


# -------------------------
# N7 Deduplicate
# -------------------------
def N7_deduplicate(state: AgentState) -> AgentState:
    """
    Role:
    - normalized_papers -> papers (deduped)
    - produce dedup_map
    For arXiv-only, we dedup by identifier (arXiv id) primarily.
    """
    node = "N7_deduplicate"
    inp = state.get("normalized_papers", [])
    add_trace(state, node, "Deduplicating normalized papers", {"count": len(inp)})

    by_key: Dict[str, Dict[str, Any]] = {}
    dedup_map: Dict[str, List[str]] = {}

    for p in inp:
        key = p.get("identifier") or p.get("title") or p.get("paper_id")
        pid = p.get("paper_id")
        if key in by_key:
            # merge map
            canonical_id = by_key[key]["paper_id"]
            dedup_map.setdefault(canonical_id, []).append(pid)
        else:
            by_key[key] = p
            dedup_map.setdefault(p["paper_id"], [])

    papers = list(by_key.values())
    return {"papers": papers, "dedup_map": dedup_map}


# -------------------------
# N8 Rank
# -------------------------
def N8_rank(state: AgentState) -> AgentState:
    """
    Role:
    - papers -> ranked_papers, paper_score
    Here we implement a simple lexical overlap score (placeholder).
    """
    node = "N8_rank"
    papers = state.get("papers", [])
    nq = state.get("normalized_query", "")
    add_trace(state, node, "Ranking papers (placeholder scoring)", {"paper_count": len(papers)})

    # naive token overlap
    q_tokens = set(re.findall(r"[A-Za-z0-9]+", nq.lower()))
    scores: Dict[str, float] = {}

    def score_p(p: Dict[str, Any]) -> float:
        text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
        t_tokens = set(re.findall(r"[A-Za-z0-9]+", text))
        if not q_tokens:
            return 0.0
        return len(q_tokens & t_tokens) / max(1, len(q_tokens))

    for p in papers:
        pid = p["paper_id"]
        scores[pid] = score_p(p)

    ranked = sorted(papers, key=lambda p: scores.get(p["paper_id"], 0.0), reverse=True)

    final_k = int(state.get("query_constrains", {}).get("final_k", 10))
    ranked = ranked[:max(1, final_k)]

    return {"ranked_papers": ranked, "paper_score": scores}


# -------------------------
# N9 EvidenceExtract (LLM-based, but minimal)
# -------------------------
def N9_evidence_extract(state: AgentState) -> AgentState:
    """
    Role:
    - From ranked_papers, extract limitation/future_work evidence spans.
    Output:
    - evidence, evidence_stats
    Implementation:
    - Use ChatGPT to extract 1~2 limitation sentences from abstract (placeholder).
    """
    node = "N9_evidence_extract"
    ranked = state.get("ranked_papers", [])
    add_trace(state, node, "Extracting evidence spans using LLM (minimal)", {"ranked_count": len(ranked)})

    evidence: List[Dict[str, Any]] = []
    failed = 0

    system = (
        "You extract limitations or future work statements grounded in the provided abstract. "
        "Return JSON with fields: spans (list of objects with span_type, text, confidence). "
        "span_type must be one of: limitation, future_work, challenge, open_problem."
    )

    for p in ranked:
        abstract = p.get("abstract", "")
        if not abstract.strip():
            continue

        user = f"Title: {p.get('title','')}\nAbstract:\n{abstract}\n\nExtract up to 2 grounded spans."
        try:
            inc_cost(state, "llm_calls", 1)
            # NOTE: model can be changed. This is a placeholder name.
            out = call_chatgpt(system=system, user=user, model="gpt-4o-mini")
            inc_cost(state, "llm_tokens_est", len(out) // 4)  # rough, placeholder

            # Minimal parsing: if JSON parse fails, keep raw string in meta.
            spans = []
            try:
                import json
                j = json.loads(out)
                spans = j.get("spans", [])
            except Exception:
                spans = [{"span_type": "other", "text": out.strip()[:500], "confidence": 0.3}]

            for s in spans:
                evidence.append({
                    "paper_id": p["paper_id"],
                    "span_type": s.get("span_type", "other"),
                    "text": s.get("text", ""),
                    "location": "abstract",
                    "confidence": float(s.get("confidence", 0.0)),
                    "extract_method": "llm",
                })
        except Exception as e:
            failed += 1
            add_error(state, node, f"LLM evidence extraction failed for paper_id={p.get('paper_id')}: {e}")

    stats = {"status": "ok", "evidence_count": len(evidence), "failed_papers": failed}
    return {"evidence": evidence, "evidence_stats": stats}


# -------------------------
# N10 GapSynthesize (LLM-based, minimal)
# -------------------------
def N10_gap_synthesize(state: AgentState) -> AgentState:
    """
    Role:
    - evidence -> gap_clusters, gap_candidates
    Implementation:
    - For skeleton: send all evidence texts to LLM and ask for 2~3 GAP candidates.
    """
    node = "N10_gap_synthesize"
    evidence = state.get("evidence", [])
    add_trace(state, node, "Synthesizing GAP candidates using LLM (minimal)", {"evidence_count": len(evidence)})

    if not evidence:
        return {"gap_clusters": {}, "gap_candidates": []}

    # Compact evidence payload
    ev_lines = []
    for ev in evidence[:50]:  # prevent huge prompt
        ev_lines.append(f"- ({ev.get('span_type')}) {ev.get('text')}")

    system = (
        "You synthesize research gaps from grounded evidence statements. "
        "Return JSON with fields: gap_candidates (list). "
        "Each candidate must include: gap_statement, why_it_matters, root_cause, research_directions (list)."
    )
    user = "Evidence statements:\n" + "\n".join(ev_lines) + "\n\nGenerate 3 gap candidates."

    try:
        inc_cost(state, "llm_calls", 1)
        out = call_chatgpt(system=system, user=user, model="gpt-4o-mini")
        inc_cost(state, "llm_tokens_est", len(out) // 4)

        # Parse or fallback
        gap_candidates: List[Dict[str, Any]] = []
        gap_clusters: Dict[str, Any] = {}

        try:
            import json
            j = json.loads(out)
            gap_candidates = j.get("gap_candidates", [])
        except Exception:
            gap_candidates = [{
                "gap_statement": out.strip()[:400],
                "why_it_matters": "",
                "root_cause": "",
                "research_directions": [],
            }]

        # Skeleton cluster output: single cluster containing all evidence indices
        gap_clusters = {"cluster_0": {"evidence_indices": list(range(len(evidence)))}}

        # Attach minimal metadata to each gap candidate for traceability
        enriched: List[Dict[str, Any]] = []
        for i, gc in enumerate(gap_candidates[:3]):
            enriched.append({
                "gap_id": f"gap_{i}",
                "cluster_id": "cluster_0",
                "gap_statement": gc.get("gap_statement", ""),
                "why_it_matters": gc.get("why_it_matters", ""),
                "root_cause": gc.get("root_cause", ""),
                "research_directions": gc.get("research_directions", []),
                "supporting_papers": list({ev["paper_id"] for ev in evidence}),
                "supporting_evidence": evidence[:10],  # keep small
            })

        return {"gap_clusters": gap_clusters, "gap_candidates": enriched}

    except Exception as e:
        add_error(state, node, f"GAP synthesis failed: {e}")
        return {"gap_clusters": {}, "gap_candidates": []}


# -------------------------
# N11 CriticValidate (LLM-based, minimal)
# -------------------------
def N11_critic_validate(state: AgentState) -> AgentState:
    """
    Role:
    - Evaluate quality of retrieval + evidence + gap candidates.
    Output:
    - critic, retry_action
    Note:
    - N12 retry is not implemented, but we still output suggested retry actions.
    """
    node = "N11_critic_validate"
    ranked = state.get("ranked_papers", [])
    gaps = state.get("gap_candidates", [])
    add_trace(state, node, "Critic validation using LLM (minimal)", {"ranked": len(ranked), "gaps": len(gaps)})

    system = (
        "You are a strict reviewer. "
        "Return JSON with fields: scores (relevance,evidence,diversity,redun), flags (list), "
        "should_retry (bool), retry_action (list), reasons (list)."
    )
    user = (
        f"Query: {state.get('normalized_query','')}\n"
        f"Ranked papers: {len(ranked)}\n"
        f"GAP candidates: {len(gaps)}\n"
        "Assess if output is sufficient."
    )

    try:
        inc_cost(state, "llm_calls", 1)
        out = call_chatgpt(system=system, user=user, model="gpt-4o-mini")
        inc_cost(state, "llm_tokens_est", len(out) // 4)

        critic = {}
        retry_action: List[str] = []
        try:
            import json
            j = json.loads(out)
            critic = {
                "scores": j.get("scores", {}),
                "flags": j.get("flags", []),
                "should_retry": bool(j.get("should_retry", False)),
                "reasons": j.get("reasons", []),
            }
            retry_action = j.get("retry_action", [])
        except Exception:
            critic = {"status": "unparsed", "raw": out[:800], "should_retry": False}
            retry_action = []

        return {"critic": critic, "retry_action": retry_action}

    except Exception as e:
        add_error(state, node, f"Critic validate failed: {e}")
        return {"critic": {"status": "error", "should_retry": False}, "retry_action": []}


# -------------------------
# N13 FormatOutput
# -------------------------
def N13_format_output(state: AgentState) -> AgentState:
    """
    Role:
    - Produce final_report string.
    """
    node = "N13_format_output"
    add_trace(state, node, "Formatting final report")

    q = state.get("normalized_query", "")
    ranked = state.get("ranked_papers", [])
    gaps = state.get("gap_candidates", [])
    critic = state.get("critic", {})
    errors = state.get("errors", [])

    lines: List[str] = []
    lines.append(f"# GAP Analysis Report\n")
    lines.append(f"## Query\n- {q}\n")

    lines.append("## Top Papers (arXiv)\n")
    for i, p in enumerate(ranked, 1):
        lines.append(f"{i}. {p.get('title','').strip()} ({p.get('year',0)})")
        lines.append(f"   - URL: {p.get('url','')}")
        pid = p.get("paper_id", "")
        score = state.get("paper_score", {}).get(pid, 0.0)
        lines.append(f"   - Score: {score:.3f}\n")

    lines.append("## GAP Candidates\n")
    for g in gaps:
        lines.append(f"- **{g.get('gap_statement','')}**")
        if g.get("why_it_matters"):
            lines.append(f"  - Why: {g.get('why_it_matters')}")
        if g.get("root_cause"):
            lines.append(f"  - Root cause: {g.get('root_cause')}")
        dirs = g.get("research_directions", [])
        if dirs:
            lines.append("  - Directions:")
            for d in dirs:
                lines.append(f"    - {d}")
        lines.append("")

    lines.append("## Critic\n")
    lines.append(str(critic))
    lines.append("")

    if errors:
        lines.append("## Errors/Warnings\n")
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")

    return {"final_report": "\n".join(lines)}


# =========================================================
# 4) Build the graph (arXiv-only pipeline)
# =========================================================
def build_arxiv_only_graph():
    g = StateGraph(AgentState)

    # Add nodes
    g.add_node("N0_initialize", N0_initialize)
    g.add_node("N1_query_analyze", N1_query_analyze)
    g.add_node("N2_subquery_builder", N2_subquery_builder)
    g.add_node("N4_retrieve_arxiv", N4_retrieve_arxiv)
    g.add_node("N6_normalize", N6_normalize)
    g.add_node("N7_deduplicate", N7_deduplicate)
    g.add_node("N8_rank", N8_rank)
    g.add_node("N9_evidence_extract", N9_evidence_extract)
    g.add_node("N10_gap_synthesize", N10_gap_synthesize)
    g.add_node("N11_critic_validate", N11_critic_validate)
    g.add_node("N13_format_output", N13_format_output)

    # Edges (1-pass)
    g.set_entry_point("N0_initialize")
    g.add_edge("N0_initialize", "N1_query_analyze")
    g.add_edge("N1_query_analyze", "N2_subquery_builder")
    g.add_edge("N2_subquery_builder", "N4_retrieve_arxiv")
    g.add_edge("N4_retrieve_arxiv", "N6_normalize")
    g.add_edge("N6_normalize", "N7_deduplicate")
    g.add_edge("N7_deduplicate", "N8_rank")
    g.add_edge("N8_rank", "N9_evidence_extract")
    g.add_edge("N9_evidence_extract", "N10_gap_synthesize")
    g.add_edge("N10_gap_synthesize", "N11_critic_validate")
    g.add_edge("N11_critic_validate", "N13_format_output")
    g.add_edge("N13_format_output", END)

    return g.compile()


# =========================================================
# 5) Demo run + Mermaid visualization helpers
# =========================================================
if __name__ == "__main__":
    app = build_arxiv_only_graph()

    # 5-1) Mermaid text
    print(app.get_graph().draw_mermaid())

    # 5-2) Write PNG graph image (if environment supports)
    try:
        png = app.get_graph().draw_mermaid_png()
        with open("gap_agent_graph.png", "wb") as f:
            f.write(png)
        print("Saved: gap_agent_graph.png")
    except Exception as e:
        print(f"Graph PNG render failed: {e}")

    # 5-3) Run the pipeline (requires OPENAI_API_KEY + openai + feedparser)
    init_state: AgentState = {
        "user_question": "Research gaps in physics-informed neural networks for RUL prediction",
        # optional: override constrains
        "query_constrains": {"top_k": 20, "final_k": 8},
        "max_retries": 0,
        "retry_count": 0,
    }

    out = app.invoke(init_state)
    print(out.get("final_report", "")[:2000])
