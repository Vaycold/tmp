# query_agent.py

from models import AgentState
from llm import llm_chat, parse_json


def _get_memory_block(state: AgentState) -> str:
    """trace 기반 임시 메모리(누적 사용자 답변) 텍스트를 구성."""
    trace = state.get("trace", {}) or {}
    mem = trace.get("memory", {}) or {}
    clarifs = mem.get("clarifications", []) or []
    if not clarifs:
        return "(none)"
    # 최근 3개만 사용(프롬프트 폭주 방지)
    recent = clarifs[-3:]
    lines = []
    for item in recent:
        q = item.get("questions_summary", "")
        a = item.get("answer", "")
        lines.append(f"- Q: {q}\n  A: {a}")
    return "\n".join(lines)


def query_analysis_node(state: AgentState) -> AgentState:
    """
    Refine user question into search query.
    Adds: query evaluation + clarify routing signal without ending the graph.
    Memory: stored in state["trace"]["memory"] only.
    """
    if state["iteration"] > 0:
        print(f"\n🔄 Re-running Query Analysis (iteration {state['iteration']})")
    print(f"\n🔍 Query Analysis Node")

    # Defensive init
    if "trace" not in state or state["trace"] is None:
        state["trace"] = {}
    if "errors" not in state or state["errors"] is None:
        state["errors"] = []

    # route 기본값 초기화 (중요: 이전 clarify 상태가 남지 않도록)
    state["route"] = ""

    memory_block = _get_memory_block(state)

    # retrieval/critic 힌트도 활용 (이미 state/trace에 존재) 
    papers_retrieved = state["trace"].get("papers_retrieved", None)
    avg_bm25 = state["trace"].get("avg_bm25", None)
    critic = state.get("critic")
    critic_scores = {
        "query_specificity": getattr(critic, "query_specificity", None) if critic else None,
        "paper_relevance": getattr(critic, "paper_relevance", None) if critic else None,
        "groundedness": getattr(critic, "groundedness", None) if critic else None,
    }

    prompt = f"""You are a research query optimizer for academic paper search (source-agnostic).

Research Question:
{state['user_question']}

Temporary Memory (user clarifications so far):
{memory_block}

Last retrieval signals:
- papers_retrieved: {papers_retrieved}
- avg_bm25: {avg_bm25}

Last critic signals:
{critic_scores}

TASK:
1) Generate a source-agnostic keyword lists for retrieval agent.
   - Do NOT include database-specific syntax (no ti:, abs:, ANDNOT, site:, etc.)
2) Decide if clarification is needed to improve retrieval quality.

Output JSON only with EXACT keys:
{{
  "refined_query": "Concise search string (6-12 words, English)",
  "keywords": ["3-6 important terms/phrases, English"],
  "negative_keywords": ["0-4 terms to exclude, English"],
  "needs_clarification": true/false,
  "clarifying_questions": ["2-3 questions if needs_clarification is true, else []"],
  "reason": "short reason"
}}
"""

    messages = [
        {"role": "system", "content": "You optimize academic search intent and ask clarifying questions when needed."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = llm_chat(messages)
        result = parse_json(response)

        state["refined_query"] = result.get("refined_query", state["user_question"])
        state["keywords"] = result.get("keywords", [])
        state["negative_keywords"] = result.get("negative_keywords", [])

        state["trace"]["query_analysis"] = state["refined_query"]
        state["trace"]["query_reason"] = result.get("reason", "")

        needs = bool(result.get("needs_clarification", False))
        questions = result.get("clarifying_questions", [])
        questions = [q.strip() for q in questions if isinstance(q, str) and q.strip()]

        state["trace"]["clarify_questions"] = questions[:5]
        state["trace"]["clarify_needed"] = needs

        print(f"  ✓ Refined: {state['refined_query']}")
        print(f"  ✓ Keywords: {', '.join(state['keywords'])}")

        if needs:
            # human_agent.py가 이 trace를 읽어서 질문을 출력하고 메모리에 누적 :contentReference[oaicite:6]{index=6}
            state["route"] = "clarify"
            print("  ⚠️ Clarification required. Routing to HumanClarify node.")

    except Exception as e:
        state["errors"].append(f"Query analysis error: {str(e)}")
        state["refined_query"] = state["user_question"]
        state["trace"]["query_analysis"] = state["refined_query"]
        state["trace"]["clarify_needed"] = False
        state["trace"]["clarify_questions"] = []

    return state
