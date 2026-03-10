# 3-3) Paper Retrieval Agent (tool-using selector)
from __future__ import annotations

from states import AgentState
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from tools import build_role_tools, bm25_rank, _safe_json_loads
from prompts.system import make_system_prompt
from llm import get_llm

llm = get_llm()

ROLE_TOOLS = build_role_tools()
RETRIEVAL_TOOLS = ROLE_TOOLS["RETRIEVAL_TOOLS"]

paper_retrieval_agent = create_agent(
    llm,
    tools=RETRIEVAL_TOOLS,
    system_prompt=make_system_prompt(
        """ROLE: Paper Retrieval Agent
You are a retrieval orchestrator. Your job is to SELECT and CALL the most appropriate search tools.

Available tools:
- web_search_tool
- arxiv_api_call_tool
- scienceon_search_tool (currently placeholder)

Inputs may include a previous Meaning Expansion Agent message containing:
- refined_query
- keywords
- expanded_terms
- arxiv_query_candidates
- web_query_candidates
- scienceon_query_candidates

Rules:
1) Do not perform meaning expansion yourself.
2) Use only the available tools above.
3) Select the minimum useful set of tools.
4) Prefer arxiv_api_call_tool for academic paper search.
5) Use web_search_tool for supplementary discovery or when the query is broad.
6) ScienceON may be selected when Korean or domestic research coverage is useful, but it is currently not implemented.

Output JSON with fields:
- selected_tools: [..]
- tool_rationale: <string>
- papers: list of {paper_id,title,year,url,abstract,authors,source}
- web_results: list
- scienceon_results: list
- notes: list[str]
Do NOT infer limitations or gaps.
"""
    ),
)


def _parse_papers_from_tool_messages(messages: list) -> list[dict]:
    """
    tool_calls 결과에서 arxiv_api_call_tool이 반환한 JSON을 파싱해
    paper dict 리스트로 반환.
    """
    papers = []
    for msg in messages:
        # ToolMessage에서 arxiv 결과 추출
        content = getattr(msg, "content", "")
        if not content:
            continue

        data = _safe_json_loads(content)
        if not data:
            continue

        # arxiv_api_call_tool 결과
        if isinstance(data, dict) and data.get("source") == "arxiv":
            papers.extend(data.get("results", []))

    return papers

def paper_retrieval_node(state: AgentState) -> AgentState:

    # ✅ 누적 메시지 병목 방지: query_analysis 메시지만 전달
    query_messages = [
        m for m in state.get("messages", [])
        if getattr(m, "name", None) == "query_analysis"
    ]

    # query_analysis 메시지 없으면 user_question으로 fallback
    if not query_messages:
        from langchain_core.messages import HumanMessage
        query_messages = [
            HumanMessage(content=state.get("user_question", ""))
        ]

    result = paper_retrieval_agent.invoke({
        **state,
        "messages": query_messages  # ✅ 필요한 메시지만 전달
    })

    messages = result.get("messages", [])

    raw_papers = _parse_papers_from_tool_messages(messages)

    query = state.get("refined_query") or state.get("user_question", "")
    if raw_papers and query:
        ranked = bm25_rank(raw_papers, query, top_k=10)
        raw_papers = ranked.get("selected", raw_papers)

    papers = []
    for p in raw_papers:
        try:
            papers.append(Paper(
                paper_id=p.get("paper_id", ""),
                title=p.get("title", ""),
                abstract=p.get("abstract", ""),
                url=p.get("url", ""),
                year=p.get("year", 0),
                authors=p.get("authors", []),
                score_bm25=p.get("score_bm25", 0.0),
                full_text_sections=p.get("full_text_sections", {}),
            ))
        except Exception as e:
            print(f"  ⚠️ Paper parsing error: {e}")
            continue

    print(f"  ✓ Retrieved {len(papers)} papers")

    last = AIMessage(content=messages[-1].content if messages else "", name="paper_retrieval")
    return {
        "messages": [last],
        "sender": "paper_retrieval",
        "papers": papers,
    }