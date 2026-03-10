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
        "ROLE: Paper Retrieval Agent\n"
        "You retrieve relevant papers/web sources using provided tools.\n"
        "Return a structured list of candidates with title, year, venue/source, abstract/snippet, and relevance score.\n"
        "Do NOT infer gaps. Do NOT summarize beyond what is needed for matching.\n"
    ),
)


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