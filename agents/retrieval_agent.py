# 3-3) Paper Retrieval Agent (tool-using selector)
from __future__ import annotations

from states import AgentState, Paper
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
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


def _parse_papers_from_ai_message(content: str) -> list[dict]:
    """
    AIMessage content (JSON)에서 papers 리스트 파싱.
    LLM이 output JSON 안에 papers 필드로 반환하는 경우 처리.
    """
    data = _safe_json_loads(content)
    if not data or not isinstance(data, dict):
        return []

    papers = []

    # ✅ LLM output JSON의 papers 필드
    for p in data.get("papers", []):
        if not isinstance(p, dict):
            continue
        papers.append({
            "paper_id": p.get("paper_id", ""),
            "title": p.get("title", ""),
            "abstract": p.get("abstract", ""),
            "url": p.get("url", ""),
            "year": p.get("year", 0) or 0,
            "authors": p.get("authors") or [],
            "score_bm25": p.get("score_bm25", 0.0),
            "source": p.get("source", "arxiv"),
            "full_text_sections": {},
        })

    # ✅ web_results도 papers로 변환
    for r in data.get("web_results", []):
        if not isinstance(r, dict):
            continue
        url = r.get("url", "")
        papers.append({
            "paper_id": f"WEB_{url[-40:].replace('/', '_').replace(':', '')}",
            "title": r.get("title", ""),
            "abstract": r.get("content") or r.get("snippet", ""),
            "url": url,
            "year": 0,
            "authors": [],
            "score_bm25": 0.0,
            "source": "web",
            "full_text_sections": {},
        })

    return papers


def _parse_papers_from_tool_messages(messages: list) -> list[dict]:
    """
    tool 결과 메시지에서 arxiv/web 결과 파싱.
    """
    papers = []
    for msg in messages:
        content = getattr(msg, "content", "")
        if not content:
            continue
        if getattr(msg, "type", "") != "tool":
            continue

        data = _safe_json_loads(content)
        if not data or not isinstance(data, dict):
            continue

        source = data.get("source", "")

        # arxiv 결과
        if source == "arxiv":
            papers.extend(data.get("results", []))

        # web 결과
        elif source == "web":
            for r in data.get("results", []):
                url = r.get("url", "")
                papers.append({
                    "paper_id": f"WEB_{url[-40:].replace('/', '_').replace(':', '')}",
                    "title": r.get("title", ""),
                    "abstract": r.get("content") or r.get("abstract") or r.get("snippet", ""),
                    "url": url,
                    "year": 0,
                    "authors": [],
                    "score_bm25": 0.0,
                    "source": "web",
                    "full_text_sections": {},
                })

    return papers


def paper_retrieval_node(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    print(f"  [DEBUG] 전체 messages 수: {len(messages)}")
    for i, m in enumerate(messages):
        print(f"  [DEBUG] msg[{i}] type={type(m).__name__} name={getattr(m, 'name', None)} content={getattr(m, 'content', '')[:80]}")
    # ✅ meaning_expand 메시지 우선, 없으면 query_analysis, 그것도 없으면 user_question
    query_messages = [
        m for m in state.get("messages", [])
        if getattr(m, "name", None) == "meaning_expand"
    ]
    if not query_messages:
        query_messages = [
            m for m in state.get("messages", [])
            if getattr(m, "name", None) == "query_analysis"
        ]
    if not query_messages:
        fallback_text = state.get("refined_query") or state.get("user_question", "")
        query_messages = [HumanMessage(content=fallback_text)]

    # 가장 최신 메시지 1개만 전달
    query_messages = query_messages[-1:]

    result = paper_retrieval_agent.invoke({
        **state,
        "messages": query_messages,
    })

    messages = result.get("messages", [])
    last_content = messages[-1].content if messages else "{}"

    # ✅ 1순위: tool 메시지에서 직접 파싱
    raw_papers = _parse_papers_from_tool_messages(messages)

    # ✅ 2순위: LLM output JSON의 papers 필드에서 파싱 (현재 구조 대응)
    if not raw_papers:
        raw_papers = _parse_papers_from_ai_message(last_content)

    print(f"  [DEBUG] raw_papers count: {len(raw_papers)}")

    # ✅ BM25 랭킹
    query = state.get("refined_query") or state.get("user_question", "")
    if raw_papers and query:
        ranked = bm25_rank(raw_papers, query, top_k=10)
        raw_papers = ranked.get("selected", raw_papers)

    # ✅ Paper 객체로 변환
    papers = []
    for p in raw_papers:
        if not p.get("title") or not p.get("paper_id"):
            continue
        try:
            papers.append(Paper(
                paper_id=p.get("paper_id", ""),
                title=p.get("title", ""),
                abstract=p.get("abstract", ""),
                url=p.get("url", ""),
                year=p.get("year", 0) or 0,
                authors=p.get("authors") or [],
                score_bm25=p.get("score_bm25", 0.0),
                full_text_sections=p.get("full_text_sections") or {},
            ))
        except Exception as e:
            print(f"  ⚠️ Paper 변환 실패: {e}")
            continue

    print(f"  ✓ Retrieved {len(papers)} papers → state['papers']에 저장")

    last = AIMessage(content=last_content, name="paper_retrieval")
    return {
        "messages": [last],
        "sender": "paper_retrieval",
        "papers": papers,  # ✅ state["papers"]에 저장
    }