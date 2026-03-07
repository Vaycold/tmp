# 3-3) Paper Retrieval Agent (tool-using selector)
from __future__ import annotations

from states import AgentState
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from tools import build_role_tools
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


def paper_retrieval_node(state: AgentState) -> AgentState:
    result = paper_retrieval_agent.invoke(state)
    last = AIMessage(content=result["messages"][-1].content, name="paper_retrieval")
    return {"messages": [last], "sender": "paper_retrieval"}
