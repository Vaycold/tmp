# 3-2) Paper Retrieval Agent (tool-using)
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
        "ROLE: Paper Retrieval Agent\n"
        "You retrieve relevant papers using available tools.\n"
        "Process:\n"
        "1) Use meaning_expand_tool to expand keywords (synonyms/acronyms).\n"
        "2) Use arxiv_api_call_tool, web_search_tool, and scienceon_search_tool as needed.\n"
        "3) Use bm25_rank_tool to rank retrieved items.\n\n"
        "Output JSON with fields: \n"
        "- expanded_terms\n"
        "- search_candidates (optional)\n"
        "- papers: list of {paper_id,title,year,url,abstract,score_bm25,source}\n"
        "- notes\n"
        "Do NOT infer gaps. Do NOT summarize beyond what is needed for matching.\n"
    ),
)


def paper_retrieval_node(state: AgentState) -> AgentState:
    result = paper_retrieval_agent.invoke(state)
    last = AIMessage(content=result["messages"][-1].content, name="paper_retrieval")
    return {"messages": [last], "sender": "paper_retrieval"}