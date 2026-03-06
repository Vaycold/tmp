# 3-2) Paper Retrieval Agent (Research Intelligence / Paper S-M + Web Search)
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
        "You retrieve relevant papers/web sources using provided tools.\n"
        "Return a structured list of candidates with title, year, venue/source, abstract/snippet, and relevance score.\n"
        "Do NOT infer gaps. Do NOT summarize beyond what is needed for matching.\n"
    ),
)


def paper_retrieval_node(state: AgentState) -> AgentState:
    result = paper_retrieval_agent.invoke(state)
    last = AIMessage(content=result["messages"][-1].content, name="paper_retrieval")
    return {"messages": [last], "sender": "paper_retrieval"}
