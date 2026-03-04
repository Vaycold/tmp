# 3-3) Limitation Extract Agent
from states import AgentState
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from tools import build_role_tools
from prompts.system import make_system_prompt
from llm import get_llm

llm = get_llm("azure", "gpt-5.1-chat")


ROLE_TOOLS = build_role_tools()
LIMITATION_TOOLS = ROLE_TOOLS["LIMITATION_TOOLS"]

limitation_extract_agent = create_agent(
    llm,
    tools=LIMITATION_TOOLS,
    system_prompt=make_system_prompt(
        "ROLE: Limitation Extract Agent\n"
        "You extract limitation/future-work statements from retrieved papers/snippets.\n"
        "Output be structured: paper_id -> [limitation_sentences], plus brief rationale.\n"
        "Do NOT infer gaps yet.\n"
    ),
)


def limitation_extract_node(state: AgentState) -> AgentState:
    result = limitation_extract_agent.invoke(state)
    last = AIMessage(content=result["messages"][-1].content, name="limitation_extract")
    return {"messages": [last], "sender": "limitation_extract"}
