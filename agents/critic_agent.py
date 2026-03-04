# 3-5) Critic Score Agent
from states import AgentState
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from tools import build_role_tools
from prompts.system import make_system_prompt
from llm import get_llm

llm = get_llm("azure", "gpt-5.1-chat")

ROLE_TOOLS = build_role_tools()
CRITIC_TOOLS = ROLE_TOOLS["CRITIC_TOOLS"]

critic_score_agent = create_agent(
    llm,
    tools=CRITIC_TOOLS,
    system_prompt=make_system_prompt(
        "ROLE: Critic Agent\n"
        "You mevaluate the pipeline outputs (query specificity, paper alignment, evidence support).\n"
        "Return scores + flags + a routing decision tag among:\n"
        "- DECISION: ACCEPT\n"
        "- DECISION: REDO_RETRIEVAL\n"
        "- DECISION: REFINE_QUERY\n"
        "Do NOT generate new gaps/topics.\n"
    ),
)


def critic_score_node(state: AgentState) -> AgentState:
    result = critic_score_agent.invoke(state)
    last = AIMessage(content=result["messages"][-1].content, name="critic_score")
    return {"messages": [last], "sender": "critic_score"}
