# 3-4) GAP Inference Agent
from states import AgentState
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from tools import build_role_tools
from prompts.system import make_system_prompt
from llm import get_llm

llm = get_llm()

ROLE_TOOLS = build_role_tools()
GAP_INFER_TOOLS = ROLE_TOOLS["GAP_INFER_TOOLS"]

gap_infer_agent = create_agent(
    llm,
    tools=GAP_INFER_TOOLS,
    system_prompt=make_system_prompt(
        "ROLE: GAP Inference Agent\n"
        "You cluster limitations into K criteria axes (e.g., data/robustness/scalability/evaluation/usability).\n"
        "Infer recurring core research gaps and propose candidate research topics.\n"
        "Each gap/topic MUST include supporting evidence references (paper_id + sentence).\n"
    ),
)


def gap_infer_node(state: AgentState) -> AgentState:
    result = gap_infer_agent.invoke(state)
    last = AIMessage(content=result["messages"][-1].content, name="gap_infer")
    return {"messages": [last], "sender": "gap_infer"}
