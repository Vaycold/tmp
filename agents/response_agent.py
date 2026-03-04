# 3-6) response Agent
from states import AgentState
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from tools import build_role_tools
from prompts.system import make_system_prompt
from llm import get_llm

llm = get_llm("azure", "gpt-5.1-chat")


ROLE_TOOLS = build_role_tools()
RESPONSE_TOOLS = ROLE_TOOLS["RESPONSE_TOOLS"]

final_response_agent = create_agent(
    model=llm,
    tools=RESPONSE_TOOLS,
    system_prompt=make_system_prompt(
        "Write a final report for the user based on prior messages.\n"
        "Include:\n"
        "1) Related papers: title/year + one-line relevance.\n"
        "2) Key limitations: grouped.\n"
        "3) Research gaps: clustered with evidence references.\n"
        "4) Critic scores : four scores and decision/flags in a compact block.\n"
        "Finish with: FINAL ANSWER\n"
    ),
)


def final_response_node(state: AgentState) -> AgentState:
    result = final_response_agent.invoke(state)
    last = AIMessage(content=result["messages"][-1].content, name="final_response")
    return {"messages": [last], "sender": "final_response"}
