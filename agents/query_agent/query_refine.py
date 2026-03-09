from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from states import AgentState
from llm import get_llm
from prompts.system import make_system_prompt
from utils.parse_json import parse_json

llm = get_llm()

query_refinement_agent = create_agent(
    llm,
    tools=[],
    system_prompt=make_system_prompt(
        "ROLE: Query Refinement Agent\n"
        "Task:\n"
        "Transform the clarified user research question into a retrieval-ready academic search query.\n\n"
        "You MUST output JSON only:\n"
        "{\n"
        '  "refined_query": "...",\n'
        '  "keywords": ["..."],\n'
        '  "negative_keywords": ["..."]\n'
        "}\n"
    ),
)


def query_refinement_node(state: AgentState) -> AgentState:
    result = query_refinement_agent.invoke({"messages": state["messages"][-6:]})

    content = result["messages"][-1].content if result.get("messages") else "{}"
    parsed = parse_json(content)

    last = AIMessage(content=content, name="query_refinement")

    return {
        "messages": [last],
        "sender": "query_refinement",
        "refined_query": parsed.get("refined_query", ""),
        "keywords": parsed.get("keywords", []),
        "negative_keywords": parsed.get("negative_keywords", []),
    }
