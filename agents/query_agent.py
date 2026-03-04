# 3-1) Query Analysis Agent
from states import AgentState
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from tools import build_role_tools
from prompts.system import make_system_prompt
from llm import get_llm

llm = get_llm("azure", "gpt-5.1-chat")


ROLE_TOOLS = build_role_tools()
QUERY_TOOLS = ROLE_TOOLS["QUERY_TOOLS"]

query_analysis_agent = create_agent(
    llm,
    tools=QUERY_TOOLS,
    system_prompt=make_system_prompt(
        "ROLE: Query Analysis Agent\n"
        "Task:\n"
        "1) Detect ambiguity: domain/data/method/time/metrics etc.\n"
        "2) Propose a refined search query.\n"
        "3) If ambiguous, ask for missing slots and request user approval.\n\n"
        "You MUST output:\n"
        "- ROUTE: RETRIEVE  OR  ROUTE: NEED_USER_CLARIFICATION\n"
        "- query_proposal: <string>\n"
        "- missing_slots: [..]\n"
        "- clarify_questions: [..]\n"
        "- keywords: [..]\n"
        "- negative_keywords: [..]\n\n"
        "Keep it concise and machine-parseable (JSON-like is OK).\n"
    ),
)


def query_analysis_node(state: AgentState) -> AgentState:
    it = state.get("iteration", 0) + 1

    result = query_analysis_agent.invoke(state)
    content = result["messages"][-1].content if result.get("messages") else ""

    need_clarify = "ROUTE: NEED_USER_CLARIFICATION" in (content or "")
    max_it = state.get("max_iterations", 3)
    if it >= max_it:
        need_clarify = False

    # ask_human: interrupt
    ask_human = bool(need_clarify) and not state.get("query_approved", False)

    last = AIMessage(content=content, name="query_analysis")

    return {
        "messages": [last],
        "sender": "query_analysis",
        "ask_human": ask_human,
        "iteration": it,
    }


def human_clarify_node(state: AgentState) -> AgentState:
    """
    Interrupt 전용 노드
    여기서는 LLM 호출/자동 응답 생성하지 않음
    UI/CLI에서 snapshot을 보고 사용자 입력을 받은 뒤 app.update_state로 아래 필드를 채움:
      - query_approved: bool
      - (optional) refined_query or user feedback message
    """
    # 아무것도 변경하지 않아도 됨 (interrupt로 멈출 것)
    return {"sender": "human_clarify"}
