# =====================================================================
# ============================== Node =================================
# =====================================================================
"""
노드(Node)에 대한 모든 것
1. 핵심 로직에 해당하는 함수, 노드를 만드는 것이 중요함.
   세부 단계(작은 단위)로 나눠 노드를 만들 수록 더욱 정교한 튜닝(흐름) 가능함
2. 노드를 만드는 법?
    - 함수로 정의
    - 사전에 정의한 상태가 'Input(입력)' & 'Output(출력)' 임
    - Input(입력) = state: AgentState
      : 상태를 입력으로 받아 필요한 정보를 상태로부터 받아 꺼내서 사용
      : 딕셔너리이므로 필요한 'Key'로 조회해서 사용
    - Output(출력)
      : 상태에 담아서 보냄
"""


# 3-1) Query Analysis Agent
from states import AgentState
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from tools import build_role_tools
from prompts.system import make_system_prompt
from llm import get_llm

llm = get_llm()


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
    
    # query_proposal 파싱해서 refined_query에 저장 for critic_agent
    refined_query = state.get("refined_query", "")
    for line in content.splitlines():
        if "query_proposal:" in line.lower():
            refined_query = line.split(":", 1)[-1].strip().strip('"')
            break

    last = AIMessage(content=content, name="query_analysis")

    return {
        "messages": [last],
        "sender": "query_analysis",
        "ask_human": ask_human,
        "iteration": it,
        "refined_query": refined_query,
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
