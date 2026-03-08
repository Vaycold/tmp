# 3-5) Critic Score Agent
from states import AgentState
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from tools import build_role_tools
from prompts.system import make_system_prompt
from llm import get_llm
from utils.critic_utils import _compute_critic_scores

llm = get_llm()

ROLE_TOOLS = build_role_tools()
CRITIC_TOOLS = ROLE_TOOLS["CRITIC_TOOLS"]

critic_score_agent = create_agent(
    llm,
    tools=CRITIC_TOOLS,
    system_prompt=make_system_prompt(
        "ROLE: Critic Agent\n"
        "You evaluate the pipeline outputs (query specificity, paper alignment, evidence support) "
        "using pre-computed scores provided as input.\n"
        "Return scores + flags + a routing decision tag among:\n"
        "- DECISION: ACCEPT (all scores sufficient)\n"
        "- DECISION: REDO_RETRIEVAL (paper_relevance too low)\n"
        "- DECISION: REFINE_QUERY (query_specificity too low)\n\n"

        "INPUT SCORES:\n"
        "- query_specificity: ratio of content words (excluding stopwords) in the query (0.0~1.0)\n"
        "- paper_relevance: normalized BM25 score relative to the highest scoring paper (0.0~1.0)\n"
        "- groundedness: ratio of limitations with valid evidence_quote (5+ words) (0.0~1.0)\n\n"

        "YOUR TASKS:\n"
        "1. Evaluate the scores in context and return flags for any low-quality signals.\n"
        "2. Return a routing decision tag as specified above.\n"
        "3. Provide a brief rationale for the decision.\n\n"

        "Do NOT generate new gaps/topics.\n"
    ),
)


def critic_score_node(state: AgentState) -> AgentState:

    # refined_query fallback
    refined_query = (
        state.get("refined_query")
        or next(
            (m.content for m in reversed(state.get("messages", []))
             if getattr(m, "name", None) == "query_analysis"),
            state.get("user_question", "")
        )
    )

    # fallback 변수 사용
    scores = _compute_critic_scores(
        refined_query=refined_query,
        papers=state.get("papers", []),
        limitations=state.get("limitations", []),
    )

    query_spec = scores["query_specificity"]
    paper_rel  = scores["paper_relevance"]
    grounded   = scores["groundedness"]

    # 계산된 점수를 에이전트 입력으로 전달
    score_summary = (
        f"Computed Scores:\n"
        f"- query_specificity: {query_spec:.2f}\n"
        f"- paper_relevance:   {paper_rel:.2f}\n"
        f"- groundedness:      {grounded:.2f}\n\n"
        f"Please evaluate these scores and return a routing decision."
    )

    result = critic_score_agent.invoke({
        **state,
        "messages": [AIMessage(content=score_summary)]
    })

    # 이전 코드 제거, 반환값에 포함
    current_trace = state.get("trace") or {}
    current_trace["critic_scores"] = {
        "query_specificity": query_spec,
        "paper_relevance":   paper_rel,
        "groundedness":      grounded,
    }

    last = AIMessage(content=result["messages"][-1].content, name="critic_score")
    return {
        "messages": [last],
        "sender": "critic_score",
        "iteration": state.get("iteration", 0) + 1, 
        "trace": current_trace,                       
    }