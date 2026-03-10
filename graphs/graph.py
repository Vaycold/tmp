from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from states import AgentState

from agents import (
    human_clarify_node,
    query_analysis_node,
    meaning_expand_node,
    paper_retrieval_node,
    limitation_extract_node,
    gap_infer_node,
    critic_score_node,
    final_response_node,
)


def route_after_query(state: AgentState) -> str:
    """
    query_analysis -> (meaning_expand) 기본
    모호하면 query_analysis에서 재질문/보완 후 다시 query_analysis로 루프
    """
    last = state["messages"][-1].content if state.get("messages") else ""

    if "FINAL ANSWER" in (last or ""):
        return END

    if state.get("ask_human", False) and not state.get("query_approved", False):
        return "human_clarify"

    return "meaning_expand"


def route_after_critic(state: AgentState) -> str:
    """
    critic_score ->
      - ACCEPT          -> final_response
      - REDO_RETRIEVAL  -> meaning_expand -> paper_retrieval
      - REFINE_QUERY    -> query_analysis
      - FINAL ANSWER    -> END
    """
    last = state["messages"][-1].content or ""

    if "FINAL ANSWER" in last:
        return END

    if "DECISION: ACCEPT" in last:
        return "final_response"

    if "DECISION: REDO_RETRIEVAL" in last:
        return "meaning_expand"

    if "DECISION: REFINE_QUERY" in last:
        return "query_analysis"

    return "query_analysis"


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("human_clarify", human_clarify_node)
    workflow.add_node("meaning_expand", meaning_expand_node)
    workflow.add_node("paper_retrieval", paper_retrieval_node)
    workflow.add_node("limitation_extract", limitation_extract_node)
    workflow.add_node("gap_infer", gap_infer_node)
    workflow.add_node("critic_score", critic_score_node)
    workflow.add_node("final_response", final_response_node)

    workflow.add_edge(START, "query_analysis")

    workflow.add_conditional_edges(
        "query_analysis",
        route_after_query,
        {
            "human_clarify": "human_clarify",
            "meaning_expand": "meaning_expand",
            END: END,
        },
    )

    workflow.add_edge("human_clarify", "query_analysis")
    workflow.add_edge("meaning_expand", "paper_retrieval")
    workflow.add_edge("paper_retrieval", "limitation_extract")
    workflow.add_edge("limitation_extract", "gap_infer")
    workflow.add_edge("gap_infer", "critic_score")

    workflow.add_conditional_edges(
        "critic_score",
        route_after_critic,
        {
            "meaning_expand": "meaning_expand",
            "query_analysis": "query_analysis",
            "final_response": "final_response",
            END: END,
        },
    )

    workflow.add_edge("final_response", END)

    graph = workflow.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["human_clarify"],
    )

    return graph
