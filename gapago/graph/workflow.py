# workflow.py

from langgraph.graph import StateGraph, END
from models import AgentState
from agents import (
    query_analysis_node,
    paper_retrieval_node,
    limitation_extract_node,
    gap_infer_node,
    critic_score_node,
    human_clarify_node
)
# from human_agent import human_clarify_node  # 추가


def route_after_query(state: AgentState) -> str:
    """query_analysis 이후 라우팅."""
    return "clarify" if state.get("route") == "clarify" else "retrieve"


def route_decision(state: AgentState) -> str:
    """
    기존 critic 기반 라우팅 (그대로 유지)
    - refine_query: query_analysis로
    - redo_retrieval: paper_retrieval로
    - accept: END
    """
    print(f"\n🔀 Router (Iteration {state['iteration']}/{state['max_iterations']})")

    if state["iteration"] >= state["max_iterations"]:
        print(f"  → ACCEPT (max iterations)")
        return "accept"

    critic = state.get("critic")
    if critic is None:
        print(f"  → ACCEPT (no critic)")
        return "accept"

    if critic.query_specificity < 0.55:
        print(f"  → REFINE (specificity: {critic.query_specificity:.2f})")
        return "refine_query"

    if critic.paper_relevance < 0.55:
        print(f"  → REDO (relevance: {critic.paper_relevance:.2f})")
        return "redo_retrieval"

    if critic.groundedness < 0.60:
        print(f"  → REDO (groundedness: {critic.groundedness:.2f})")
        return "redo_retrieval"

    print(f"  → ACCEPT (all scores pass)")
    return "accept"


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("human_clarify", human_clarify_node)  # 추가
    workflow.add_node("paper_retrieval", paper_retrieval_node)
    workflow.add_node("limitation_extract", limitation_extract_node)
    workflow.add_node("gap_infer", gap_infer_node)
    workflow.add_node("critic_score", critic_score_node)


    workflow.set_entry_point("query_analysis")

    # ✅ query_analysis 이후 조건부 라우팅
    workflow.add_conditional_edges(
        "query_analysis",
        route_after_query,
        {
            "clarify": "human_clarify",
            "retrieve": "paper_retrieval",
            
        },
    )

    # ✅ human_clarify는 다시 query_analysis로 복귀
    workflow.add_edge("human_clarify", "query_analysis")

    workflow.add_edge("paper_retrieval", "limitation_extract")
    workflow.add_edge("limitation_extract", "gap_infer")
    workflow.add_edge("gap_infer", "critic_score")

    workflow.add_conditional_edges(
        "critic_score",
        route_decision,
        {
            "refine_query": "query_analysis",
            "redo_retrieval": "paper_retrieval",
            "accept": END,
        },
    )

    
    return workflow.compile()
