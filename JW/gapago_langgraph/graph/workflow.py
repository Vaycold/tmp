"""
LangGraph workflow construction.
"""

from langgraph.graph import StateGraph, END
from models import AgentState
from agents import (
    query_analysis_node,
    paper_retrieval_node,
    limitation_extract_node,
    gap_infer_node,
    critic_score_node
)


def route_decision(state: AgentState) -> str:
    """
    Conditional routing based on critic scores.
    
    Args:
        state: Current agent state
        
    Returns:
        Next route: "refine_query" | "redo_retrieval" | "accept"
    """
    print(f"\n🔀 Router (Iteration {state['iteration']}/{state['max_iterations']})")
    
    # Check max iterations
    if state["iteration"] >= state["max_iterations"]:
        print(f"  → ACCEPT (max iterations)")
        return "accept"
    
    critic = state.get("critic")
    
    if critic is None:
        print(f"  → ACCEPT (no critic)")
        return "accept"
    
    # Decision logic
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
    """
    Build LangGraph workflow.
    
    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("paper_retrieval", paper_retrieval_node)
    workflow.add_node("limitation_extract", limitation_extract_node)
    workflow.add_node("gap_infer", gap_infer_node)
    workflow.add_node("critic_score", critic_score_node)
    
    # Define edges
    workflow.set_entry_point("query_analysis")
    workflow.add_edge("query_analysis", "paper_retrieval")
    workflow.add_edge("paper_retrieval", "limitation_extract")
    workflow.add_edge("limitation_extract", "gap_infer")
    workflow.add_edge("gap_infer", "critic_score")
    
    # Conditional routing
    workflow.add_conditional_edges(
        "critic_score",
        route_decision,
        {
            "refine_query": "query_analysis",
            "redo_retrieval": "paper_retrieval",
            "accept": END
        }
    )
    
    return workflow.compile()