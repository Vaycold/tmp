"""
Critic Scoring Agent.
"""

from models import AgentState, CriticScores


def critic_score_node(state: AgentState) -> AgentState:
    """
    Calculate quality scores.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state
    """
    print(f"\n⭐ Critic Scoring Node")
    
    # Query specificity
    query_len = len(state["refined_query"].split())
    keyword_count = len(state["keywords"])
    query_spec = min((query_len / 10.0) * 0.6 + (keyword_count / 5.0) * 0.4, 1.0)
    
    # Paper relevance
    if state["papers"]:
        avg_bm25 = sum(p.score_bm25 for p in state["papers"]) / len(state["papers"])
        paper_rel = min(avg_bm25 / 50.0, 1.0)
    else:
        paper_rel = 0.0
    
    # Groundedness
    if state["limitations"]:
        with_evidence = sum(1 for lim in state["limitations"] if lim.evidence_quote)
        grounded = with_evidence / len(state["limitations"])
    else:
        grounded = 0.0
    
    state["critic"] = CriticScores(
        query_specificity=query_spec,
        paper_relevance=paper_rel,
        groundedness=grounded
    )
    
    print(f"  ✓ Query Specificity: {query_spec:.2f}")
    print(f"  ✓ Paper Relevance: {paper_rel:.2f}")
    print(f"  ✓ Groundedness: {grounded:.2f}")
    
    # Increment iteration
    state["iteration"] += 1
    
    state["trace"]["critic_scores"] = {
        "query_specificity": query_spec,
        "paper_relevance": paper_rel,
        "groundedness": grounded
    }
    
    return state