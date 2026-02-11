"""
Query Analysis Agent.
"""

from models import AgentState
from llm import llm_chat, parse_json


def query_analysis_node(state: AgentState) -> AgentState:
    """
    Refine user question into search query.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state
    """
    if state["iteration"] > 0:
        print(f"\n🔄 Re-running Query Analysis (iteration {state['iteration']})")
    
    print(f"\n🔍 Query Analysis Node")
    
    prompt = f"""Given this research question, generate an optimized arXiv search query.

Research Question: {state['user_question']}

Output a JSON with:
- refined_query: Concise search string (5-10 words)
- keywords: List of 3-5 important terms
- negative_keywords: List of 1-3 terms to exclude

Output JSON only:"""
    
    messages = [
        {"role": "system", "content": "You are a research query optimizer."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = llm_chat(messages)
        result = parse_json(response)
        
        state["refined_query"] = result.get("refined_query", state["user_question"])
        state["keywords"] = result.get("keywords", [])
        state["negative_keywords"] = result.get("negative_keywords", [])
        
        print(f"  ✓ Refined: {state['refined_query']}")
        print(f"  ✓ Keywords: {', '.join(state['keywords'])}")
        
    except Exception as e:
        state["errors"].append(f"Query analysis error: {str(e)}")
        state["refined_query"] = state["user_question"]
    
    state["trace"]["query_analysis"] = state["refined_query"]
    return state