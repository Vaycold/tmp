"""
Paper Retrieval Agent.
"""

from rank_bm25 import BM25Okapi
from models import AgentState
from utils import search_arxiv, tokenize
from config import config


def paper_retrieval_node(state: AgentState) -> AgentState:
    """
    Search arXiv and rank with BM25.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state
    """
    if state["iteration"] > 0:
        print(f"\n🔄 Re-running Retrieval (iteration {state['iteration']})")
    
    print(f"\n📚 Paper Retrieval Node")
    
    # Search arXiv
    papers = search_arxiv(state["refined_query"], max_results=config.ARXIV_MAX_RESULTS)
    
    if not papers:
        print("  ⚠️ No papers found")
        state["papers"] = []
        state["trace"]["papers_retrieved"] = 0
        return state
    
    print(f"  ✓ Retrieved {len(papers)} papers")
    
    # BM25 ranking
    corpus = [tokenize(p.abstract) for p in papers]
    bm25 = BM25Okapi(corpus)
    
    query_tokens = tokenize(state["refined_query"])
    scores = bm25.get_scores(query_tokens)
    
    # Sort and take top-K
    paper_scores = list(zip(papers, scores))
    paper_scores.sort(key=lambda x: x[1], reverse=True)
    top_papers = paper_scores[:config.TOP_K_PAPERS]
    
    # Update scores
    for paper, score in top_papers:
        paper.score_bm25 = float(score)
    
    state["papers"] = [p for p, _ in top_papers]
    print(f"  ✓ Selected top {len(state['papers'])} by BM25")
    
    state["trace"]["papers_retrieved"] = len(state["papers"])
    state["trace"]["avg_bm25"] = sum(p.score_bm25 for p in state["papers"]) / len(state["papers"]) if state["papers"] else 0
    
    return state