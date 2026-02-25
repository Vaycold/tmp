# workflow.py

from langgraph.graph import StateGraph, END
from models import AgentState
from agents import (
    ambiguity_check_node,
    query_analysis_node,
    paper_retrieval_node,
    limitation_extract_node,
    gap_infer_node,
    critic_score_node,
    human_clarify_node
)
# from human_agent import human_clarify_node  # 추가


def route_after_ambiguity(state : AgentState) -> str : 
    """
    [NEW] ambiguity_check 이후 라우팅.
      - route == "ambiguous" → human_clarify  (질문 보완 요청)
      - route == "clear"     → query_analysis (쿼리 생성 진행)
    """
    return "clarify" if state.get("route") == "ambiguous" else "query_analysis"

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


def build_graph(interrupt_before=None) -> StateGraph:
    
    workflow = StateGraph(AgentState)

    workflow.add_node("ambiguity_check", ambiguity_check_node)   # ← NEW
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("human_clarify", human_clarify_node)
    workflow.add_node("paper_retrieval", paper_retrieval_node)
    workflow.add_node("limitation_extract", limitation_extract_node)
    workflow.add_node("gap_infer", gap_infer_node)
    workflow.add_node("critic_score", critic_score_node)
    

    
    workflow.set_entry_point("ambiguity_check")                  # 진입점
    
    workflow.add_conditional_edges(
        "ambiguity_check",
        route_after_ambiguity,
        {
            "clarify": "human_clarify",
            "query_analysis": "query_analysis",
        },
    )
    '''
    clarify -> query_analysis # 모호할 경우 -> HIL
    query_analysis -> query_analysis # 확실할 경우 -> query_analysis로 가기
    '''
    
    
    # ── human_clarify → ambiguity_check 복귀 (답변 후 재검증) ────
    workflow.add_edge("human_clarify", "ambiguity_check")        # ← 기존 query_analysis → 변경

    # # ✅ query_analysis 이후 조건부 라우팅
    # workflow.add_conditional_edges(
    #     "query_analysis",
    #     route_after_query,
    #     {
    #         "clarify": "human_clarify",
    #         "retrieve": "paper_retrieval",
            
    #     },
    # )
    
    workflow.add_edge("query_analysis", "paper_retrieval")
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

    
        # ── compile: interrupt_before 버전 호환 처리 ──────────────────
    # LangGraph 0.1.x: compile()에 interrupt_before 미지원 → checkpoint 필요
    # LangGraph 0.2+:  compile(interrupt_before=[...]) 직접 지원
    _interrupt = interrupt_before or []
    try:
        return workflow.compile(interrupt_before=_interrupt)
    except TypeError:
        # 구버전 fallback: interrupt 없이 컴파일 (main.py의 루프로 대응)
        return workflow.compile()
