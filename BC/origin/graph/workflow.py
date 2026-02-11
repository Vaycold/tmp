"""
GAPAGO Workflow
전체 시스템 통합
"""

import sys
sys.path.append('/home/claude/gapago_project')

from langgraph.graph import StateGraph, END
from state.state import GAPAGOState

# Agent 노드들 import
from agents.query_analysis.node import query_analysis_node
from agents.critic.node import critic_query_node, critic_paper_node, critic_gap_node
from agents.paper_search.node import paper_search_node
from agents.web_search.node import web_search_node
from agents.gap_classification.node import gap_classification_node
from agents.topic_generation.node import topic_generation_node

# Orchestrator
from agents.orchestrator.logic import (
    orchestrator_after_query,
    orchestrator_after_paper,
    orchestrator_after_gap
)


def create_gapago_workflow():
    """GAPAGO 워크플로우 생성"""
    
    workflow = StateGraph(GAPAGOState)
    
    # ═══════════════════════════════════════════════════════
    # 노드 추가
    # ═══════════════════════════════════════════════════════
    
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("critic_query", critic_query_node)
    
    # Paper S-M (병렬)
    workflow.add_node("paper_search", paper_search_node)
    workflow.add_node("web_search", web_search_node)
    
    workflow.add_node("critic_paper", critic_paper_node)
    workflow.add_node("gap_classification", gap_classification_node)
    workflow.add_node("critic_gap", critic_gap_node)
    workflow.add_node("topic_generation", topic_generation_node)
    
    # ═══════════════════════════════════════════════════════
    # 시작점
    # ═══════════════════════════════════════════════════════
    
    workflow.set_entry_point("query_analysis")
    
    # ═══════════════════════════════════════════════════════
    # 엣지 연결
    # ═══════════════════════════════════════════════════════
    
    # 1. Query Analysis → Critic Query (순차)
    workflow.add_edge("query_analysis", "critic_query")
    
    # 2. Critic Query → Orchestrator (조건부)
    workflow.add_conditional_edges(
        "critic_query",
        orchestrator_after_query,
        {
            "human_loop": "query_analysis",  # 모호함 → 재분석 (Human Loop 후)
            "paper_search": "paper_search"   # 명확함 → Paper Search
        }
    )
    
    # 3. Paper Search → Web Search (병렬)
    workflow.add_edge("paper_search", "web_search")
    
    # 4. Web Search → Critic Paper (병렬 종료)
    workflow.add_edge("web_search", "critic_paper")
    
    # 5. Critic Paper → Orchestrator (조건부)
    workflow.add_conditional_edges(
        "critic_paper",
        orchestrator_after_paper,
        {
            "paper_search": "paper_search",           # 부족 → 재검색
            "gap_classification": "gap_classification" # 충분 → GAP 분류
        }
    )
    
    # 6. GAP Classification → Critic GAP (순차)
    workflow.add_edge("gap_classification", "critic_gap")
    
    # 7. Critic GAP → Orchestrator (조건부)
    workflow.add_conditional_edges(
        "critic_gap",
        orchestrator_after_gap,
        {
            "paper_search": "paper_search",           # 낮음 → 재검색
            "topic_generation": "topic_generation"    # 높음 → Topic 생성
        }
    )
    
    # 8. Topic Generation → END
    workflow.add_edge("topic_generation", END)
    
    return workflow.compile()


# ═══════════════════════════════════════════════════════════
# 시각화 (선택)
# ═══════════════════════════════════════════════════════════

def visualize_workflow():
    """워크플로우 시각화"""
    app = create_gapago_workflow()
    
    try:
        from IPython.display import Image, display
        display(Image(app.get_graph().draw_mermaid_png()))
    except:
        print("Mermaid 다이어그램:")
        print(app.get_graph().draw_mermaid())


if __name__ == "__main__":
    print("✅ GAPAGO Workflow 생성 완료")
    print("\n워크플로우 구조:")
    print("""
    Query Analysis
        ↓
    Critic Query
        ↓
    [Orchestrator] → Human Loop (모호함) or Paper Search (명확함)
        ↓
    Paper Search (병렬)
        ↓
    Web Search (병렬)
        ↓
    Critic Paper
        ↓
    [Orchestrator] → Paper Search (재검색) or GAP Classification
        ↓
    GAP Classification
        ↓
    Critic GAP
        ↓
    [Orchestrator] → Paper Search (재검색) or Topic Generation
        ↓
    Topic Generation
        ↓
    END
    """)
