from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from states import AgentState
from agents import human_clarify_node, query_analysis_node, query_refinement_node


def route_after_query_analysis(state: AgentState) -> str:
    if state.get("is_ambiguous", False):
        return "human_clarify"
    return "next"


def build_subgraph():
    builder = StateGraph(AgentState)

    builder.add_node("query_analysis", query_analysis_node)
    builder.add_node("human_clarify", human_clarify_node)
    # builder.add_node("query_refinement", query_refinement_node)

    builder.add_edge(START, "query_analysis")

    builder.add_conditional_edges(
        "query_analysis",
        route_after_query_analysis,
        {
            "human_clarify": "human_clarify",
            "next": END,
        },
    )

    # interrupt 이후 사용자 답변이 들어오면 다시 ambiguity 재평가
    builder.add_edge("human_clarify", "query_analysis")

    # 검색용 쿼리 정제 완료 후 서브그래프 종료
    # builder.add_edge("query_refinement", END)

    graph = builder.compile(
        interrupt_before=["human_clarify"],
    )
    return graph
