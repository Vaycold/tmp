# human_agent.py

from models import AgentState


def human_clarify_node(state: AgentState) -> AgentState:
    """
    Human-in-the-loop clarification node.
    - Prints questions (from trace) and collects user's answer.
    - Stores answer into trace['memory']['clarifications'] for continuity.
    - Clears route and returns to query_analysis.
    """
    print("\n🧾 Human Clarification Node")

    if "trace" not in state or state["trace"] is None:
        state["trace"] = {}
    if "errors" not in state or state["errors"] is None:
        state["errors"] = []

    questions = state["trace"].get("clarify_questions", []) or []
    if questions:
        print("\n[Clarifying Questions]")
        for i, q in enumerate(questions, 1):
            print(f"{i}. {q}")
        questions_summary = " / ".join(questions[:3])
    else:
        print("\n추가 정보가 필요합니다. 연구 범위/도메인/방법을 구체화해 주세요.")
        questions_summary = "generic_clarification"

    answer = input("\n추가 정보를 입력하세요: ").strip()

    # 임시 메모리(연속성) 저장: top-level state key 추가 없이 trace 사용
    mem = state["trace"].get("memory", {}) or {}
    clarifs = mem.get("clarifications", []) or []
    clarifs.append({
        "iteration": state.get("iteration", 0),
        "questions_summary": questions_summary,
        "answer": answer,
    })
    mem["clarifications"] = clarifs
    state["trace"]["memory"] = mem

    # 다음 query_analysis에서 사용하도록 route 초기화
    state["route"] = ""
    return state
