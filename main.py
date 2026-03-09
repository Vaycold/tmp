"""
GAPAGO - Research GAP Analysis Multi-Agent System
기존 모듈(agents/, states.py, graph.py, llm.py, utils/)을 활용한 실행 진입점
"""

# =====================================================================
# 0. 환경 설정
# =====================================================================
import config  # noqa: F401

# =====================================================================
# 1. 그래프 빌드
# =====================================================================
from graphs.graph import build_graph
from langchain_core.messages import HumanMessage

app = build_graph()


# =====================================================================
# 2. 출력 유틸
# =====================================================================
def print_divider(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)


def print_message(msg):
    # ToolMessage인 경우에만 요약 출력 (디버깅 효율)
    if msg.type == "tool":
        try:
            import json

            data = json.loads(msg.content)
            print(
                f"🛠️ [Tool: {msg.name}] {len(data)} results retrieved. (Top 1: {data[0].get('title', 'No Title')})"
            )
        except:
            print(f"🛠️ [Tool: {msg.name}] (Content too long to display)")
        return

    # 그 외 Human, AI Message는 깔끔하게 출력
    msg.pretty_print()


def print_stream_events_and_capture_interrupt(app, stream_input, config_dict):
    """
    subgraphs=True로 이벤트를 출력하면서
    - clarify_prompt
    - interrupt 발생 여부
    를 함께 수집
    """
    interrupted = False
    latest_clarify_prompt = None

    for i, event in enumerate(app.stream(stream_input, config_dict, subgraphs=True)):
        path, update = event

        print(f"\n===== EVENT {i} =====")
        print("PATH:", " -> ".join(path) if path else "(root)")

        for node, values in update.items():
            if node == "__interrupt__":
                interrupted = True
                print("\n*** INTERRUPT ***")
                continue

            print(f"\n--- NODE: {node} ---")

            if not isinstance(values, dict):
                print(values)
                continue

            # 상태값 일부 출력
            for key in ["iteration", "is_ambiguous", "forced_proceed", "refined_query"]:
                if key in values:
                    print(f"{key} = {values[key]}")

            if "clarify_questions" in values:
                print("clarify_questions =", values["clarify_questions"])

            # 메시지 출력
            for msg in values.get("messages", []):
                print_message(msg)

                if getattr(msg, "name", None) == "clarify_prompt":
                    latest_clarify_prompt = msg.content

    return interrupted, latest_clarify_prompt


# =====================================================================
# 3. 실행 로직
# =====================================================================
def run():
    config_dict = {"configurable": {"thread_id": "1"}, "recursion_limit": 30} # 최대 노드 실행 개수 지정 (순환 로직에 빠지지 않기 위함)

    # --- 사용자 입력 ---
    default_query = "Domain adaptation"
    user_input = input("연구 질문을 입력하세요: ").strip() or default_query
    if not user_input:
        user_input = "Domain adaptation in clinical drug"

    inputs = {
        "messages": [HumanMessage(content=user_input)],
        "max_iterations": 3,
    }

    print_divider("[STEP 1] 초기 실행")

    # 첫 실행은 inputs 사용
    interrupted, latest_clarify_prompt = print_stream_events_and_capture_interrupt(
        app, inputs, config_dict
    )

    # -----------------------------------------------------------------
    # Human-in-the-loop clarification loop
    # -----------------------------------------------------------------
    while interrupted:
        print_divider("[STEP 2] HUMAN CLARIFICATION 필요")

        if latest_clarify_prompt:
            print("\nAI 질문:")
            print(latest_clarify_prompt)
        else:
            print("\n질문을 더 구체화할 필요가 있습니다. 추가 정보를 입력해주세요.")

        user_response = ""
        while not user_response:
            user_response = input(
                "\n보완 답변 입력 > "
            ).strip()  ## ex. domain adaptation for fault detection in smart factory
            if not user_response:
                print("보완 답변을 입력해야 다음 단계로 진행할 수 있습니다.")

        # 사용자 답변을 messages에 추가
        app.update_state(
            config_dict,
            {
                "messages": [HumanMessage(content=user_response)],
            },
        )

        print_divider("[STEP 3] 파이프라인 재개")

        # resume 시에는 stream_input = None
        interrupted, latest_clarify_prompt = print_stream_events_and_capture_interrupt(
            app, None, config_dict
        )

    # -----------------------------------------------------------------
    # 최종 결과 출력
    # -----------------------------------------------------------------
    print_divider("[STEP 4] 최종 상태")

    final_state = app.get_state(config_dict)
    values = final_state.values if final_state else {}

    print("next =", final_state.next if final_state else None)
    print("iteration =", values.get("iteration"))
    print("is_ambiguous =", values.get("is_ambiguous"))
    print("refined_query =", values.get("refined_query"))

    print_divider("[파이프라인 완료]")

    if values.get("messages"):
        print(values["messages"][-1].content)
    else:
        print(values)


if __name__ == "__main__":
    run()
