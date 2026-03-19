"""
GAPAGO - main2.py
=================
main.py와 동일한 실행 흐름이지만,
query_subgraph (Query Analysis + Human Clarification) 단계까지만 실행.

Paper S-M Agent(meaning_expand → paper_retrieval)로 넘어가기 직전 상태에서 멈추고
아래 결과를 출력:
  - refined_query      : 최종 확정된 검색 쿼리
  - keywords           : 검색 키워드
  - negative_keywords  : 제외 키워드
  - ambiguity_signals  : CLAMBER + APA + 5축 통합 판정 신호 (NEW)
  - weighted_score     : 5축 가중 점수
  - iteration          : clarification 반복 횟수
  - query_analysis JSON: LLM의 전체 분석 결과 (CLAMBER/APA 포함)
"""

# =====================================================================
# 0. 환경 설정
# =====================================================================
import json
import config  # noqa: F401
import uuid
from pathlib import Path
from datetime import datetime

# =====================================================================
# 1. query_subgraph만 빌드 (paper_retrieval 이후 노드 제외)
# =====================================================================
from graphs.query_subgraph import build_subgraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

# query_subgraph를 standalone으로 실행하기 위해 checkpointer 붙임
query_app = build_subgraph()
# MemorySaver를 붙여 thread 기반 상태 유지 (Human-in-the-loop 지원)
from langgraph.graph import StateGraph, START, END
from states import AgentState
from agents import human_clarify_node, query_analysis_node

def build_standalone_query_graph():
    """
    query_subgraph를 그대로 사용하되 MemorySaver를 붙여
    Human-in-the-loop interrupt가 동작하도록 구성
    """
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    from states import AgentState
    from agents import human_clarify_node, query_analysis_node

    def route_after_query_analysis(state: AgentState) -> str:
        if state.get("is_ambiguous", False):
            return "human_clarify"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("query_analysis", query_analysis_node)
    builder.add_node("human_clarify", human_clarify_node)

    builder.add_edge(START, "query_analysis")
    builder.add_conditional_edges(
        "query_analysis",
        route_after_query_analysis,
        {"human_clarify": "human_clarify", END: END},
    )
    builder.add_edge("human_clarify", "query_analysis")

    graph = builder.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["human_clarify"],
    )
    return graph

app = build_standalone_query_graph()
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# =====================================================================
# 2. 출력 유틸 (main.py와 동일)
# =====================================================================
def random_uuid():
    return str(uuid.uuid4())

def print_divider(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)

def print_message(msg):
    if msg.type == "tool":
        try:
            data = json.loads(msg.content)
            print(f"🛠️ [Tool: {msg.name}] {len(data)} results. (Top: {data[0].get('title','?')})")
        except:
            print(f"🛠️ [Tool: {msg.name}] (Content too long)")
        return
    msg.pretty_print()

def print_stream_events_and_capture_interrupt(app, stream_input, config_dict):
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

            for key in ["iteration", "is_ambiguous", "refined_query"]:
                if key in values:
                    print(f"  {key} = {values[key]}")

            if "ambiguity_signals" in values and values["ambiguity_signals"]:
                sig = values["ambiguity_signals"]
                print(f"  ambiguity_signals = "
                      f"infogain={sig.get('infogain',0):.3f} | "
                      f"hard={sig.get('hard_fail')} | "
                      f"soft={sig.get('soft_fail')} | "
                      f"clamber={sig.get('clamber_fail')} | "
                      f"apa={sig.get('apa_fail')}")
                if sig.get("clamber_detected_types"):
                    print(f"  clamber_detected = {sig['clamber_detected_types']}")

            if "clarify_questions" in values:
                print("  clarify_questions =", values["clarify_questions"])

            for msg in values.get("messages", []):
                print_message(msg)
                if getattr(msg, "name", None) == "clarify_prompt":
                    latest_clarify_prompt = msg.content

    return interrupted, latest_clarify_prompt


# =====================================================================
# 3. 최종 결과 출력 (Paper S-M 직전 상태)
# =====================================================================
def print_query_result(values: dict):
    """query_subgraph 완료 후 Paper S-M으로 넘기기 직전 상태 출력"""
    print_divider("[ QUERY AGENT 완료 — Paper S-M Agent 진입 직전 상태 ]")

    print(f"\n  refined_query     : {values.get('refined_query', '')}")
    print(f"  keywords          : {values.get('keywords', [])}")
    print(f"  negative_keywords : {values.get('negative_keywords', [])}")
    print(f"  iteration         : {values.get('iteration', 0)}")
    print(f"  weighted_score    : {values.get('weighted_score', 0.0):.4f}")
    print(f"  is_ambiguous      : {values.get('is_ambiguous', False)}")

    # ambiguity_signals 상세
    sig = values.get("ambiguity_signals", {})
    if sig:
        print("\n  [통합 모호성 판정 신호]")
        print(f"    INFOGAIN           : {sig.get('infogain', 0.0):.4f}  "
              f"(임계값 {0.35}: 이상이면 APA-ambiguous)")
        print(f"    hard_fail (도메인) : {sig.get('hard_fail')}")
        print(f"    soft_fail (5축점수): {sig.get('soft_fail')}")
        print(f"    clamber_fail       : {sig.get('clamber_fail')}  "
              f"severity={sig.get('clamber_max_severity', 0.0):.2f}")
        print(f"    apa_fail           : {sig.get('apa_fail')}")
        if sig.get("clamber_detected_types"):
            print(f"    CLAMBER 감지 유형  : {sig['clamber_detected_types']}")

    # query_analysis JSON (CLAMBER + APA 포함)
    print("\n  [Query Analysis 전체 JSON (CLAMBER + APA 포함)]")
    for msg in values.get("messages", []):
        if getattr(msg, "name", None) == "query_analysis":
            try:
                data = json.loads(msg.content)
                # CLAMBER 요약
                clamber = data.get("clamber", {})
                detected = [k for k, v in clamber.items()
                           if isinstance(v, dict) and v.get("detected")]
                print(f"\n  CLAMBER 감지 유형: {detected}")

                # APA 요약
                pa = data.get("perceived_ambiguity", {})
                print(f"  APA infogain     : {pa.get('infogain_score', 0.0):.4f}")
                print(f"  APA dominant     : {pa.get('dominant_interpretation', '')}")
                interps = pa.get("interpretations", [])
                if interps:
                    print("  APA 해석 후보:")
                    for iv in interps:
                        print(f"    [{iv.get('plausibility',0):.2f}] {iv.get('interpretation','')}")

                # 5축 scores 요약
                scores = data.get("scores", {})
                print("\n  5축 점수:")
                for k in ["domain_clarity","task_clarity","methodology_clarity",
                          "data_clarity","temporal_clarity"]:
                    s = scores.get(k, {})
                    print(f"    {k:<25}: {s.get('score', 0.0):.2f}")
            except Exception as e:
                print(f"  (JSON 파싱 실패: {e})")
            break


def save_query_result(query: str, values: dict) -> Path:
    """query_subgraph 결과를 JSON으로 저장"""
    messages_out = []
    for msg in values.get("messages", []):
        messages_out.append({
            "type":    msg.type,
            "name":    getattr(msg, "name", None),
            "content": getattr(msg, "content", ""),
        })

    result = {
        "query":             query,
        "timestamp":         datetime.now().isoformat(),
        "refined_query":     values.get("refined_query", ""),
        "keywords":          values.get("keywords", []),
        "negative_keywords": values.get("negative_keywords", []),
        "weighted_score":    values.get("weighted_score", 0.0),
        "iteration":         values.get("iteration", 0),
        "ambiguity_signals": values.get("ambiguity_signals", {}),
        "messages":          messages_out,
    }

    fname = f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path  = OUTPUT_DIR / fname
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  ✅ Query 결과 저장 → {path}")
    return path


# =====================================================================
# 4. 실행 로직
# =====================================================================
def run():
    config_dict = {
        "configurable": {"thread_id": random_uuid()},
        "recursion_limit": 20,
    }

    default_query = "Domain adaptation"
    user_input = input("연구 질문을 입력하세요: ").strip() or default_query

    inputs = {
        "messages": [HumanMessage(content=user_input)],
        "max_iterations": 3,
    }

    print_divider("[STEP 1] Query Analysis 시작")

    interrupted, latest_clarify_prompt = print_stream_events_and_capture_interrupt(
        app, inputs, config_dict
    )

    # Human-in-the-loop clarification loop
    while interrupted:
        print_divider("[STEP 2] HUMAN CLARIFICATION 필요")

        if latest_clarify_prompt:
            print("\nAI 질문:")
            print(latest_clarify_prompt)
        else:
            print("\n질문을 더 구체화할 필요가 있습니다.")

        user_response = ""
        while not user_response:
            user_response = input("\n보완 답변 입력 > ").strip()
            if not user_response:
                print("보완 답변을 입력해야 다음 단계로 진행할 수 있습니다.")

        app.update_state(
            config_dict,
            {"messages": [HumanMessage(content=user_response)]},
        )

        print_divider("[STEP 3] 파이프라인 재개")
        interrupted, latest_clarify_prompt = print_stream_events_and_capture_interrupt(
            app, None, config_dict
        )

    # 최종 결과 출력
    final_state = app.get_state(config_dict)
    values = final_state.values if final_state else {}

    print_query_result(values)
    save_query_result(user_input, values)

    print_divider("[ Paper S-M Agent 진입 준비 완료 ]")
    print(f"  → 다음 단계로 전달될 refined_query: \"{values.get('refined_query', '')}\"")
    print(f"  → 키워드: {values.get('keywords', [])}")


if __name__ == "__main__":
    run()