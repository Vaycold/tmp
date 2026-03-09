"""
GAPAGO - Research GAP Analysis Multi-Agent System
기존 모듈(agents/, states.py, graph.py, llm.py, utils/)을 활용한 실행 진입점
"""

# =====================================================================
# 0. 환경 설정
#    - config.py 가 load_dotenv() + langsmith 초기화를 side-effect로 수행
# =====================================================================
import config  # noqa: F401

# =====================================================================
# 1. 그래프 빌드
#    - graph.py 의 build_graph() 가 모든 노드/엣지/라우팅을 조립
#    - 노드 함수들은 agents/ 폴더에서 import
#    - AgentState 는 states.py 에서 import
# =====================================================================
from graph import build_graph
from langchain_core.messages import HumanMessage

app = build_graph()


# =====================================================================
# 2. 실행 로직
# =====================================================================
def run():
    config_dict = {"configurable": {"thread_id": "1"}}

    # --- 사용자 입력 ---
    user_input = input("연구 질문을 입력하세요: ").strip()
    if not user_input:
        user_input = "Domain adaptation in clinical drug"

    inputs = {"messages": [HumanMessage(content=user_input)]}

    # --- Step 1: 초기 실행 (query_analysis까지, human_clarify 직전에 interrupt) ---
    print("\n[STEP 1] Query Analysis 실행 중...\n")
    output = app.invoke(inputs, config_dict)
    print("=>", output["messages"][-1].content)

    # --- Step 2: Human-in-the-loop 처리 ---
    snapshot = app.get_state(config_dict)
    if snapshot.next and "human_clarify" in snapshot.next:
        print("\n[HUMAN CLARIFICATION 필요]")
        print("Query를 승인하려면 'y', 수정할 내용이 있으면 직접 입력하세요.")
        user_response = input("> ").strip()

        if user_response.lower() == "y":
            app.update_state(
                config_dict,
                {
                    "query_approved": True,
                    "ask_human": False,
                    "messages": [HumanMessage(content="APPROVE")],
                },
            )
        else:
            # 사용자 피드백을 메시지로 넣고 query_analysis 재실행
            app.update_state(
                config_dict,
                {
                    "query_approved": False,
                    "ask_human": False,
                    "messages": [HumanMessage(content=user_response)],
                },
            )

        # --- Step 3: 파이프라인 재개 (paper_retrieval -> ... -> final_response) ---
        print("\n[STEP 3] 파이프라인 재개 중...\n")
        for event in app.stream(None, config_dict, stream_mode="values"):
            if "messages" in event:
                msg = event["messages"][-1]
                agent_name = getattr(msg, "name", "agent")
                print(f"[{agent_name}] {msg.content}\n")

    else:
        # human clarify 없이 바로 완료된 경우
        print("\n[파이프라인 완료]\n")
        print(output["messages"][-1].content)


if __name__ == "__main__":
    run()