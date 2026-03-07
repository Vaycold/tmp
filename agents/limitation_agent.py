# 3-3) Limitation Extract Agent
from states import AgentState
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from tools import build_role_tools
from prompts.system import make_system_prompt
from llm import get_llm

llm = get_llm(temperature = 0)


ROLE_TOOLS = build_role_tools()
LIMITATION_TOOLS = ROLE_TOOLS["LIMITATION_TOOLS"]

limitation_extract_agent = create_agent(
    llm,
    tools=LIMITATION_TOOLS,
    system_prompt=make_system_prompt(
        "ROLE: Limitation Extract Agent\n"
        "You extract limitation/future-work statements from retrieved papers/snippets.\n"
        "RULES:\n"
        "1. Process ALL papers in the input without exception.\n"
        "2. Extract 1-2 key limitations per paper. No more, no less.\n"
        "3. Each limitation MUST include:\n"
        "   - paper_id: the paper's unique identifier\n"
        "   - claim: a brief limitation statement (1-2 sentences)\n"
        "   - evidence_quote: an exact quote from the abstract or body\n"
        "4. Do NOT infer or assume limitations not stated in the text.\n"
        "5. Do NOT skip any paper even if the abstract is short or unclear.\n"
        "6. Do NOT infer gaps yet.\n\n"

        "OUTPUT FORMAT (strictly follow):\n"
        "paper_id: <id>\n"
        "  - claim: <limitation statement>\n"
        "    evidence_quote: <exact quote from paper>\n"
        "  - claim: <limitation statement>\n"
        "    evidence_quote: <exact quote from paper>\n"
        "Output be structured: paper_id -> [limitation_sentences], plus brief rationale.\n"
        "Do NOT infer gaps yet.\n"
    ),
)


def limitation_extract_node(state: AgentState) -> AgentState:
    """
    - LangChain 에이전트 + AIMessage 반환 구조 유지
    - 모든 논문 순회 + 에러 격리
    - 병목 방지: limitation_extract 메시지만 반환
    """
    papers = state.get("papers", [])

    # 논문 없을 때 조기 반환
    if not papers:
        print("  ⚠️ No papers to analyze")
        empty_msg = AIMessage(content="No papers to analyze.", name="limitation_extract")
        return {"messages": [empty_msg], "sender": "limitation_extract"}

    all_limitations = []
    errors = []

    # 논문별 개별 순회 처리
    for paper in papers:
        try:
            result = limitation_extract_agent.invoke({
                **state,
                # 현재 논문 메시지만 전달: GAP agent와의 병목 방지 
                "messages": [
                    AIMessage(content=(
                        f"Extract limitations from the following paper.\n\n"
                        f"paper_id: {paper.paper_id}\n"
                        f"Title: {paper.title}\n"
                        f"Abstract: {paper.abstract}"
                    ))
                ]
            })
            last_content = result["messages"][-1].content
            all_limitations.append(last_content)
            print(f"  ✓ Extracted limitations for {paper.paper_id}")

        except Exception as e:
            error_msg = f"Limitation extraction error for {paper.paper_id}: {str(e)}"
            errors.append(error_msg)
            print(f"  ⚠️ {error_msg}")
            continue
    combined_content = "\n\n".join(all_limitations)

    if errors:
        combined_content += "\n\nERRORS:\n" + "\n".join(errors)

    print(f"  ✓ Processed {len(all_limitations)}/{len(papers)} papers")

    # AIMessage 반환 구조 유지
    last = AIMessage(content=combined_content, name="limitation_extract")
    return {"messages": [last], "sender": "limitation_extract"}
