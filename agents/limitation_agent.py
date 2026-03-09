# 3-3) Limitation Extract Agent
from states import AgentState
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from tools import build_role_tools
from prompts.system import make_system_prompt
from llm import get_llm

llm = get_llm()

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
        # ✅ abstract or body → full text sections 반영
        "   - evidence_quote: an exact quote from the provided text sections\n"
        "4. Do NOT infer or assume limitations not stated in the text.\n"
        "5. Do NOT skip any paper even if the abstract is short or unclear.\n"
        "6. Do NOT infer gaps yet.\n\n"

        # ✅ 섹션별 신뢰도 기준 추가
        "SECTION PRIORITY (high to low):\n"
        "  1. INTRODUCTION  — author-defined gaps, most reliable\n"
        "  2. CONCLUSION    — key contributions + limitations\n"
        "  3. LIMITATIONS   — author-stated weaknesses\n"
        "  4. DISCUSSION    — result interpretation + limitations\n"
        "  5. ABSTRACT      — fallback only, least detail\n"
        "  6. FUTURE_WORK   — supplementary evidence only\n\n"

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

def _build_paper_context(paper) -> str:
    sections = getattr(paper, "full_text_sections", {}) or {}

    if not sections:
        print(f"    [DEBUG] {paper.paper_id} → abstract fallback")
        return f"[Source: abstract only]\n{paper.abstract}"

    parts = [f"[Source: full text sections — {list(sections.keys())}]"]

    priority = ["introduction", "conclusion", "limitations", "discussion", "future_work"]
    for key in priority:
        if key in sections:
            # ✅ 섹션별 텍스트 길이 확인
            print(f"    [DEBUG] {paper.paper_id} → [{key}] {len(sections[key])} chars")
            parts.append(f"[{key.upper()}]\n{sections[key]}")

    return "\n\n".join(parts)


def limitation_extract_node(state: AgentState) -> AgentState:
    papers = state.get("papers", [])

    if not papers:
        print("  ⚠️ No papers to analyze")
        empty_msg = AIMessage(content="No papers to analyze.", name="limitation_extract")
        return {"messages": [empty_msg], "sender": "limitation_extract"}

    all_limitations = []
    errors = []

    for paper in papers:
        try:
            paper_context = _build_paper_context(paper)

            # ✅ 에이전트에 넘기는 전체 입력 확인
            input_message = (
                f"Extract limitations from the following paper.\n\n"
                f"paper_id: {paper.paper_id}\n"
                f"Title: {paper.title}\n\n"
                f"{paper_context}"
            )
            print(f"\n    [DEBUG] ===== INPUT TO AGENT: {paper.paper_id} =====")
            print(input_message[:500])  # 너무 길면 앞 500자만 출력
            print(f"    [DEBUG] ... (total {len(input_message)} chars)")

            result = limitation_extract_agent.invoke({
                **state,
                "messages": [AIMessage(content=input_message)]
            })

            last_content = result["messages"][-1].content

            # ✅ 에이전트 출력 확인
            print(f"\n    [DEBUG] ===== OUTPUT FROM AGENT: {paper.paper_id} =====")
            print(last_content[:500])
            print(f"    [DEBUG] ... (total {len(last_content)} chars)")

            all_limitations.append(last_content)

            sections = getattr(paper, "full_text_sections", {}) or {}
            source = f"full text {list(sections.keys())}" if sections else "abstract only"
            print(f"  ✓ Extracted limitations for {paper.paper_id} [{source}]")

        except Exception as e:
            error_msg = f"Limitation extraction error for {paper.paper_id}: {str(e)}"
            errors.append(error_msg)
            print(f"  ⚠️ {error_msg}")
            continue

    combined_content = "\n\n".join(all_limitations)

    if errors:
        combined_content += "\n\nERRORS:\n" + "\n".join(errors)

    print(f"  ✓ Processed {len(all_limitations)}/{len(papers)} papers")

    last = AIMessage(content=combined_content, name="limitation_extract")
    return {"messages": [last], "sender": "limitation_extract"}