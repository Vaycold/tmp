# Recency Check Agent
# limitation의 최신성을 웹 검색 결과와 대조하여 검증
from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from states import AgentState
from llm import get_llm
from utils.parse_json import parse_json

SYSTEM_PROMPT = """ROLE: Recency Check Agent

You verify whether research limitations are still relevant by cross-referencing with recent web sources.

## Input
- A list of limitations extracted from academic papers
- Recent web search results (blog posts, news, preprints, etc.)

## Task
For each limitation, determine if recent developments have addressed it:
- "unresolved": No evidence that this limitation has been addressed
- "partial": Some progress found, but not fully resolved
- "resolved": Clear evidence that this limitation has been overcome

## Output Format (strictly JSON list)
[
  {
    "paper_id": "<id>",
    "claim": "<original limitation claim>",
    "recency_status": "unresolved" or "partial" or "resolved",
    "evidence": "<brief explanation referencing web source, or 'No relevant web evidence found'>"
  },
  ...
]

## Rules
1. Be conservative: only mark "resolved" if there is clear, specific evidence.
2. If no web result is relevant to a limitation, mark it "unresolved".
3. Do NOT invent or hallucinate web sources. Only use the provided web results.
4. Output ONLY the JSON list. No explanation before or after.
"""


def recency_check_node(state: AgentState) -> AgentState:
    web_results = state.get("web_results", [])
    limitations = state.get("limitations", [])

    # 웹 결과가 없으면 모든 limitation을 unresolved로 통과
    if not web_results:
        print("  [recency] 웹 결과 없음 → 전체 unresolved로 통과")
        for lim in limitations:
            lim["recency_status"] = "unresolved"
            lim["recency_evidence"] = "No web results available"

        return {
            "messages": [AIMessage(
                content=f"Recency check skipped: no web results. {len(limitations)} limitations passed as unresolved.",
                name="recency_check",
            )],
            "sender": "recency_check",
            "limitations": limitations,
        }

    # 웹 결과 요약 구성
    web_context = "\n".join(
        f"  - [{r.get('title', 'N/A')}] {r.get('content', '')[:500]}"
        for r in web_results[:10]  # 최대 10개
    )

    # limitation 요약 구성
    lim_context = "\n".join(
        f"  {i+1}. [{l.get('paper_id', '')}] {l.get('claim', '')}"
        for i, l in enumerate(limitations)
    )

    user_prompt = (
        f"## Limitations to verify ({len(limitations)}개)\n{lim_context}\n\n"
        f"## Recent web sources ({len(web_results)}개)\n{web_context}"
    )

    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"  ⚠️ [recency] LLM 호출 실패: {e} → 전체 unresolved 처리")
        for lim in limitations:
            lim["recency_status"] = "unresolved"
            lim["recency_evidence"] = "LLM call failed"

        return {
            "messages": [AIMessage(content="Recency check failed. All unresolved.", name="recency_check")],
            "sender": "recency_check",
            "limitations": limitations,
        }

    # LLM 응답 파싱
    parsed = parse_json(content)
    if not isinstance(parsed, list):
        parsed = []

    # 파싱 결과를 limitations에 매핑
    recency_map = {}
    for item in parsed:
        key = (item.get("paper_id", ""), item.get("claim", "")[:50])
        recency_map[key] = {
            "status": item.get("recency_status", "unresolved"),
            "evidence": item.get("evidence", ""),
        }

    resolved_count = 0
    partial_count = 0
    for lim in limitations:
        key = (lim.get("paper_id", ""), lim.get("claim", "")[:50])
        match = recency_map.get(key)
        if match:
            lim["recency_status"] = match["status"]
            lim["recency_evidence"] = match["evidence"]
            if match["status"] == "resolved":
                resolved_count += 1
            elif match["status"] == "partial":
                partial_count += 1
        else:
            lim["recency_status"] = "unresolved"
            lim["recency_evidence"] = "No matching recency check result"

    summary = (
        f"Recency check complete: {len(limitations)} limitations verified. "
        f"resolved={resolved_count}, partial={partial_count}, "
        f"unresolved={len(limitations) - resolved_count - partial_count}"
    )
    print(f"  ✅ [recency] {summary}")

    return {
        "messages": [AIMessage(content=summary, name="recency_check")],
        "sender": "recency_check",
        "limitations": limitations,
    }
