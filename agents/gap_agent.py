# 3-4) GAP Inference Agent
"""
처리 흐름:
  [gap_infer_agent] messages에서 limitation 내용 정리/보완
  Step 1. 고정 축 5개 정의
  Step 2. agent 출력에서 limitation 텍스트 추출
  Step 3. LLM → 동적 축 생성 (최대 2개)
  Step 4. LLM → 각 limitation을 축으로 배치 분류
  Step 5. 축별 GAP statement + 상세 설명 + 제안 연구 주제 생성
"""

import json
import re
from collections import defaultdict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from states import AgentState, GapCandidate
from tools import build_role_tools
from prompts.system import make_system_prompt
from llm import get_llm

# ── gap_infer_agent (기존 유지) ───────────────────────────────────────────────

llm = get_llm()

ROLE_TOOLS = build_role_tools()
GAP_INFER_TOOLS = ROLE_TOOLS["GAP_INFER_TOOLS"]

gap_infer_agent = create_agent(
    llm,
    tools=GAP_INFER_TOOLS,
    system_prompt=make_system_prompt(
        "ROLE: GAP Inference Agent\n"
        "Your job is to read the limitation statements from retrieved papers and summarize them clearly.\n"
        "List each limitation with its paper source if available.\n"
        "Do NOT infer gaps yet — just organize and clarify the limitations.\n"
    ),
)

# ── 상수 ─────────────────────────────────────────────────────────────────────

FIXED_AXES = {
    "data": {
        "label": "Data Dependency",
        "description": "Limitations related to data quality, quantity, bias, or availability",
    },
    "methodology": {
        "label": "Methodology",
        "description": "Limitations in research methods, model design, or algorithmic approach",
    },
    "evaluation": {
        "label": "Evaluation",
        "description": "Limitations in evaluation metrics, benchmarks, or experimental setup",
    },
    "scalability": {
        "label": "Scalability",
        "description": "Limitations regarding computational cost, efficiency, or scaling to larger settings",
    },
    "generalization": {
        "label": "Generalization",
        "description": "Limitations in applying results across domains, languages, or real-world environments",
    },
}

DYNAMIC_MAX = 2          # 동적 축 최대 개수
DYNAMIC_MIN_PAPERS = 2   # 동적 축으로 인정할 최소 논문 수
BATCH_SIZE = 20          # 배치 분류 크기


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _llm_call(messages: list[dict]) -> str:
    """get_llm()을 이용해 LLM을 호출하고 텍스트 응답을 반환한다."""
    llm = get_llm()
    lc_messages = []
    for m in messages:
        if m["role"] == "system":
            lc_messages.append(SystemMessage(content=m["content"]))
        else:
            lc_messages.append(HumanMessage(content=m["content"]))
    return llm.invoke(lc_messages).content


def _parse_json(text: str) -> dict:
    """LLM 응답에서 JSON 블록을 파싱한다."""
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    return json.loads(cleaned)


# ── Step 2. messages에서 limitation 텍스트 추출 ───────────────────────────────

def _extract_limitation_text(state: AgentState) -> str:
    """
    messages 히스토리에서 limitation_extract 노드의 출력을 찾아 반환한다.
    없으면 최근 메시지들을 합쳐서 반환한다.
    """
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "name", "") == "limitation_extract":
            return msg.content

    # fallback: 최근 메시지 3개 합치기
    recent = state.get("messages", [])[-3:]
    return "\n".join(m.content for m in recent if hasattr(m, "content"))


# ── Step 3. 동적 축 생성 ─────────────────────────────────────────────────────

def _generate_dynamic_axes(limitation_text: str, research_question: str) -> list[dict]:
    fixed_summary = "\n".join(
        f"  - {k}: {v['description']}" for k, v in FIXED_AXES.items()
    )
    prompt = f"""You are a research gap analyst.

Research Question: "{research_question}"

Below are limitation/future_work statements extracted from related papers:
{limitation_text}

We already have these 5 universal axes:
{fixed_summary}

TASK:
Find domain-specific patterns NOT well covered by the 5 fixed axes.
Recurring themes appearing in {DYNAMIC_MIN_PAPERS}+ papers only.

Rules:
- Propose {DYNAMIC_MAX} extra axes at most. If no clear pattern, return empty list.
- Name must be snake_case English (e.g. "reproducibility", "ethical_concern").

Output JSON only:
{{
  "dynamic_axes": [
    {{
      "name": "axis_key",
      "label": "Short readable label",
      "description": "What kind of limitation this axis captures"
    }}
  ]
}}
"""
    messages = [
        {"role": "system", "content": "You are an expert research analyst. Always respond in valid JSON."},
        {"role": "user", "content": prompt},
    ]
    try:
        response = _llm_call(messages)
        result = _parse_json(response)
        axes = result.get("dynamic_axes", [])
        fixed_keys = set(FIXED_AXES.keys())
        valid = []
        for ax in axes:
            name = ax.get("name", "").strip()
            if not name or name in fixed_keys:
                continue
            valid.append({
                "name": name,
                "label": ax.get("label", name),
                "description": ax.get("description", ""),
            })
        return valid[:DYNAMIC_MAX]
    except Exception as e:
        print(f"  ⚠️ Dynamic axis generation failed: {e}")
        return []


# ── Step 4. 배치 분류 ────────────────────────────────────────────────────────

def _classify_limitations_batch(limitation_text: str, final_axes: dict) -> list[dict]:
    """
    limitation 텍스트를 줄 단위로 나눠 BATCH_SIZE씩 LLM에 보내
    각 줄을 축으로 분류한다.

    Returns:
        [{"text": str, "axis": str}, ...]
    """
    lines = [l.strip() for l in limitation_text.split("\n") if l.strip()]
    axes_block = "\n".join(f"  {k}: {v['description']}" for k, v in final_axes.items())
    axis_keys = list(final_axes.keys())
    fallback = "methodology"

    classified = []
    batches = [lines[i:i + BATCH_SIZE] for i in range(0, len(lines), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        items_block = "\n".join(f'  [{i}] "{line}"' for i, line in enumerate(batch))
        prompt = f"""Classify each limitation into ONE axis from the list below.

Available axes:
{axes_block}

Limitations to classify:
{items_block}

Output JSON only:
{{
  "classifications": {{
    "0": "axis_key",
    "1": "axis_key"
  }}
}}
"""
        messages = [
            {"role": "system", "content": "You are a research limitation classifier. Always respond in valid JSON."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = _llm_call(messages)
            result = _parse_json(response)
            cls_map = result.get("classifications", {})
            for i, line in enumerate(batch):
                axis = cls_map.get(str(i), fallback)
                if axis not in axis_keys:
                    axis = fallback
                classified.append({"text": line, "axis": axis})
        except Exception as e:
            print(f"  ⚠️ Batch {batch_idx} classification failed: {e}")
            for line in batch:
                classified.append({"text": line, "axis": fallback})

    return classified


# ── Step 5. 축별 GAP 생성 ────────────────────────────────────────────────────

def _generate_gap_for_axis(ax_key: str, ax_info: dict, ax_lines: list[str], research_question: str) -> dict | None:
    claims_block = "\n".join(f"  - {line}" for line in ax_lines[:8])
    prompt = f"""You are a research gap analyst.

Research Question: "{research_question}"

Axis: {ax_info['label']} ({ax_key})
Axis description: {ax_info['description']}

Limitations under this axis ({len(ax_lines)} items):
{claims_block}

TASK: Generate:
1. gap_statement  : One concise sentence describing the core research gap.
2. elaboration    : 2-3 sentences explaining WHY this gap matters and WHAT is missing.
3. proposed_topic : One concrete research topic title that directly addresses this gap.

Output JSON only:
{{
  "gap_statement":  "...",
  "elaboration":    "...",
  "proposed_topic": "..."
}}
"""
    messages = [
        {"role": "system", "content": "You are an expert at identifying research gaps. Always respond in valid JSON."},
        {"role": "user", "content": prompt},
    ]
    try:
        response = _llm_call(messages)
        result = _parse_json(response)
        return {
            "gap_statement":  result.get("gap_statement",  f"Gap in {ax_key}"),
            "elaboration":    result.get("elaboration",    ""),
            "proposed_topic": result.get("proposed_topic", ""),
        }
    except Exception as e:
        print(f"  ⚠️ GAP generation failed for axis '{ax_key}': {e}")
        return None


# ── 메인 노드 ────────────────────────────────────────────────────────────────

def gap_infer_node(state: AgentState) -> AgentState:
    print("\n💡 GAP Inference Node")

    research_question = state.get("user_question", "")

    # ── gap_infer_agent: limitation 정리/보완 ────────────────────────────────
    print("  🔄 gap_infer_agent — limitation 정리 중...")
    agent_result = gap_infer_agent.invoke(state)
    agent_output = agent_result["messages"][-1].content if agent_result.get("messages") else ""

    # ── Step 2. limitation 텍스트 추출 ──────────────────────────────────────
    # agent 출력을 우선 사용, 없으면 messages에서 직접 추출
    limitation_text = agent_output if agent_output.strip() else _extract_limitation_text(state)
    if not limitation_text.strip():
        print("  ⚠️ No limitation text found")
        return {"messages": [AIMessage(content="No limitations to analyze.", name="gap_infer")],
                "sender": "gap_infer", "gaps": []}

    # ── Step 1 & 3. 고정 축 + 동적 축 확정 ──────────────────────────────────
    print(f"  ✓ 고정 축 {len(FIXED_AXES)}개 로드: {list(FIXED_AXES.keys())}")
    print(f"  🔄 동적 축 생성 중...")
    dynamic_axes = _generate_dynamic_axes(limitation_text, research_question)

    final_axes = dict(FIXED_AXES)
    for ax in dynamic_axes:
        final_axes[ax["name"]] = {"label": ax["label"], "description": ax["description"], "type": "dynamic"}
    for k in FIXED_AXES:
        final_axes[k]["type"] = "fixed"

    if dynamic_axes:
        for ax in dynamic_axes:
            print(f"  🟢 동적 축 추가: [{ax['name']}] {ax['label']}")
    else:
        print("  ℹ️  동적 축 없음 (고정 5개로 충분)")
    print(f"  ✓ 최종 축 {len(final_axes)}개 확정")

    # ── Step 4. 배치 분류 ────────────────────────────────────────────────────
    print(f"  🔄 limitation 배치 분류 중...")
    classified = _classify_limitations_batch(limitation_text, final_axes)

    axis_groups = defaultdict(list)
    for item in classified:
        axis_groups[item["axis"]].append(item["text"])

    print(f"\n  {'축':<28} {'유형':>6}  {'항목 수':>6}")
    print(f"  {'-'*45}")
    for ax_key, lines in sorted(axis_groups.items(), key=lambda x: -len(x[1])):
        ax_info = final_axes.get(ax_key, {})
        tag = "🟢 동적" if ax_info.get("type") == "dynamic" else "🔵 고정"
        label = ax_info.get("label", ax_key)
        print(f"  {label:<28} {tag:>6}  {len(lines):>5}개")

    # ── Step 5. 축별 GAP 생성 ────────────────────────────────────────────────
    gaps = []
    print(f"\n  🔄 {len(axis_groups)}개 축 GAP 생성 중...")

    for ax_key, ax_lines in sorted(axis_groups.items(), key=lambda x: -len(x[1])):
        ax_info = final_axes.get(ax_key, {"label": ax_key, "description": "", "type": "fixed"})
        result = _generate_gap_for_axis(ax_key, ax_info, ax_lines, research_question)
        if result is None:
            continue

        gaps.append(GapCandidate(
            axis=ax_key,
            axis_label=ax_info["label"],
            axis_type=ax_info.get("type", "fixed"),
            gap_statement=result["gap_statement"],
            elaboration=result["elaboration"],
            proposed_topic=result["proposed_topic"],
            repeat_count=len(ax_lines),
            supporting_papers=[],
            supporting_quotes=[],
        ))
        print(f"  ✓ [{ax_key}] {result['gap_statement'][:65]}...")

    gaps.sort(key=lambda g: g.repeat_count, reverse=True)
    print(f"\n  ✅ GAP {len(gaps)}개 생성 완료")

    # 결과 요약 메시지 생성
    summary_lines = [f"## Research GAP Analysis ({len(gaps)} gaps found)\n"]
    for g in gaps:
        summary_lines.append(f"### [{g.axis_label}] {g.gap_statement}")
        summary_lines.append(f"{g.elaboration}")
        summary_lines.append(f"**Proposed Topic:** {g.proposed_topic}\n")
    summary = "\n".join(summary_lines)

    return {
        "messages": [AIMessage(content=summary, name="gap_infer")],
        "sender": "gap_infer",
        "gaps": [g.model_dump() for g in gaps],
    }