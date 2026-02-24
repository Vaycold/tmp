"""
Ambiguity Check Agent.

사용자 질문이 들어오면 query_analysis 이전에 먼저 실행되어,
4가지 기준으로 모호성을 판단하고 라우팅 신호를 반환한다.

판단 기준:
  1. 도메인 명확성  - 연구 분야가 특정되는지
  2. 방법론 명확성  - 기법/알고리즘 언급 여부
  3. 데이터 명확성  - 데이터셋/대상 명시 여부
  4. 기간 명확성    - 연구 연도/최신성 언급 여부

각 항목은 0~1 점수로 평가되며, 평균이 AMBIGUITY_THRESHOLD 미만이면
"ambiguous" 판정 → human_clarify 노드로 라우팅.
"""

from models import AgentState
from llm import llm_chat, parse_json

# 판정 기준: 4개 항목 중 CLEAR_MIN_COUNT개 이상이 CLEAR_MIN_SCORE 이상이면 "명확"으로 판정
# → 2개 이상의 축에서 0.3 이상이면 분석을 진행할 수 있는 충분한 정보로 간주
CLEAR_MIN_SCORE: float = 0.3   # 개별 항목 최소 점수
CLEAR_MIN_COUNT: int   = 2     # 최소 통과 항목 수
AMBIGUITY_THRESHOLD : float = 0.3
# 각 항목 설명 (프롬프트 + 출력 표시에 활용)
CRITERIA_META = {
    "domain_clarity": {
        "label": "도메인 명확성",
        "description": "연구 분야(예: NLP, computer vision, drug discovery 등)가 구체적으로 특정되는지",
    },
    "methodology_clarity": {
        "label": "방법론 명확성",
        "description": "특정 기법, 알고리즘, 모델 아키텍처(예: transformer, GAN, RL 등)가 언급되는지",
    },
    "data_clarity": {
        "label": "데이터 명확성",
        "description": "사용 데이터셋, 실험 대상, 입력 데이터 유형(예: ImageNet, clinical records 등)이 명시되는지",
    },
    "temporal_clarity": {
        "label": "기간 명확성",
        "description": "연구 연도 범위 혹은 최신 연구에 대한 언급(예: 'since 2020', 'recent', '2023~')이 있는지",
    },
}


def _build_prompt(user_question: str) -> str:
    """모호성 판단 프롬프트를 생성한다."""

    criteria_block = "\n".join(
        f"- {key}: {meta['description']}"
        for key, meta in CRITERIA_META.items()
    )

    return f"""You are a research query ambiguity evaluator.

Evaluate the following research question on 4 clarity criteria.
Score each criterion from 0.0 (completely unclear) to 1.0 (perfectly clear).

Research Question:
\"{user_question}\"

Criteria to evaluate:
{criteria_block}

Instructions:
- Be strict: a question with no mention of a criterion should score 0.0~0.3.
- A partial mention scores 0.4~0.6.
- A clear, specific mention scores 0.7~1.0.
- Also provide 1 short clarifying question per low-scoring criterion (score < 0.6).

Output JSON only with EXACT keys (no extra text):
{{
  "domain_clarity":       {{"score": 0.0, "reason": "...", "clarifying_question": "...or null"}},
  "methodology_clarity":  {{"score": 0.0, "reason": "...", "clarifying_question": "...or null"}},
  "data_clarity":         {{"score": 0.0, "reason": "...", "clarifying_question": "...or null"}},
  "temporal_clarity":     {{"score": 0.0, "reason": "...", "clarifying_question": "...or null"}},
  "overall_ambiguous":    true
}}

Set "overall_ambiguous" to true if the average score is below {AMBIGUITY_THRESHOLD}.
"""


def ambiguity_check_node(state: AgentState) -> AgentState:
    """
    Ambiguity Check Node — query_analysis 이전에 실행.

    - 4가지 기준으로 모호성 점수를 계산
    - 평균 점수 < AMBIGUITY_THRESHOLD  → route = "ambiguous" (human_clarify로 이동)
    - 평균 점수 >= AMBIGUITY_THRESHOLD → route = "clear"    (query_analysis로 이동)
    - 결과는 state["trace"]["ambiguity"] 에 저장
    """
    print("\n🔎 Ambiguity Check Node")

    # ── 방어적 초기화 ──────────────────────────────────────────────
    if "trace" not in state or state["trace"] is None:
        state["trace"] = {}
    if "errors" not in state or state["errors"] is None:
        state["errors"] = []

    # route 초기화 (이전 루프 잔여값 제거)
    state["route"] = ""

    # ── LLM 호출 ──────────────────────────────────────────────────
    prompt = _build_prompt(state["user_question"])
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert at evaluating the clarity and specificity "
                "of academic research questions. Always respond in valid JSON."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        response = llm_chat(messages)
        result = parse_json(response)

        # ── 점수 파싱 ──────────────────────────────────────────────
        scores = {}
        clarifying_questions = []

        for key, meta in CRITERIA_META.items():
            item = result.get(key, {})
            score = float(item.get("score", 0.0))
            reason = item.get("reason", "")
            cq = item.get("clarifying_question")

            scores[key] = {"score": score, "reason": reason, "label": meta["label"]}

            # 점수가 낮은 항목의 clarifying question 수집 (CLEAR_MIN_SCORE 미만인 항목)
            if score < CLEAR_MIN_SCORE and cq and isinstance(cq, str) and cq.strip():
                clarifying_questions.append(cq.strip())

        avg_score = sum(v["score"] for v in scores.values()) / len(scores)
        # ✅ 새 판정 기준: CLEAR_MIN_COUNT개 이상 항목이 CLEAR_MIN_SCORE 이상이면 명확
        clear_count = sum(1 for v in scores.values() if v["score"] >= CLEAR_MIN_SCORE)
        is_ambiguous = clear_count < CLEAR_MIN_COUNT

        # ── 결과 출력 ──────────────────────────────────────────────
        print(f"  {'항목':<20} {'점수':>6}  {'사유'}")
        print(f"  {'-'*60}")
        for key, val in scores.items():
            flag = "⚠️ " if val["score"] < CLEAR_MIN_SCORE else "✅ "
            print(f"  {flag}{val['label']:<18} {val['score']:>5.2f}  {val['reason'][:50]}")
        print(f"  {'-'*60}")
        print(f"  📊 평균 점수: {avg_score:.2f}  |  기준 통과 항목: {clear_count}/4  →  {'모호함 (clarify 필요)' if is_ambiguous else '명확함 (분석 진행)'}")

        # ── state 저장 ─────────────────────────────────────────────
        state["trace"]["ambiguity"] = {
            "scores": scores,
            "avg_score": round(avg_score, 4),
            "clear_count": clear_count,
            "is_ambiguous": is_ambiguous,
            "criterion": f"{CLEAR_MIN_COUNT} of 4 items >= {CLEAR_MIN_SCORE}",
        }

        # clarifying_questions를 trace에 저장 → human_clarify_node가 바로 사용
        state["trace"]["clarify_questions"] = clarifying_questions[:5]
        state["trace"]["clarify_needed"] = is_ambiguous

        # ── 라우팅 신호 설정 ──────────────────────────────────────
        if is_ambiguous:
            state["route"] = "ambiguous"
            print("  ⚠️  Routing → human_clarify")
        else:
            state["route"] = "clear"
            print("  ✅  Routing → query_analysis")

    except Exception as e:
        # 파싱 실패 시 안전하게 "clear"로 진행 (파이프라인 중단 방지)
        state["errors"].append(f"Ambiguity check error: {str(e)}")
        state["trace"]["ambiguity"] = {"error": str(e), "is_ambiguous": False}
        state["trace"]["clarify_questions"] = []
        state["trace"]["clarify_needed"] = False
        state["route"] = "clear"
        print(f"  ❌ Error in ambiguity check: {e} → defaulting to clear")

    return state