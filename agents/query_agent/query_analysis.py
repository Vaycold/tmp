"""
================================================== 1. QUERY ANALYSIS (v2) ==================================================

[적용 논문 및 코드 매핑]

① CLAMBER (Zhang et al., ACL 2024 | arXiv:2405.12063)
   논문 근거: "These dimensions are further conceptualized into eight fine-grained
               categories to facilitate in-depth evaluation."
   코드 반영:
   - states.py의 ClamberAnalysis 클래스 (8가지 유형)
   - 시스템 프롬프트 PART 2: LLM에게 8유형 분류 요청
   - _compute_combined_ambiguity_signal()의 clamber_fail 신호
   - _build_clarification_message()에서 severity 높은 유형 우선 노출

② APA — Alignment with Perceived Ambiguity (Kim et al., EMNLP 2024 | arXiv:2404.11972)
   논문 근거: "(2) the degree of ambiguity perceived by the LLMs may vary depending
               on the possessed knowledge."
               "We measure the information gain (INFOGAIN) between the initial input
                and the disambiguation, identifying samples with high INFOGAIN as ambiguous."
   코드 반영:
   - states.py의 PerceivedAmbiguity 클래스 (해석 후보 + INFOGAIN)
   - 시스템 프롬프트 PART 3: LLM에게 2~4개 해석 생성 요청
   - _compute_infogain(): 1 - max_plausibility로 INFOGAIN 근사
   - _compute_combined_ambiguity_signal()의 apa_fail 신호

③ STaR-GATE (Andukuri et al., 2024 | arXiv:2403.19154)
   논문 근거: "The Questioner is iteratively finetuned on questions that increase
               the probability of high-quality responses to the task."
               "models often struggle to ask good questions"
   코드 반영:
   - 시스템 프롬프트 PART 4: "list questions ordered by INFORMATION GAIN"
   - PerceivedAmbiguity.clarification_priority: LLM이 우선순위화한 질문 목록
   - _build_clarification_message(): 3단 우선순위 (APA → CLAMBER → 기존5축)
   - clarify_questions 수집 로직 (최대 3개, 정보이득 큰 것 우선)

[변경 요약]
- 기존: 5축 점수 threshold → is_ambiguous
- 신규: 5축 점수 + CLAMBER 8유형 + APA INFOGAIN → 통합 판정
        STaR-GATE 우선순위 질문 생성
=======================================================================================================================
"""

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from states import AgentState
from llm import get_llm
from tools import build_role_tools
from prompts.system import make_system_prompt
import re

from states import (
    QueryAnalysis,
    ImportanceWeights,
    ClamberAnalysis,
    PerceivedAmbiguity,
)

llm = get_llm()

# =====================================================================================================================
# ==================================================== 0) 초기 설정 ====================================================
# =====================================================================================================================

ROLE_TOOLS = build_role_tools()
QUERY_TOOLS = ROLE_TOOLS["QUERY_TOOLS"]

# ── 기존 5축 threshold (원본 코드 그대로 유지) ──────────────────────────────
CLEAR_MIN_SCORE: float = 0.4
CORE_SLOT_MIN_COUNT: int = 1
QUESTION_SCORE_THRESHOLD: float = 0.6
WEIGHTED_SCORE_THRESHOLD: float = 0.6
SEARCH_CONFIDENCE_THRESHOLD: float = 0.5

# ── NEW ① APA INFOGAIN threshold (Kim et al., EMNLP 2024) ────────────────
# 논문: INFOGAIN = 1 - max_plausibility
# INFOGAIN > 0.35 → 해석 분산 → perceived ambiguous
# INFOGAIN < 0.15 → 해석 수렴 → perceived clear
INFOGAIN_AMBIGUOUS_THRESHOLD: float = 0.35
INFOGAIN_CLEAR_THRESHOLD: float = 0.15

# ── NEW ② CLAMBER severity threshold (Zhang et al., ACL 2024) ────────────
# 8유형 중 severity >= 0.5 이상인 유형이 있으면 모호
CLAMBER_SEVERITY_THRESHOLD: float = 0.5
# 감지된 유형 수 >= 2이면 추가 질문 강제
CLAMBER_DETECTED_COUNT_STRONG: int = 2

DEFAULT_WEIGHTS = {
    "domain_clarity": 0.30,
    "task_clarity": 0.25,
    "methodology_clarity": 0.20,
    "data_clarity": 0.15,
    "temporal_clarity": 0.10,
}

CRITERIA_KEYS = [
    "domain_clarity",
    "task_clarity",
    "methodology_clarity",
    "data_clarity",
    "temporal_clarity",
]

CRITERIA_META = {
    "domain_clarity": {
        "label": "도메인(산업군)",
        "example": "예: 금융, 의료, 자동차 제조 등",
    },
    "task_clarity": {
        "label": "태스크(해결 문제)",
        "example": "예: 이상 징후 탐지, 수요 예측, 텍스트 요약 등",
    },
    "methodology_clarity": {
        "label": "방법론(기술)",
        "example": "예: 트랜스포머, 강화학습, 도메인 적응 등",
    },
    "data_clarity": {
        "label": "데이터 유형",
        "example": "예: 센서 데이터, 이미지/영상 데이터 등",
    },
    "temporal_clarity": {
        "label": "연구 기간",
        "example": "예: 최근 3년 이내, 2020년 이후 등",
    },
}

# CLAMBER 8유형 한글 레이블
CLAMBER_LABELS = {
    "entity_ambiguity":    "개체 모호성 (동일 표현, 여러 개체)",
    "temporal_ambiguity":  "시간 범위 모호성",
    "scope_ambiguity":     "범위·수준 모호성",
    "intent_ambiguity":    "의도·목적 모호성",
    "reference_ambiguity": "지시 대상 모호성",
    "underspecification":  "필수 정보 누락",
    "multifaceted_query":  "다중 해석 가능 질문",
    "conflicting_info":    "모순·충돌 정보 포함",
}


# =====================================================================================================================
# =============================================== 1) 서브 함수 정의 =====================================================
# =====================================================================================================================

def _normalize_weights(raw_weights: ImportanceWeights | None) -> dict[str, float]:
    """기존 코드 그대로 유지"""
    if raw_weights is None:
        raw_dict = DEFAULT_WEIGHTS.copy()
    else:
        raw_dict = raw_weights.model_dump()

    cleaned = {
        key: max(0.0, float(raw_dict.get(key, DEFAULT_WEIGHTS[key])))
        for key in CRITERIA_KEYS
    }
    total = sum(cleaned.values())
    if total <= 1e-9:
        return DEFAULT_WEIGHTS.copy()
    return {k: v / total for k, v in cleaned.items()}


# ── NEW ① APA INFOGAIN 계산 ────────────────────────────────────────────────
def _compute_infogain(perceived: PerceivedAmbiguity) -> float:
    """
    APA의 INFOGAIN 근사 계산 (Kim et al., EMNLP 2024)

    논문 원리:
      - LLM이 쿼리에 대해 여러 interpretation을 생성
      - max_plausibility가 낮을수록 → 해석 분산 → 모호한 쿼리
      - INFOGAIN = 1 - max_plausibility (엔트로피 기반 근사)

    반환값:
      0.0 ~ 1.0 (높을수록 모호)
      > 0.35 : perceived ambiguous
      < 0.15 : perceived clear
    """
    if perceived.interpretations:
        plausibilities = [iv.plausibility for iv in perceived.interpretations]
        max_p = max(plausibilities) if plausibilities else 1.0
        return round(1.0 - max_p, 4)
    # LLM이 직접 계산한 infogain_score 사용 (fallback)
    return perceived.infogain_score


# ── NEW ② CLAMBER + APA + 기존 5축 통합 모호성 판정 ─────────────────────────
def _compute_combined_ambiguity_signal(
    weighted_score: float,
    domain_ok: bool,
    core_clear_count: int,
    clamber: ClamberAnalysis,
    perceived: PerceivedAmbiguity,
    search_readiness_ok: bool,
    confidence: float,
) -> dict:
    """
    세 논문의 모호성 신호를 통합하여 최종 is_ambiguous 판정

    신호 구성:
      hard_fail   : 도메인 미언급 (기존 로직)
      soft_fail   : 5축 점수 부족 (기존 로직)
      clamber_fail: CLAMBER severity 높은 유형 감지 (Zhang et al., ACL 2024)
      apa_fail    : APA INFOGAIN > threshold (Kim et al., EMNLP 2024)

    반환:
      dict with keys: is_ambiguous, infogain, hard_fail, soft_fail,
                      clamber_fail, apa_fail,
                      clamber_detected_types, clamber_max_severity
    """
    infogain = _compute_infogain(perceived)

    # 기존 로직 (그대로 유지)
    hard_fail = not domain_ok
    soft_fail = (
        core_clear_count < CORE_SLOT_MIN_COUNT
        or weighted_score < WEIGHTED_SCORE_THRESHOLD
        or not search_readiness_ok
        or (confidence < SEARCH_CONFIDENCE_THRESHOLD and weighted_score < WEIGHTED_SCORE_THRESHOLD)
    )

    # NEW: CLAMBER fail
    clamber_detected = clamber.detected_types
    clamber_max_severity = clamber.max_severity
    clamber_fail = (
        clamber_max_severity >= CLAMBER_SEVERITY_THRESHOLD
        or len(clamber_detected) >= CLAMBER_DETECTED_COUNT_STRONG
    )

    # NEW: APA fail
    apa_fail = (
        perceived.perceived_ambiguous
        and infogain >= INFOGAIN_AMBIGUOUS_THRESHOLD
    )

    is_ambiguous = hard_fail or soft_fail or clamber_fail or apa_fail

    return {
        "is_ambiguous": is_ambiguous,
        "infogain": infogain,
        "hard_fail": hard_fail,
        "soft_fail": soft_fail,
        "clamber_fail": clamber_fail,
        "apa_fail": apa_fail,
        "clamber_detected_types": clamber_detected,
        "clamber_max_severity": clamber_max_severity,
    }


# ── NEW ③ STaR-GATE 방식 clarification 메시지 생성 ───────────────────────────
def _build_clarification_message(
    parsed: QueryAnalysis,
    ambiguity_signals: dict,
    scores: object,
) -> str:
    """
    STaR-GATE 우선순위화된 clarification 메시지 (Andukuri et al., 2024)

    논문 원리:
      "The Questioner is iteratively finetuned on questions that increase
       the probability of high-quality responses to the task."
      → 정보이득이 가장 큰 질문(해석 분기를 가장 많이 줄이는 질문)을 먼저

    질문 우선순위:
      1순위: APA clarification_priority (LLM이 INFOGAIN 기준으로 직접 정렬)
      2순위: CLAMBER severity 높은 유형의 resolution_hint
      3순위: 기존 5축 clarifying_question (점수 낮은 순)
    """
    intro = (
        "네, 제안해주신 주제로 논문을 찾아보고 있습니다! "
        "다만, 조금 더 구체적인 정보를 주시면 훨씬 정확한 검색 결과를 드릴 수 있을 것 같아요."
    )

    # ── 1순위: APA clarification_priority ────────────────────────────────
    priority_questions = []
    if parsed.perceived_ambiguity and parsed.perceived_ambiguity.clarification_priority:
        for q in parsed.perceived_ambiguity.clarification_priority[:2]:
            q = q.strip()
            if q:
                priority_questions.append(q)

    # ── 2순위: CLAMBER severity 높은 유형의 resolution_hint ──────────────
    if parsed.clamber:
        clamber_fields = [
            "underspecification", "scope_ambiguity", "intent_ambiguity",
            "entity_ambiguity", "temporal_ambiguity", "reference_ambiguity",
            "multifaceted_query", "conflicting_info",
        ]
        severity_sorted = sorted(
            [(f, getattr(parsed.clamber, f)) for f in clamber_fields
             if getattr(parsed.clamber, f).detected],
            key=lambda x: x[1].severity,
            reverse=True,
        )
        for _, ctype in severity_sorted[:2]:
            hint = ctype.resolution_hint.strip()
            if hint and hint not in priority_questions:
                priority_questions.append(hint)

    # ── 3순위: 기존 5축 clarifying_question ─────────────────────────────
    axis_q = []
    for key in CRITERIA_KEYS:
        item = getattr(scores, key, None)
        if item is not None and item.score < QUESTION_SCORE_THRESHOLD:
            q = getattr(item, "clarifying_question", "") or ""
            if q.strip() and q.strip() not in priority_questions:
                axis_q.append((item.score, q.strip()))
    for _, q in sorted(axis_q):
        if q not in priority_questions:
            priority_questions.append(q)

    # 최대 3개 (STaR-GATE: 질문 수가 많으면 사용자 부담)
    final_questions = priority_questions[:3]

    # ── CLAMBER 근거 섹션 ────────────────────────────────────────────────
    clamber_info = []
    if ambiguity_signals.get("clamber_detected_types") and parsed.clamber:
        for dtype in ambiguity_signals["clamber_detected_types"][:3]:
            label = CLAMBER_LABELS.get(dtype, dtype)
            ctype_obj = getattr(parsed.clamber, dtype, None)
            if ctype_obj and ctype_obj.evidence:
                clamber_info.append(f"  - [{label}] \"{ctype_obj.evidence}\"")
            else:
                clamber_info.append(f"  - [{label}]")

    # ── APA dominant interpretation ──────────────────────────────────────
    dominant = ""
    if parsed.perceived_ambiguity and parsed.perceived_ambiguity.dominant_interpretation:
        dominant = parsed.perceived_ambiguity.dominant_interpretation.strip()

    # ── 메시지 조립 ──────────────────────────────────────────────────────
    parts = [intro]

    if dominant:
        parts.append(f"\n현재 질문을 이렇게 이해했어요:\n  → \"{dominant}\"")
        parts.append("이 해석이 맞다면 바로 진행할게요! 다른 의도라면 아래 질문에 답해주세요.")

    if clamber_info:
        parts.append("\n아래 부분이 불명확합니다:\n" + "\n".join(clamber_info))

    # 부족한 축 안내 (기존 로직 유지)
    weak_areas = []
    for key in CRITERIA_KEYS:
        item = getattr(scores, key, None)
        if item is not None and item.score < QUESTION_SCORE_THRESHOLD:
            label = CRITERIA_META[key]["label"]
            example = CRITERIA_META[key]["example"]
            weak_areas.append(f"  - {label}: {example}")
    if weak_areas:
        parts.append("\n구체적으로 아래 정보들이 부족합니다:\n" + "\n".join(weak_areas))

    if final_questions:
        qs = "\n".join(f"{i}. {q}" for i, q in enumerate(final_questions, 1))
        parts.append(f"\n아래 질문에 답해주시면 좋습니다:\n{qs}")

    parts.append("\n조금 더 자세한 질문을 해주시면 바로 조사를 시작할게요!")
    return "\n".join(parts)


# =====================================================================================================================
# =============================================== 2) 시스템 프롬프트 =====================================================
# =====================================================================================================================

system_prompt = make_system_prompt(
    # ── 기존 PART 1 (5축 평가, 원본 유지) ──────────────────────────────────
    "ROLE: Query Analysis Agent (v2 — CLAMBER + APA + STaR-GATE)\n"
    "You evaluate research query ambiguity using three complementary frameworks.\n\n"

    "=== PART 1: 5-axis Clarity Scoring (original) ===\n"
    "Evaluate the user question on 5 criteria:\n"
    "   - domain_clarity: Specific industry/application sector (e.g., Finance, Healthcare).\n"
    "   - task_clarity: Problem specificity (e.g., anomaly detection, forecasting).\n"
    "   - methodology_clarity: Specific techniques/architectures (e.g., UDA, Transformers, RL).\n"
    "   - data_clarity: Data modalities or datasets (e.g., EHR, sensor logs).\n"
    "   - temporal_clarity: Time constraints (e.g., since 2020, last 5 years).\n"
    "Scoring anchor: 0.0=absent, 0.2=generic, 0.4=broad, 0.6=partial, 0.8=high, 1.0=perfect.\n"
    "Be conservative. Do not infer unstated details.\n\n"

    # ── NEW PART 2: CLAMBER taxonomy ──────────────────────────────────────
    "=== PART 2: CLAMBER Taxonomy (Zhang et al., ACL 2024) ===\n"
    "Classify the query into 8 ambiguity types:\n"
    "  [Dim A - Epistemic]\n"
    "    entity_ambiguity     : Same expression = multiple entities (e.g., 'GAN', 'Apple').\n"
    "    temporal_ambiguity   : Vague time range (e.g., 'recent', 'modern', 'latest').\n"
    "  [Dim B - Linguistic]\n"
    "    scope_ambiguity      : Breadth/level unclear (too broad or too narrow).\n"
    "    intent_ambiguity     : Purpose unclear (survey? new method? comparison?).\n"
    "    reference_ambiguity  : Referent unclear (e.g., 'this approach', 'the model').\n"
    "  [Dim C - Aleatoric Output]\n"
    "    underspecification   : Required info missing (dataset, metric, baseline).\n"
    "    multifaceted_query   : Multiple valid research contexts coexist.\n"
    "    conflicting_info     : Self-contradictory or conflicting info present.\n"
    "For each DETECTED type: severity(0~1), evidence(exact quote from query), resolution_hint.\n"
    "Leave detected=False for types that are NOT present.\n\n"

    # ── NEW PART 3: APA Perceived Ambiguity ───────────────────────────────
    "=== PART 3: APA Perceived Ambiguity (Kim et al., EMNLP 2024) ===\n"
    "Generate 2~4 distinct interpretations of the query using YOUR OWN knowledge.\n"
    "Each interpretation = a specific research context (e.g., 'UDA for PMSM fault detection using vibration data').\n"
    "Assign plausibility to each (values should approximately sum to 1.0).\n"
    "infogain_score = 1 - max_plausibility:\n"
    "  > 0.35 → multiple valid interpretations exist → set perceived_ambiguous = True\n"
    "  < 0.15 → single dominant interpretation → set perceived_ambiguous = False\n"
    "dominant_interpretation: the most plausible interpretation in one sentence.\n\n"

    # ── NEW PART 4: STaR-GATE question prioritization ────────────────────
    "=== PART 4: STaR-GATE Clarification Priority (Andukuri et al., 2024) ===\n"
    "In clarification_priority, order questions by INFORMATION GAIN (most ambiguity-reducing first):\n"
    "  - First question eliminates the LARGEST interpretation branch.\n"
    "  - Max 3 questions total. Each answerable in 1~2 sentences.\n"
    "  - Do NOT ask about facts already clearly stated.\n\n"

    # ── 기존 OUTPUT RULES (원본 유지) ────────────────────────────────────
    "=== OUTPUT RULES ===\n"
    "- suggested_query: natural academic research question preserving user intent.\n"
    "- keywords: 2~5 domain/task terms, no generic words (AI, model, system).\n"
    "- negative_keywords: 1~3 exclusion terms only when explicitly useful.\n"
    "- importance_weights: all positive, sum=1.0.\n"
)

structured_llm = llm.with_structured_output(QueryAnalysis)


# =====================================================================================================================
# =============================================== 3) 노드 함수 =========================================================
# =====================================================================================================================

def query_analysis_node(state: AgentState) -> AgentState:
    it = state.get("iteration", 0) + 1
    max_it = state.get("max_iterations", 3)

    # 기존과 동일: 누적된 HumanMessage 합산
    user_inputs = [
        m.content.strip() for m in state["messages"] if isinstance(m, HumanMessage)
    ]
    combined_user_input = "\n".join(user_inputs)
    input_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=combined_user_input),
    ]

    parsed: QueryAnalysis = structured_llm.invoke(input_messages)

    # ── 기존 ①: 5축 가중 점수 계산 ────────────────────────────────────────
    dynamic_weights = _normalize_weights(parsed.importance_weights)
    weighted_score = sum(
        getattr(parsed.scores, key).score * dynamic_weights.get(key, 0.0)
        for key in CRITERIA_KEYS
    )

    domain_score = parsed.scores.domain_clarity.score
    domain_ok = domain_score >= CLEAR_MIN_SCORE

    core_slots = ["task_clarity", "methodology_clarity", "data_clarity"]
    core_clear_count = sum(
        1 for key in core_slots if getattr(parsed.scores, key).score >= CLEAR_MIN_SCORE
    )

    # ── NEW ②: CLAMBER + APA + 5축 통합 판정 ─────────────────────────────
    ambiguity_signals = _compute_combined_ambiguity_signal(
        weighted_score=weighted_score,
        domain_ok=domain_ok,
        core_clear_count=core_clear_count,
        clamber=parsed.clamber,
        perceived=parsed.perceived_ambiguity,
        search_readiness_ok=parsed.search_readiness.can_retrieve_meaningful_papers,
        confidence=parsed.search_readiness.confidence,
    )
    is_ambiguous = ambiguity_signals["is_ambiguous"] and (it < max_it)

    # ── 기존 ③: keyword 추출 ───────────────────────────────────────────────
    keywords = [
        kw.strip() for kw in (parsed.keywords or [])
        if isinstance(kw, str) and kw.strip()
    ][:3]
    negative_keywords = [
        kw.strip() for kw in (parsed.negative_keywords or [])
        if isinstance(kw, str) and kw.strip()
    ][:3]

    if not keywords and parsed.suggested_query:
        fallback = re.split(r"[,/;]| and | or ", parsed.suggested_query)
        keywords = [x.strip() for x in fallback if isinstance(x, str) and x.strip()][:3]

    # ── 메시지 구성 ────────────────────────────────────────────────────────
    analysis_content = parsed.model_dump_json()
    messages = [AIMessage(content=analysis_content, name="query_analysis")]

    clarify_questions = []
    if is_ambiguous:
        seen = set()

        # 1순위: APA clarification_priority
        if parsed.perceived_ambiguity and parsed.perceived_ambiguity.clarification_priority:
            for q in parsed.perceived_ambiguity.clarification_priority[:3]:
                q = q.strip()
                if q and q not in seen:
                    seen.add(q)
                    clarify_questions.append(q)

        # 2순위: CLAMBER resolution_hint (severity 높은 순)
        if parsed.clamber:
            clamber_fields = [
                "underspecification", "scope_ambiguity", "intent_ambiguity",
                "entity_ambiguity", "temporal_ambiguity", "reference_ambiguity",
                "multifaceted_query", "conflicting_info",
            ]
            sv_sorted = sorted(
                [(f, getattr(parsed.clamber, f)) for f in clamber_fields
                 if getattr(parsed.clamber, f).detected],
                key=lambda x: x[1].severity,
                reverse=True,
            )
            for _, ctype in sv_sorted:
                hint = ctype.resolution_hint.strip()
                if hint and hint not in seen and len(clarify_questions) < 3:
                    seen.add(hint)
                    clarify_questions.append(hint)

        # 3순위: 기존 5축 clarifying_question
        axis_scored = [
            (getattr(parsed.scores, k).score, k)
            for k in CRITERIA_KEYS
            if getattr(parsed.scores, k).score < QUESTION_SCORE_THRESHOLD
        ]
        for _, k in sorted(axis_scored):
            item = getattr(parsed.scores, k)
            q = (getattr(item, "clarifying_question", "") or "").strip()
            if q and q not in seen and len(clarify_questions) < 3:
                seen.add(q)
                clarify_questions.append(q)

        # STaR-GATE 방식 clarification 메시지
        clarification_text = _build_clarification_message(
            parsed, ambiguity_signals, parsed.scores
        )
        messages.append(AIMessage(content=clarification_text, name="clarify_prompt"))

    return {
        "messages": messages,
        "sender": "query_analysis",
        "iteration": it,
        "is_ambiguous": is_ambiguous,
        "clarify_questions": clarify_questions,
        "keywords": keywords,
        "negative_keywords": negative_keywords,
        "core_clear_count": core_clear_count,
        "weighted_score": weighted_score,
        "refined_query": parsed.suggested_query,
        "user_question": combined_user_input,
        "ambiguity_signals": ambiguity_signals,  # NEW: 디버깅/트레이싱용
    }


def human_clarify_node(state: AgentState) -> AgentState:
    """Interrupt 전용 노드 — 기존과 동일"""
    return {"sender": "human_clarify"}
