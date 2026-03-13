"""
================================================== 1. QUERY ANALYSIS ==================================================
- 역할
1. 사용자 질문의 모호성(ambiguity) 평가
2. 논문 검색이 가능한 수준인지 판단
3. 모호하면 사용자에게 보완 질문을 던질 수 있게 상태를 반환
-> 즉, 5사지 축에 따라 사용자 질문을 평가하고, 지금 바로 검색해도 되는가?를 판단하는 노드
=======================================================================================================================

"""

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
    SystemMessage,
)  # 랭체인 메시지 객체(사용자, 모델, 공통 부모 타입)
from states import AgentState  # 랭그래프 노드가 입력받고 출력하는 공용 상태 스키마
from llm import get_llm
from tools import build_role_tools
from prompts.system import make_system_prompt
import re

from states import QueryAnalysis, ImportanceWeights

llm = get_llm()

# =====================================================================================================================
# ==================================================== 0) 초기 설정 ====================================================
# =====================================================================================================================

ROLE_TOOLS = build_role_tools()
QUERY_TOOLS = ROLE_TOOLS["QUERY_TOOLS"]

# 모호성 판정 기준
CLEAR_MIN_SCORE: float = (
    0.4  # 각 항목 점수가 최소 이 값 이상이면, “완전히 없는 건 아니다” 정도로 보는 기준
)
CORE_SLOT_MIN_COUNT: int = (
    1  # 태스크, 방법론, 데이터 중 몇 개 이상은 어느 정도 채워져 있어야 하는지 보는 기준
)
QUESTION_SCORE_THRESHOLD: float = (
    0.6  # 이 값보다 낮으면 추가 질문을 생성할 대상으로 보는 기준
)
WEIGHTED_SCORE_THRESHOLD: float = 0.6  # 전체 가중 점수의 하한선
SEARCH_CONFIDENCE_THRESHOLD: float = (
    0.5  # “현재 정보로 검색 의미가 있다”는 신뢰도가 이 값보다 낮으면 모호하다고 보는 기준
)

DEFAULT_WEIGHTS = {
    "domain_clarity": 0.30,  # 도메인 (30%)
    "task_clarity": 0.25,  # 태스크 (25%)
    "methodology_clarity": 0.20,  # 방법론 (20%)
    "data_clarity": 0.15,  # 데이터 (15%)
    "temporal_clarity": 0.10,  # 기간 (10%)
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

# =====================================================================================================================
# =============================================== 1) 서브 함수 정의 =====================================================
# =====================================================================================================================


def _normalize_weights(raw_weights: ImportanceWeights | None) -> dict[str, float]:
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


def _build_clarification_message(
    clarify_questions: list[str],
    scores: object,
) -> str:
    weak_areas_info = []
    # 부족한 영역을 찾고 예시와 함께 정리
    for key in CRITERIA_KEYS:
        item = getattr(scores, key, None)
        if item is not None and item.score < QUESTION_SCORE_THRESHOLD:
            label = CRITERIA_META[key]["label"]
            example = CRITERIA_META[key]["example"]
            weak_areas_info.append(f"- {label}: {example}")

    # 인트로 문구
    intro = "네, 제안해주신 주제로 논문을 찾아보고 있습니다! 다만, 조금 더 구체적인 정보를 주시면 훨씬 정확한 검색 결과를 드릴 수 있을 것 같아요."

    if weak_areas_info:
        guide_section = "\n구체적으로 아래와 같은 정보들이 부족합니다:\n" + "\n".join(
            weak_areas_info
        )
    else:
        guide_section = ""

    # 질문 목록 구성
    if clarify_questions:
        qs = "\n".join(f"{i}. {q}" for i, q in enumerate(clarify_questions[:3], 1))
        message = (
            f"{intro}\n"
            f"{guide_section}\n\n"
            f"아래 질문에 대한 답을 포함해주시면 좋습니다:\n{qs}\n\n"
            "조금 더 자세한 질문을 해주시면 바로 조사를 시작할게요!"
        )
        return message

    # Fallback (질문이 없을 경우 기본 가이드)
    return (
        f"{intro}\n"
        "예를 들어, 어떤 산업 분야인지, 어떤 데이터나 기술에 관심이 있는지 조금만 더 자세히 말씀해 주시겠어요?\n"
        "알려주시는 내용에 맞춰 최적의 논문들을 정리해 드릴게요."
    )


# =====================================================================================================================
# =============================================== 2) 에이전트(노드) 생성 =================================================
# =====================================================================================================================

system_prompt = make_system_prompt(
    "ROLE: Query Analysis Agent\n"
    "You evaluate ambiguity and rewrite academic research questions.\n\n"
    "Your tasks:\n"
    "1) Evaluate the user question on 5 criteria with following definitions:\n"
    "   - domain_clarity: Identification of specific industry sectors or application sector (e.g., Finance, Healthcare).\n"
    "   - task_clarity: Specificity of the problem to be solved (e.g., anomaly detection, forecasting, generation).\n"
    "   - methodology_clarity: Mention of specific techniques, algorithms, or architectures (e.g., UDA, Transformers, RL).\n"
    "   - data_clarity: Specification of data modalities or datasets (e.g., EHR, sensor logs, molecular graphs).\n"
    "   - temporal_clarity: Mention of time constraints or focus on recent work (e.g., since 2020, last 5 years).\n"
    "2) Use reasoning first, then assign scores conservatively.\n"
    "3) Estimate whether meaningful academic paper retrieval is possible with current information.\n"
    "4) Suggest dynamic importance weights for the 5 criteria.\n"
    "5) Suggest a keyword for academic search query.\n"
    "6) If the question is ambiguous, ask concise clarifying questions.\n\n"
    "Scoring guidance (Anchor Points):\n"
    "- 0.0: No mention at all. Information is completely missing.\n"
    "- 0.2: Extremely vague or generic (e.g., 'something', 'research', 'data').\n"
    "- 0.4: Mentioned but broad/underspecified (e.g., just 'medical', 'prediction', 'AI'). The category is known but context is missing.\n"
    "- 0.6: Partial clarity with some context (e.g., 'clinical drug prediction', 'transformer-based models'). Provides a general direction.\n"
    "- 0.8: High clarity with specific details (e.g., 'adverse event prediction for cancer drugs', 'adversarial domain adaptation'). Clear enough for effective search.\n"
    "- 1.0: Perfect clarity with specific instances/constraints (e.g., 'Toxicity prediction using ImageNet-pretrained ResNet since 2022').\n\n"
    "Scoring Rules:\n"
    "- Be conservative: If information is halfway between 0.4 and 0.6, pick 0.4.\n"
    "- Do not infer unstated details: Only score based on text explicitly provided by the user.\n"
    "- Methodological paradigms (e.g., 'transfer learning', 'GAN') should be scored at least 0.7 for methodology_clarity.\n"
    "- If a concept is mentioned but you need to ask a 'Which one?' question, it should not exceed 0.6.\n\n"
    "Suggested keyword guideline:\n"
    "- suggested_keword: write it as a natural academic research question.\n"
    "- Keep the original user intent while making the question clearer and more specific.\n\n"
    "Keyword extraction guideline:\n"
    "- Extract 2 to 5 concise keywords or short phrases only.\n"
    "- Build keywords directly from the user answer without expansion.\n"
    "- Prioritize domain terms, task-defining terms, and important data or sensor terms.\n"
    "- Good examples: PMSM fault diagnosis, vibration sensor, urban airflow, battery health.\n"
    "- Avoid generic terms such as AI, model, system, study unless they are the actual core concept.\n"
    "- negative_keywords: include 1 to 3 short exclusion terms only when explicit exclusion is useful; otherwise return an empty list.\n"
    "- Do not include full sentences and do not expand synonyms.\n\n"
    "Importance weight constraints:\n"
    "- All weights must be positive real numbers (>= 0).\n"
    "- The sum of all weights must equal exactly 1.0.\n"
    "- Do not omit any weight.\n"
)

structured_llm = llm.with_structured_output(QueryAnalysis)


def query_analysis_node(state: AgentState) -> AgentState:
    it = state.get("iteration", 0) + 1
    max_it = state.get("max_iterations", 3)

    # 입력값 방식 1) 누적된 HumanMessage 만
    user_inputs = [
        m.content.strip() for m in state["messages"] if isinstance(m, HumanMessage)
    ]
    combined_user_input = "\n".join(user_inputs)
    input_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=combined_user_input),
    ]
    # 입력값 방식 2) 누적된 모든 Message
    # messages = [SystemMessage(content=system_prompt), *state["messages"]]

    parsed: QueryAnalysis = structured_llm.invoke(input_messages)

    # 1. 각 축의 모호성 가중 점수 계산
    dynamic_weights = _normalize_weights(parsed.importance_weights)
    weighted_score = sum(
        getattr(parsed.scores, key).score * dynamic_weights.get(key, 0.0)
        for key in CRITERIA_KEYS
    )  # 각축의 평균 score 점수

    # 2. 도메인 최소 조건 (>0.4)
    ## 도메인이 최소 수준이라도 언급되었는지 봄
    domain_score = parsed.scores.domain_clarity.score
    domain_ok = domain_score >= CLEAR_MIN_SCORE

    # 3. 핵심 슬롯 수 계산
    ## 무엇을 하려는지(task), 어떤 접근인지(methodology), 어떤 데이터인지(data) 핵심 슬롯 3개 중 몇 개가 최소 수준 이상인지 count
    core_slots = ["task_clarity", "methodology_clarity", "data_clarity"]
    core_clear_count = sum(
        1 for key in core_slots if getattr(parsed.scores, key).score >= CLEAR_MIN_SCORE
    )

    # 4. 최종 모호성 판정
    hard_fail = not domain_ok  # 도메인이 없으면 바로 실패
    soft_fail = (
        core_clear_count < CORE_SLOT_MIN_COUNT  # 핵심 슬롯이 1개 미만이면 부족
        or weighted_score
        < WEIGHTED_SCORE_THRESHOLD  # 전체 질문 품질이 너무 낮으면 부족
        or not parsed.search_readiness.can_retrieve_meaningful_papers  # LLM이 “의미 있는 논문 검색 어렵다”고 판단하면 부족
        or (
            parsed.search_readiness.confidence < SEARCH_CONFIDENCE_THRESHOLD
            and weighted_score < WEIGHTED_SCORE_THRESHOLD
        )  # confidence가 낮더라도 weighted score가 충분히 높으면 너무 엄격하게 ambiguity 처리하지 않음
    )
    is_ambiguous = (hard_fail or soft_fail) and (it < max_it)

    # 5. retrieval-ready keyword state 구성
    keywords = [kw.strip() for kw in (parsed.keywords or []) if isinstance(kw, str) and kw.strip()][:3]
    negative_keywords = [kw.strip() for kw in (parsed.negative_keywords or []) if isinstance(kw, str) and kw.strip()][:3]

    # fallback: keyword가 비어 있으면 suggested_query에서 짧은 핵심 단어만 보완
    if not keywords and parsed.suggested_query:
        fallback = re.split(r"[,/;]| and | or ", parsed.suggested_query)
        keywords = [x.strip() for x in fallback if isinstance(x, str) and x.strip()][:3]

    # 6. 결과 메시지 구성
    analysis_content = parsed.model_dump_json()
    messages = [AIMessage(content=analysis_content, name="query_analysis")]

    clarify_questions = []
    if is_ambiguous:
        # 모호할 때만 질문 수집
        seen = set()
        for key in CRITERIA_KEYS:
            item = getattr(parsed.scores, key)
            if item.score < QUESTION_SCORE_THRESHOLD and item.clarifying_question:
                q = item.clarifying_question.strip()
                if q not in seen:
                    seen.add(q)
                    clarify_questions.append(q)

        # 모호할 때만 안내 메시지 추가
        clarification_text = _build_clarification_message(
            clarify_questions, parsed.scores  # .model_dump()
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
        "refined_query": parsed.suggested_query,  # ← 이게 없으면 query_refinement에서 보존할 값이 없음
        "user_question": combined_user_input,
    }


def human_clarify_node(state: AgentState) -> AgentState:
    """
    Interrupt 전용 노드
    여기서는 LLM 호출/자동 응답 생성하지 않음
    UI/CLI에서 snapshot을 보고 사용자 입력을 받은 뒤 app.update_state로 아래 필드를 채움:
      - query_approved: bool
      - (optional) refined_query or user feedback message
    """
    # 아무것도 변경하지 않아도 됨 (interrupt로 멈출 것)
    return {"sender": "human_clarify"}
