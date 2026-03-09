from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from states import AgentState
from llm import get_llm
from tools import build_role_tools
from prompts.system import make_system_prompt
from utils.parse_json import parse_json

llm = get_llm()

ROLE_TOOLS = build_role_tools()
QUERY_TOOLS = ROLE_TOOLS["QUERY_TOOLS"]

CLEAR_MIN_SCORE: float = 0.3
CORE_SLOT_MIN_COUNT: int = 2
QUESTION_SCORE_THRESHOLD: float = 0.6
WEIGHTED_SCORE_THRESHOLD: float = 0.35
SEARCH_CONFIDENCE_THRESHOLD: float = 0.5

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
        "label": "도메인 명확성",
        "description": "연구 분야(예: NLP, computer vision, drug discovery, fault diagnosis 등)가 구체적으로 특정되는지",
    },
    "task_clarity": {
        "label": "태스크 명확성",
        "description": "무엇을 하려는지(예: prediction, classification, detection, generation, retrieval, summarization 등)가 구체적으로 드러나는지",
    },
    "methodology_clarity": {
        "label": "방법론 명확성",
        "description": "특정 기법, 알고리즘, 모델 아키텍처(예: transformer, GAN, RL, domain adaptation 등)가 언급되는지",
    },
    "data_clarity": {
        "label": "데이터 명확성",
        "description": "사용 데이터셋, 실험 대상, 입력 데이터 유형(예: ImageNet, clinical records, EHR, sensor signals 등)이 명시되는지",
    },
    "temporal_clarity": {
        "label": "기간 명확성",
        "description": "연구 연도 범위 혹은 최신 연구에 대한 언급(예: 'since 2020', 'recent', '2023~')이 있는지",
    },
}


def get_latest_human_message(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _normalize_weights(raw_weights: dict | None) -> dict:
    if not isinstance(raw_weights, dict):
        return DEFAULT_WEIGHTS.copy()

    cleaned = {}
    for key in CRITERIA_KEYS:
        val = raw_weights.get(key, DEFAULT_WEIGHTS[key])
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = DEFAULT_WEIGHTS[key]
        cleaned[key] = max(0.0, val)

    total = sum(cleaned.values())
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()

    return {k: v / total for k, v in cleaned.items()}


def _safe_parse(content: str) -> dict:
    try:
        parsed = parse_json(content)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _build_clarification_message(
    clarify_questions: list[str],
    scores: dict,
) -> str:
    weak_areas = [
        meta["label"]
        for key, meta in scores.items()
        if meta.get("score", 1.0) < QUESTION_SCORE_THRESHOLD
    ]

    intro = "현재 질문만으로는 검색 범위가 조금 넓습니다."
    if weak_areas:
        intro += f" 특히 {', '.join(weak_areas[:3])} 정보가 부족합니다."

    if clarify_questions:
        qs = "\n".join(f"{i}. {q}" for i, q in enumerate(clarify_questions[:3], 1))
        return f"{intro}\n아래를 알려주시면 더 정확한 논문 검색이 가능합니다.\n{qs}"

    return (
        f"{intro}\n"
        "아래를 알려주시면 더 정확한 논문 검색이 가능합니다.\n"
        "1. 대상 도메인이나 데이터는 무엇인가요?\n"
        "2. 원하는 태스크는 무엇인가요?\n"
        "3. 특정 방법론이나 기간 제한이 있나요?"
    )


query_analysis_agent = create_agent(
    llm,
    tools=QUERY_TOOLS,
    system_prompt=make_system_prompt(
        "ROLE: Query Analysis Agent\n"
        "You evaluate ambiguity and rewrite academic research questions.\n\n"
        "Your tasks:\n"
        "1) Evaluate the user question on 5 criteria:\n"
        "   - domain_clarity\n"
        "   - task_clarity\n"
        "   - methodology_clarity\n"
        "   - data_clarity\n"
        "   - temporal_clarity\n"
        "2) Use reasoning first, then assign scores conservatively.\n"
        "3) Estimate whether meaningful academic paper retrieval is possible with current information.\n"
        "4) Suggest dynamic importance weights for the 5 criteria.\n"
        "5) Propose a refined academic search query.\n"
        "6) If the question is ambiguous, ask concise clarifying questions.\n\n"
        "Scoring guidance:\n"
        "- missing criterion: 0.0~0.3\n"
        "- partial mention: 0.4~0.6\n"
        "- clear specific mention: 0.7~1.0\n"
        "- Be conservative. Do not infer unstated details.\n\n"
        "Output valid JSON only:\n"
        "{\n"
        '  "scores": {\n'
        '    "domain_clarity": {"score": 0.0, "reason": "...", "clarifying_question": "...or null"},\n'
        '    "task_clarity": {"score": 0.0, "reason": "...", "clarifying_question": "...or null"},\n'
        '    "methodology_clarity": {"score": 0.0, "reason": "...", "clarifying_question": "...or null"},\n'
        '    "data_clarity": {"score": 0.0, "reason": "...", "clarifying_question": "...or null"},\n'
        '    "temporal_clarity": {"score": 0.0, "reason": "...", "clarifying_question": "...or null"}\n'
        "  },\n"
        '  "importance_weights": {\n'
        '    "domain_clarity": 0.0,\n'
        '    "task_clarity": 0.0,\n'
        '    "methodology_clarity": 0.0,\n'
        '    "data_clarity": 0.0,\n'
        '    "temporal_clarity": 0.0\n'
        "  },\n"
        '  "missing_slots": ["..."],\n'
        '  "search_readiness": {\n'
        '    "can_retrieve_meaningful_papers": true,\n'
        '    "confidence": 0.0,\n'
        '    "reason": "..."\n'
        "  }\n"
        "}\n"
    ),
)


def query_analysis_node(state: AgentState) -> AgentState:
    it = state.get("iteration", 0) + 1
    max_it = state.get("max_iterations", 3)

    result = query_analysis_agent.invoke({"messages": state["messages"][-6:]})
    content = result["messages"][-1].content if result.get("messages") else "{}"
    parsed = _safe_parse(content)

    raw_scores = parsed.get("scores", {}) or {}
    scores = {}
    for key, meta in CRITERIA_META.items():
        item = raw_scores.get(key, {}) or {}
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        scores[key] = {
            "score": score,
            "reason": item.get("reason", ""),
            "label": meta["label"],
        }

    dynamic_weights = _normalize_weights(parsed.get("importance_weights", {}))
    weighted_score = sum(
        scores[key]["score"] * dynamic_weights.get(key, 0.0) for key in CRITERIA_KEYS
    )

    domain_ok = scores["domain_clarity"]["score"] >= CLEAR_MIN_SCORE
    core_slots = ["task_clarity", "methodology_clarity", "data_clarity"]
    core_clear_count = sum(
        1 for key in core_slots if scores[key]["score"] >= CLEAR_MIN_SCORE
    )

    search_readiness = parsed.get("search_readiness", {}) or {}
    can_retrieve = bool(search_readiness.get("can_retrieve_meaningful_papers", False))
    try:
        retrieval_confidence = float(search_readiness.get("confidence", 0.0))
    except (TypeError, ValueError):
        retrieval_confidence = 0.0

    hard_fail = not domain_ok
    soft_fail = (
        core_clear_count < CORE_SLOT_MIN_COUNT
        or weighted_score < WEIGHTED_SCORE_THRESHOLD
        or not can_retrieve
        or (
            retrieval_confidence < SEARCH_CONFIDENCE_THRESHOLD and weighted_score < 0.45
        )
    )
    is_ambiguous = hard_fail or soft_fail

    clarify_questions = []
    seen = set()
    for key in CRITERIA_KEYS:
        item = raw_scores.get(key, {}) or {}
        cq = item.get("clarifying_question")
        score = scores[key]["score"]
        if (
            score < QUESTION_SCORE_THRESHOLD
            and isinstance(cq, str)
            and cq.strip()
            and cq.strip() not in seen
        ):
            seen.add(cq.strip())
            clarify_questions.append(cq.strip())

    if it >= max_it:
        is_ambiguous = False

    messages = [AIMessage(content=content, name="query_analysis")]

    if is_ambiguous:
        clarification_text = _build_clarification_message(
            clarify_questions=clarify_questions,
            scores=scores,
        )
        messages.append(AIMessage(content=clarification_text, name="clarify_prompt"))

    return {
        "messages": messages,
        "sender": "query_analysis",
        "iteration": it,
        "is_ambiguous": is_ambiguous,
        "clarify_questions": clarify_questions,
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
