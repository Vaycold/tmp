"""
GAPAGO State Definitions
This module defines the states used at each stage of the research process.
"""

from typing import List, Annotated, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import add_messages
from typing import Sequence
from typing_extensions import Annotated
import operator


# =====================================================================
# ======================== Data Models(Schema) ========================
# =====================================================================


# =====================================================================
# -1- QUERY AGENT
class Score(BaseModel):
    """Evaluate the user question on 5 criteria."""

    score: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Clarity score for a specific evaluation criterion.",
        ),
    ]
    reason: Annotated[
        str, Field(default="", description="Reason explaining the assigned score.")
    ]
    clarifying_question: Annotated[
        Optional[str],
        Field(
            description="Question to clarify missing information for this criterion."
        ),
    ]


class Scores(BaseModel):

    domain_clarity: Score
    task_clarity: Score
    methodology_clarity: Score
    data_clarity: Score
    temporal_clarity: Score


class ImportanceWeights(BaseModel):
    """Dynamic Importance Weights for the 5 criteria."""

    domain_clarity: Annotated[
        float, Field(description="Weight for domain clarity importance.")
    ] = 0.30
    task_clarity: Annotated[
        float, Field(description="Weight for task clarity importance.")
    ] = 0.25
    methodology_clarity: Annotated[
        float, Field(description="Weight for methodology clarity importance.")
    ] = 0.20
    data_clarity: Annotated[
        float, Field(description="Weight for data specification importance.")
    ] = 0.15
    temporal_clarity: Annotated[
        float, Field(description="Weight for temporal scope importance.")
    ] = 0.10


class SearchReadiness(BaseModel):
    """Estimation whether meaningful academic paper retrieval is possible with current information."""

    can_retrieve_meaningful_papers: Annotated[
        bool,
        Field(
            description="Whether the query contains enough information for meaningful paper retrieval."
        ),
    ] = False
    confidence: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Confidence that meaningful retrieval is possible.",
        ),
    ] = 0.0
    reason: Annotated[
        str,
        Field(description="Explanation supporting the retrieval readiness decision."),
    ] = ""


# =====================================================================
# -1-1- CLAMBER Taxonomy (Zhang et al., ACL 2024 | arXiv:2405.12063)
#
# 논문 근거:
#   "we establish a taxonomy that consolidates both input understanding
#    and task completion perspectives into three primary dimensions.
#    These dimensions are further conceptualized into eight fine-grained
#    categories to facilitate in-depth evaluation."
#
# 3 Dimensions × 8 Categories:
#   Dim A (Epistemic Misalignment) : entity_ambiguity, temporal_ambiguity
#   Dim B (Linguistic Ambiguity)   : scope_ambiguity, intent_ambiguity, reference_ambiguity
#   Dim C (Aleatoric Output)       : underspecification, multifaceted_query, conflicting_info
# =====================================================================
class ClamberAmbiguityType(BaseModel):
    """CLAMBER taxonomy 단일 모호성 유형 평가"""
    detected: bool = Field(False, description="해당 유형의 모호성 감지 여부")
    severity: float = Field(0.0, ge=0.0, le=1.0, description="심각도 (0=없음, 1=매우 심각)")
    evidence: str = Field("", description="모호성을 유발하는 원문 텍스트 근거")
    resolution_hint: str = Field("", description="이 모호성 해소를 위한 최소 질문/힌트")


class ClamberAnalysis(BaseModel):
    """CLAMBER 8가지 fine-grained 모호성 유형 전체 분석"""
    # Dimension A: Epistemic Misalignment
    entity_ambiguity: ClamberAmbiguityType = Field(default_factory=ClamberAmbiguityType,
        description="동일 표현이 여러 개체를 지칭 (e.g., 'GAN'=생성모델 vs 적대학습)")
    temporal_ambiguity: ClamberAmbiguityType = Field(default_factory=ClamberAmbiguityType,
        description="시간 범위 불명확 (e.g., 'recent', 'modern', 'latest')")
    # Dimension B: Linguistic Ambiguity
    scope_ambiguity: ClamberAmbiguityType = Field(default_factory=ClamberAmbiguityType,
        description="질문 범위·수준 불명확 (너무 넓거나 좁음)")
    intent_ambiguity: ClamberAmbiguityType = Field(default_factory=ClamberAmbiguityType,
        description="목적 불명확 (survey인지, 새 방법인지, 비교인지)")
    reference_ambiguity: ClamberAmbiguityType = Field(default_factory=ClamberAmbiguityType,
        description="지시 대상 불명확 (e.g., 'this approach', 'the model')")
    # Dimension C: Aleatoric Output
    underspecification: ClamberAmbiguityType = Field(default_factory=ClamberAmbiguityType,
        description="필수 정보 누락 (데이터셋, 평가지표, baseline 미언급)")
    multifaceted_query: ClamberAmbiguityType = Field(default_factory=ClamberAmbiguityType,
        description="다중 해석 공존 (복수의 유효한 연구 맥락)")
    conflicting_info: ClamberAmbiguityType = Field(default_factory=ClamberAmbiguityType,
        description="자기 모순·충돌 정보 포함")

    @property
    def detected_types(self) -> list:
        fields = [
            "entity_ambiguity", "temporal_ambiguity", "scope_ambiguity",
            "intent_ambiguity", "reference_ambiguity", "underspecification",
            "multifaceted_query", "conflicting_info",
        ]
        return [f for f in fields if getattr(self, f).detected]

    @property
    def max_severity(self) -> float:
        fields = [
            "entity_ambiguity", "temporal_ambiguity", "scope_ambiguity",
            "intent_ambiguity", "reference_ambiguity", "underspecification",
            "multifaceted_query", "conflicting_info",
        ]
        vals = [getattr(self, f).severity for f in fields]
        return max(vals) if vals else 0.0


# =====================================================================
# -1-2- APA: Alignment with Perceived Ambiguity
#        (Kim et al., EMNLP 2024 | arXiv:2404.11972)
#
# 논문 근거:
#   "(1) LLMs are not explicitly trained to deal with ambiguous utterances;
#    (2) the degree of ambiguity perceived by the LLMs may vary depending
#    on the possessed knowledge."
#
#   "We measure the information gain (INFOGAIN) between the initial input
#    and the disambiguation, identifying samples with high INFOGAIN as
#    ambiguous."
#
# 핵심 아이디어:
#   - LLM이 쿼리에 대해 여러 해석(interpretation)을 생성
#   - 해석들의 다양성 = INFOGAIN = 1 - max_plausibility
#   - INFOGAIN 높음 → 해석 분산 → perceived ambiguous
# =====================================================================
class InterpretationVariant(BaseModel):
    """APA에서 LLM이 생성하는 쿼리 해석 후보"""
    interpretation: str = Field("", description="쿼리의 가능한 해석 (구체적 연구 맥락)")
    plausibility: float = Field(0.5, ge=0.0, le=1.0, description="이 해석이 사용자 의도일 가능성")


class PerceivedAmbiguity(BaseModel):
    """APA Perceived Ambiguity 분석 결과"""
    interpretations: List[InterpretationVariant] = Field(
        default_factory=list,
        description="LLM이 생성한 2~4개 쿼리 해석 후보 (plausibility 합계 ≈ 1.0)"
    )
    infogain_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="INFOGAIN = 1 - max_plausibility. 높을수록 모호 (>0.35 → ambiguous)"
    )
    perceived_ambiguous: bool = Field(
        False,
        description="LLM이 자신의 지식 기준으로 모호하다고 판단하는가"
    )
    dominant_interpretation: str = Field(
        "",
        description="가장 가능성 높은 해석 한 문장 (query_refine에서 refined_query 기반으로 활용)"
    )
    clarification_priority: List[str] = Field(
        default_factory=list,
        description=(
            "STaR-GATE 방식: 정보이득 최대화 순으로 정렬된 clarification 질문 목록. "
            "해석 분기를 가장 많이 줄이는 질문이 앞에 위치."
        )
    )


class QueryAnalysis(BaseModel):
    """Evaluation ambiguity and rewriting academic research questions"""

    scores: Annotated[
        Scores, Field(description="Evaluation scores for each ambiguity criterion.")
    ]
    importance_weights: Annotated[
        ImportanceWeights,
        Field(description="Dynamic importance weights for each criterion."),
    ] = ImportanceWeights()
    search_readiness: Annotated[
        SearchReadiness, Field(description="Overall evaluation of search readiness.")
    ] = SearchReadiness()

    # ── NEW: CLAMBER taxonomy (Zhang et al., ACL 2024) ─────────────────
    clamber: ClamberAnalysis = Field(
        default_factory=ClamberAnalysis,
        description="CLAMBER 8가지 fine-grained 모호성 유형 분석 결과"
    )

    # ── NEW: APA Perceived Ambiguity (Kim et al., EMNLP 2024) ──────────
    perceived_ambiguity: PerceivedAmbiguity = Field(
        default_factory=PerceivedAmbiguity,
        description="APA 방식 Perceived Ambiguity 분석 (INFOGAIN + 해석 후보 + 우선순위 질문)"
    )

    suggested_query: Annotated[
        str,
        Field(
            description="A clear natural-language academic research question inferred from the user question."
        ),
    ]
    keywords: Annotated[
        List[str],
        Field(
            default_factory=list,
            description="2 to 5 core keywords extracted from the user question for downstream retrieval. Prioritize domain terms, sensor/data terms, and task-defining terms without expansion.",
        ),
    ]
    negative_keywords: Annotated[
        List[str],
        Field(
            default_factory=list,
            description="Optional exclusion keywords for downstream retrieval.",
        ),
    ]


# =====================================================================
# -2- RETRIEVE AGENT
class Paper(BaseModel):
    """Individual paper metadata from arXiv."""

    paper_id: str
    title: str
    abstract: str
    url: str
    year: int
    authors: List[str] = Field(default_factory=list)
    score_bm25: float = 0.0
    full_text_sections: dict = Field(default_factory=dict)


# =====================================================================
# -3- LIMITATION AGENT
class LimitationItem(BaseModel):
    """Extracted limitation from a paper."""

    paper_id: str
    claim: str
    evidence_quote: str
    track: str = "author_stated"
    source_section: str = ""


# =====================================================================
# -4- GAP INFER AGENT
class GapCandidate(BaseModel):
    """Research gap identified from limitations."""

    axis: str  # 축 key
    axis_label: str = ""  # 축 한글/영문 레이블
    axis_type: str = "fixed"  # "fixed" | "dynamic"
    gap_statement: str  # 핵심 GAP 1문장 요약
    elaboration: str = ""  # GAP 상세 설명 (2~3문장)
    proposed_topic: str = ""  # 제안 연구 주제
    repeat_count: int = 0  # 반복 등장 논문 수
    supporting_papers: List[str] = Field(default_factory=list)
    supporting_quotes: List[str] = Field(default_factory=list)


# =====================================================================
# -5- CRITIC AGENT
class CriticScores(BaseModel):
    """Quality scores for the analysis."""

    query_specificity: float = Field(0.0, ge=0.0, le=1.0)
    paper_relevance: float = Field(0.0, ge=0.0, le=1.0)
    groundedness: float = Field(0.0, ge=0.0, le=1.0)


class DimensionScore(BaseModel):
    """개별 평가 차원 점수"""

    dimension: str
    label: str
    score: int = Field(0, ge=0, le=10)
    reasoning: str = ""


class EvaluationResult(BaseModel):
    """LLM-as-a-Judge 최종 평가 결과"""

    dimension_scores: list[DimensionScore] = Field(default_factory=list)
    average_score: float = Field(0.0, ge=0.0, le=10.0)
    summary: str = ""


# =====================================================================


# =====================================================================
# ========================= State Definitions =========================
# =====================================================================


class AgentState(TypedDict):
    # ==================================================================
    # -0- ORCHESTRATION
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sender: Annotated[str, "The sender of the last message"]
    errors: List[str]

    # ==================================================================
    # -1- QUERY AGENT
    iteration: int
    is_ambiguous: bool
    clarify_questions: List[str]

    keywords: List[str]
    negative_keywords: List[str]
    refined_query: str
    user_question: str
    max_iterations: int
    core_clear_count: int
    weighted_score: float

    # NEW: 통합 모호성 판정 신호 (CLAMBER + APA + 기존 5축)
    # keys: is_ambiguous, infogain, hard_fail, clamber_fail, apa_fail, soft_fail,
    #       clamber_detected_types, clamber_max_severity
    ambiguity_signals: Optional[dict]

    # ==================================================================
    # -2- RETRIEVE AGENT
    papers: List[dict]

    # ==================================================================
    # -3- LIMITATION AGENT
    limitations: List[dict]

    # ==================================================================
    # -4- GAP INFER AGENT
    gaps: List[dict]

    # ==================================================================
    # -5- CRITIC AGENT
    critic: Optional[dict]
    critic_loop_count: int

    # ==================================================================
    # 추후 다시 확인
    # iteration: int
    # max_iterations: int
    # errors: List[str]
    trace: dict