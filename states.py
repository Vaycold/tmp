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
