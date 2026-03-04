"""
GAPAGO State Definitions
This module defines the states used at each stage of the research process.
"""

from typing import List, Annotated, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing import Sequence
from typing_extensions import Annotated


# ====================== Data Models ======================
class Paper(BaseModel):
    """Individual paper metadata from arXiv."""

    paper_id: str
    title: str
    abstract: str
    url: str
    year: int
    authors: List[str] = Field(default_factory=list)
    score_bm25: float = 0.0


class LimitationItem(BaseModel):
    """Extracted limitation from a paper."""

    paper_id: str
    claim: str
    evidence_quote: str


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


# ====================== State Definitions ======================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sender: Annotated[str, "The sender of the last message"]

    user_question: str
    refined_query: str
    keywords: List[str]
    negative_keywords: List[str]

    missing_slots: List[str]
    clarify_questions: List[str]
    query_proposal: str
    query_approved: bool
    ask_human: bool

    papers: List[dict]
    limitations: List[dict]
    gaps: List[dict]
    critic: Optional[dict]

    iteration: int
    max_iterations: int
    errors: List[str]
    trace: dict
