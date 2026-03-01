"""
Pydantic models for GAPago LangGraph.
"""

from typing import Optional, TypedDict
from pydantic import BaseModel, Field


class Paper(BaseModel):
    """Individual paper metadata from arXiv."""
    paper_id: str
    title: str
    abstract: str
    url: str
    year: int
    authors: list[str] = Field(default_factory=list)
    score_bm25: float = 0.0


class LimitationItem(BaseModel):
    """Extracted limitation from a paper."""
    paper_id: str
    claim: str
    evidence_quote: str


class GapCandidate(BaseModel):
    """Research gap identified from limitations."""
    axis: str                                        # 축 key
    axis_label: str = ""                             # 축 한글/영문 레이블
    axis_type: str = "fixed"                         # "fixed" | "dynamic"
    gap_statement: str                               # 핵심 GAP 1문장 요약
    elaboration: str = ""                            # GAP 상세 설명 (2~3문장)
    proposed_topic: str = ""                         # 제안 연구 주제
    repeat_count: int = 0                            # 반복 등장 논문 수
    supporting_papers: list[str] = Field(default_factory=list)
    supporting_quotes: list[str] = Field(default_factory=list)
    
# class GapCandidate(BaseModel):
#     """Research gap identified from limitations."""
#     axis: str
#     gap_statement: str
#     supporting_papers: list[str] = Field(default_factory=list)
#     supporting_quotes: list[str] = Field(default_factory=list)


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

class AgentState(TypedDict):
    """LangGraph state for the pipeline."""
    user_question: str
    refined_query: str
    keywords: list[str]
    negative_keywords: list[str]
    papers: list[Paper]
    limitations: list[LimitationItem]
    gaps: list[GapCandidate]
    critic: Optional[CriticScores]
    iteration: int
    max_iterations: int
    route: str
    errors: list[str]
    trace: dict