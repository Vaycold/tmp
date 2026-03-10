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


# =====================================================================
# ============================ Data Models ============================
# =====================================================================
"""
`Annotated` 사용 이유
- 추가 정보 제공(타입 힌트) / 문서화
- 타입 힌트에 추가적인 정보(코드에 대한 추가 설명)를 포함시킬 수 있음. 
- 이는 코드를 읽는 사람이나 도구에 더 많은 컨텍스트를 제공함

- salary: Annotated[float, Field(gt=0, lt=10000, description="연봉 (단위: 만원, 최대 10억)")]
  skills: Annotated[List[str], Field(min_items=1, max_items=10, description="보유 기술 (1-10개)")]
    ... = 누락되면 안된다는 뜻
    salary = 0~10000이어야함을 명시

"""


class Paper(BaseModel):
    """Individual paper metadata from arXiv."""

    paper_id: str
    title: str
    abstract: str
    url: str
    year: int
    authors: List[str] = Field(default_factory=list)
    score_bm25: float = 0.0
    # full text 섹션 추가
    full_text_sections: dict = Field(default_factory=dict)
    """
    {
        "introduction": "...",
        "conclusion":   "...",
        "limitations":  "...",
        "future_work":  "...",
        "discussion":   "...",
    }
    비어 있으면 abstract만 사용 (fallback)
    """


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


# =====================================================================
# ========================= State Definitions =========================
# =====================================================================
"""
상태(State)에 대한 모든 것
1. 상태란 ?
    - '노드와 노드 사이에 정보 전달을 할 건데 여기서 쓰는 키는 이것입니다.'를 사전에 정의하는 것
    -  노드 별 상태 값의 변화
2. 모든 값을 다 채우지 않아도 됨
3. 노드에서 필요한 상태 값을 조회해서 동작에 활용
4. 각 노드에서 새롭게 업데이트 하는 값은 기존 Key 값을 덮어쓰는 방식
5. Reducer -> add_messages 란?
    - 기존에 있던 메시지에 추가하는 것
    - add_messages 추가 시, [ ] 리스트 내부에 계속 메시지가 쌓이게 됨!
    - = append 라고 보면 됨

=> 앞으로 모든 agent에서 사용할 key값을 사전에 정의

"""


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

    # ==================================================================
    # 추후 다시 확인
    # iteration: int
    # max_iterations: int
    # errors: List[str]
    trace: dict
