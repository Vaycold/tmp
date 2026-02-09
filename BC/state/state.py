"""
GAPAGO State 정의
모든 Agent가 공유하는 상태
"""

from typing import TypedDict, List, Dict, Optional


class GAPAGOState(TypedDict):
    """GAPAGO 시스템 전체 State"""
    
    # ═══════════════════════════════════════════════════════
    # 입력
    # ═══════════════════════════════════════════════════════
    user_query: str
    
    # ═══════════════════════════════════════════════════════
    # Query Analysis Agent 출력
    # ═══════════════════════════════════════════════════════
    keywords: List[str]                      # 검색 키워드
    query_analysis_result: Optional[Dict]    # 상세 분석 결과
    missing_elements: List[Dict]             # 부족한 요소
    
    # ═══════════════════════════════════════════════════════
    # Critic Agent 출력 (Query)
    # ═══════════════════════════════════════════════════════
    query_clarity_score: float               # 모호성 점수 (0-1, 낮을수록 명확)
    
    # ═══════════════════════════════════════════════════════
    # Paper Search Agent 출력
    # ═══════════════════════════════════════════════════════
    papers: List[Dict]
    # 각 paper 구조:
    # {
    #     'title': str,
    #     'abstract': str,
    #     'keywords': List[str],
    #     'limitation': str,
    #     'discussion': str,
    #     'future_work': str,
    #     'embedding': List[float]  # 키워드 임베딩
    # }
    
    # ═══════════════════════════════════════════════════════
    # Web Search Agent 출력
    # ═══════════════════════════════════════════════════════
    web_info: List[Dict]
    # 각 web_info 구조:
    # {
    #     'source': str,
    #     'url': str,
    #     'content': str,
    #     'limitation_mentions': str
    # }
    
    # ═══════════════════════════════════════════════════════
    # Critic Agent 출력 (Paper)
    # ═══════════════════════════════════════════════════════
    paper_relevance_score: float             # 논문 정합성 점수 (0-1)
    
    # ═══════════════════════════════════════════════════════
    # GAP Classification Agent 출력
    # ═══════════════════════════════════════════════════════
    classified_gaps: Dict[str, List[str]]
    # {
    #     '데이터_의존성': [...],
    #     '실제_환경_검증': [...],
    #     '확장성': [...],
    #     '구조적_한계': [...]
    # }
    priority_axis: str                       # 우선순위 축
    
    # ═══════════════════════════════════════════════════════
    # Critic Agent 출력 (GAP)
    # ═══════════════════════════════════════════════════════
    gap_quality_score: float                 # GAP 품질 점수 (0-1)
    
    # ═══════════════════════════════════════════════════════
    # Topic Generation Agent 출력
    # ═══════════════════════════════════════════════════════
    research_topics: List[str]               # 추천 연구 주제
    
    # ═══════════════════════════════════════════════════════
    # 제어 변수
    # ═══════════════════════════════════════════════════════
    retry_count: int                         # 현재 재시도 횟수
    max_retries: int                         # 최대 재시도 횟수
    query_embedding: Optional[List[float]]   # 질문 임베딩
