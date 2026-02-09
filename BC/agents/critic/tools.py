"""
Critic Agent - Tools
모든 Agent의 출력을 점수로 평가
"""

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
import json

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    groq_api_key=os.environ.get('GROQ_API_KEY')
)


@tool
def evaluate_query_clarity(analysis_result: str, missing_elements: str) -> str:
    """Query Analysis 결과의 명확성을 평가합니다.
    
    Args:
        analysis_result: 분석 결과 JSON
        missing_elements: 부족한 요소 JSON
        
    Returns:
        명확성 점수 (0-1, 낮을수록 명확)
    """
    prompt = ChatPromptTemplate.from_template(
        """Query 분석 결과를 평가하세요:

분석 결과: {analysis}
부족 요소: {missing}

평가 기준:
1. 5가지 기준 중 몇 개가 충족되었는가?
2. 각 기준의 점수가 얼마나 높은가?
3. 부족한 요소가 몇 개인가?

명확성 점수 계산:
- 0.0: 매우 명확 (모든 기준 충족, 점수 > 0.7)
- 0.3: 보통 (3-4개 기준 충족)
- 0.5-1.0: 모호함 (2개 이하 충족)

JSON으로 출력:
{{
    "clarity_score": 0.0-1.0,
    "reason": "평가 근거"
}}"""
    )
    
    result = (prompt | llm).invoke({
        "analysis": analysis_result,
        "missing": missing_elements
    })
    return result.content


@tool
def evaluate_paper_relevance(papers: str, keywords: str) -> str:
    """논문 검색 결과의 정합성을 평가합니다.
    
    Args:
        papers: 검색된 논문 JSON
        keywords: 검색 키워드
        
    Returns:
        정합성 점수 (0-1)
    """
    prompt = ChatPromptTemplate.from_template(
        """논문 검색 결과를 평가하세요:

키워드: {keywords}
논문 수: {paper_count}

평가 기준:
1. 논문 개수가 충분한가? (3개 이상: 1.0, 1-2개: 0.5, 0개: 0.0)
2. 논문이 키워드와 관련 있는가?

정합성 점수 계산:
- 1.0: 매우 적합 (5개 이상, 관련성 높음)
- 0.7: 적합 (3-4개, 관련성 보통)
- 0.5: 부족 (1-2개)
- 0.0: 없음 (0개)

JSON으로 출력:
{{
    "relevance_score": 0.0-1.0,
    "reason": "평가 근거"
}}"""
    )
    
    papers_data = json.loads(papers) if papers else []
    
    result = (prompt | llm).invoke({
        "keywords": keywords,
        "paper_count": len(papers_data)
    })
    return result.content


@tool
def evaluate_gap_quality(classified_gaps: str, priority_axis: str) -> str:
    """GAP 분류 결과의 품질을 평가합니다.
    
    Args:
        classified_gaps: 분류된 GAP JSON
        priority_axis: 우선순위 축
        
    Returns:
        품질 점수 (0-1)
    """
    prompt = ChatPromptTemplate.from_template(
        """GAP 분류 결과를 평가하세요:

분류 결과: {gaps}
우선순위 축: {priority}

평가 기준:
1. 각 축에 GAP이 골고루 분류되었는가?
2. 우선순위 축에 충분한 GAP이 있는가? (3개 이상)
3. GAP이 구체적이고 근거가 있는가?

품질 점수:
- 1.0: 매우 좋음 (10개 이상, 골고루 분포)
- 0.7: 좋음 (5-9개, 우선축에 3개 이상)
- 0.4: 부족 (3-4개)
- 0.0: 없음 (2개 이하)

JSON으로 출력:
{{
    "quality_score": 0.0-1.0,
    "reason": "평가 근거"
}}"""
    )
    
    result = (prompt | llm).invoke({
        "gaps": classified_gaps,
        "priority": priority_axis
    })
    return result.content
