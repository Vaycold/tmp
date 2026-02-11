"""
Query Analysis Agent - Tools
질문 분석 도구
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
def analyze_query_criteria(query: str) -> str:
    """질문을 N가지 기준으로 분석합니다.
    
    기준:
    - domain: 연구 도메인
    - data: 대상 데이터
    - method: 방법론/기술
    - timeframe: 시간 범위
    - purpose: 연구 목적
    
    Args:
        query: 사용자의 연구 질문
        
    Returns:
        JSON 형식의 분석 결과
    """
    prompt = ChatPromptTemplate.from_template(
        """질문을 5가지 기준으로 평가하세요:
        
질문: {query}

기준:
1. **domain** (연구 도메인): NLP, CV, Robotics 등
2. **data** (대상 데이터): 텍스트, 이미지, 센서 데이터 등
3. **method** (방법론): Transformer, CNN, RL 등
4. **timeframe** (시간 범위): 최근, 2024년, 특정 기간
5. **purpose** (연구 목적): 한계점 분석, 성능 비교 등

각 기준별로 평가:

{{
    "domain": {{
        "mentioned": true/false,
        "value": "구체적 내용 또는 null",
        "score": 0.0-1.0
    }},
    "data": {{
        "mentioned": true/false,
        "value": "구체적 내용 또는 null",
        "score": 0.0-1.0
    }},
    "method": {{
        "mentioned": true/false,
        "value": "구체적 내용 또는 null",
        "score": 0.0-1.0
    }},
    "timeframe": {{
        "mentioned": true/false,
        "value": "구체적 내용 또는 null",
        "score": 0.0-1.0
    }},
    "purpose": {{
        "mentioned": true/false,
        "value": "구체적 내용 또는 null",
        "score": 0.0-1.0
    }}
}}

점수:
- 1.0: 매우 명확
- 0.5: 어느 정도 언급
- 0.0: 없음

JSON만 출력:"""
    )
    
    result = (prompt | llm).invoke({"query": query})
    return result.content


@tool
def extract_search_keywords(query: str, analysis_result: str) -> str:
    """분석 결과를 바탕으로 검색 키워드를 추출합니다.
    
    Args:
        query: 사용자의 연구 질문
        analysis_result: analyze_query_criteria의 JSON 결과
        
    Returns:
        쉼표로 구분된 영어 키워드 (3-5개)
    """
    prompt = ChatPromptTemplate.from_template(
        """논문 검색용 키워드를 추출하세요.

질문: {query}

분석 결과: {analysis}

규칙:
1. 영어로 추출
2. 학술 논문 검색에 적합한 용어
3. 3-5개 키워드
4. 쉼표로 구분

키워드만 출력:"""
    )
    
    result = (prompt | llm).invoke({
        "query": query,
        "analysis": analysis_result
    })
    return result.content


@tool
def generate_clarification_questions(missing_elements: str) -> str:
    """부족한 요소에 대한 질문을 생성합니다 (Human-in-the-Loop).
    
    Args:
        missing_elements: 부족한 요소 JSON
        
    Returns:
        사용자에게 물어볼 질문들
    """
    prompt = ChatPromptTemplate.from_template(
        """부족한 요소에 대해 구체적인 질문을 생성하세요:

부족한 요소: {missing}

각 요소별로 명확하고 구체적인 질문을 만드세요.

예시:
- domain 부족 → "어떤 연구 분야에 관심이 있으신가요? (예: NLP, Computer Vision)"
- data 부족 → "어떤 종류의 데이터를 다루고 싶으신가요?"

JSON 형식으로 출력:
{{
    "questions": [
        {{"criterion": "domain", "question": "..."}},
        ...
    ]
}}"""
    )
    
    result = (prompt | llm).invoke({"missing": missing_elements})
    return result.content
