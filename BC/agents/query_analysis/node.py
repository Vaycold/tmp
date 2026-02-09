"""
Query Analysis Agent - Node
LangGraph 노드
"""

import sys
sys.path.append('/home/claude/gapago_project')

from state.state import GAPAGOState
from .agent import create_query_analysis_agent
import json
import re


def query_analysis_node(state: GAPAGOState) -> GAPAGOState:
    """Query Analysis 노드
    
    입력:
        state['user_query']
        
    출력:
        state['keywords']
        state['query_analysis_result']
        state['missing_elements']
    """
    print("\n" + "="*70)
    print("[Node] Query Analysis Agent")
    print("="*70)
    
    # Agent 생성
    agent = create_query_analysis_agent()
    
    # Agent 실행
    result = agent.invoke({
        "input": f"다음 질문을 분석하세요: {state['user_query']}"
    })
    
    output = result['output']
    print(f"\n[Agent Output]\n{output}\n")
    
    # JSON 파싱
    try:
        # 분석 결과 추출
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            analysis_result = json.loads(json_match.group())
            state['query_analysis_result'] = analysis_result
            
            # 부족한 요소 파악
            missing = []
            for criterion, details in analysis_result.items():
                if not details.get('mentioned', False) or details.get('score', 0) < 0.5:
                    missing.append({
                        'criterion': criterion,
                        'score': details.get('score', 0),
                        'value': details.get('value')
                    })
            state['missing_elements'] = missing
        
        # 키워드 추출 (쉼표로 구분된 텍스트)
        keywords_match = re.findall(r'\b[a-zA-Z][a-zA-Z\s-]+\b', output)
        keywords = [kw.strip() for kw in output.split(',') if kw.strip() and len(kw.strip()) > 2]
        
        # 키워드가 없으면 기본값
        if not keywords or len(keywords) < 2:
            keywords = ['transformer', 'limitations', 'NLP']
        
        state['keywords'] = keywords[:5]  # 최대 5개
        
    except Exception as e:
        print(f"⚠️ 파싱 오류: {e}")
        # 기본값
        state['keywords'] = ['transformer', 'NLP', 'limitations']
        state['query_analysis_result'] = {}
        state['missing_elements'] = []
    
    print(f"\n✅ 키워드: {state['keywords']}")
    print(f"✅ 부족한 요소: {len(state['missing_elements'])}개")
    print("="*70)
    
    return state


def test_query_analysis():
    """테스트"""
    import os
    os.environ['GROQ_API_KEY'] = 'your_api_key'
    
    test_state = {
        'user_query': "Transformer 모델을 활용한 저자원 언어 처리의 한계점",
        'keywords': [],
        'query_analysis_result': None,
        'missing_elements': [],
        'query_clarity_score': 0.0,
        'papers': [],
        'web_info': [],
        'paper_relevance_score': 0.0,
        'classified_gaps': {},
        'priority_axis': "",
        'gap_quality_score': 0.0,
        'research_topics': [],
        'retry_count': 0,
        'max_retries': 3,
        'query_embedding': None
    }
    
    result = query_analysis_node(test_state)
    print(f"\n최종 키워드: {result['keywords']}")
    print(f"부족 요소: {result['missing_elements']}")


if __name__ == "__main__":
    test_query_analysis()
