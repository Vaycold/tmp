"""
GAPAGO Main
전체 시스템 실행
"""

import sys
sys.path.append('/home/claude/gapago_project')

import os
from state.state import GAPAGOState
from graph.workflow import create_gapago_workflow


def run_gapago(query: str, api_key: str = None):
    """GAPAGO 실행
    
    Args:
        query: 사용자의 연구 질문
        api_key: Groq API 키 (선택)
    """
    
    # API 키 설정
    if api_key:
        os.environ['GROQ_API_KEY'] = api_key
    
    if 'GROQ_API_KEY' not in os.environ:
        raise ValueError("GROQ_API_KEY가 설정되지 않았습니다!")
    
    # 초기 상태
    initial_state: GAPAGOState = {
        'user_query': query,
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
        'max_retries': 2,  # 최대 2회 재시도
        'query_embedding': None
    }
    
    # 워크플로우 생성
    print("="*70)
    print("🚀 GAPAGO 시작")
    print("="*70)
    print(f"\n📝 질문: {query}\n")
    
    app = create_gapago_workflow()
    
    # 실행
    try:
        result = app.invoke(initial_state)
        
        print("\n" + "="*70)
        print("✅ GAPAGO 완료")
        print("="*70)
        
        # 결과 출력
        print_results(result)
        
        return result
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_results(result: GAPAGOState):
    """결과 출력"""
    
    print(f"\n📊 최종 결과")
    print("-"*70)
    
    print(f"\n🔑 검색 키워드:")
    for i, kw in enumerate(result['keywords'], 1):
        print(f"  {i}. {kw}")
    
    print(f"\n📚 수집 논문: {len(result['papers'])}개")
    
    print(f"\n🌐 웹 정보: {len(result['web_info'])}개")
    
    print(f"\n🎯 GAP 분류:")
    for axis, gaps in result['classified_gaps'].items():
        marker = "⭐" if axis == result['priority_axis'] else "  "
        print(f"  {marker} {axis}: {len(gaps)}개")
    
    print(f"\n⭐ 우선순위 축: {result['priority_axis']}")
    
    print(f"\n💡 추천 연구 주제:")
    for i, topic in enumerate(result['research_topics'], 1):
        print(f"  {i}. {topic}")
    
    print("\n" + "="*70)


def main():
    """메인 함수"""
    
    # API 키 입력
    api_key = input("Groq API Key (Enter로 환경변수 사용): ").strip()
    if api_key:
        os.environ['GROQ_API_KEY'] = api_key
    
    # 질문 입력
    query = input("\n연구 질문을 입력하세요: ").strip()
    
    if not query:
        query = "Transformer 모델을 활용한 저자원 언어 처리의 한계점과 개선 방향"
        print(f"기본 질문 사용: {query}")
    
    # 실행
    run_gapago(query)


if __name__ == "__main__":
    main()
