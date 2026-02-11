"""
Topic Generation Agent - Node
우선순위 축 기반 연구 주제 생성
"""

import sys
sys.path.append('/home/claude/gapago_project')

from state.state import GAPAGOState
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    groq_api_key=os.environ.get('GROQ_API_KEY')
)


def topic_generation_node(state: GAPAGOState) -> GAPAGOState:
    """Topic Generation 노드
    
    작업:
    1. 우선순위 축의 GAP들 종합
    2. 새로운 연구 주제 3개 생성
    
    입력:
        state['classified_gaps']
        state['priority_axis']
        state['user_query']
        
    출력:
        state['research_topics']
    """
    print("\n" + "="*70)
    print("[Node] Topic Generation Agent")
    print("="*70)
    
    priority_axis = state['priority_axis']
    priority_gaps = state['classified_gaps'].get(priority_axis, [])
    
    if not priority_gaps:
        priority_gaps = ["일반적인 한계점"]
    
    # GAP 요약
    gaps_summary = '\n'.join([f"- {gap[:100]}" for gap in priority_gaps[:5]])
    
    print(f"\n우선순위 축: {priority_axis}")
    print(f"GAP 개수: {len(priority_gaps)}개")
    
    # 연구 주제 생성
    prompt = ChatPromptTemplate.from_template(
        """다음 한계점들을 해결할 연구 주제 3개를 제안하세요:

원래 질문: {user_query}

우선순위 축: {priority_axis}

한계점들:
{gaps_summary}

각 연구 주제는:
1. 구체적이고 실행 가능해야 함
2. 위 한계점을 직접 해결해야 함
3. 학술 연구로 타당해야 함

연구 주제 3개 (번호를 붙여서):"""
    )
    
    result = (prompt | llm).invoke({
        "user_query": state['user_query'],
        "priority_axis": priority_axis,
        "gaps_summary": gaps_summary
    })
    
    # 파싱
    topics = []
    for line in result.content.split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            topic = line.lstrip('0123456789.-) ')
            if topic:
                topics.append(topic)
    
    state['research_topics'] = topics[:3]
    
    print(f"\n✅ 연구 주제 생성:")
    for i, topic in enumerate(state['research_topics'], 1):
        print(f"   {i}. {topic}")
    
    print("="*70)
    return state
