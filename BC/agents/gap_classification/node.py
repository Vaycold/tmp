"""
GAP Classification Agent - Node
4축 기반 한계점 분류 + 우선순위 결정
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


# 4가지 축 정의
AXES = {
    "데이터_의존성": "특정 데이터에 과도하게 의존",
    "실제_환경_검증": "실제 환경에서의 검증 부족",
    "확장성": "대규모 적용의 어려움",
    "구조적_한계": "모델/방법론의 본질적 한계"
}


def gap_classification_node(state: GAPAGOState) -> GAPAGOState:
    """GAP Classification 노드
    
    작업:
    1. 논문의 limitation, discussion, future_work 수집
    2. 웹 정보의 limitation_mentions 수집
    3. 각 한계점을 4가지 축으로 분류
    4. 우선순위 축 결정
    
    입력:
        state['papers']
        state['web_info']
        
    출력:
        state['classified_gaps']
        state['priority_axis']
    """
    print("\n" + "="*70)
    print("[Node] GAP Classification Agent")
    print("="*70)
    
    # 1. 모든 한계점 수집
    all_limitations = []
    
    # 논문에서
    for paper in state['papers']:
        if paper.get('limitation'):
            all_limitations.append(paper['limitation'])
        if paper.get('discussion'):
            all_limitations.append(paper['discussion'])
        if paper.get('future_work'):
            all_limitations.append(paper['future_work'])
    
    # 웹에서
    for info in state['web_info']:
        if info.get('limitation_mentions'):
            all_limitations.append(info['limitation_mentions'])
    
    print(f"\n1. 총 {len(all_limitations)}개 한계점 수집")
    
    # 2. 각 한계점을 4축으로 분류
    print(f"2. 4축 분류 중...")
    
    classified = {axis: [] for axis in AXES.keys()}
    
    for limitation in all_limitations:
        if not limitation or len(limitation) < 10:
            continue
        
        prompt = ChatPromptTemplate.from_template(
            """다음 한계점을 카테고리로 분류하세요:

한계점: {limitation}

카테고리:
- 데이터_의존성: 특정 데이터에 과도하게 의존
- 실제_환경_검증: 실제 환경에서의 검증 부족
- 확장성: 대규모 적용의 어려움
- 구조적_한계: 모델/방법론의 본질적 한계

카테고리 이름만 출력 (하나만):"""
        )
        
        try:
            result = (prompt | llm).invoke({"limitation": limitation[:200]})
            category = result.content.strip()
            
            # 정확한 매칭
            if category in AXES:
                classified[category].append(limitation)
            else:
                # 부분 매칭
                for axis in AXES.keys():
                    if axis in category:
                        classified[axis].append(limitation)
                        break
                else:
                    # 기본값
                    classified["구조적_한계"].append(limitation)
        
        except Exception as e:
            print(f"   ⚠️ 분류 오류: {e}")
            classified["구조적_한계"].append(limitation)
    
    # 3. 우선순위 축 결정
    axis_counts = {axis: len(items) for axis, items in classified.items()}
    
    if sum(axis_counts.values()) == 0:
        priority_axis = "구조적_한계"
    else:
        priority_axis = max(axis_counts.items(), key=lambda x: x[1])[0]
    
    state['classified_gaps'] = classified
    state['priority_axis'] = priority_axis
    
    print(f"\n축별 분포:")
    for axis, items in classified.items():
        marker = "⭐" if axis == priority_axis else "  "
        print(f"  {marker} {axis}: {len(items)}개")
    
    print(f"\n✅ 우선순위 축: {priority_axis}")
    print("="*70)
    
    return state
