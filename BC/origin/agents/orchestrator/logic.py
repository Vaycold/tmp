"""
Orchestrator
Critic 점수를 기반으로 워크플로우 제어
"""

import sys
sys.path.append('/home/claude/gapago_project')

from state.state import GAPAGOState
from typing import Literal


def orchestrator_after_query(state: GAPAGOState) -> Literal["human_loop", "paper_search"]:
    """Query Analysis 후 분기
    
    Critic의 query_clarity_score를 기반으로 판단
    - 점수 < 0.3: 명확함 → paper_search
    - 점수 >= 0.3: 모호함 → human_loop
    """
    print("\n" + "="*70)
    print("[Orchestrator] Query 평가 후 분기")
    print("="*70)
    
    threshold = 0.3
    score = state['query_clarity_score']
    
    # 최대 재시도 체크
    if state['retry_count'] >= state['max_retries']:
        print(f"  ⚠️ 최대 재시도 도달 ({state['retry_count']}/{state['max_retries']})")
        print(f"  → 강제로 paper_search로 진행")
        return "paper_search"
    
    # 점수 기반 분기
    if score < threshold:
        print(f"  ✅ 명확함 (점수: {score:.2f} < {threshold})")
        print(f"  → paper_search로 진행")
        return "paper_search"
    else:
        print(f"  ⚠️ 모호함 (점수: {score:.2f} >= {threshold})")
        print(f"  → human_loop로 이동 (재시도 {state['retry_count'] + 1}/{state['max_retries']})")
        state['retry_count'] += 1
        return "human_loop"


def orchestrator_after_paper(state: GAPAGOState) -> Literal["gap_classification", "paper_search"]:
    """Paper Search 후 분기
    
    Critic의 paper_relevance_score를 기반으로 판단
    - 점수 >= 0.5: 충분함 → gap_classification
    - 점수 < 0.5: 부족함 → paper_search (재검색)
    """
    print("\n" + "="*70)
    print("[Orchestrator] Paper Search 평가 후 분기")
    print("="*70)
    
    threshold = 0.5
    score = state['paper_relevance_score']
    
    # 최대 재시도 체크
    if state['retry_count'] >= state['max_retries']:
        print(f"  ⚠️ 최대 재시도 도달 ({state['retry_count']}/{state['max_retries']})")
        print(f"  → 강제로 gap_classification으로 진행")
        return "gap_classification"
    
    # 점수 기반 분기
    if score >= threshold:
        print(f"  ✅ 충분함 (점수: {score:.2f} >= {threshold})")
        print(f"  → gap_classification으로 진행")
        return "gap_classification"
    else:
        print(f"  ⚠️ 부족함 (점수: {score:.2f} < {threshold})")
        print(f"  → paper_search로 재검색 (재시도 {state['retry_count'] + 1}/{state['max_retries']})")
        state['retry_count'] += 1
        return "paper_search"


def orchestrator_after_gap(state: GAPAGOState) -> Literal["topic_generation", "paper_search"]:
    """GAP Classification 후 분기
    
    Critic의 gap_quality_score를 기반으로 판단
    - 점수 >= 0.4: 높음 → topic_generation
    - 점수 < 0.4: 낮음 → paper_search (재검색)
    """
    print("\n" + "="*70)
    print("[Orchestrator] GAP Classification 평가 후 분기")
    print("="*70)
    
    threshold = 0.4
    score = state['gap_quality_score']
    
    # 최대 재시도 체크
    if state['retry_count'] >= state['max_retries']:
        print(f"  ⚠️ 최대 재시도 도달 ({state['retry_count']}/{state['max_retries']})")
        print(f"  → 강제로 topic_generation으로 진행")
        return "topic_generation"
    
    # 점수 기반 분기
    if score >= threshold:
        print(f"  ✅ 높음 (점수: {score:.2f} >= {threshold})")
        print(f"  → topic_generation으로 진행")
        return "topic_generation"
    else:
        print(f"  ⚠️ 낮음 (점수: {score:.2f} < {threshold})")
        print(f"  → paper_search로 재검색 (재시도 {state['retry_count'] + 1}/{state['max_retries']})")
        state['retry_count'] += 1
        return "paper_search"
