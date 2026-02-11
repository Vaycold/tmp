"""
Critic Agent - Nodes
각 단계의 출력을 평가하는 노드들
"""

import sys
sys.path.append('/home/claude/gapago_project')

from state.state import GAPAGOState
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .tools import (
    llm,
    evaluate_query_clarity,
    evaluate_paper_relevance,
    evaluate_gap_quality
)
import json
import re


# ═══════════════════════════════════════════════════════════
# Critic Agent 생성
# ═══════════════════════════════════════════════════════════

TOOLS = [
    evaluate_query_clarity,
    evaluate_paper_relevance,
    evaluate_gap_quality
]

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 Agent 출력을 평가하는 Critic입니다.

주어진 도구를 사용하여 각 단계의 품질을 점수로 평가하세요."""),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])


def create_critic_agent():
    """Critic Agent 생성"""
    agent = create_react_agent(llm, TOOLS, CRITIC_PROMPT)
    
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True
    )


# ═══════════════════════════════════════════════════════════
# Nodes
# ═══════════════════════════════════════════════════════════

def critic_query_node(state: GAPAGOState) -> GAPAGOState:
    """Critic Node - Query Analysis 평가
    
    입력:
        state['query_analysis_result']
        state['missing_elements']
        
    출력:
        state['query_clarity_score']
    """
    print("\n" + "="*70)
    print("[Critic] Query Analysis 평가")
    print("="*70)
    
    agent = create_critic_agent()
    
    result = agent.invoke({
        "input": f"""Query Analysis 결과를 평가하세요:

분석 결과: {json.dumps(state['query_analysis_result'], ensure_ascii=False)}
부족 요소: {json.dumps(state['missing_elements'], ensure_ascii=False)}

evaluate_query_clarity 도구를 사용하세요."""
    })
    
    output = result['output']
    print(f"\n[Critic Output]\n{output}\n")
    
    # 점수 추출
    try:
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            eval_result = json.loads(json_match.group())
            state['query_clarity_score'] = eval_result.get('clarity_score', 0.5)
        else:
            # 숫자 찾기
            score_match = re.search(r'(\d+\.?\d*)', output)
            if score_match:
                state['query_clarity_score'] = float(score_match.group(1))
            else:
                state['query_clarity_score'] = 0.5
    except:
        state['query_clarity_score'] = 0.5
    
    print(f"✅ Query 명확성 점수: {state['query_clarity_score']:.2f}")
    print("="*70)
    
    return state


def critic_paper_node(state: GAPAGOState) -> GAPAGOState:
    """Critic Node - Paper Search 평가
    
    입력:
        state['papers']
        state['keywords']
        
    출력:
        state['paper_relevance_score']
    """
    print("\n" + "="*70)
    print("[Critic] Paper Search 평가")
    print("="*70)
    
    agent = create_critic_agent()
    
    result = agent.invoke({
        "input": f"""논문 검색 결과를 평가하세요:

키워드: {state['keywords']}
논문 수: {len(state['papers'])}

evaluate_paper_relevance 도구를 사용하세요."""
    })
    
    output = result['output']
    print(f"\n[Critic Output]\n{output}\n")
    
    # 점수 추출
    try:
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            eval_result = json.loads(json_match.group())
            state['paper_relevance_score'] = eval_result.get('relevance_score', 0.5)
        else:
            score_match = re.search(r'(\d+\.?\d*)', output)
            if score_match:
                state['paper_relevance_score'] = float(score_match.group(1))
            else:
                state['paper_relevance_score'] = 0.5
    except:
        state['paper_relevance_score'] = 0.5
    
    print(f"✅ 논문 정합성 점수: {state['paper_relevance_score']:.2f}")
    print("="*70)
    
    return state


def critic_gap_node(state: GAPAGOState) -> GAPAGOState:
    """Critic Node - GAP Classification 평가
    
    입력:
        state['classified_gaps']
        state['priority_axis']
        
    출력:
        state['gap_quality_score']
    """
    print("\n" + "="*70)
    print("[Critic] GAP Classification 평가")
    print("="*70)
    
    agent = create_critic_agent()
    
    result = agent.invoke({
        "input": f"""GAP 분류 결과를 평가하세요:

분류 결과: {json.dumps(state['classified_gaps'], ensure_ascii=False)}
우선순위 축: {state['priority_axis']}

evaluate_gap_quality 도구를 사용하세요."""
    })
    
    output = result['output']
    print(f"\n[Critic Output]\n{output}\n")
    
    # 점수 추출
    try:
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            eval_result = json.loads(json_match.group())
            state['gap_quality_score'] = eval_result.get('quality_score', 0.5)
        else:
            score_match = re.search(r'(\d+\.?\d*)', output)
            if score_match:
                state['gap_quality_score'] = float(score_match.group(1))
            else:
                state['gap_quality_score'] = 0.5
    except:
        state['gap_quality_score'] = 0.5
    
    print(f"✅ GAP 품질 점수: {state['gap_quality_score']:.2f}")
    print("="*70)
    
    return state
