"""
Query Analysis Agent
질문 분석 Agent
"""

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .tools import (
    llm,
    analyze_query_criteria,
    extract_search_keywords,
    generate_clarification_questions
)


TOOLS = [
    analyze_query_criteria,
    extract_search_keywords,
    generate_clarification_questions
]


AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 연구 질문을 분석하는 전문가입니다.

주어진 도구를 사용하여:
1. analyze_query_criteria로 질문을 5가지 기준으로 평가
2. extract_search_keywords로 검색 키워드 추출
3. 부족한 요소가 있으면 generate_clarification_questions로 질문 생성

단계적으로 진행하세요."""),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])


def create_query_analysis_agent():
    """Query Analysis Agent 생성"""
    agent = create_react_agent(llm, TOOLS, AGENT_PROMPT)
    
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        max_iterations=15,
        handle_parsing_errors=True
    )
