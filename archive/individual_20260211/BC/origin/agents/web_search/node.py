"""
Web Search Agent - Node
웹에서 한계점 정보 수집
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


def web_search_node(state: GAPAGOState) -> GAPAGOState:
    """Web Search 노드
    
    작업:
    1. keywords로 웹 검색 (LLM 기반 시뮬레이션)
    2. 기사, 블로그에서 한계점 정보 추출
    
    입력:
        state['keywords']
        
    출력:
        state['web_info']
    """
    print("\n" + "="*70)
    print("[Node] Web Search Agent (병렬)")
    print("="*70)
    
    keywords = state['keywords']
    query = ', '.join(keywords)
    
    # TODO: 실제 웹 크롤링 구현
    # 현재는 LLM으로 일반적인 한계점 생성
    
    print(f"\n웹 검색 쿼리: {query}")
    print("(LLM 기반 시뮬레이션)")
    
    prompt = ChatPromptTemplate.from_template(
        """다음 주제에 대한 일반적으로 알려진 한계점 3가지를 나열하세요:

주제: {query}

각 한계점은 다음 형식으로:
1. [한계점 1]
2. [한계점 2]
3. [한계점 3]"""
    )
    
    result = (prompt | llm).invoke({"query": query})
    limitations_text = result.content
    
    # 파싱
    web_info = []
    for line in limitations_text.split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            lim = line.lstrip('0123456789.-) ')
            if lim:
                web_info.append({
                    'source': 'Web Search (LLM)',
                    'url': '',
                    'content': lim,
                    'limitation_mentions': lim
                })
    
    state['web_info'] = web_info
    
    print(f"\n✅ {len(web_info)}개 웹 정보 수집")
    for i, info in enumerate(web_info, 1):
        print(f"   {i}. {info['content'][:60]}...")
    
    print("="*70)
    return state
