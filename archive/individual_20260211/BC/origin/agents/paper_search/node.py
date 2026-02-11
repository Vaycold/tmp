"""
Paper Search Agent - Node
arXiv 논문 검색 + 임베딩 필터링
"""

import sys
sys.path.append('/home/claude/gapago_project')

from state.state import GAPAGOState
import arxiv
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 임베딩 모델
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def paper_search_node(state: GAPAGOState) -> GAPAGOState:
    """Paper Search 노드
    
    작업:
    1. keywords를 임베딩
    2. arXiv에서 논문 검색
    3. 각 논문의 keywords를 임베딩
    4. 코사인 유사도로 필터링
    5. limitation, discussion, future_work 추출 (TODO)
    
    입력:
        state['keywords']
        
    출력:
        state['papers']
        state['query_embedding']
    """
    print("\n" + "="*70)
    print("[Node] Paper Search Agent")
    print("="*70)
    
    keywords = state['keywords']
    query_text = ' '.join(keywords)
    
    # 1. Query 임베딩
    print(f"\n1. Query 임베딩 중...")
    query_embedding = embedding_model.encode([query_text])[0]
    state['query_embedding'] = query_embedding.tolist()
    
    # 2. arXiv 검색
    print(f"2. arXiv 검색: {query_text}")
    
    try:
        search = arxiv.Search(
            query=query_text,
            max_results=10,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        papers = []
        for result in search.results():
            paper = {
                'title': result.title,
                'abstract': result.summary[:500],
                'keywords': result.categories,  # arXiv categories
                'limitation': "",  # TODO: 추출
                'discussion': "",  # TODO: 추출
                'future_work': "",  # TODO: 추출
                'embedding': None
            }
            papers.append(paper)
            time.sleep(0.5)  # Rate limit
        
        print(f"   ✅ {len(papers)}개 논문 검색 완료")
        
    except Exception as e:
        print(f"   ⚠️ 검색 실패: {e}")
        # 더미 데이터
        papers = [
            {
                'title': f'Survey on {keywords[0]}',
                'abstract': f'This paper discusses {keywords[0]}.',
                'keywords': keywords,
                'limitation': 'Data dependency, scalability issues',
                'discussion': 'Further research needed',
                'future_work': 'Improve model efficiency',
                'embedding': None
            }
        ]
    
    # 3. 각 논문 keywords 임베딩 및 유사도 계산
    print(f"3. 임베딩 유사도 계산 중...")
    
    for paper in papers:
        keyword_text = ' '.join(paper['keywords'])
        paper_embedding = embedding_model.encode([keyword_text])[0]
        paper['embedding'] = paper_embedding.tolist()
        
        # 코사인 유사도
        similarity = cosine_similarity(
            [query_embedding],
            [paper_embedding]
        )[0][0]
        paper['similarity'] = float(similarity)
    
    # 4. 유사도 기준 정렬 및 필터링
    papers.sort(key=lambda x: x.get('similarity', 0), reverse=True)
    top_papers = papers[:5]  # 상위 5개
    
    print(f"   ✅ 상위 {len(top_papers)}개 논문 선정")
    for i, paper in enumerate(top_papers, 1):
        print(f"      {i}. {paper['title'][:50]}... (유사도: {paper.get('similarity', 0):.2f})")
    
    state['papers'] = top_papers
    
    print("="*70)
    return state
