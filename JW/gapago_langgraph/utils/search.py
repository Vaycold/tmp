"""
Paper search client.
- OpenAlex (global, primary)
- ScienceON (Korean, TODO)
"""

import time
import requests
from models import Paper


def search_papers(query: str, max_results: int = 50, max_retries: int = 3) -> list[Paper]:
    """
    Search papers from multiple sources.
    
    Args:
        query: Search query
        max_results: Maximum results to retrieve
        max_retries: Maximum retry attempts
        
    Returns:
        List of Paper objects
    """
    papers = []
    
    # 1) OpenAlex (primary)
    openalex_papers = _search_openalex(query, max_results, max_retries)
    papers.extend(openalex_papers)
    
    # 2) ScienceON (TODO: API 키 발급 후 구현)
    # scienceon_papers = _search_scienceon(query, max_results, max_retries)
    # papers.extend(scienceon_papers)
    
    # 중복 제거 (title 기준)
    seen_titles = set()
    unique_papers = []
    for p in papers:
        title_key = p.title.lower().strip()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_papers.append(p)
    
    return unique_papers


def _search_openalex(query: str, max_results: int = 50, max_retries: int = 3) -> list[Paper]:
    """
    Search using OpenAlex API.
    
    Docs: https://docs.openalex.org/api-entities/works/search-works
    - Free, no API key required
    - Rate limit: 10 requests/second
    - Covers 250M+ works including arXiv
    """
    base_url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per_page": min(max_results, 100),
        "sort": "relevance_score:desc",
        "filter": "has_abstract:true",
        "select": "id,title,authorships,publication_year,primary_location,abstract_inverted_index"
    }
    headers = {
        "User-Agent": "GAPago/1.0 (Research Gap Analysis Tool)"
    }
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = 3 * (2 ** attempt)
                print(f"  ⏳ OpenAlex retry {attempt}/{max_retries} - waiting {wait_time}s...")
                time.sleep(wait_time)
            
            response = requests.get(base_url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 429:
                print("  ⚠️ OpenAlex rate limit (429). Retrying...")
                continue
            
            response.raise_for_status()
            data = response.json()
            
            papers = []
            for work in data.get("results", []):
                try:
                    # Abstract 복원 (OpenAlex는 inverted index 형태로 저장)
                    abstract = _rebuild_abstract(work.get("abstract_inverted_index"))
                    if not abstract:
                        continue
                    
                    # Paper ID 추출
                    openalex_id = work.get("id", "")
                    paper_id = openalex_id.replace("https://openalex.org/", "")
                    
                    # URL 결정 (arXiv 링크 우선)
                    url = openalex_id
                    primary_loc = work.get("primary_location") or {}
                    landing_url = primary_loc.get("landing_page_url")
                    if landing_url:
                        url = landing_url
                    
                    # 저자 추출
                    authors = []
                    for authorship in (work.get("authorships") or []):
                        author = authorship.get("author") or {}
                        name = author.get("display_name")
                        if name:
                            authors.append(name)
                    
                    # 제목
                    title = work.get("title") or "Untitled"
                    
                    # 연도
                    year = work.get("publication_year") or 2024
                    
                    papers.append(Paper(
                        paper_id=paper_id,
                        title=title,
                        abstract=abstract,
                        url=url,
                        year=year,
                        authors=authors
                    ))
                except (AttributeError, ValueError, KeyError):
                    continue
            
            print(f"  ✓ OpenAlex: {len(papers)} papers with abstracts")
            return papers
        
        except requests.RequestException as e:
            print(f"  ⚠️ OpenAlex error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                print(f"  ❌ OpenAlex: all {max_retries} attempts failed.")
                return []
            continue
    
    return []


def _rebuild_abstract(inverted_index: dict) -> str:
    """
    Rebuild abstract from OpenAlex inverted index format.
    
    OpenAlex stores abstracts as:
    {"word1": [0, 5], "word2": [1, 3], ...}
    meaning word1 appears at positions 0 and 5, etc.
    
    Args:
        inverted_index: OpenAlex abstract_inverted_index
        
    Returns:
        Reconstructed abstract string
    """
    if not inverted_index:
        return ""
    
    try:
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        
        word_positions.sort(key=lambda x: x[0])
        return " ".join(word for _, word in word_positions)
    except (TypeError, AttributeError):
        return ""


# ──────────────────────────────────────────────
# ScienceON (TODO: API 키 발급 후 구현)
# ──────────────────────────────────────────────
# def _search_scienceon(query: str, max_results: int = 50, max_retries: int = 3) -> list[Paper]:
#     """
#     Search using ScienceON API (Korean research papers).
#     
#     Docs: https://open.scienceon.kisti.re.kr
#     Requires: SCIENCEON_API_KEY in .env
#     
#     Args:
#         query: Search query
#         max_results: Maximum results
#         max_retries: Retry attempts
#         
#     Returns:
#         List of Paper objects
#     """
#     from config import config
#     
#     if not config.SCIENCEON_API_KEY:
#         print("  ⚠️ ScienceON: API key not configured, skipping.")
#         return []
#     
#     base_url = "https://open.scienceon.kisti.re.kr/api/search"
#     params = {
#         "query": query,
#         "target": "article",
#         "apiKey": config.SCIENCEON_API_KEY,
#         "count": min(max_results, 100)
#     }
#     
#     # TODO: 구현
#     return []