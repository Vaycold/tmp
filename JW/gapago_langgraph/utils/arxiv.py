"""
arXiv API client.
"""

import time
import requests
from xml.etree import ElementTree as ET
from models import Paper


def search_arxiv(query: str, max_results: int = 50) -> list[Paper]:
    """
    Search arXiv API and return papers.
    
    Args:
        query: Search query
        max_results: Maximum results to retrieve
        
    Returns:
        List of Paper objects
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        
        papers = []
        for entry in root.findall("atom:entry", ns):
            try:
                paper_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
                title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
                
                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns)
                    if name is not None:
                        authors.append(name.text)
                
                published = entry.find("atom:published", ns).text
                year = int(published[:4])
                
                url = f"https://arxiv.org/abs/{paper_id}"
                
                papers.append(Paper(
                    paper_id=paper_id,
                    title=title,
                    abstract=abstract,
                    url=url,
                    year=year,
                    authors=authors
                ))
            except (AttributeError, ValueError):
                continue
        
        time.sleep(3)  # Rate limiting
        return papers
        
    except requests.RequestException as e:
        print(f"⚠️ arXiv API error: {e}")
        return []