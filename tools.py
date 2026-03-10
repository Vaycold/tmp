from __future__ import annotations

from typing import Optional, List
import json
import re
import time
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
import json
import re
import time
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from rank_bm25 import BM25Okapi

from config import Configuration
from utils.tavily import TavilySearch


_ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _tokenize(text: str) -> list[str]:
    return _norm(text).lower().split()


def _safe_json_loads(text: str) -> dict | list | None:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text or "")
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def _arxiv_url(search_query: str, start: int, max_results: int,
               sort_by: str = "relevance", sort_order: str = "descending") -> str:
    params = {
        "search_query": search_query,
        "start": int(start),
        "max_results": int(max_results),
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }
    return f"{_ARXIV_API}?{urlencode(params)}"


def _parse_atom(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    entries = root.findall("atom:entry", _ATOM_NS)

    out: list[dict] = []
    for e in entries:
        arxiv_id_url = (e.findtext("atom:id", default="", namespaces=_ATOM_NS) or "").strip()
        arxiv_id = arxiv_id_url.replace("http://arxiv.org/abs/", "").replace("https://arxiv.org/abs/", "").strip()
        if not arxiv_id:
            continue

        title = _norm(e.findtext("atom:title", default="", namespaces=_ATOM_NS))
        abstract = _norm(e.findtext("atom:summary", default="", namespaces=_ATOM_NS))
        if not title or not abstract:
            continue

        authors = []
        for a in e.findall("atom:author", _ATOM_NS):
            name = (a.findtext("atom:name", default="", namespaces=_ATOM_NS) or "").strip()
            if name:
                authors.append(name)

        published = (e.findtext("atom:published", default="", namespaces=_ATOM_NS) or "").strip()
        year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else 0

        url = arxiv_id_url
        for link in e.findall("atom:link", _ATOM_NS):
            if link.attrib.get("rel") == "alternate" and link.attrib.get("href"):
                url = link.attrib["href"]
                break
            
        # ✅ full text 섹션 수집
        full_text_sections = fetch_full_text_sections(url)

        out.append({
            "paper_id": f"arxiv:{arxiv_id}",
            "title": title,
            "abstract": abstract,
            "url": url,
            "year": year,
            "authors": authors,
            "score_bm25": 0.0,
            "source": "arxiv",
            "full_text_sections": full_text_sections,
        })

    return out


def arxiv_api_call(search_query: str, max_total: int, page_size: int, max_pages: int) -> list[dict]:
    raw: list[dict] = []
    for page in range(max_pages):
        start = page * page_size
        if start >= max_total:
            break

        url = _arxiv_url(
            search_query=search_query,
            start=start,
            max_results=min(page_size, max_total - start),
        )

        last_error = None
        for _ in range(2):
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                batch = _parse_atom(r.text)
                raw.extend(batch)
                if not batch:
                    return list({p["paper_id"]: p for p in raw}.values())
                time.sleep(0.3)
                last_error = None
                break
            except Exception as e:
                last_error = e
                time.sleep(0.5)
        if last_error:
            raise last_error

    uniq = {p["paper_id"]: p for p in raw}
    return list(uniq.values())


def bm25_rank(papers: list[dict], query_text: str, top_k: int) -> dict:
    if not papers:
        return {"selected": [], "avg_bm25": 0.0}

    corpus = [_tokenize(f"{p.get('title', '')} {p.get('abstract', '')}") for p in papers]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query_text))
    pairs = list(zip(papers, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:top_k]

    selected = []
    for paper, score in top:
        item = dict(paper)
        item["score_bm25"] = float(score)
        selected.append(item)

    avg = sum(p["score_bm25"] for p in selected) / len(selected) if selected else 0.0
    return {"selected": selected, "avg_bm25": avg}


class ArxivApiCallInput(BaseModel):
    search_query: str = Field(description="arXiv API search_query")
    max_total: int = Field(default=80, description="총 최대 결과 수")
    page_size: int = Field(default=40, description="페이지당 결과 수")
    max_pages: int = Field(default=3, description="최대 페이지 수")


class WebSearchInput(BaseModel):
    query: str = Field(description="웹 검색 쿼리")


class ScienceOnSearchInput(BaseModel):
    query: str = Field(description="ScienceON 검색 쿼리")
    max_results: int = Field(default=5, description="가져올 결과 수")


def build_retrieval_tools(config: Optional[RunnableConfig] = None) -> List:
    """
    Retrieval Agent가 선택할 수 있는 외부 검색 툴만 노출한다.
    - web_search_tool
    - arxiv_api_call_tool
    - scienceon_search_tool (placeholder)
    Retrieval Agent가 선택할 수 있는 외부 검색 툴만 노출한다.
    - web_search_tool
    - arxiv_api_call_tool
    - scienceon_search_tool (placeholder)
    """
    cfg = Configuration.from_runnable_config()
    tavily_tool = TavilySearch(max_results=cfg.tavily_max_results)

    @tool(args_schema=ArxivApiCallInput)
    def arxiv_api_call_tool(
        search_query: str,
        max_total: int = 80,
        page_size: int = 40,
        max_pages: int = 3,
    ) -> str:
        """Call arXiv API directly and return a paper list as JSON string."""
    @tool(args_schema=ArxivApiCallInput)
    def arxiv_api_call_tool(
        search_query: str,
        max_total: int = 80,
        page_size: int = 40,
        max_pages: int = 3,
    ) -> str:
        """Call arXiv API directly and return a paper list as JSON string."""
        try:
            results = arxiv_api_call(
                search_query=search_query,
                max_total=max_total,
                page_size=page_size,
                max_pages=max_pages,
            results = arxiv_api_call(
                search_query=search_query,
                max_total=max_total,
                page_size=page_size,
                max_pages=max_pages,
            )
            return json.dumps({
                "source": "arxiv",
                "query": search_query,
                "results": results,
            }, ensure_ascii=False)
            return json.dumps({
                "source": "arxiv",
                "query": search_query,
                "results": results,
            }, ensure_ascii=False)
        except Exception as e:
            return f"<Error>Arxiv API call failed: {str(e)}</Error>"

    @tool(args_schema=WebSearchInput)
    def web_search_tool(query: str) -> str:
        """Search the web API and return results as JSON string."""
        try:
            results = tavily_tool.search(query)
            return json.dumps({
                "source": "web",
                "query": query,
                "results": results,
            }, ensure_ascii=False)
        except Exception as e:
            return f"<Error>Web search failed: {str(e)}</Error>"

    @tool(args_schema=ScienceOnSearchInput)
    def scienceon_search_tool(query: str, max_results: int = 5) -> str:
        """ScienceON search tool placeholder until API development is complete."""
        payload = {
            "status": "not_implemented",
            "source": "scienceon",
            "query": query,
            "max_results": max_results,
            "message": "ScienceON API is under development.",
            "results": [],
        }
        return json.dumps(payload, ensure_ascii=False)

    return [
        web_search_tool,
        arxiv_api_call_tool,
        scienceon_search_tool,
    ]
            return f"<Error>Arxiv API call failed: {str(e)}</Error>"

    @tool(args_schema=WebSearchInput)
    def web_search_tool(query: str) -> str:
        """Search the web API and return results as JSON string."""
        try:
            results = tavily_tool.search(query)
            return json.dumps({
                "source": "web",
                "query": query,
                "results": results,
            }, ensure_ascii=False)
        except Exception as e:
            return f"<Error>Web search failed: {str(e)}</Error>"

    @tool(args_schema=ScienceOnSearchInput)
    def scienceon_search_tool(query: str, max_results: int = 5) -> str:
        """ScienceON search tool placeholder until API development is complete."""
        payload = {
            "status": "not_implemented",
            "source": "scienceon",
            "query": query,
            "max_results": max_results,
            "message": "ScienceON API is under development.",
            "results": [],
        }
        return json.dumps(payload, ensure_ascii=False)

    return [
        web_search_tool,
        arxiv_api_call_tool,
        scienceon_search_tool,
    ]


def build_role_tools(config: Optional[RunnableConfig] = None) -> dict:
    retrieval_tools = build_retrieval_tools(config)
    return {
        "QUERY_TOOLS": [],
        "RETRIEVAL_TOOLS": retrieval_tools,
        "LIMITATION_TOOLS": [],
        "GAP_INFER_TOOLS": [],
        "CRITIC_TOOLS": [],
        "RESPONSE_TOOLS": [],
    }