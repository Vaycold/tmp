from __future__ import annotations

from typing import Optional, List
import json
import re
import time
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_community.retrievers import ArxivRetriever
from rank_bm25 import BM25Okapi

from config import Configuration
from utils.tavily import TavilySearch
from llm import get_llm


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

        out.append({
            "paper_id": f"arxiv:{arxiv_id}",
            "title": title,
            "abstract": abstract,
            "url": url,
            "year": year,
            "authors": authors,
            "score_bm25": 0.0,
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
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        batch = _parse_atom(r.text)
        raw.extend(batch)
        if not batch:
            break
        time.sleep(0.3)

    uniq = {p["paper_id"]: p for p in raw}
    return list(uniq.values())


def bm25_rank(papers: list[dict], query_text: str, top_k: int) -> dict:
    if not papers:
        return {"selected": [], "avg_bm25": 0.0}

    corpus = [_tokenize(p.get("abstract", "")) for p in papers]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query_text))
    pairs = list(zip(papers, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:top_k]

    selected = []
    for paper, score in top:
        paper["score_bm25"] = float(score)
        selected.append(paper)

    avg = sum(p["score_bm25"] for p in selected) / len(selected) if selected else 0.0
    return {"selected": selected, "avg_bm25": avg}


# =========================== 1. RETRIEVAL AGENT ==========================
class ArxivSearchInput(BaseModel):
    query: str = Field(description="arXiv 검색 쿼리")
    max_docs: int = Field(default=3, description="가져올 문서 수")


class ArxivApiCallInput(BaseModel):
    search_query: str = Field(description="arXiv API search_query")
    max_total: int = Field(default=80, description="총 최대 결과 수")
    page_size: int = Field(default=40, description="페이지당 결과 수")
    max_pages: int = Field(default=3, description="최대 페이지 수")


class BM25RankInput(BaseModel):
    papers_json: str = Field(description="논문 리스트 JSON 문자열")
    query_text: str = Field(description="BM25 질의 텍스트")
    top_k: int = Field(default=10, description="상위 결과 수")


class MeaningExpandInput(BaseModel):
    refined_query: str = Field(description="정제된 검색 질의")
    keywords: list[str] = Field(default_factory=list, description="핵심 키워드")
    user_memory: Optional[dict] = Field(default=None, description="선택적 사용자 메모리")


class WebSearchInput(BaseModel):
    query: str = Field(description="웹 검색 쿼리")


class ScienceOnSearchInput(BaseModel):
    query: str = Field(description="ScienceON 검색 쿼리")
    max_results: int = Field(default=5, description="가져올 결과 수")


def _format_arxiv_docs(arxiv_search_results) -> str:
    return "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata.get("entry_id","")}" '
            f'date="{doc.metadata.get("Published", "")}" '
            f'authors="{doc.metadata.get("Authors", "")}"/>\n'
            f'<Title>\n{doc.metadata.get("Title","")}\n</Title>\n\n'
            f'<Summary>\n{doc.metadata.get("Summary","")}\n</Summary>\n\n'
            f"<Content>\n{doc.page_content}\n</Content>\n"
            f"</Document>"
            for doc in arxiv_search_results
        ]
    )


def build_retrieval_tools(config: Optional[RunnableConfig] = None) -> List:
    """
    RunnableConfig 기반으로 retriever/tool 인스턴스를 생성하고
    LangChain tool 리스트로 반환
    """
    cfg = Configuration.from_runnable_config()

    tavily_tool = TavilySearch(max_results=cfg.tavily_max_results)

    arxiv_retriever = ArxivRetriever(
        load_max_docs=cfg.arxiv_max_docs,
        load_all_available_meta=True,
        get_full_documents=True,
    )

    @tool(args_schema=ArxivSearchInput)
    def arxiv_search_tool(query: str, max_docs: Optional[int] = None) -> str:
        """Search arXiv and return formatted documents for LLM context."""
        max_docs = cfg.arxiv_max_docs
        try:
            results = arxiv_retriever.invoke(
                query,
                load_max_docs=max_docs,
                load_all_available_meta=True,
                get_full_documents=True,
            )
            return _format_arxiv_docs(results)
        except Exception as e:
            return f"<Error>Arxiv search failed: {str(e)}</Error>"

#     @tool(args_schema=MeaningExpandInput)
#     def meaning_expand_tool(
#         refined_query: str,
#         keywords: Optional[list[str]] = None,
#         user_memory: Optional[dict] = None,
#     ) -> str:
#         """Expand keywords using LLM (no arXiv query generation)."""
#         llm = get_llm("azure", "gpt-5.1-chat")
#         rq = _norm(refined_query)
#         kws = [_norm(k) for k in (keywords or []) if _norm(k)]
#         mem = user_memory or {}

#         prompt = f"""You are an information retrieval engineer.

# INPUT:
# - refined_query: {rq}
# - keywords: {kws}
# - user_memory (optional): {json.dumps(mem, ensure_ascii=False)[:1500]}

# GOAL:
# 1) Expand terms with acronyms/synonyms/related terms (domain-agnostic).
# 2) Provide a BM25 query text (plain) for ranking.

# OUTPUT JSON ONLY with keys:
# {
#     "expanded_terms": ["<=12 terms, include acronyms"],
#     "bm25_query_text": "plain text for BM25"
# }
# """

#         try:
#             resp = llm.invoke(prompt)
#             data = _safe_json_loads(getattr(resp, "content", "") or "") or {}
#         except Exception as e:
#             return f"<Error>meaning_expand failed: {str(e)}</Error>"

#         expanded = data.get("expanded_terms", []) or []
#         expanded = [_norm(x) for x in expanded if isinstance(x, str) and _norm(x)]
#         expanded = expanded[:12]

#         bm25_text = _norm(data.get("bm25_query_text", "")) or _norm(" ".join([rq] + expanded[:6]))

#         payload = {
#             "expanded_terms": expanded,
#             "bm25_query_text": bm25_text,
#         }
#         return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ArxivApiCallInput)
    def arxiv_api_call_tool(
        search_query: str,
        max_total: int = 80,
        page_size: int = 40,
        max_pages: int = 3,
    ) -> str:
        """Call arXiv API directly and return raw paper list as JSON string."""
        try:
            results = arxiv_api_call(
                search_query=search_query,
                max_total=max_total,
                page_size=page_size,
                max_pages=max_pages,
            )
            return json.dumps(results, ensure_ascii=False)
        except Exception as e:
            return f"<Error>Arxiv API call failed: {str(e)}</Error>"

    @tool(args_schema=BM25RankInput)
    def bm25_rank_tool(papers_json: str, query_text: str, top_k: int = 10) -> str:
        """Rank paper list with BM25 and return selected papers as JSON string."""
        try:
            papers = json.loads(papers_json or "[]")
            result = bm25_rank(papers, query_text=query_text, top_k=top_k)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"<Error>BM25 rank failed: {str(e)}</Error>"

    @tool(args_schema=WebSearchInput)
    def web_search_tool(query: str) -> str:
        """Alias web search tool (Tavily)."""
        try:
            return json.dumps(tavily_tool.search(query), ensure_ascii=False)
        except Exception as e:
            return f"<Error>Web search failed: {str(e)}</Error>"

    @tool(args_schema=ScienceOnSearchInput)
    def scienceon_search_tool(query: str, max_results: int = 5) -> str:
        """ScienceON search tool (requires external API configuration)."""
        return "<Error>ScienceON API is not configured.</Error>"

    return [
        # meaning_expand_tool,
        arxiv_search_tool,
        arxiv_api_call_tool,
        web_search_tool,
        scienceon_search_tool,
        bm25_rank_tool,
        tavily_tool,
    ]


# ==============================================================================


def build_role_tools(config: Optional[RunnableConfig] = None) -> dict:
    """
    에이전트 역할별 tool 생성
    """
    retrieval_tools = build_retrieval_tools(config)

    return {
        "QUERY_TOOLS": [],
        "RETRIEVAL_TOOLS": retrieval_tools,
        "LIMITATION_TOOLS": [],
        "GAP_INFER_TOOLS": [],
        "CRITIC_TOOLS": [],
        "RESPONSE_TOOLS": [],
    }