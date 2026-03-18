from __future__ import annotations

from typing import Optional, List
import base64
import datetime
import json
import re
import time
import requests
import xml.etree.ElementTree as ET
from urllib.parse import urlencode
from pydantic import BaseModel, Field

from Crypto.Cipher import AES
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from rank_bm25 import BM25Okapi

from config import Configuration
from utils.tavily import TavilySearch


_ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_SCIENCEON_OPENAPI = "https://apigateway.kisti.re.kr/openapicall.do"
_SCIENCEON_TOKEN_API = "https://apigateway.kisti.re.kr/tokenrequest.do"
_SCIENCEON_AES_IV = "jvHJ1EFA0IXBrxxz"
_SCIENCEON_TOKEN_CACHE: dict[str, Optional[str]] = {
    "access_token": None,
    "refresh_token": None,
}


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


# ============================== arXiv ==============================
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
            "source": "arxiv",
            "full_text_sections": {},
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


# =========================== ScienceON Helpers ==========================
def _scienceon_pad_pkcs7(text: str, block_size: int = 16) -> bytes:
    pad_len = block_size - (len(text.encode("utf-8")) % block_size)
    return (text + chr(pad_len) * pad_len).encode("utf-8")


def _scienceon_encrypt_accounts(mac_address: str, key: str) -> str:
    timestamp = "".join(re.findall(r"\d", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    plain_txt = json.dumps({"datetime": timestamp, "mac_address": mac_address}, separators=(",", ":"), ensure_ascii=False)
    cipher = AES.new(key.encode("utf-8"), AES.MODE_CBC, _SCIENCEON_AES_IV.encode("utf-8"))
    encrypted_bytes = cipher.encrypt(_scienceon_pad_pkcs7(plain_txt))
    return base64.urlsafe_b64encode(encrypted_bytes).decode("utf-8")


def _scienceon_request_create_token(client_id: str, mac_address: str, key: str, timeout: int = 30) -> dict:
    encrypted_txt = _scienceon_encrypt_accounts(mac_address=mac_address, key=key)
    response = requests.get(
        _SCIENCEON_TOKEN_API,
        params={"client_id": client_id, "accounts": encrypted_txt},
        timeout=timeout,
    )
    response.raise_for_status()
    try:
        return response.json()
    except Exception:
        return {"raw": response.text}


def _scienceon_request_access_token(client_id: str, refresh_token: str, timeout: int = 30) -> dict:
    response = requests.get(
        _SCIENCEON_TOKEN_API,
        params={"refreshToken": refresh_token, "client_id": client_id},
        timeout=timeout,
    )
    response.raise_for_status()
    try:
        return response.json()
    except Exception:
        return {"raw": response.text}


def _scienceon_resolve_tokens(*, client_id: Optional[str], mac_address: Optional[str], key: Optional[str], timeout: int = 30) -> dict:
    access_token = _SCIENCEON_TOKEN_CACHE.get("access_token")
    refresh_token = _SCIENCEON_TOKEN_CACHE.get("refresh_token")
    events: list[str] = []

    if access_token:
        return {"access_token": access_token, "refresh_token": refresh_token, "events": events}

    if client_id and refresh_token:
        refreshed = _scienceon_request_access_token(client_id=client_id, refresh_token=refresh_token, timeout=timeout)
        new_access = refreshed.get("access_token")
        new_refresh = refreshed.get("refresh_token") or refresh_token
        if new_access:
            _SCIENCEON_TOKEN_CACHE["access_token"] = new_access
            _SCIENCEON_TOKEN_CACHE["refresh_token"] = new_refresh
            events.append("access_token_reissued_from_refresh_token")
            return {"access_token": new_access, "refresh_token": new_refresh, "events": events}

    if client_id and mac_address and key:
        created = _scienceon_request_create_token(client_id=client_id, mac_address=mac_address, key=key, timeout=timeout)
        new_access = created.get("access_token")
        new_refresh = created.get("refresh_token")
        if new_access:
            _SCIENCEON_TOKEN_CACHE["access_token"] = new_access
            _SCIENCEON_TOKEN_CACHE["refresh_token"] = new_refresh
            events.append("refresh_and_access_token_issued")
            return {"access_token": new_access, "refresh_token": new_refresh, "events": events}

    return {"access_token": None, "refresh_token": refresh_token, "events": events}


def _scienceon_item_values(record: ET.Element) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in record.findall("./item"):
        meta = item.attrib.get("metaCode")
        if meta:
            values[meta] = _norm(item.text or "")
    return values


def _scienceon_parse_search_xml(xml_text: str, target: str) -> dict:
    root = ET.fromstring(xml_text)

    total_count_text = _norm(root.findtext("./resultSummary/TotalCount", default="0"))
    service_datatype = _norm(root.findtext("./resultSummary/serviceDatatype", default=""))
    status_code = _norm(root.findtext("./resultSummary/statusCode", default=""))

    results = []
    for idx, record in enumerate(root.findall("./recordList/record"), start=1):
        values = _scienceon_item_values(record)
        cn = values.get("CN") or values.get("ArticleId") or f"{target}_{idx}"
        title = values.get("Title") or values.get("Title2") or ""
        abstract = values.get("Abstract") or values.get("Abstract2") or ""
        authors_raw = values.get("Author") or values.get("Author2") or ""
        authors = [a.strip() for a in re.split(r"[;|]", authors_raw) if a.strip()]
        year_text = values.get("Pubyear") or values.get("Pubdate") or ""
        year = int(year_text[:4]) if len(year_text) >= 4 and year_text[:4].isdigit() else 0
        url = values.get("FulltextURL") or values.get("ContentURL") or values.get("MobileURL") or values.get("DOI") or ""

        results.append({
            "paper_id": f"scienceon:{cn}",
            "title": title,
            "abstract": abstract,
            "url": url,
            "year": year,
            "authors": authors,
            "score_bm25": 0.0,
            "source": "scienceon",
            "full_text_sections": {},
            "journal": values.get("JournalName", ""),
            "publisher": values.get("Publisher", ""),
            "doi": values.get("DOI", ""),
            "keywords_text": values.get("Keyword") or values.get("Keyword2") or "",
            "content_url": values.get("ContentURL", ""),
            "fulltext_url": values.get("FulltextURL", ""),
            "raw": values,
        })

    return {
        "source": "scienceon",
        "target": target,
        "total_count": int(total_count_text) if total_count_text.isdigit() else 0,
        "service_datatype": service_datatype,
        "status_code": status_code,
        "results": results,
    }


def scienceon_search(*, client_id: str, query: str, target: str = "ARTI", cur_page: int = 1, row_count: int = 10, mac_address: Optional[str] = None, key: Optional[str] = None, timeout: int = 30) -> dict:
    token_state = _scienceon_resolve_tokens(
        client_id=client_id,
        mac_address=mac_address,
        key=key,
        timeout=timeout,
    )
    access_token = token_state.get("access_token")
    refresh_token = token_state.get("refresh_token")
    events = list(token_state.get("events", []))

    if not access_token:
        raise RuntimeError(
            "ScienceON token is not available. Set SCIENCEON_CLIENT_ID and provide SCIENCEON_MAC_ADDRESS + SCIENCEON_KEY so the tool can issue a token at call time."
        )

    params = {
        "client_id": client_id,
        "token": access_token,
        "version": "1.0",
        "action": "search",
        "target": target,
        "searchQuery": json.dumps({"BI": _norm(query)}, ensure_ascii=False, separators=(",", ":")),
        "curPage": int(cur_page),
        "rowCount": int(row_count),
    }

    response = requests.get(_SCIENCEON_OPENAPI, params=params, timeout=timeout)
    response.raise_for_status()
    parsed = _scienceon_parse_search_xml(response.text, target=target)

    if parsed.get("status_code") != "200" and refresh_token:
        refreshed = _scienceon_request_access_token(client_id=client_id, refresh_token=refresh_token, timeout=timeout)
        new_access = refreshed.get("access_token")
        new_refresh = refreshed.get("refresh_token") or refresh_token
        if new_access:
            _SCIENCEON_TOKEN_CACHE["access_token"] = new_access
            _SCIENCEON_TOKEN_CACHE["refresh_token"] = new_refresh
            params["token"] = new_access
            response = requests.get(_SCIENCEON_OPENAPI, params=params, timeout=timeout)
            response.raise_for_status()
            parsed = _scienceon_parse_search_xml(response.text, target=target)
            events.append("access_token_reissued_after_non200_response")

    parsed["query"] = query
    parsed["cur_page"] = int(cur_page)
    parsed["row_count"] = int(row_count)
    parsed["token_events"] = events
    return parsed


# =========================== Tool APIs ==========================
class ArxivApiCallInput(BaseModel):
    search_query: str = Field(description="arXiv API search_query")
    max_total: int = Field(default=80, description="총 최대 결과 수")
    page_size: int = Field(default=40, description="페이지당 결과 수")
    max_pages: int = Field(default=3, description="최대 페이지 수")


class WebSearchInput(BaseModel):
    query: str = Field(description="웹 검색 쿼리")


class ScienceOnSearchInput(BaseModel):
    query: str = Field(description="ScienceON 검색 쿼리")
    target: str = Field(default="ARTI", description="ScienceON target")
    cur_page: int = Field(default=1, description="현재 페이지 번호")
    row_count: int = Field(default=10, description="가져올 결과 수")


def build_retrieval_tools(config: Optional[RunnableConfig] = None) -> List:
    """
    Retrieval Agent가 선택할 수 있는 외부 검색 툴만 노출한다.
    - web_search_tool
    - arxiv_api_call_tool
    - scienceon_search_tool (placeholder)
    """
    cfg = Configuration.from_runnable_config(config)
    tavily_tool = TavilySearch(max_results=cfg.tavily_max_results)

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
            )
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
    def scienceon_search_tool(query: str, target: str = "ARTI", cur_page: int = 1, row_count: int = 10) -> str:
        """Search ScienceON paper records using the exact openapicall.do format: action=search, target=ARTI, searchQuery={\"BI\":\"...\"}, curPage, rowCount."""
        if not cfg.scienceon_client_id:
            return "<Error>ScienceON client_id is not configured. Set SCIENCEON_CLIENT_ID.</Error>"
        try:
            result = scienceon_search(
                client_id=cfg.scienceon_client_id,
                query=query,
                target=target or cfg.scienceon_default_target,
                cur_page=cur_page,
                row_count=row_count or cfg.scienceon_default_row_count,
                mac_address=cfg.scienceon_mac_address,
                key=cfg.scienceon_key,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"<Error>ScienceON search failed: {str(e)}</Error>"

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
