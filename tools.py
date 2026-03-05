from __future__ import annotations

from typing import Optional, List
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_community.retrievers import ArxivRetriever

from config import Configuration
from utils.tavily import TavilySearch


# =========================== 1. RETRIEVAL AGENT ==========================
class ArxivSearchInput(BaseModel):
    query: str = Field(description="arXiv 검색 쿼리")
    max_docs: int = Field(default=3, description="가져올 문서 수")


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

    return [arxiv_search_tool, tavily_tool]


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
