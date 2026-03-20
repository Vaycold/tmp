# 3-3) Limitation Extract Agent
from __future__ import annotations

import re
import time
from typing import Optional

import requests
import fitz

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from states import AgentState
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from tools import build_role_tools
from prompts.system import make_system_prompt
from llm import get_llm


from states import AgentState, Paper, LimitationItem
from llm import get_llm

llm = get_llm()

# =====================================================================
# 섹션 키워드 정의
# =====================================================================
SECTION_KEYWORDS = {
    # Track 1 - 저자 기술
    "conclusion":   ["conclusion", "concluding remarks", "summary"],
    "limitations":  ["limitation", "limitations", "weakness", "weaknesses"],
    "future_work":  ["future work", "future research", "future direction"],
    # Track 2 - 구조 분석
    "introduction": ["introduction", "background"],
    "method":       ["method", "methods", "methodology", "approach", "proposed method",
                     "our approach", "our method", "model", "framework", "architecture"],
    "experiment":   ["experiment", "experiments", "experimental setup", "experimental results",
                     "evaluation", "results", "empirical evaluation", "benchmarks"],
    "discussion":   ["discussion", "analysis", "ablation", "ablation study"],
}

TRACK1_KEYS = {"conclusion", "limitations", "future_work"}
TRACK2_KEYS = {"introduction", "method", "experiment", "discussion"}

MAX_SECTION_CHARS = 3000  # 섹션별 최대 글자 수 (토큰 비용 제한)


# =====================================================================
# 섹션 분리 유틸
# =====================================================================
def _split_sections(full_text: str) -> dict:
    """전체 텍스트에서 섹션별로 분리."""
    heading_pattern = re.compile(
        r"(?m)^(\d+[\.\d]*\s+)?("
        + "|".join(
            kw.title()
            for kws in SECTION_KEYWORDS.values()
            for kw in kws
        )
        + r")(\s*:)?\s*$",
        re.IGNORECASE,
    )

    matches = list(heading_pattern.finditer(full_text))
    if not matches:
        return {}

    sections = {}
    for i, match in enumerate(matches):
        h = match.group(0).lower().strip()
        key = None
        for k, kws in SECTION_KEYWORDS.items():
            if any(kw in h for kw in kws):
                key = k
                break
        if not key or key in sections:
            continue

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        text = full_text[start:end].strip()
        if len(text) > 100:
            sections[key] = text[:MAX_SECTION_CHARS]

    return sections


# =====================================================================
# Full text 로드 (소스별 분기)
# =====================================================================
_REQUEST_HEADERS = {"User-Agent": "GAPAGO-Research-Agent/1.0"}


def _extract_arxiv_id(paper: Paper) -> Optional[str]:
    """paper_id ('arxiv:2401.12345v1') 에서 순수 arXiv ID를 추출."""
    pid = paper.paper_id or ""
    if pid.lower().startswith("arxiv:"):
        pid = pid[len("arxiv:"):]
    pid = pid.strip()
    if not pid:
        return None
    if not re.match(r"^(\d{4}\.\d{4,5}|[\w\-]+\.?[\w\-]*/\d{7})", pid):
        return None
    pid = re.sub(r"v\d+$", "", pid)
    return pid if pid else None


def _load_arxiv_full_text(paper: Paper) -> dict:
    """arXiv 논문 full text 로드 (ArxivLoader 사용)."""
    arxiv_id = _extract_arxiv_id(paper)
    try:
        from langchain_community.document_loaders import ArxivLoader

        if arxiv_id:
            loader = ArxivLoader(query=arxiv_id, load_max_docs=1, load_all_available_meta=True)
        else:
            return {}

        docs = loader.load()
        if not docs:
            return {}

        sections = _split_sections(docs[0].page_content)
        print(f"  [fulltext:arxiv] '{paper.title[:50]}' → {list(sections.keys())}")
        return sections

    except Exception as e:
        print(f"  [fulltext:arxiv] 로드 실패: {paper.title[:50]} ({e})")
        return {}


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """PyMuPDF로 PDF 바이트에서 텍스트 추출."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def _find_pdf_url_from_doi(doi: str) -> Optional[str]:
    """DOI 페이지에 접근하여 PDF 다운로드 링크를 찾는다."""
    doi_url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
    try:
        resp = requests.get(doi_url, headers=_REQUEST_HEADERS, timeout=15, allow_redirects=True)
        resp.raise_for_status()

        # 리다이렉트된 최종 URL이 PDF인 경우
        if resp.headers.get("content-type", "").startswith("application/pdf"):
            return resp.url

        # HTML 페이지에서 PDF 링크 추출
        pdf_patterns = [
            r'href=["\']([^"\']*\.pdf[^"\']*)["\']',
            r'href=["\']([^"\']*\/pdf[^"\']*)["\']',
            r'content=["\']([^"\']*\.pdf[^"\']*)["\']',
        ]
        for pattern in pdf_patterns:
            matches = re.findall(pattern, resp.text, re.IGNORECASE)
            for url in matches:
                if not url.startswith("http"):
                    # 상대 URL → 절대 URL
                    from urllib.parse import urljoin
                    url = urljoin(resp.url, url)
                return url

    except Exception as e:
        print(f"  [fulltext:doi] DOI 페이지 접근 실패: {doi} ({e})")
    return None


def _load_doi_full_text(paper: Paper) -> dict:
    """DOI를 통해 출판사 PDF에서 full text를 로드."""
    # paper에서 DOI 추출 (full_text_sections에 raw가 있을 수 있음)
    doi = ""
    if hasattr(paper, "full_text_sections") and isinstance(paper.full_text_sections, dict):
        doi = paper.full_text_sections.get("doi", "")
    if not doi:
        # Paper 객체의 url 필드가 DOI일 수 있음
        if paper.url and ("doi.org" in paper.url or "dx.doi.org" in paper.url):
            doi = paper.url

    if not doi:
        return {}

    pdf_url = _find_pdf_url_from_doi(doi)
    if not pdf_url:
        print(f"  [fulltext:doi] PDF 링크 없음: {doi}")
        return {}

    try:
        resp = requests.get(pdf_url, headers=_REQUEST_HEADERS, timeout=30)
        resp.raise_for_status()

        if not resp.content[:4] == b"%PDF":
            print(f"  [fulltext:doi] PDF가 아닌 응답: {pdf_url}")
            return {}

        full_text = _extract_text_from_pdf_bytes(resp.content)
        if len(full_text) < 200:
            return {}

        sections = _split_sections(full_text)
        print(f"  [fulltext:doi] '{paper.title[:50]}' → {list(sections.keys())}")
        return sections

    except Exception as e:
        print(f"  [fulltext:doi] PDF 다운로드 실패: {pdf_url} ({e})")
        return {}


def _load_scienceon_full_text(paper: Paper) -> dict:
    """ScienceON 논문 full text 로드. DOI가 있으면 DOI 경유, 없으면 빈 dict."""
    # paper.full_text_sections에 raw 메타데이터가 저장되어 있을 수 있음
    raw = {}
    if hasattr(paper, "full_text_sections") and isinstance(paper.full_text_sections, dict):
        raw = paper.full_text_sections

    doi = raw.get("doi", "")

    # Paper.url에서 DOI 추출 시도
    if not doi and paper.url and ("doi.org" in paper.url or "dx.doi.org" in paper.url):
        doi = paper.url

    if doi:
        return _load_doi_full_text(paper)

    print(f"  [fulltext:scienceon] DOI 없음 → abstract fallback: {paper.title[:50]}")
    return {}


def _load_full_text_sections(paper: Paper) -> dict:
    """
    논문 소스별로 full text를 로드하고 섹션으로 분리.
    - arXiv → ArxivLoader
    - ScienceON → DOI 경유 PDF 다운로드
    - 그 외 → abstract fallback (빈 dict)
    """
    if paper.full_text_sections and any(
        k in paper.full_text_sections for k in SECTION_KEYWORDS
    ):
        return paper.full_text_sections

    pid = (paper.paper_id or "").lower()

    if pid.startswith("arxiv:"):
        return _load_arxiv_full_text(paper)

    if pid.startswith("scienceon:"):
        return _load_scienceon_full_text(paper)

    # 웹 소스 등은 abstract fallback
    return {}


# =====================================================================
# 프롬프트 구성
# =====================================================================
def _build_prompt(paper: Paper, sections: dict) -> str:
    """
    2-track 방식 프롬프트 구성.
    Track 1: 저자 기술 섹션
    Track 2: 구조 분석 섹션
    fallback: abstract (Track 1+2 모두 없을 때)
    """
    track1 = {k: v for k, v in sections.items() if k in TRACK1_KEYS}
    track2 = {k: v for k, v in sections.items() if k in TRACK2_KEYS}

    # fallback: 모두 없으면 abstract 사용
    use_fallback = not track1 and not track2

    lines = [
        f"paper_id: {paper.paper_id}",
        f"title: {paper.title}",
        f"year: {paper.year}",
        "",
    ]

ROLE_TOOLS = build_role_tools()
LIMITATION_TOOLS = ROLE_TOOLS["LIMITATION_TOOLS"]

limitation_extract_agent = create_agent(
    llm,
    tools=LIMITATION_TOOLS,
    system_prompt=make_system_prompt(
        "ROLE: Limitation Extract Agent\n"
        "You extract limitation/future-work statements from retrieved papers/snippets.\n"
        "RULES:\n"
        "1. Process ALL papers in the input without exception.\n"
        "2. Extract 1-2 key limitations per paper. No more, no less.\n"
        "3. Each limitation MUST include:\n"
        "   - paper_id: the paper's unique identifier\n"
        "   - claim: a brief limitation statement (1-2 sentences)\n"
        "   - evidence_quote: an exact quote from the provided text sections\n"
        "4. Do NOT infer or assume limitations not stated in the text.\n"
        "5. Do NOT skip any paper even if the abstract is short or unclear.\n"
        "6. Do NOT infer gaps yet.\n\n"

        "SECTION PRIORITY (high to low):\n"
        "  1. INTRODUCTION  — author-defined gaps, most reliable\n"
        "  2. CONCLUSION    — key contributions + limitations\n"
        "  3. LIMITATIONS   — author-stated weaknesses\n"
        "  4. DISCUSSION    — result interpretation + limitations\n"
        "  5. ABSTRACT      — fallback only, least detail\n"
        "  6. FUTURE_WORK   — supplementary evidence only\n\n"

        "OUTPUT FORMAT (strictly follow):\n"
        "paper_id: <id>\n"
        "  - claim: <limitation statement>\n"
        "    evidence_quote: <exact quote from paper>\n"
        "  - claim: <limitation statement>\n"
        "    evidence_quote: <exact quote from paper>\n"
        "Output be structured: paper_id -> [limitation_sentences], plus brief rationale.\n"
        "Do NOT infer gaps yet.\n"
    ),
)


def limitation_extract_node(state: AgentState) -> AgentState:
    papers_raw = state.get("papers", [])
    errors = list(state.get("errors", []) or [])
    print(f"  [DEBUG] state['papers'] count: {len(papers_raw)}")
    print(f"  [DEBUG] state['papers'] type: {type(papers_raw)}")
    if papers_raw:
        print(f"  [DEBUG] first paper type: {type(papers_raw[0])}")
        print(f"  [DEBUG] first paper: {papers_raw[0]}")
    if not papers_raw:
        print("  ⚠️ [limitation] papers 없음 → 빈 limitations 반환")
        errors.append("[limitation_extract] No papers provided")
        return {
            "messages": [AIMessage(content="[]", name="limitation_extract")],
            "sender": "limitation_extract",
            "limitations": [],
            "errors": errors,
        }

    # dict → Paper 객체 변환
    papers = []
    for p in papers_raw:
        if isinstance(p, Paper):
            papers.append(p)
        else:
            try:
                papers.append(Paper(**p))
            except Exception as e:
                errors.append(f"[limitation_extract] Paper conversion failed: {e}")
                print(f"  ⚠️ Paper 변환 실패: {e}")
                continue

    all_limitations = []
    all_messages = []
    fulltext_fail_count = 0
    llm_fail_count = 0

    for paper in papers:
        print(f"\n  [limitation] 처리 중: {paper.paper_id}")

        # full text 로드
        sections = _load_full_text_sections(paper)
        if not sections:
            fulltext_fail_count += 1
        time.sleep(0.5)  # ArxivLoader rate limit 대응

        # 프롬프트 구성
        paper_prompt = _build_prompt(paper, sections)

        # LLM 호출
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": paper_prompt},
        ]

        try:
            response = llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            llm_fail_count += 1
            errors.append(f"[limitation_extract] LLM failed for {paper.paper_id}: {e}")
            print(f"  ⚠️ LLM 호출 실패: {paper.paper_id} ({e})")
            content = "[]"

        # JSON 파싱
        parsed = parse_json(content)
        if not isinstance(parsed, list):
            parsed = []

        # LimitationItem 변환
        for item in parsed:
            try:
                lim = LimitationItem(
                    paper_id=item.get("paper_id", paper.paper_id),
                    claim=item.get("claim", ""),
                    evidence_quote=item.get("evidence_quote", ""),
                    track=item.get("track", "author_stated"),
                    source_section=item.get("source_section", ""),
                )
                if lim.claim:
                    all_limitations.append(lim.model_dump())
            except Exception as e:
                errors.append(f"[limitation_extract] LimitationItem parse failed for {paper.paper_id}: {e}")
                print(f"  ⚠️ LimitationItem 변환 실패: {e}")
                continue

        all_messages.append(AIMessage(content=content, name="limitation_extract"))
        print(f"  ✓ {paper.paper_id}: {len(parsed)}개 limitation 추출")

    # fulltext/LLM 실패 요약을 errors에 기록
    if fulltext_fail_count:
        errors.append(f"[limitation_extract] Full text load failed for {fulltext_fail_count}/{len(papers)} papers (abstract fallback used)")
    if llm_fail_count:
        errors.append(f"[limitation_extract] LLM call failed for {llm_fail_count}/{len(papers)} papers")

    # 최종 summary 메시지
    summary = AIMessage(
        content=f"Extracted {len(all_limitations)} limitations from {len(papers)} papers.",
        name="limitation_extract",
    )

    print(f"\n  ✅ [limitation] 총 {len(all_limitations)}개 추출 완료")

    return {
        "messages": [summary],
        "sender": "limitation_extract",
        "limitations": all_limitations,
        "errors": errors,
    }
