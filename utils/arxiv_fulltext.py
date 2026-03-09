# utils/arxiv_fulltext.py

from __future__ import annotations

import re
import time
import requests
from bs4 import BeautifulSoup

# =====================================================================
# 섹션 키워드 정의 (섹션별 신뢰도 순)
# Introduction > Conclusion > Limitations > Discussion > Future Work
# =====================================================================
SECTION_KEYWORDS = {
    "introduction":  ["introduction", "background"],
    "conclusion":    ["conclusion", "concluding remarks", "summary"],
    "limitations":   ["limitation", "limitations", "weakness"],
    "future_work":   ["future work", "future research", "future direction"],
    "discussion":    ["discussion", "analysis"],
}


def _normalize_section_key(heading: str) -> str | None:
    """섹션 헤딩 텍스트를 SECTION_KEYWORDS 키로 매핑"""
    h = heading.lower().strip()
    for key, keywords in SECTION_KEYWORDS.items():
        if any(kw in h for kw in keywords):
            return key
    return None


# =====================================================================
# HTML 파싱 (1순위: 2023년 이후 논문)
# =====================================================================
def _parse_sections_from_html(html_text: str) -> dict:
    """
    arXiv HTML에서 섹션별 텍스트 추출.
    <section> 또는 <h2>/<h3> 태그 기반으로 섹션 경계 인식.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    sections = {}

    # 방법 1: <section> 태그 기반
    for section in soup.find_all("section"):
        heading_tag = section.find(["h1", "h2", "h3", "h4"])
        if not heading_tag:
            continue

        heading_text = heading_tag.get_text(separator=" ", strip=True)
        key = _normalize_section_key(heading_text)
        if key and key not in sections:
            # 헤딩 제외하고 본문만 추출
            heading_tag.decompose()
            text = section.get_text(separator=" ", strip=True)
            if len(text) > 100:  # 너무 짧은 섹션 제외
                sections[key] = text

    # 방법 2: <section> 없으면 <h2> 기반으로 fallback
    if not sections:
        headings = soup.find_all(["h2", "h3"])
        for heading in headings:
            key = _normalize_section_key(heading.get_text())
            if not key or key in sections:
                continue

            # 다음 heading까지의 텍스트 수집
            content = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ["h2", "h3"]:
                    break
                content.append(sibling.get_text(separator=" ", strip=True))

            text = " ".join(content).strip()
            if len(text) > 100:
                sections[key] = text

    return sections


# =====================================================================
# PDF 파싱 (2순위: HTML 없는 구논문 fallback)
# =====================================================================
def _parse_sections_from_pdf(pdf_bytes: bytes) -> dict:
    """
    PDF에서 섹션별 텍스트 추출.
    PyMuPDF(fitz)로 텍스트 추출 후 헤딩 패턴으로 섹션 분리.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return {}

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = "\n".join(page.get_text() for page in doc)
    except Exception:
        return {}

    # 섹션 헤딩 패턴
    # 예) "1. Introduction", "5 Conclusion", "Limitations"
    heading_pattern = re.compile(
        r"(?m)^(\d+\.?\s+)?("
        + "|".join(
            kw.title()
            for keywords in SECTION_KEYWORDS.values()
            for kw in keywords
        )
        + r")\s*$",
        re.IGNORECASE,
    )

    matches = list(heading_pattern.finditer(full_text))
    if not matches:
        return {}

    sections = {}
    for i, match in enumerate(matches):
        key = _normalize_section_key(match.group(0))
        if not key or key in sections:
            continue

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        text = full_text[start:end].strip()

        if len(text) > 100:
            sections[key] = text

    return sections


# =====================================================================
# 메인 함수: HTML 우선 + PDF fallback
# =====================================================================
def fetch_full_text_sections(arxiv_url: str) -> dict:
    """
    arXiv 논문의 full text를 섹션별로 추출.

    1순위: arXiv HTML (2023년 이후 논문)
    2순위: arXiv PDF  (HTML 없는 구논문)
    실패 시: 빈 dict 반환 → limitation_agent가 abstract로 fallback

    Returns:
        {
            "introduction": "...",
            "conclusion":   "...",
            "limitations":  "...",
            "future_work":  "...",
            "discussion":   "...",
        }
    """
    arxiv_id = (
        arxiv_url
        .replace("https://arxiv.org/abs/", "")
        .replace("http://arxiv.org/abs/", "")
        .strip()
    )

    # 1순위: HTML 파싱
    try:
        html_url = f"https://arxiv.org/html/{arxiv_id}"
        response = requests.get(html_url, timeout=15)
        if response.status_code == 200:
            sections = _parse_sections_from_html(response.text)
            if sections:
                print(f"    [fulltext] HTML 파싱 성공: {arxiv_id} → {list(sections.keys())}")
                return sections
    except Exception as e:
        print(f"    [fulltext] HTML 파싱 실패: {arxiv_id} ({e})")

    time.sleep(0.3)  # arXiv rate limit 대응

    # 2순위: PDF fallback
    try:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        response = requests.get(pdf_url, timeout=30)
        if response.status_code == 200:
            sections = _parse_sections_from_pdf(response.content)
            if sections:
                print(f"    [fulltext] PDF 파싱 성공: {arxiv_id} → {list(sections.keys())}")
                return sections
    except Exception as e:
        print(f"    [fulltext] PDF 파싱 실패: {arxiv_id} ({e})")

    print(f"    [fulltext] 파싱 실패, abstract fallback 사용: {arxiv_id}")
    return {}