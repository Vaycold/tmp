# 3-3) Limitation Extract Agent
from __future__ import annotations

import re
import time
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from states import AgentState, Paper, LimitationItem
from llm import get_llm
from utils.parse_json import parse_json

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
# ArxivLoader full text 로드
# =====================================================================
def _load_full_text_sections(paper: Paper) -> dict:
    """
    ArxivLoader로 논문 full text 로드 후 섹션 분리.
    실패 시 빈 dict 반환 → abstract fallback.
    """
    # 이미 로드된 경우 재사용
    if paper.full_text_sections:
        return paper.full_text_sections

    try:
        from langchain_community.document_loaders import ArxivLoader

        loader = ArxivLoader(
            query=paper.title,
            load_max_docs=1,
            load_all_available_meta=True,
        )
        docs = loader.load()

        if not docs:
            return {}

        sections = _split_sections(docs[0].page_content)
        print(f"  [fulltext] '{paper.title[:50]}' → {list(sections.keys())}")
        return sections

    except Exception as e:
        print(f"  [fulltext] 로드 실패: {paper.title[:50]} ({e})")
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

    if use_fallback:
        lines += [
            "## FALLBACK: Abstract Only",
            paper.abstract,
        ]
    else:
        if track1:
            lines.append("## Track 1: Author-Stated Sections")
            for k, v in track1.items():
                lines += [f"### [{k.upper()}]", v, ""]

        if track2:
            lines.append("## Track 2: Structural Analysis Sections")
            for k, v in track2.items():
                lines += [f"### [{k.upper()}]", v, ""]

    return "\n".join(lines)


SYSTEM_PROMPT = """ROLE: Limitation Extract Agent

You extract research limitations from academic papers using a 2-track approach.

## Track 1: Author-Stated Limitations
Sections: conclusion, limitations, future_work
→ Extract what the authors explicitly admit as limitations or future work.
→ Be critical: authors often write defensively. Note if a stated "future work" is actually avoiding a limitation.

## Track 2: Structural Limitations
Sections: introduction, method, experiment, discussion
→ Identify limitations NOT explicitly stated by authors but revealed by:
  - Narrow assumptions in the method (e.g., "we assume i.i.d. data")
  - Limited datasets or evaluation scope (e.g., single domain, small scale)
  - Missing baselines or comparisons
  - Scope restrictions mentioned in introduction

## Rules
1. Extract 1-3 limitations per paper. Prioritize Track 2 structural findings.
2. Each limitation MUST include:
   - claim: concise limitation statement (1-2 sentences)
   - evidence_quote: exact short quote from the provided text
   - track: "author_stated" or "structural"
   - source_section: section name (e.g., "conclusion", "method", "experiment")
3. Do NOT infer gaps. Only extract limitations from the provided text.
4. If only abstract is provided (FALLBACK), extract 1 limitation maximum.

## Output Format (strictly JSON list)
[
  {
    "paper_id": "<id>",
    "claim": "<limitation statement>",
    "evidence_quote": "<exact short quote>",
    "track": "author_stated" or "structural",
    "source_section": "<section name>"
  },
  ...
]
Output ONLY the JSON list. No explanation before or after.
"""


# =====================================================================
# limitation_extract_node
# =====================================================================
def limitation_extract_node(state: AgentState) -> AgentState:
    papers_raw = state.get("papers", [])
    print(f"  [DEBUG] state['papers'] count: {len(papers_raw)}")
    print(f"  [DEBUG] state['papers'] type: {type(papers_raw)}")
    if papers_raw:
        print(f"  [DEBUG] first paper type: {type(papers_raw[0])}")
        print(f"  [DEBUG] first paper: {papers_raw[0]}")
    if not papers_raw:
        print("  ⚠️ [limitation] papers 없음 → 빈 limitations 반환")
        return {
            "messages": [AIMessage(content="[]", name="limitation_extract")],
            "sender": "limitation_extract",
            "limitations": [],
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
                print(f"  ⚠️ Paper 변환 실패: {e}")
                continue

    all_limitations = []
    all_messages = []

    for paper in papers:
        print(f"\n  [limitation] 처리 중: {paper.paper_id}")

        # full text 로드
        sections = _load_full_text_sections(paper)
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
                print(f"  ⚠️ LimitationItem 변환 실패: {e}")
                continue

        all_messages.append(AIMessage(content=content, name="limitation_extract"))
        print(f"  ✓ {paper.paper_id}: {len(parsed)}개 limitation 추출")

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
    }