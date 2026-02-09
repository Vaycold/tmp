# GAPago LangGraph (MVP) — Research Gap Analysis Pipeline

연구 질문 1개를 입력하면 **arXiv 논문을 수집/랭킹(BM25)**하고, **Abstract 기반으로 한계점(Limitations) 추출 → GAP(Research Gap) 생성**까지 수행하는 **LangGraph 기반 MVP 파이프라인**입니다.  
또한 **Critic 점수에 따라 refine/redo/accept 조건 분기**를 수행합니다.

---
## Quick Start

```bash
pip install -r requirements.txt
export LLM_PROVIDER=mock
python main.py "limitations of rag systems"
```

## 핵심 기능 (What it does)

### 1) End-to-End 파이프라인
- **User Question 입력**
- **Query Analysis**: 검색용 쿼리(refined_query) + keywords/negative_keywords 생성
- **Paper Retrieval**: arXiv 검색 → BM25로 Top-K 논문 선정
- **Limitation Extraction**: 각 논문 abstract에서 한계점 claim + 근거(evidence quote) 추출
- **GAP Inference**: 고정 K축(axis)으로 분류 후 GAP 문장 생성 + supporting evidence 포함
- **Critic Scoring**: 품질 점수 산출(특정성/정합성/근거성)
- **Router(조건 분기)**: 점수 기준으로 `refine_query` / `redo_retrieval` / `accept` 결정

### 2) LangGraph Orchestration
`StateGraph`로 아래 노드가 순차 실행되며, critic 결과에 따라 루프를 수행합니다.
```
query_analysis
 → paper_retrieval
 → limitation_extract
 → gap_infer
 → critic_score
 → router
```

### 3) 결과 출력
- 콘솔: 요약(Question/Query/Gaps/Quality Scores)
- 파일: `outputs/run_YYYYMMDD_HHMMSS.json`

---

## 프로젝트 구조
```
gapago_langgraph/
  main.py
  config.py
  models.py
  graph/workflow.py
  agents/
    query_agent.py
    retrieval_agent.py
    limitation_agent.py
    gap_agent.py
    critic_agent.py
  llm/
    base.py
    providers.py
    utils.py
  utils/
    arxiv.py
    text.py
  requirements.txt
```
---

## 설치

```bash
pip install -r requirements.txt
```
LLM Provider에 따라 추가 패키지가 필요할 수 있습니다(예: openai/anthropic/google-generativeai/transformers 등).

## 실행 방법

1) 기본 실행 (Mock LLM)

API 없이 동작하는 더미(Mock) 모드입니다.

```bash
export LLM_PROVIDER=mock
python main.py "limitations of rag based qa systems"
```

2) 질문 인자를 주지 않으면 입력 프롬프트로 받습니다
```bash
python main.py
```

LLM Provider 설정

config.py는 기본적으로 환경변수로 LLM provider/model을 제어합니다.

공통
	•	LLM_PROVIDER (default: mock)

OpenAI
	•	OPENAI_API_KEY
	•	OPENAI_MODEL (default: gpt-4o-mini)

Anthropic (직접 API)
	•	ANTHROPIC_API_KEY
	•	ANTHROPIC_MODEL (default: claude-sonnet-4-20250514)

Google Gemini
	•	GOOGLE_API_KEY
	•	GEMINI_MODEL (default: gemini-2.0-flash-exp)

Exaone (로컬)
	•	EXAONE_MODEL_PATH (로컬 모델 경로)

주의: 현재 코드의 anthropic_llm은 Anthropic 직접 API 기반이며, AWS Bedrock Claude 호출은 별도 구현이 필요.

⸻

파이프라인 설정 (환경변수)
	•	ARXIV_MAX_RESULTS (default: 50) — arXiv 후보 수
	•	TOP_K_PAPERS (default: 10) — BM25 상위 K
	•	MAX_ITERATIONS (default: 2) — 라우팅 루프 최대 반복
	•	OUTPUT_DIR (default: outputs) — 결과 저장 디렉토리

⸻

Router(조건 분기) 기준

graph/workflow.py의 route_decision()에서 아래 임계치로 분기합니다.
	•	iteration >= max_iterations → accept (루프 종료)
	•	critic.query_specificity < 0.55 → refine_query (쿼리 재정제)
	•	critic.paper_relevance < 0.55 → redo_retrieval (논문 재수집/재랭킹)
	•	critic.groundedness < 0.60 → redo_retrieval
	•	else → accept

⸻

출력(JSON) 스키마 요약

main.py에서 최종 state를 아래 형태로 저장합니다:
```bash
{
  "question": "...",
  "query": "...",
  "gaps": [
    {
      "axis": "...",
      "gap_statement": "...",
      "supporting_papers": ["..."],
      "supporting_quotes": ["..."]
    }
  ],
  "critic": {
    "query_specificity": 0.0,
    "paper_relevance": 0.0,
    "groundedness": 0.0
  },
  "iteration": 0,
  "route": "accept",
  "errors": [],
  "trace": {}
}
```

제한사항(MVP)
	•	논문 본문(PDF) 파싱 없이 abstract 기반으로 limitation/GAP을 생성합니다.
	•	Web Search/trend signal, VectorDB(FAISS), 본격 클러스터링(HDBSCAN) 등은 포함하지 않습니다.
	•	LLM Provider별 JSON 파싱 실패 가능성이 있어, 안정화(재시도/복구 로직) 여지가 있습니다.

⸻
