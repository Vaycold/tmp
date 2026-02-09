# GAPago LangGraph - Research Gap Analysis Pipeline

LangGraph 기반 연구 논문 GAP 분석 파이프라인

## 🎯 기능

- 연구 질문 → arXiv 검색 → BM25 랭킹
- Abstract 기반 Limitation 추출
- Research GAP 자동 생성
- Critic 기반 조건 분기 (refine/redo/accept)

## 📊 출력

- 콘솔: Pretty print
- 파일: `outputs/run_TIMESTAMP.json`

## 🔧 LLM Provider 설정

| Provider | 환경변수 | 비고 |
|----------|---------|------|
| Mock | `LLM_PROVIDER=mock` | API 불필요 |
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini |
| Claude | `ANTHROPIC_API_KEY` | claude-sonnet-4 |
| Gemini | `GOOGLE_API_KEY` | gemini-2.0-flash |
| Exaone | `EXAONE_MODEL_PATH` | 로컬 모델 |

## 📂 프로젝트 구조
├── main.py              # 실행 진입점
├── config.py            # 설정 관리
├── models.py            # Pydantic 모델
├── llm/                 # LLM 추상화
│   ├── base.py          # 메인 인터페이스
│   ├── providers.py     # Provider 구현
│   └── utils.py         # JSON 파싱
├── agents/              # LangGraph 노드
│   ├── query_agent.py
│   ├── retrieval_agent.py
│   ├── limitation_agent.py
│   ├── gap_agent.py
│   └── critic_agent.py
├── utils/               # 유틸리티
│   ├── arxiv.py         # arXiv API
│   └── text.py          # 텍스트 처리
└── graph/               # LangGraph 구성
└── workflow.py      # 그래프 빌더

