# GAPAGO 프로젝트 실행 가이드

## 📁 프로젝트 구조

```
gapago_project/
├── main.py                       # 실행 파일
├── requirements.txt              # 패키지 의존성
│
├── state/
│   └── state.py                  # 공통 State 정의
│
├── agents/
│   ├── query_analysis/          # Query Analysis Agent
│   │   ├── tools.py
│   │   ├── agent.py
│   │   └── node.py
│   │
│   ├── critic/                  # Critic Agent
│   │   ├── tools.py
│   │   └── node.py
│   │
│   ├── paper_search/            # Paper Search Agent
│   │   └── node.py
│   │
│   ├── web_search/              # Web Search Agent
│   │   └── node.py
│   │
│   ├── gap_classification/      # GAP Classification Agent
│   │   └── node.py
│   │
│   ├── topic_generation/        # Topic Generation Agent
│   │   └── node.py
│   │
│   └── orchestrator/            # Orchestrator
│       └── logic.py
│
└── graph/
    └── workflow.py              # 전체 워크플로우
```

## 🚀 실행 방법

### 1단계: 패키지 설치

```bash
pip install -r requirements.txt
```

### 2단계: API 키 설정

```bash
export GROQ_API_KEY=your_api_key_here
```

### 3단계: 실행

```bash
python main.py
```

또는 Python 코드에서:

```python
from main import run_gapago

result = run_gapago(
    "Transformer 모델의 한계점",
    api_key="your_api_key"
)
```

## 📊 워크플로우

```
Query Analysis
    ↓
Critic (모호성 평가)
    ↓
[Orchestrator]
├─ 모호함 → Query Analysis (재분석)
└─ 명확함 → Paper Search
            ↓
        Web Search (병렬)
            ↓
        Critic (정합성 평가)
            ↓
        [Orchestrator]
        ├─ 부족 → Paper Search (재검색)
        └─ 충분 → GAP Classification
                    ↓
                Critic (품질 평가)
                    ↓
                [Orchestrator]
                ├─ 낮음 → Paper Search (재검색)
                └─ 높음 → Topic Generation
                            ↓
                          END
```

## 🎯 핵심 기능

### 1. Critic Agent
- **역할**: 모든 Agent 출력을 점수로 평가
- **평가 항목**:
  - Query 명확성 (0-1)
  - 논문 정합성 (0-1)
  - GAP 품질 (0-1)

### 2. Orchestrator
- **역할**: Critic 점수 기반 흐름 제어
- **분기 로직**:
  - Query: 0.3 이상 → Human Loop
  - Paper: 0.5 미만 → 재검색
  - GAP: 0.4 미만 → 재검색

### 3. Paper S-M Agent
- **병렬 실행**:
  - Paper Search: arXiv 검색 + 임베딩 필터링
  - Web Search: 웹 정보 수집
- **임베딩 기반 유사도 계산**

### 4. GAP Classification
- **4가지 축**:
  - 데이터_의존성
  - 실제_환경_검증
  - 확장성
  - 구조적_한계
- **우선순위 결정**

## 🔧 개발 가이드

### 개별 Agent 테스트

```bash
cd agents/query_analysis
python node.py
```

### 새로운 Agent 추가

1. `agents/your_agent/` 폴더 생성
2. `tools.py`, `agent.py`, `node.py` 작성
3. `graph/workflow.py`에 노드 추가

## ⚙️ 설정

### State 수정

`state/state.py`에서 필드 추가/수정

### 임계값 조정

`agents/orchestrator/logic.py`에서:
- `threshold = 0.3`  # Query 명확성
- `threshold = 0.5`  # 논문 정합성
- `threshold = 0.4`  # GAP 품질

### 재시도 횟수

`main.py`에서:
- `max_retries = 2`  # 최대 2회 재시도

## 📝 사용 예시

```python
from main import run_gapago

# 실행
result = run_gapago(
    "Transformer 모델을 활용한 저자원 언어 처리의 한계점"
)

# 결과 확인
print(f"키워드: {result['keywords']}")
print(f"우선순위 축: {result['priority_axis']}")
print(f"연구 주제: {result['research_topics']}")
```

## 🐛 문제 해결

### API 키 오류
```
ValueError: GROQ_API_KEY가 설정되지 않았습니다!
```
→ 환경변수 설정 또는 코드에서 직접 전달

### 임포트 오류
```
ModuleNotFoundError: No module named 'state'
```
→ `sys.path.append()`로 프로젝트 루트 추가

### arXiv Rate Limit
```
HTTPError: Page request resulted in HTTP 429
```
→ `time.sleep()` 추가 또는 더미 데이터 사용

## 📚 참고

- LangChain: https://python.langchain.com/
- LangGraph: https://langchain-ai.github.io/langgraph/
- Groq: https://console.groq.com/
- arXiv API: https://info.arxiv.org/help/api/
