"""
Configuration management for GAPago LangGraph.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from utils import logging  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

load_dotenv(ENV_PATH, override=False)
LANGSMITH_PROJECT = "GAPAGO"                  
logging.langsmith(LANGSMITH_PROJECT) 
class Config:
    """Global configuration."""
    
    # LLM Provider
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")
    
    # ── Azure OpenAI 설정 ──. 
    AZURE_OPENAI_API_KEY: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat")

    
    # AWS Bedrock Claude
    # AWS credentials are handled by boto3 automatically
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    BEDROCK_CLAUDE_MODEL: str = os.getenv(
        "BEDROCK_CLAUDE_MODEL", 
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0"  # 🔥 수정: 올바른 모델 ID
    )
    
    # Google
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    
    # Exaone
    EXAONE_MODEL_PATH: Optional[str] = os.getenv("EXAONE_MODEL_PATH")
    
    # Pipeline
    ARXIV_MAX_RESULTS: int = int(os.getenv("ARXIV_MAX_RESULTS", "50"))
    TOP_K_PAPERS: int = int(os.getenv("TOP_K_PAPERS", "10"))
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "2"))
    
    # Output
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "outputs")
    
    # GAP Axes
    # GAP_AXES: list[str] = [
    #     "data_dependency",
    #     "robustness",
    #     "scalability",
    #     "generalization",
    #     "evaluation_gap",
    #     "practicality",
    #     "interpretability",
    #     "methodology_gap"
    # ]

    # ── GAP 분석 고정 축 (5개) ──────────────────────────────────
    # 모든 연구 분야에 공통 적용되는 보편적 한계 기준
    # 동적 축은 gap_agent.py 내에서 LLM이 추가로 생성
    GAP_AXES_FIXED: dict = {
        "methodology": {
            "label": "방법론적 한계",
            "description": "연구에서 사용한 알고리즘, 모델 구조, 실험 설계 자체의 제약"
        },
        "data": {
            "label": "데이터 한계",
            "description": "데이터 부족, 편향, 특정 데이터셋 의존, 데이터 품질 문제"
        },
        "evaluation": {
            "label": "검증/평가 한계",
            "description": "실험 범위 협소, 평가 지표 부적절, 비교 실험 부족"
        },
        "generalization": {
            "label": "일반화 한계",
            "description": "특정 조건/환경/도메인에서만 작동, 다른 분야 적용 어려움"
        },
        "practicality": {
            "label": "실용성 한계",
            "description": "계산 비용, 구현 복잡도, 실제 배포/적용의 어려움, 윤리적 문제"
        },
    }

    # 동적 축 생성 설정
    GAP_AXES_DYNAMIC_MAX: int = 2      # 동적으로 추가할 최대 축 수
    GAP_AXES_DYNAMIC_MIN_PAPERS: int = 3  # 축으로 인정할 최소 논문 반복 수
    
    # SCIENCEON_API_KEY: Optional[str] = os.getenv("SCIENCEON_API_KEY")

# Singleton instance
config = Config()

