"""
Configuration management for GAPago LangGraph.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Global configuration."""
    
    # LLM Provider
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")
    
    # ── Azure OpenAI 설정 ──
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
    GAP_AXES: list[str] = [
        "data_dependency",
        "robustness",
        "scalability",
        "generalization",
        "evaluation_gap",
        "practicality",
        "interpretability",
        "methodology_gap"
    ]
    # SCIENCEON_API_KEY: Optional[str] = os.getenv("SCIENCEON_API_KEY")

# Singleton instance
config = Config()