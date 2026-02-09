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
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    # Anthropic
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    
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


# Singleton instance
config = Config()