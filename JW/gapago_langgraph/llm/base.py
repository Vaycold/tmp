"""
Main LLM interface.
"""

from typing import Optional
from config import config
from .providers import (
    mock_llm,
    openai_llm,
    anthropic_llm,
    gemini_llm,
    exaone_llm
)


def llm_chat(messages: list[dict], model: Optional[str] = None) -> str:
    """
    LLM abstraction layer supporting multiple providers.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Optional model name override
        
    Returns:
        LLM response as string
    """
    provider = config.LLM_PROVIDER.lower()
    
    if provider == "mock":
        return mock_llm(messages)
    elif provider == "openai":
        return openai_llm(messages, model)
    elif provider == "anthropic":
        return anthropic_llm(messages, model)
    elif provider == "gemini":
        return gemini_llm(messages, model)
    elif provider == "exaone":
        return exaone_llm(messages, model)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider}. "
            f"Choose: mock, openai, anthropic, gemini, exaone"
        )