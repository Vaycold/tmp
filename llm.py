import os
from functools import lru_cache
from dotenv import load_dotenv

from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI

# from langchain_anthropic import ChatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


@lru_cache(maxsize=8)
def get_llm(provider: str | None = None, model: str | None = None) -> BaseChatModel:
    provider = (provider or os.getenv("LLM_PROVIDER", "azure")).lower()
    model = model or os.getenv("LLM_MODEL", "")

    if provider == "azure":
        # model은 보통 deployment랑 같게 두는 게 편함
        deployment = model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1-chat")
        return AzureChatOpenAI(
            openai_api_version=os.getenv(
                "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
            ),
            azure_deployment=deployment,
            # endpoint/api_key는 env에서 읽히도록 설정해두는게 보통 편함
        )

    if provider == "claude" or provider == "anthropic":
        # model 예: "claude-3-7-sonnet-latest"
        return ChatAnthropic(model=model or "claude-3-7-sonnet-latest")

    if provider == "gemini" or provider == "google":
        # model 예: "gemini-2.0-flash"
        return ChatGoogleGenerativeAI(model=model or "gemini-2.0-flash")

    if provider == "exaone":
        raise NotImplementedError(
            "EXAONE loader is project-specific (vLLM/transformers/API)."
        )

    raise ValueError(f"Unsupported provider: {provider}")
