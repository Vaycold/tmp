from dataclasses import dataclass, field
from typing import Optional
from langchain_core.runnables import RunnableConfig
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


@dataclass
class Configuration:
    # Search Settings
    tavily_max_results: int = field(
        default=os.getenv("TAVILY_MAX_RESULTS"),
        metadata={
            "description": "Maximum number of Tavily search results",
            "range": [1, 50],
        },
    )
    arxiv_max_docs: int = field(
        default=os.getenv("ARXIV_MAX_RESULTS"),
        metadata={"description": "Maximum number of ArXiv documents", "range": [1, 50]},
    )

    # (선택) 기타 런타임 옵션들 추가 가능
    # retrieval_timeout: int = 30
    # enable_checkpointing: bool = True

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        configurable = config.get("configurable", {}) if config else {}
        defaults = cls()
        return cls(
            tavily_max_results=configurable.get(
                "tavily_max_results", defaults.tavily_max_results
            ),
            arxiv_max_docs=configurable.get("arxiv_max_docs", defaults.arxiv_max_docs),
        )
