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
logging.langsmith()


@dataclass
class Configuration:
    tavily_max_results: int = field(
        default=int(os.getenv("TAVILY_MAX_RESULTS")),
        metadata={
            "description": "Maximum number of Tavily search results",
            "range": [1, 50],
        },
    )
    arxiv_max_docs: int = field(
        default=int(os.getenv("ARXIV_MAX_RESULTS")),
        metadata={"description": "Maximum number of ArXiv documents", "range": [1, 50]},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        configurable = config.get("configurable", {}) if config else {}
        defaults = cls()

        return cls(
            tavily_max_results=int(
                configurable.get("tavily_max_results", defaults.tavily_max_results)
            ),
            arxiv_max_docs=int(
                configurable.get("arxiv_max_docs", defaults.arxiv_max_docs)
            ),
        )
