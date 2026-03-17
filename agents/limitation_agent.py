# 3-3) Limitation Extract Agent
from states import AgentState
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from tools import build_role_tools
from prompts.system import make_system_prompt
from llm import get_llm

llm = get_llm()


ROLE_TOOLS = build_role_tools()
LIMITATION_TOOLS = ROLE_TOOLS["LIMITATION_TOOLS"]

limitation_extract_agent = create_agent(
    llm,
    tools=LIMITATION_TOOLS,
    system_prompt=make_system_prompt(
        "ROLE: Limitation Extract Agent\n"
        "You extract limitation/future-work statements from retrieved papers/snippets.\n"
        "RULES:\n"
        "1. Process ALL papers in the input without exception.\n"
        "2. Extract 1-2 key limitations per paper. No more, no less.\n"
        "3. Each limitation MUST include:\n"
        "   - paper_id: the paper's unique identifier\n"
        "   - claim: a brief limitation statement (1-2 sentences)\n"
        "   - evidence_quote: an exact quote from the provided text sections\n"
        "4. Do NOT infer or assume limitations not stated in the text.\n"
        "5. Do NOT skip any paper even if the abstract is short or unclear.\n"
        "6. Do NOT infer gaps yet.\n\n"

        "SECTION PRIORITY (high to low):\n"
        "  1. INTRODUCTION  — author-defined gaps, most reliable\n"
        "  2. CONCLUSION    — key contributions + limitations\n"
        "  3. LIMITATIONS   — author-stated weaknesses\n"
        "  4. DISCUSSION    — result interpretation + limitations\n"
        "  5. ABSTRACT      — fallback only, least detail\n"
        "  6. FUTURE_WORK   — supplementary evidence only\n\n"

        "OUTPUT FORMAT (strictly follow):\n"
        "paper_id: <id>\n"
        "  - claim: <limitation statement>\n"
        "    evidence_quote: <exact quote from paper>\n"
        "  - claim: <limitation statement>\n"
        "    evidence_quote: <exact quote from paper>\n"
        "Output be structured: paper_id -> [limitation_sentences], plus brief rationale.\n"
        "Do NOT infer gaps yet.\n"
    ),
)


def limitation_extract_node(state: AgentState) -> AgentState:
    result = limitation_extract_agent.invoke(state)
    last = AIMessage(content=result["messages"][-1].content, name="limitation_extract")
    return {"messages": [last], "sender": "limitation_extract"}