# 3-6) Final Response Agent
from states import AgentState
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from tools import build_role_tools
from prompts.system import make_system_prompt
from llm import get_llm

llm = get_llm()

ROLE_TOOLS = build_role_tools()
RESPONSE_TOOLS = ROLE_TOOLS["RESPONSE_TOOLS"]

RESPONSE_SYSTEM_PROMPT = (
    "You are a research gap analysis report writer.\n"
    "Write a structured final report based on all prior messages.\n\n"

    "=== OUTPUT FORMAT (follow exactly) ===\n\n"

    "## Research GAP Analysis Report\n"
    "### Query: <original user query>\n"
    "### (refined) <refined query from query_analysis>\n\n"

    "────────────────────────────────────────────────────────────────\n"
    "1) Related Papers (<N>편)\n"
    "────────────────────────────────────────────────────────────────\n"
    "| # | paper_id | Title | Year | Relevance |\n"
    "|---|----------|-------|------|-----------|\n"
    "| 1 | <id> | <title> | <year> | <one-line why it's relevant> |\n"
    "... (list ALL retrieved papers)\n\n"

    "────────────────────────────────────────────────────────────────\n"
    "2) Key Limitations  (축별 그룹)\n"
    "────────────────────────────────────────────────────────────────\n"
    "Read the axis_counter message and group limitations by axis.\n"
    "Format each line as:\n"
    "[<axis>]  <one-line summary of the recurring limitation>  (<N>건)\n\n"

    "────────────────────────────────────────────────────────────────\n"
    "3) Research Gaps & Proposed Topics  (중요도 순)\n"
    "────────────────────────────────────────────────────────────────\n"
    "Read the gap_infer message. List each GAP in FREQUENCY ORDER (highest first).\n"
    "For each GAP use this format:\n\n"
    "<stars>  GAP #<N> [<axis> · <count>개 논문]\n"
    "\"<proposed_topic title>\"\n"
    "→ <methodology>  |  <dataset>\n"
    "→ 목표: <objective>\n\n"
    "Stars: ★★★ for rank 1, ★★☆ for ranks 2-3, ★☆☆ for ranks 4+\n"
    "If a GAP has only 1 supporting paper, append: ⚠️ 근거 단일 논문\n\n"

    "────────────────────────────────────────────────────────────────\n"
    "4) Critic Scores\n"
    "────────────────────────────────────────────────────────────────\n"
    "Read the critic_score message and reproduce its scores and DECISION compactly:\n"
    "<metric> : <score> <✅ or ⚠️>  |  <metric> : <score> <✅ or ⚠️>\n"
    "<metric> : <score> <✅ or ⚠️>  |  DECISION: <ACCEPT/REDO/REFINE>\n"
    "Flags: <list any flags from critic>\n\n"

    "Finish with exactly: FINAL ANSWER\n"
)

final_response_agent = create_agent(
    model=llm,
    tools=RESPONSE_TOOLS,
    system_prompt=make_system_prompt(RESPONSE_SYSTEM_PROMPT),
)


def final_response_node(state: AgentState) -> AgentState:
    result = final_response_agent.invoke(state)
    last = AIMessage(content=result["messages"][-1].content, name="final_response")
    return {"messages": [last], "sender": "final_response"}
