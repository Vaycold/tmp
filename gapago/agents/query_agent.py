"""
Query Analysis Agent (Enhanced).
- Adds: query evaluation + clarification questions
- Policy: do NOT add new top-level state keys (store extras under state["trace"] only)
"""

from models import AgentState
from llm import llm_chat, parse_json


def query_analysis_node(state: AgentState) -> AgentState:
    """
    Refine user question into search query.
    Also evaluate query quality and (if needed) generate clarifying questions.

    State writes (existing keys only):
      - state["refined_query"]
      - state["keywords"]
      - state["negative_keywords"]
      - state["trace"]["query_analysis"]
      - state["trace"]["query_eval"]            (new trace entry only)
      - state["trace"]["clarify"]               (new trace entry only)
    """
    if state["iteration"] > 0:
        print(f"\n🔄 Re-running Query Analysis (iteration {state['iteration']})")

    print(f"\n🔍 Query Analysis Node")

    # Ensure trace/errors exist (robustness)
    if "trace" not in state or state["trace"] is None:
        state["trace"] = {}
    if "errors" not in state or state["errors"] is None:
        state["errors"] = []

    # (Optional) If your orchestrator stores user's clarification somewhere,
    # you can append it to the prompt. To avoid adding new state keys here,
    # we only read from trace if present.
    # Example orchestrator behavior:
    #   state["trace"]["user_clarification"] = "..."   (outside this node)
    user_clarification = ""
    if isinstance(state["trace"], dict):
        user_clarification = str(state["trace"].get("user_clarification", "")).strip()

    prompt = f"""You are a research query optimizer for arXiv.

Given the research question, produce:
1) an optimized search query
2) an evaluation of the produced query quality
3) clarification questions if the query is underspecified

Research Question:
{state["user_question"]}

Additional User Clarification (if any):
{user_clarification if user_clarification else "(none)"}

Return JSON only with EXACT keys:
{{
  "refined_query": "Concise English search string (5-12 words)",
  "keywords": ["3-7 important terms/phrases in English"],
  "negative_keywords": ["0-4 terms to exclude"],
  "quality_score": 0.0,
  "issues": ["list of short issue strings"],
  "needs_clarification": true,
  "clarifying_questions": ["3-5 questions to ask user"]
}}

Rules:
- quality_score must be a float in [0, 1]
- needs_clarification must be true if quality_score < 0.60 OR if issues indicate ambiguity
- clarifying_questions must be empty list if needs_clarification is false
- Output JSON only, no markdown, no extra text
"""

    messages = [
        {"role": "system", "content": "You optimize academic search queries and interview users for missing constraints."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = llm_chat(messages)
        result = parse_json(response)

        # ---- Update existing state keys ----
        state["refined_query"] = result.get("refined_query", state["user_question"])
        state["keywords"] = result.get("keywords", [])
        state["negative_keywords"] = result.get("negative_keywords", [])

        # ---- Store evaluation/clarify under trace only (no new top-level keys) ----
        state["trace"]["query_analysis"] = state["refined_query"]
        state["trace"]["query_eval"] = {
            "quality_score": result.get("quality_score", None),
            "issues": result.get("issues", []),
        }
        state["trace"]["clarify"] = {
            "needed": bool(result.get("needs_clarification", False)),
            "questions": result.get("clarifying_questions", []),
        }

        print(f"  ✓ Refined: {state['refined_query']}")
        print(f"  ✓ Keywords: {', '.join(state['keywords'])}")

        # (Optional) show clarification questions for developer visibility
        if state["trace"]["clarify"]["needed"]:
            print("  ⚠️ Clarification needed. Questions:")
            for q in state["trace"]["clarify"]["questions"]:
                print(f"    - {q}")

    except Exception as e:
        state["errors"].append(f"Query analysis error: {str(e)}")
        state["refined_query"] = state.get("refined_query", state["user_question"])

        # Keep trace consistent even on failure
        state["trace"]["query_analysis"] = state["refined_query"]
        state["trace"]["query_eval"] = {
            "quality_score": None,
            "issues": [f"exception: {str(e)}"],
        }
        state["trace"]["clarify"] = {
            "needed": False,
            "questions": [],
        }

    return state
