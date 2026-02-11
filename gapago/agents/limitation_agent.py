"""
Limitation Extraction Agent.
"""

from models import AgentState, LimitationItem
from llm import llm_chat, parse_json


def limitation_extract_node(state: AgentState) -> AgentState:
    """
    Extract limitations from paper abstracts.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state
    """
    print(f"\n🔬 Limitation Extraction Node")
    
    if not state["papers"]:
        print("  ⚠️ No papers to analyze")
        return state
    
    limitations = []
    
    for paper in state["papers"]:
        try:
            prompt = f"""Extract 1-2 key limitations from this abstract.

Title: {paper.title}

Abstract: {paper.abstract}

Output JSON:
{{
  "limitations": [
    {{
      "claim": "Brief limitation statement",
      "evidence_quote": "Exact quote from abstract"
    }}
  ]
}}

Output JSON only:"""
            
            messages = [
                {"role": "system", "content": "You are a research analysis assistant."},
                {"role": "user", "content": prompt}
            ]
            
            response = llm_chat(messages)
            result = parse_json(response)
            
            for lim in result.get("limitations", []):
                limitations.append(LimitationItem(
                    paper_id=paper.paper_id,
                    claim=lim.get("claim", ""),
                    evidence_quote=lim.get("evidence_quote", "")
                ))
        
        except Exception as e:
            state["errors"].append(f"Limitation extraction error for {paper.paper_id}: {str(e)}")
            continue
    
    state["limitations"] = limitations
    print(f"  ✓ Extracted {len(limitations)} limitations")
    
    state["trace"]["limitations_extracted"] = len(limitations)
    return state