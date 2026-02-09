"""
GAP Inference Agent.
"""

from collections import Counter
from models import AgentState, GapCandidate
from llm import llm_chat, parse_json
from config import config


def gap_infer_node(state: AgentState) -> AgentState:
    """
    Infer research gaps from limitations.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state
    """
    print(f"\n💡 GAP Inference Node")
    
    if not state["limitations"]:
        print("  ⚠️ No limitations to analyze")
        state["gaps"] = []
        return state
    
    # Classify limitations by axis
    axis_mapping = {}
    
    for lim in state["limitations"]:
        try:
            prompt = f"""Classify this limitation into ONE category: {', '.join(config.GAP_AXES)}

Limitation: {lim.claim}

Output JSON: {{"axis": "category_name"}}"""
            
            messages = [
                {"role": "system", "content": "You are a research classifier."},
                {"role": "user", "content": prompt}
            ]
            
            response = llm_chat(messages)
            result = parse_json(response)
            axis = result.get("axis", "methodology_gap")
            
            if axis not in config.GAP_AXES:
                axis = "methodology_gap"
            
            axis_mapping[id(lim)] = axis
        
        except Exception as e:
            axis_mapping[id(lim)] = "methodology_gap"
    
    # Count by axis
    axis_counts = Counter(axis_mapping.values())
    top_axes = [axis for axis, _ in axis_counts.most_common(3)]
    
    print(f"  ✓ Top axes: {', '.join(top_axes)}")
    
    # Generate gaps
    gaps = []
    
    for axis in top_axes:
        axis_lims = [
            lim for lim in state["limitations"]
            if axis_mapping.get(id(lim)) == axis
        ]
        
        if not axis_lims:
            continue
        
        try:
            claims_text = "\n".join([f"- {lim.claim}" for lim in axis_lims[:5]])
            
            prompt = f"""Generate a research gap statement for '{axis}' based on:

{claims_text}

Output JSON: {{"gap_statement": "one-sentence gap description"}}"""
            
            messages = [
                {"role": "system", "content": "You are a research gap analyst."},
                {"role": "user", "content": prompt}
            ]
            
            response = llm_chat(messages)
            result = parse_json(response)
            
            gap_statement = result.get("gap_statement", f"Gap in {axis} across multiple papers")
            
            gaps.append(GapCandidate(
                axis=axis,
                gap_statement=gap_statement,
                supporting_papers=[lim.paper_id for lim in axis_lims],
                supporting_quotes=[lim.evidence_quote for lim in axis_lims if lim.evidence_quote]
            ))
        
        except Exception as e:
            state["errors"].append(f"GAP generation error for {axis}: {str(e)}")
            continue
    
    state["gaps"] = gaps
    print(f"  ✓ Generated {len(gaps)} gaps")
    
    state["trace"]["gaps_generated"] = len(gaps)
    return state