#!/usr/bin/env python3
"""
agent_baseline.py

Single-file LangGraph MVP for Research Gap Analysis.
Processes: Query → arXiv Search → BM25 Ranking → Limitation Extraction → GAP Generation → Critic → Conditional Routing

Usage:
    python3 agent_baseline.py "Your research question here"
    
Requirements:
    pip install langgraph pydantic requests rank-bm25 python-dotenv
"""

import os
import sys
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict
from xml.etree import ElementTree as ET

import requests
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

LLM_MODE = os.getenv("LLM_MODE", "mock")  # mock or real
ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "50"))
TOP_K_PAPERS = int(os.getenv("TOP_K_PAPERS", "10"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "2"))

# Fixed axis categories
GAP_AXES = [
    "data_dependency",
    "robustness",
    "scalability",
    "generalization",
    "evaluation_gap",
    "practicality",
    "interpretability",
    "methodology_gap"
]

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class Paper(BaseModel):
    """Individual paper metadata from arXiv."""
    paper_id: str
    title: str
    abstract: str
    url: str
    year: int
    authors: list[str] = Field(default_factory=list)
    score_bm25: float = 0.0


class LimitationItem(BaseModel):
    """Extracted limitation from a paper."""
    paper_id: str
    claim: str
    evidence_quote: str


class GapCandidate(BaseModel):
    """Research gap identified from limitations."""
    axis: str
    gap_statement: str
    supporting_papers: list[str] = Field(default_factory=list)
    supporting_quotes: list[str] = Field(default_factory=list)


class CriticScores(BaseModel):
    """Quality scores for the analysis."""
    query_specificity: float = Field(0.0, ge=0.0, le=1.0)
    paper_relevance: float = Field(0.0, ge=0.0, le=1.0)
    groundedness: float = Field(0.0, ge=0.0, le=1.0)


class AgentState(TypedDict):
    """LangGraph state for the pipeline."""
    user_question: str
    refined_query: str
    keywords: list[str]
    negative_keywords: list[str]
    papers: list[Paper]
    limitations: list[LimitationItem]
    gaps: list[GapCandidate]
    critic: Optional[CriticScores]
    iteration: int
    max_iterations: int
    route: str
    errors: list[str]
    trace: dict


# ============================================================================
# LLM ABSTRACTION (MOCK/REAL)
# ============================================================================

def llm_chat(messages: list[dict], model: Optional[str] = None) -> str:
    """
    LLM abstraction layer.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Optional model name
        
    Returns:
        LLM response as string
    """
    if LLM_MODE == "mock":
        return _mock_llm(messages)
    elif LLM_MODE == "real":
        raise NotImplementedError("Real LLM integration not implemented yet. Set LLM_MODE=mock")
    else:
        raise ValueError(f"Unknown LLM_MODE: {LLM_MODE}")


def _mock_llm(messages: list[dict]) -> str:
    """
    Mock LLM for testing without API.
    Returns heuristic-based JSON responses.
    """
    content = messages[-1]["content"].lower()
    
    # Query refinement
    if "refine" in content and "search query" in content:
        question_match = re.search(r"research question:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
        question = question_match.group(1).strip() if question_match else "research question"
        
        # Extract key terms
        words = re.findall(r'\b\w{4,}\b', question.lower())
        keywords = list(set(words[:5]))
        
        return json.dumps({
            "refined_query": " ".join(keywords[:3]),
            "keywords": keywords[:5],
            "negative_keywords": ["review", "survey"]
        })
    
    # Limitation extraction
    elif "limitation" in content or "future work" in content:
        # Extract abstract if present
        abstract_match = re.search(r"abstract:\s*(.+?)(?:\n\n|$)", content, re.IGNORECASE | re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            sentences = re.split(r'[.!?]+', abstract)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            
            limitations = []
            # Heuristic: look for limitation keywords
            limit_keywords = ["however", "limitation", "future", "challenge", "difficult", "cannot", "limited"]
            
            for sent in sentences:
                if any(kw in sent.lower() for kw in limit_keywords):
                    limitations.append({
                        "claim": f"Limitation identified: {sent[:80]}...",
                        "evidence_quote": sent[:150]
                    })
                    if len(limitations) >= 2:
                        break
            
            if not limitations and sentences:
                # Fallback: use last sentence
                limitations.append({
                    "claim": f"Potential limitation: {sentences[-1][:80]}...",
                    "evidence_quote": sentences[-1][:150]
                })
            
            return json.dumps({"limitations": limitations})
    
    # Axis classification
    elif "classify" in content and "axis" in content:
        # Simple heuristic mapping
        claim_lower = content.lower()
        
        if "data" in claim_lower or "dataset" in claim_lower:
            axis = "data_dependency"
        elif "robust" in claim_lower or "noise" in claim_lower:
            axis = "robustness"
        elif "scale" in claim_lower or "large" in claim_lower:
            axis = "scalability"
        elif "general" in claim_lower or "transfer" in claim_lower:
            axis = "generalization"
        elif "evaluat" in claim_lower or "metric" in claim_lower:
            axis = "evaluation_gap"
        elif "practic" in claim_lower or "deploy" in claim_lower:
            axis = "practicality"
        elif "interpret" in claim_lower or "explain" in claim_lower:
            axis = "interpretability"
        else:
            axis = "methodology_gap"
        
        return json.dumps({"axis": axis})
    
    # GAP generation
    elif "research gap" in content or "gap statement" in content:
        axis_match = re.search(r"'(\w+)'", content)
        axis = axis_match.group(1) if axis_match else "methodology_gap"
        
        gap_templates = {
            "data_dependency": "Current approaches require large-scale datasets which limits applicability to low-resource scenarios.",
            "robustness": "Existing models lack robustness to domain shifts and adversarial perturbations.",
            "scalability": "Computational requirements prevent scaling to larger problem instances.",
            "generalization": "Models demonstrate limited generalization across different domains and tasks.",
            "evaluation_gap": "Evaluation metrics do not adequately capture real-world performance.",
            "practicality": "Deployment barriers exist due to resource constraints and integration challenges.",
            "interpretability": "Lack of interpretability hinders trust and adoption in critical applications.",
            "methodology_gap": "Methodological limitations restrict the scope of applicable scenarios."
        }
        
        return json.dumps({
            "gap_statement": gap_templates.get(axis, "Research gap identified in this area.")
        })
    
    # Default
    return json.dumps({"result": "mock response"})


def parse_json(text: str) -> dict:
    """
    Parse JSON from LLM response with error recovery.
    
    Args:
        text: Raw LLM response
        
    Returns:
        Parsed JSON dict
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try finding JSON object
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return {}


# ============================================================================
# ARXIV CLIENT
# ============================================================================

def search_arxiv(query: str, max_results: int = 50) -> list[Paper]:
    """
    Search arXiv API and return papers.
    
    Args:
        query: Search query
        max_results: Maximum results to retrieve
        
    Returns:
        List of Paper objects
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        
        papers = []
        for entry in root.findall("atom:entry", ns):
            try:
                paper_id = entry.find("atom:id", ns).text.split("/abs/")[-1]
                title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
                
                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns)
                    if name is not None:
                        authors.append(name.text)
                
                published = entry.find("atom:published", ns).text
                year = int(published[:4])
                
                url = f"https://arxiv.org/abs/{paper_id}"
                
                papers.append(Paper(
                    paper_id=paper_id,
                    title=title,
                    abstract=abstract,
                    url=url,
                    year=year,
                    authors=authors
                ))
            except (AttributeError, ValueError):
                continue
        
        return papers
        
    except requests.RequestException as e:
        print(f"⚠️ arXiv API error: {e}")
        return []


# ============================================================================
# TEXT UTILITIES
# ============================================================================

def tokenize(text: str) -> list[str]:
    """Simple tokenization for BM25."""
    return text.lower().split()


# ============================================================================
# LANGGRAPH NODES
# ============================================================================

def query_analysis_node(state: AgentState) -> AgentState:
    """
    Node: Refine user question into search query.
    """
    # iteration이 0보다 크면 재실행 중
    if state["iteration"] > 0:
        print(f"\n🔄 Re-running Query Analysis (iteration {state['iteration']})")

    print(f"\n🔍 Query Analysis Node (iteration {state['iteration']})")
    
    prompt = f"""Given this research question, generate an optimized arXiv search query.

Research Question: {state['user_question']}

Output a JSON with:
- refined_query: Concise search string (5-10 words)
- keywords: List of 3-5 important terms
- negative_keywords: List of 1-3 terms to exclude

Output JSON only:"""
    
    messages = [
        {"role": "system", "content": "You are a research query optimizer."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = llm_chat(messages)
        result = parse_json(response)
        
        state["refined_query"] = result.get("refined_query", state["user_question"])
        state["keywords"] = result.get("keywords", [])
        state["negative_keywords"] = result.get("negative_keywords", [])
        
        print(f"  ✓ Refined: {state['refined_query']}")
        print(f"  ✓ Keywords: {', '.join(state['keywords'])}")
        
    except Exception as e:
        state["errors"].append(f"Query analysis error: {str(e)}")
        state["refined_query"] = state["user_question"]
    
    state["trace"]["query_analysis"] = state["refined_query"]
    return state


def paper_retrieval_node(state: AgentState) -> AgentState:
    """
    Node: Search arXiv and rank with BM25.
    """
    if state["iteration"] > 0:
        print(f"\n🔄 Re-running Retrieval (iteration {state['iteration']})")
    
    print(f"\n📚 Paper Retrieval Node")

    
    # Search arXiv
    papers = search_arxiv(state["refined_query"], max_results=ARXIV_MAX_RESULTS)
    
    if not papers:
        print("  ⚠️ No papers found")
        state["papers"] = []
        state["trace"]["papers_retrieved"] = 0
        return state
    
    print(f"  ✓ Retrieved {len(papers)} papers")
    
    # BM25 ranking
    corpus = [tokenize(p.abstract) for p in papers]
    bm25 = BM25Okapi(corpus)
    
    query_tokens = tokenize(state["refined_query"])
    scores = bm25.get_scores(query_tokens)
    
    # Sort and take top-K
    paper_scores = list(zip(papers, scores))
    paper_scores.sort(key=lambda x: x[1], reverse=True)
    top_papers = paper_scores[:TOP_K_PAPERS]
    
    # Update scores
    for paper, score in top_papers:
        paper.score_bm25 = float(score)
    
    state["papers"] = [p for p, _ in top_papers]
    print(f"  ✓ Selected top {len(state['papers'])} by BM25")
    
    state["trace"]["papers_retrieved"] = len(state["papers"])
    state["trace"]["avg_bm25"] = sum(p.score_bm25 for p in state["papers"]) / len(state["papers"]) if state["papers"] else 0
    
    return state


def limitation_extract_node(state: AgentState) -> AgentState:
    """
    Node: Extract limitations from paper abstracts.
    """
    print(f"\n🔬 Limitation Extraction Node")
    
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


def gap_infer_node(state: AgentState) -> AgentState:
    """
    Node: Infer research gaps from limitations.
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
            prompt = f"""Classify this limitation into ONE category: {', '.join(GAP_AXES)}

Limitation: {lim.claim}

Output JSON: {{"axis": "category_name"}}"""
            
            messages = [
                {"role": "system", "content": "You are a research classifier."},
                {"role": "user", "content": prompt}
            ]
            
            response = llm_chat(messages)
            result = parse_json(response)
            axis = result.get("axis", "methodology_gap")
            
            if axis not in GAP_AXES:
                axis = "methodology_gap"
            
            axis_mapping[id(lim)] = axis
        
        except Exception as e:
            axis_mapping[id(lim)] = "methodology_gap"
    
    # Count by axis
    from collections import Counter
    axis_counts = Counter(axis_mapping.values())
    top_axes = [axis for axis, _ in axis_counts.most_common(3)]
    
    print(f"  ✓ Top axes: {', '.join(top_axes)}")
    
    # Generate gaps
    gaps = []
    
    for axis in top_axes:
        # Get limitations for this axis
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


def critic_score_node(state: AgentState) -> AgentState:
    """
    Node: Calculate quality scores.
    """
    print(f"\n⭐ Critic Scoring Node")
    
    # Query specificity
    query_len = len(state["refined_query"].split())
    keyword_count = len(state["keywords"])
    query_spec = min((query_len / 10.0) * 0.6 + (keyword_count / 5.0) * 0.4, 1.0)
    
    # Paper relevance
    if state["papers"]:
        avg_bm25 = sum(p.score_bm25 for p in state["papers"]) / len(state["papers"])
        paper_rel = min(avg_bm25 / 50.0, 1.0)
    else:
        paper_rel = 0.0
    
    # Groundedness
    if state["limitations"]:
        with_evidence = sum(1 for lim in state["limitations"] if lim.evidence_quote)
        grounded = with_evidence / len(state["limitations"])
    else:
        grounded = 0.0
    
    state["critic"] = CriticScores(
        query_specificity=query_spec,
        paper_relevance=paper_rel,
        groundedness=grounded
    )
    
    print(f"  ✓ Query Specificity: {query_spec:.2f}")
    print(f"  ✓ Paper Relevance: {paper_rel:.2f}")
    print(f"  ✓ Groundedness: {grounded:.2f}")
    
    state["trace"]["critic_scores"] = {
        "query_specificity": query_spec,
        "paper_relevance": paper_rel,
        "groundedness": grounded
    }

    state["iteration"] += 1
    
    state["trace"]["critic_scores"] = {
        "query_specificity": query_spec,
        "paper_relevance": paper_rel,
        "groundedness": grounded
    }
    return state


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================

def route_decision(state: AgentState) -> str:
    """
    Decide next route based on critic scores.
    
    Returns:
        "refine_query" | "redo_retrieval" | "accept"
    """
    # ⚠️ 주의: state를 직접 수정하지 말고, 읽기만 해야 함
    
    # Check max iterations
    if state["iteration"] >= state["max_iterations"]:
        print(f"\n🔀 Router: Max iterations reached → ACCEPT")
        return "accept"
    
    critic = state["critic"]
    
    # critic이 None인 경우 처리
    if critic is None:
        print(f"\n🔀 Router: No critic scores → ACCEPT")
        return "accept"
    
    # Decision logic
    if critic.query_specificity < 0.55:
        print(f"\n🔀 Router: Low query specificity ({critic.query_specificity:.2f}) → REFINE")
        return "refine_query"
    
    if critic.paper_relevance < 0.55:
        print(f"\n🔀 Router: Low paper relevance ({critic.paper_relevance:.2f}) → REDO RETRIEVAL")
        return "redo_retrieval"
    
    if critic.groundedness < 0.60:
        print(f"\n🔀 Router: Low groundedness ({critic.groundedness:.2f}) → REDO RETRIEVAL")
        return "redo_retrieval"
    
    print(f"\n🔀 Router: All scores acceptable → ACCEPT")
    return "accept"


# ============================================================================
# LANGGRAPH CONSTRUCTION
# ============================================================================

def build_graph() -> StateGraph:
    """
    Build LangGraph workflow.
    
    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("paper_retrieval", paper_retrieval_node)
    workflow.add_node("limitation_extract", limitation_extract_node)
    workflow.add_node("gap_infer", gap_infer_node)
    workflow.add_node("critic_score", critic_score_node)
    
    # Define edges
    workflow.set_entry_point("query_analysis")
    workflow.add_edge("query_analysis", "paper_retrieval")
    workflow.add_edge("paper_retrieval", "limitation_extract")
    workflow.add_edge("limitation_extract", "gap_infer")
    workflow.add_edge("gap_infer", "critic_score")
    
    # Conditional routing from critic
    workflow.add_conditional_edges(
        "critic_score",
        route_decision,
        {
            "refine_query": "query_analysis",
            "redo_retrieval": "paper_retrieval",
            "accept": END
        }
    )
    
    return workflow.compile()


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_output(state: AgentState) -> dict:
    """
    Format final state into output JSON.
    
    Args:
        state: Final agent state
        
    Returns:
        Output dict
    """
    return {
        "question": state["user_question"],
        "query": state["refined_query"],
        "gaps": [
            {
                "axis": gap.axis,
                "gap_statement": gap.gap_statement,
                "supporting_papers": gap.supporting_papers,
                "supporting_quotes": gap.supporting_quotes
            }
            for gap in state["gaps"]
        ],
        "critic": {
            "query_specificity": state["critic"].query_specificity if state["critic"] else 0.0,
            "paper_relevance": state["critic"].paper_relevance if state["critic"] else 0.0,
            "groundedness": state["critic"].groundedness if state["critic"] else 0.0
        },
        "iteration": state["iteration"],
        "route": state["route"],
        "errors": state["errors"],
        "trace": state["trace"]
    }


def save_output(output: dict, output_dir: str = "outputs") -> str:
    """
    Save output to JSON file.
    
    Args:
        output: Output dict
        output_dir: Directory path
        
    Returns:
        Saved file path
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = Path(output_dir) / f"run_{timestamp}.json"
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    return str(filepath)


def print_results(output: dict):
    """
    Pretty print results.
    
    Args:
        output: Output dict
    """
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    print(f"\n❓ Question: {output['question']}")
    print(f"🔍 Query: {output['query']}")
    print(f"🔄 Iterations: {output['iteration']}")
    print(f"🎯 Final Route: {output['route']}")
    
    if output['errors']:
        print(f"\n⚠️ Errors ({len(output['errors'])}):")
        for err in output['errors'][:3]:
            print(f"  - {err}")
    
    print(f"\n💡 Research Gaps ({len(output['gaps'])}):\n")
    
    for i, gap in enumerate(output['gaps'], 1):
        print(f"{i}. [{gap['axis']}]")
        print(f"   {gap['gap_statement']}")
        print(f"   📄 Papers: {len(gap['supporting_papers'])}")
        print(f"   💬 Quotes: {len(gap['supporting_quotes'])}")
        print()
    
    critic = output['critic']
    print("⭐ Quality Scores:")
    print(f"   Query Specificity: {critic['query_specificity']:.2f}")
    print(f"   Paper Relevance: {critic['paper_relevance']:.2f}")
    print(f"   Groundedness: {critic['groundedness']:.2f}")
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("🚀 LangGraph MVP - Research Gap Analysis")
    print("="*60)
    
    # Get question
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("\nEnter your research question:")
        question = input("> ").strip()
        if not question:
            print("❌ No question provided. Exiting.")
            sys.exit(1)
    
    print(f"\n📝 Question: {question}")
    
    # Initialize state
    initial_state = AgentState(
        user_question=question,
        refined_query="",
        keywords=[],
        negative_keywords=[],
        papers=[],
        limitations=[],
        gaps=[],
        critic=None,
        iteration=0,
        max_iterations=MAX_ITERATIONS,
        route="",
        errors=[],
        trace={}
    )
    
    # Build and run graph
    try:
        graph = build_graph()
        final_state = graph.invoke(initial_state)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Format output
    output = format_output(final_state)
    
    # Print results
    print_results(output)
    
    # Save output
    filepath = save_output(output)
    print(f"💾 Results saved to: {filepath}\n")


if __name__ == "__main__":
    main()