#!/usr/bin/env python3
"""
Main entry point for GAPago LangGraph.

Usage:
    python main.py "Your research question"
"""

import sys
import json
from datetime import datetime
from pathlib import Path

from config import config
from models import AgentState
from graph import build_graph


def format_output(state: AgentState) -> dict:
    """Format final state into output JSON."""
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


def save_output(output: dict) -> str:
    """Save output to JSON file."""
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = Path(config.OUTPUT_DIR) / f"run_{timestamp}.json"
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    return str(filepath)


def print_results(output: dict):
    """Pretty print results."""
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


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("🚀 GAPago LangGraph - Research Gap Analysis")
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
    print(f"🤖 LLM Provider: {config.LLM_PROVIDER}")
    
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
        max_iterations=config.MAX_ITERATIONS,
        route="",
        errors=[],
        trace={}
    )
    
    # Build and run graph
    try:
        graph = build_graph()
        final_state = graph.invoke(
            initial_state,
            config={"recursion_limit": 50}
        )
        
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