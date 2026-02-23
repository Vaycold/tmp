#!/usr/bin/env python3
"""
Main entry point for GAPago LangGraph.

Usage:
    python main.py "Your research question"
    python main.py  # Interactive mode
"""
import os
import sys
import json
from datetime import datetime
from pathlib import Path

from config import config
from models import AgentState
from graph import build_graph


def select_llm_provider() -> str:
    """
    Interactive LLM provider selection.
    
    Returns:
        Selected provider name
    """
    providers = {
        "1": ("mock", "Mock (No API required, for testing)"),
        "2": ("openai", "OpenAI GPT (gpt-4o-mini)"),
        "3": ("bedrock_claude", "AWS Bedrock Claude (Claude 3.5 Sonnet)"),
        "4": ("gemini", "Google Gemini (gemini-2.0-flash)"),
        "5": ("exaone", "LG Exaone (Local model)")
    }
    
    print("\n" + "="*60)
    print("🤖 Select LLM Provider")
    print("="*60)
    
    for key, (provider, description) in providers.items():
        print(f"{key}. {description}")
    
    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice in providers:
            selected_provider, description = providers[choice]
            print(f"\n✓ Selected: {description}")
            
            # Validate credentials
            if not validate_provider_credentials(selected_provider):
                print(f"\n❌ Missing credentials for {selected_provider}.")
                print("Please set the required environment variables and try again.\n")
                continue
            
            return selected_provider
        else:
            print("❌ Invalid choice. Please enter 1-5.")


def validate_provider_credentials(provider: str) -> bool:
    """
    Validate that required credentials exist for the provider.
    
    Args:
        provider: Provider name
        
    Returns:
        True if credentials are valid
    """
    import os
    
    if provider == "mock":
        return True
    
    elif provider == "openai":
        missing = []
        if not os.getenv("AZURE_OPENAI_API_KEY"):
            missing.append("AZURE_OPENAI_API_KEY")
        if not os.getenv("AZURE_OPENAI_ENDPOINT"):
            missing.append("AZURE_OPENAI_ENDPOINT")
        if missing:
            print(f"\n⚠️ Required: {', '.join(missing)}")
            print("   Set in .env file:")
            print("     AZURE_OPENAI_API_KEY=your-api-key")
            print("     AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
            print("     AZURE_OPENAI_DEPLOYMENT=gpt-5.2-chat  # optional")
            print("     AZURE_OPENAI_API_VERSION=2024-12-01-preview  # optional")
            return False
        return True
    
    elif provider == "bedrock_claude":
        if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
            print("\n⚠️ Required: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            print("   Option 1: Run 'aws configure'")
            print("   Option 2: Set in .env file:")
            print("     AWS_ACCESS_KEY_ID=xxxxx")
            print("     AWS_SECRET_ACCESS_KEY=xxxxx")
            print("     AWS_REGION=us-east-1  # optional")
            return False
        return True
    
    elif provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            print("\n⚠️ Required: GOOGLE_API_KEY")
            print("   Get it from: https://makersuite.google.com/app/apikey")
            print("   Set it in .env file or export GOOGLE_API_KEY=xxxxx")
            return False
        return True
    
    elif provider == "exaone":
        if not os.getenv("EXAONE_MODEL_PATH"):
            print("\n⚠️ Required: EXAONE_MODEL_PATH")
            print("   Set it in .env file:")
            print("     EXAONE_MODEL_PATH=LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
            return False
        return True
    
    return True


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
    
    base_dir = os.path.dirname(os.path.abspath(__file__))  # gapago/ 경로
    out_dir = config.OUTPUT_DIR

    if not os.path.isabs(out_dir):
        out_dir = os.path.join(base_dir, out_dir)
        
    Path(out_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = Path(out_dir) / f"run_{timestamp}.json"
    
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
    
    # Select LLM Provider (if not already set via environment)
    if not config.LLM_PROVIDER or config.LLM_PROVIDER == "mock":
        provider = select_llm_provider()
        config.LLM_PROVIDER = provider
    else:
        print(f"\n🤖 Using LLM Provider: {config.LLM_PROVIDER} (from environment)")
        if not validate_provider_credentials(config.LLM_PROVIDER):
            print("\n❌ Missing credentials. Exiting.")
            sys.exit(1)
    
    # Get question
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("\n" + "="*60)
        print("📝 Enter your research question:")
        print("="*60)
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
    print(f"💾 Results saved to: {filepath}")
if __name__ == "__main__":
    main()