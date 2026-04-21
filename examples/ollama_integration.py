"""
examples/ollama_integration.py
================================
CogniField v10 — Ollama LLM Integration

Demonstrates using CogniField with a local Ollama model
to add natural language understanding and generation.

Prerequisites:
  1. Install Ollama: https://ollama.ai
  2. Pull a model:   ollama pull llama3
  3. Start server:   ollama serve  (or it starts automatically)

Run:
  python examples/ollama_integration.py
  python examples/ollama_integration.py --model mistral
  python examples/ollama_integration.py --check-only
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cognifield import CogniField
from cognifield.llm.base import OllamaClient


def check_ollama(model: str = "llama3", url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running and the model is available."""
    client = OllamaClient(model=model, base_url=url)
    available = client.is_available()
    if available:
        print(f"  ✓ Ollama server reachable at {url}")
        # Try a quick generation
        resp = client.generate("Say 'hello' in one word.", max_tokens=10)
        if "[Ollama error" not in resp:
            print(f"  ✓ Model '{model}' responding: {resp!r}")
            return True
        else:
            print(f"  ✗ Model '{model}' error: {resp}")
            return False
    else:
        print(f"  ✗ Ollama server not reachable at {url}")
        print("    → Install: https://ollama.ai")
        print(f"    → Pull:    ollama pull {model}")
        print("    → Start:   ollama serve")
        return False


def demo_with_ollama(model: str, url: str) -> None:
    """Full demo using real Ollama LLM."""
    print(f"\n  Using model: {model}")

    cf = CogniField({
        "agents":      3,
        "uncertainty": "low",
        "llm":         "ollama",
        "llm_model":   model,
        "llm_base_url": url,
    })

    # Teach some facts
    cf.teach("apple",       {"edible": True,  "category": "food"})
    cf.teach("stone",       {"edible": False, "category": "material"})
    cf.teach("purple_berry",{"edible": True,  "category": "food"})

    questions = [
        "Is the apple safe to eat?",
        "What should I do with the stone?",
        "Should I eat the purple berry?",
        "I found an unknown glowing mushroom. What should I do?",
    ]

    print(f"\n  {'Question':50s} | LLM Response")
    print(f"  {'─'*50} | {'─'*30}")

    for q in questions:
        result = cf.think(q)
        llm_out = result.get("llm_output", "")
        # Trim to first sentence for display
        first_sent = llm_out.split(".")[0] + "." if llm_out else "(no LLM output)"
        print(f"\n  Q: {q}")
        print(f"     CogniField decision: {result['decision']} "
              f"({result['confidence']:.0%})")
        print(f"     LLM says: {first_sent[:80]}")


def demo_without_ollama(model: str) -> None:
    """Fallback demo showing what would happen with Ollama."""
    print(f"\n  Ollama not available — showing mock output")
    print(f"  (Install Ollama + '{model}' for real LLM responses)\n")

    cf = CogniField({
        "agents":      3,
        "llm":         "mock",   # mock shows what the prompt/response would look like
    })
    cf.teach("apple", {"edible": True,  "category": "food"})
    cf.teach("stone", {"edible": False, "category": "material"})

    result = cf.think("Is the apple safe to eat?")
    print(f"  CogniField decision: {result['decision']} ({result['confidence']:.0%})")
    print(f"\n  Mock LLM prompt would look like:")
    from cognifield.llm.base import MockLLM, OllamaClient
    mock     = OllamaClient(model=model)
    mock_llm = MockLLM()
    ctx = {
        "decision":    result["decision"],
        "confidence":  result["confidence"],
        "beliefs":     {k: {"value": v["value"], "confidence": v["confidence"]}
                        for k, v in result.get("consensus", {}).items()},
        "uncertainty": "low",
        "strategy":    "explore",
    }
    prompt = mock_llm.format_decision_prompt("Is the apple safe to eat?", ctx)
    print(f"  {'─'*55}")
    for line in prompt.strip().split("\n")[:12]:
        print(f"    {line}")
    print(f"    ...")
    print(f"  {'─'*55}")
    print(f"\n  Mock response: {mock_llm.generate(prompt)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CogniField + Ollama integration demo"
    )
    parser.add_argument("--model",      default="llama3", help="Ollama model")
    parser.add_argument("--url",        default="http://localhost:11434",
                        help="Ollama server URL")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check if Ollama is available")
    args = parser.parse_args()

    print("=" * 60)
    print("  CogniField v10 — Ollama Integration")
    print("=" * 60)
    print(f"\n  Checking Ollama availability...")

    available = check_ollama(args.model, args.url)

    if args.check_only:
        sys.exit(0 if available else 1)

    if available:
        demo_with_ollama(args.model, args.url)
    else:
        demo_without_ollama(args.model)

    print("\n✓ Ollama integration demo complete.")


if __name__ == "__main__":
    main()
