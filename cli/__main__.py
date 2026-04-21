"""
cli/__main__.py
================
CogniField CLI — python -m cognifield

Usage:

    # Think about a question
    python -m cognifield "Is this berry safe to eat?"

    # Think with more agents
    python -m cognifield "Should I cross this bridge?" --agents 5

    # Decide (with risk level)
    python -m cognifield "Eat the red fruit?" --mode decide

    # Simulate a scenario
    python -m cognifield "Foraging in unknown forest" --mode simulate --steps 8

    # With Ollama LLM
    python -m cognifield "Is this safe?" --llm ollama

    # With API LLM
    python -m cognifield "Is this safe?" --llm api --llm-key sk-...

    # JSON output
    python -m cognifield "Is this safe?" --json

    # Quiet (decision + confidence only)
    python -m cognifield "Is this safe?" --quiet
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from typing import Any, Dict


def print_result(result: Dict, mode: str = "think", quiet: bool = False,
                 as_json: bool = False) -> None:
    """Pretty-print a CogniField result."""
    if as_json:
        print(json.dumps(result, indent=2, default=str))
        return

    if quiet:
        decision   = result.get("decision", "?")
        confidence = result.get("confidence", 0)
        risk       = result.get("risk_level", "")
        suffix     = f" | risk={risk}" if risk else ""
        print(f"{decision}  (confidence={confidence:.0%}{suffix})")
        return

    # Full output
    W = 62
    print("─" * W)

    decision   = result.get("decision", "unknown").upper()
    confidence = result.get("confidence", 0)
    conf_bar   = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
    print(f"  DECISION:    {decision}")
    print(f"  CONFIDENCE:  {conf_bar} {confidence:.0%}")

    if mode == "decide":
        action     = result.get("action", "")
        risk       = result.get("risk_level", "")
        risk_color = {"low": "✅", "medium": "⚠️ ", "high": "🔴", "critical": "🚨"}.get(risk, "")
        if action:
            print(f"  ACTION:      {action}")
        if risk:
            print(f"  RISK LEVEL:  {risk_color} {risk.upper()}")
        alts = result.get("alternatives", [])
        if alts:
            print(f"  ALTERNATIVES: {', '.join(alts)}")

    if mode == "simulate":
        sr = result.get("success_rate", 0)
        bc = result.get("belief_changes", 0)
        outcomes = result.get("outcomes", [])
        print(f"  SUCCESS RATE: {sr:.0%}")
        print(f"  BELIEFS UPDATED: {bc}")
        if outcomes:
            print("  OUTCOMES:")
            for o in outcomes[:4]:
                print(f"    {o}")

    reasoning = result.get("reasoning", [])
    if reasoning:
        print(f"\n  REASONING:")
        for r in reasoning:
            wrapped = textwrap.fill(r, width=W - 6,
                                    initial_indent="    • ",
                                    subsequent_indent="      ")
            print(wrapped)

    consensus = result.get("consensus", {})
    if consensus:
        print(f"\n  CONSENSUS ({len(consensus)} beliefs):")
        for k, v in list(consensus.items())[:4]:
            conf_c = v.get("confidence", 0)
            val    = v.get("value")
            agr    = v.get("agreement", 0)
            print(f"    {k:25s} = {str(val):6s} "
                  f"(conf={conf_c:.2f}, agree={agr:.0%})")

    llm_output = result.get("llm_output", "")
    if llm_output and "[Mock" not in llm_output and "error" not in llm_output.lower():
        print(f"\n  LLM INSIGHT:")
        wrapped = textwrap.fill(llm_output, width=W - 6,
                                initial_indent="    ",
                                subsequent_indent="    ")
        print(wrapped)

    meta = result.get("meta", {})
    elapsed = result.get("elapsed_ms", 0)
    agents  = meta.get("agents", "?")
    strat   = meta.get("strategy", "?")
    unc     = meta.get("uncertainty", "?")
    print(f"\n  META: agents={agents}, strategy={strat}, "
          f"uncertainty={unc}, time={elapsed:.0f}ms")
    print("─" * W)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="cognifield",
        description="CogniField — Multi-Agent Cognitive Reasoning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python -m cognifield "Is this berry safe?"
          python -m cognifield "Should I eat this?" --mode decide
          python -m cognifield "Foraging in forest" --mode simulate --steps 8
          python -m cognifield "Is this safe?" --agents 5 --uncertainty medium
          python -m cognifield "Is this safe?" --llm ollama --quiet
          python -m cognifield "Is this safe?" --json
        """),
    )

    parser.add_argument("input",      nargs="?", help="Input text / question")
    parser.add_argument("--mode",     default="think",
                        choices=["think", "decide", "simulate"],
                        help="Operation mode (default: think)")
    parser.add_argument("--steps",    default=8, type=int,
                        help="Simulation steps (mode=simulate, default: 8)")
    parser.add_argument("--agents",   default=3, type=int,
                        help="Number of reasoning agents (default: 3)")
    parser.add_argument("--uncertainty", default="low",
                        choices=["none","low","medium","high","chaotic"],
                        help="Uncertainty level (default: low)")
    parser.add_argument("--llm",      default="mock",
                        choices=["mock","ollama","api"],
                        help="LLM backend (default: mock)")
    parser.add_argument("--llm-model",default="llama3",
                        help="LLM model name")
    parser.add_argument("--llm-url",  default="http://localhost:11434",
                        help="Ollama base URL")
    parser.add_argument("--llm-key",  default="",
                        help="API key for remote LLM")
    parser.add_argument("--quiet",    action="store_true",
                        help="Print decision + confidence only")
    parser.add_argument("--json",     action="store_true",
                        help="Output raw JSON")
    parser.add_argument("--status",   action="store_true",
                        help="Print system status and exit")
    parser.add_argument("--version",  action="store_true",
                        help="Print version and exit")

    args = parser.parse_args()

    # Version
    if args.version:
        import cognifield; __version__ = cognifield.__version__
        print(f"CogniField v{__version__}")
        return 0

    # Build CogniField
    cfg = {
        "agents":       args.agents,
        "uncertainty":  args.uncertainty,
        "llm":          args.llm,
        "llm_model":    args.llm_model,
        "llm_base_url": args.llm_url,
        "llm_api_key":  args.llm_key,
    }

    from cognifield import CogniField
    cf = CogniField(cfg)

    # Status only
    if args.status:
        status = cf.status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"CogniField v{status['version']}")
            print(f"  Agents:   {status['agents']}")
            print(f"  LLM:      {status['llm_backend']}")
            print(f"  Config:   {status['config']}")
        return 0

    # Input required for reasoning modes
    input_text = args.input
    if not input_text:
        # Try reading from stdin if piped
        if not sys.stdin.isatty():
            input_text = sys.stdin.read().strip()
        else:
            parser.print_help()
            return 1

    if not input_text:
        print("Error: input text required", file=sys.stderr)
        return 1

    # Run
    try:
        if args.mode == "think":
            result = cf.think(input_text)
        elif args.mode == "decide":
            result = cf.decide(input_text)
        else:  # simulate
            result = cf.simulate(input_text, steps=args.steps)

        print_result(result, mode=args.mode, quiet=args.quiet, as_json=args.json)
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
