"""
api/server.py
==============
CogniField REST API Server

Endpoints:
  GET  /health           – liveness + system status
  POST /think            – full reasoning pipeline
  POST /decide           – action decision with risk assessment
  POST /simulate         – multi-step scenario simulation
  GET  /beliefs          – current authoritative belief set
  POST /teach            – inject a fact into the agent fleet

Run:

    # Inline
    python -m cognifield.api.server

    # With custom port
    python -m cognifield.api.server --port 8080

    # With LLM
    python -m cognifield.api.server --llm ollama --llm-model llama3
"""

from __future__ import annotations

import json
import os
import traceback
from typing import Any, Dict, Optional

try:
    from flask import Flask, request, jsonify, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


def create_app(
    cognifield_config: Optional[Dict] = None,
) -> Any:
    """
    Create and return the Flask application.

    Parameters
    ----------
    cognifield_config : Optional dict passed to CogniField().
    """
    if not FLASK_AVAILABLE:
        raise ImportError("Flask is required: pip install flask")

    from .._core import CogniField

    app = Flask("cognifield-api")
    cf  = CogniField(cognifield_config or {})

    # ----------------------------------------------------------------
    # Health
    # ----------------------------------------------------------------

    @app.route("/health", methods=["GET"])
    def health():
        """Liveness probe + system status."""
        status = cf.status()
        return jsonify({
            "status":  "ok",
            "version": status["version"],
            "agents":  status["agents"],
            "llm":     status["llm_backend"],
            "config":  status["config"],
        }), 200

    # ----------------------------------------------------------------
    # Think
    # ----------------------------------------------------------------

    @app.route("/think", methods=["POST"])
    def think():
        """
        Run the full reasoning pipeline on an input.

        Request body:
          { "input": "Is this berry safe to eat?" }

        Response:
          { "decision": "cautious", "confidence": 0.62, ... }
        """
        body = _get_json(request)
        if body is None:
            return _error("Request body must be JSON with 'input' field.", 400)

        text = body.get("input", "").strip()
        if not text:
            return _error("'input' field is required and must be non-empty.", 400)

        try:
            result = cf.think(text)
            return jsonify(result), 200
        except Exception as e:
            return _error(f"Internal error: {e}", 500, trace=traceback.format_exc())

    # ----------------------------------------------------------------
    # Decide
    # ----------------------------------------------------------------

    @app.route("/decide", methods=["POST"])
    def decide():
        """
        Make a concrete action decision.

        Request body:
          { "input": "Should I eat the red fruit?" }

        Response includes:
          action, risk_level, alternatives in addition to think fields.
        """
        body = _get_json(request)
        if body is None:
            return _error("Request body must be JSON with 'input' field.", 400)

        text = body.get("input", "").strip()
        if not text:
            return _error("'input' field is required.", 400)

        try:
            result = cf.decide(text)
            return jsonify(result), 200
        except Exception as e:
            return _error(f"Internal error: {e}", 500)

    # ----------------------------------------------------------------
    # Simulate
    # ----------------------------------------------------------------

    @app.route("/simulate", methods=["POST"])
    def simulate():
        """
        Run a multi-step scenario simulation.

        Request body:
          {
            "scenario": "foraging in unknown forest",
            "steps": 10
          }

        Response includes success_rate, belief_changes, outcomes.
        """
        body = _get_json(request)
        if body is None:
            return _error("Request body must be JSON.", 400)

        scenario = body.get("scenario", body.get("input", "")).strip()
        if not scenario:
            return _error("'scenario' field is required.", 400)

        steps = int(body.get("steps", 10))
        steps = max(1, min(steps, 50))   # clamp 1–50

        try:
            result = cf.simulate(scenario, steps=steps)
            return jsonify(result), 200
        except Exception as e:
            return _error(f"Internal error: {e}", 500)

    # ----------------------------------------------------------------
    # Beliefs
    # ----------------------------------------------------------------

    @app.route("/beliefs", methods=["GET"])
    def beliefs():
        """
        Return the current authoritative belief set.

        Query params:
          ?min_confidence=0.60   (default: 0.60)
        """
        try:
            min_conf = float(request.args.get("min_confidence", 0.60))
            belief_map = cf.get_beliefs(min_confidence=min_conf)
            return jsonify({
                "beliefs":        belief_map,
                "n_beliefs":      len(belief_map),
                "min_confidence": min_conf,
            }), 200
        except Exception as e:
            return _error(f"Internal error: {e}", 500)

    # ----------------------------------------------------------------
    # Teach
    # ----------------------------------------------------------------

    @app.route("/teach", methods=["POST"])
    def teach():
        """
        Inject a fact into the agent fleet.

        Request body:
          {
            "label": "apple",
            "properties": {"edible": true, "category": "food"},
            "text": "apple is a red edible fruit"  // optional
          }
        """
        body = _get_json(request)
        if body is None:
            return _error("Request body must be JSON.", 400)

        label = body.get("label", "").strip()
        props = body.get("properties", {})
        text  = body.get("text")

        if not label or not isinstance(props, dict):
            return _error("'label' (str) and 'properties' (dict) are required.", 400)

        try:
            cf.teach(label, props, text)
            return jsonify({
                "status":  "taught",
                "label":   label,
                "properties": props,
            }), 200
        except Exception as e:
            return _error(f"Internal error: {e}", 500)

    # ----------------------------------------------------------------
    # Error handlers
    # ----------------------------------------------------------------

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "endpoint not found",
                        "available": ["/health","/think","/decide",
                                      "/simulate","/beliefs","/teach"]}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "method not allowed"}), 405

    return app, cf


def _get_json(req) -> Optional[Dict]:
    """Safely parse request JSON."""
    try:
        return req.get_json(force=True, silent=True) or {}
    except Exception:
        return None


def _error(msg: str, status: int = 400, trace: str = "") -> Any:
    from flask import jsonify
    body = {"error": msg}
    if trace:
        body["trace"] = trace[:500]
    return jsonify(body), status


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="CogniField API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cognifield.api.server
  python -m cognifield.api.server --port 8080
  python -m cognifield.api.server --agents 5 --uncertainty medium
  python -m cognifield.api.server --llm ollama --llm-model llama3
        """,
    )
    parser.add_argument("--host",        default="0.0.0.0",  help="Bind host")
    parser.add_argument("--port",        default=8000, type=int, help="Port")
    parser.add_argument("--agents",      default=3,    type=int, help="Number of agents")
    parser.add_argument("--uncertainty", default="low", help="Uncertainty level")
    parser.add_argument("--llm",         default="mock", help="LLM backend")
    parser.add_argument("--llm-model",   default="llama3", help="LLM model name")
    parser.add_argument("--llm-url",     default="http://localhost:11434")
    parser.add_argument("--llm-key",     default="")
    parser.add_argument("--debug",       action="store_true")
    args = parser.parse_args()

    cfg = {
        "agents":       args.agents,
        "uncertainty":  args.uncertainty,
        "llm":          args.llm,
        "llm_model":    args.llm_model,
        "llm_base_url": args.llm_url,
        "llm_api_key":  args.llm_key,
    }

    app, cf = create_app(cfg)
    print(f"╔══════════════════════════════════════════════╗")
    print(f"║   CogniField API Server v10                  ║")
    print(f"╚══════════════════════════════════════════════╝")
    print(f"  System: {cf}")
    print(f"  Endpoints: /health  /think  /decide  /simulate  /beliefs  /teach")
    print(f"  Running on http://{args.host}:{args.port}")
    print()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
