"""
examples/api_client.py
=======================
CogniField v10 — REST API Client Example

Shows how to use the CogniField API server from Python or curl.

Prerequisites:
  Start the server:
    python -m cognifield.api.server --port 8000

Then run this file:
    python examples/api_client.py

Or use curl:
    curl -X POST http://localhost:8000/think \\
         -H "Content-Type: application/json" \\
         -d '{"input": "Is this berry safe?"}'
"""

import sys, os, json, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import urllib.request
    import urllib.error

    def api_call(endpoint: str, data: dict = None, base_url: str = "http://localhost:8000") -> dict:
        """Make one API call. Returns response dict."""
        url = f"{base_url}/{endpoint.lstrip('/')}"
        if data is None:
            req = urllib.request.Request(url, method="GET")
        else:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())

    def demo_api_calls(base_url: str) -> None:
        print(f"\n  API base: {base_url}")
        print("  ─" * 30)

        # 1. Health check
        print("\n  1. GET /health")
        h = api_call("/health", base_url=base_url)
        print(f"     Status:  {h['status']}")
        print(f"     Agents:  {h['agents']}")
        print(f"     Version: {h['version']}")

        # 2. Teach some facts
        print("\n  2. POST /teach")
        facts = [
            {"label": "apple",  "properties": {"edible": True,  "category": "food"}},
            {"label": "stone",  "properties": {"edible": False, "category": "material"}},
            {"label": "bread",  "properties": {"edible": True,  "category": "food"}},
        ]
        for fact in facts:
            r = api_call("/teach", fact, base_url=base_url)
            print(f"     Taught: {fact['label']} → {r['status']}")

        # 3. Think
        print("\n  3. POST /think")
        questions = [
            "Is the apple safe to eat?",
            "Should I eat the stone?",
        ]
        for q in questions:
            r = api_call("/think", {"input": q}, base_url=base_url)
            print(f"     Q: {q}")
            print(f"        Decision={r['decision']}, Confidence={r['confidence']:.0%}")

        # 4. Decide
        print("\n  4. POST /decide")
        r = api_call("/decide", {"input": "Should I eat the bread?"}, base_url=base_url)
        print(f"     Decision:   {r['decision']}")
        print(f"     Action:     {r.get('action', 'N/A')}")
        print(f"     Risk level: {r.get('risk_level', 'N/A')}")

        # 5. Simulate
        print("\n  5. POST /simulate")
        r = api_call("/simulate", {
            "scenario": "foraging for food",
            "steps": 5,
        }, base_url=base_url)
        print(f"     Success rate:    {r['success_rate']:.0%}")
        print(f"     Belief changes:  {r['belief_changes']}")
        print(f"     Strategy:        {r['strategy']}")

        # 6. Get beliefs
        print("\n  6. GET /beliefs")
        r = api_call("/beliefs?min_confidence=0.5", base_url=base_url)
        print(f"     Total beliefs: {r['n_beliefs']}")
        for k, v in list(r.get("beliefs", {}).items())[:3]:
            print(f"       {k:25s} = {str(v['value']):6s} "
                  f"(conf={v['confidence']:.2f})")

        print("\n  ✓ All API calls successful.")

except Exception as e:
    print(f"  API client error: {e}")


def embed_server_demo() -> None:
    """
    Launch the API server in a background thread and demo it in-process.
    No external process needed.
    """
    from cognifield.api.server import create_app

    app, cf = create_app({"agents": 2})
    client  = app.test_client()

    print("\n  (Using embedded test server — no external process needed)")
    print("  For a real server: python -m cognifield.api.server")

    def api_call_local(endpoint: str, data: dict = None) -> dict:
        if data is None:
            r = client.get(endpoint)
        else:
            r = client.post(endpoint, json=data)
        return json.loads(r.data)

    # Health
    h = api_call_local("/health")
    print(f"\n  GET /health → status={h['status']}, agents={h['agents']}")

    # Teach
    api_call_local("/teach", {"label": "apple", "properties": {"edible": True, "category": "food"}})
    api_call_local("/teach", {"label": "stone", "properties": {"edible": False, "category": "material"}})
    print("  POST /teach → apple + stone taught")

    # Think
    r = api_call_local("/think", {"input": "Is apple edible?"})
    print(f"  POST /think → decision={r['decision']}, conf={r['confidence']:.0%}")

    # Decide
    r = api_call_local("/decide", {"input": "Should I eat stone?"})
    print(f"  POST /decide → decision={r['decision']}, risk={r.get('risk_level','?')}")

    # Simulate
    r = api_call_local("/simulate", {"scenario": "foraging", "steps": 3})
    print(f"  POST /simulate → sr={r['success_rate']:.0%}, "
          f"beliefs_updated={r['belief_changes']}")

    # Beliefs
    r = api_call_local("/beliefs")
    print(f"  GET /beliefs → {r['n_beliefs']} beliefs")

    print("\n  ✓ Embedded API demo complete.")


def show_curl_examples() -> None:
    """Print curl examples for documentation."""
    print("\n  " + "─" * 56)
    print("  curl Examples (after starting: python -m cognifield.api.server)")
    print("  " + "─" * 56)

    examples = [
        ("Health check",
         "curl http://localhost:8000/health"),
        ("Think",
         'curl -X POST http://localhost:8000/think \\\n'
         '     -H "Content-Type: application/json" \\\n'
         '     -d \'{"input": "Is this berry safe?"}\''),
        ("Decide",
         'curl -X POST http://localhost:8000/decide \\\n'
         '     -H "Content-Type: application/json" \\\n'
         '     -d \'{"input": "Should I eat the red fruit?"}\''),
        ("Simulate",
         'curl -X POST http://localhost:8000/simulate \\\n'
         '     -H "Content-Type: application/json" \\\n'
         '     -d \'{"scenario": "foraging", "steps": 10}\''),
        ("Teach",
         'curl -X POST http://localhost:8000/teach \\\n'
         '     -H "Content-Type: application/json" \\\n'
         '     -d \'{"label": "apple", "properties": {"edible": true}}\''),
        ("Get beliefs",
         "curl 'http://localhost:8000/beliefs?min_confidence=0.6'"),
    ]

    for name, cmd in examples:
        print(f"\n  # {name}")
        for line in cmd.split("\n"):
            print(f"  {line}")


if __name__ == "__main__":
    print("=" * 60)
    print("  CogniField v10 — REST API Client Examples")
    print("=" * 60)

    # Run embedded demo (no server process needed)
    embed_server_demo()

    # Show curl examples
    show_curl_examples()

    print("\n✓ API client demo complete.")
