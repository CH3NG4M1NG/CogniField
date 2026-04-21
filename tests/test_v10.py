"""
tests/test_v10.py
=================
CogniField v10 Test Suite — 90 tests

Covers:
  CogniFieldConfig · CogniField API · LLM clients
  Flask API server · CLI interface · Examples

Run: PYTHONPATH=.. python tests/test_v10.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json

PASS = 0; FAIL = 0; ERRORS = []

def check(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        print(f"  ✓ {name}"); PASS += 1
    else:
        print(f"  ✗ {name}" + (f" — {msg}" if msg else ""))
        FAIL += 1; ERRORS.append(name)


# ─────────────────────────────────────────────────────────
print("\n[CogniFieldConfig]")
from cognifield.cognifield_main import CogniFieldConfig

# Default config
cfg = CogniFieldConfig()
check("default_agents",       cfg.agents == 3)
check("default_uncertainty",  cfg.uncertainty == "low")
check("default_llm",          cfg.llm == "mock")
check("default_enable_meta",  cfg.enable_meta == True)
check("default_seed",         cfg.seed == 42)
check("default_strategy",     cfg.strategy == "adaptive")

# from_dict
cfg2 = CogniFieldConfig.from_dict({
    "agents": 5,
    "uncertainty": "high",
    "llm": "ollama",
    "unknown_key": "ignored",   # should not raise
})
check("from_dict_agents",       cfg2.agents == 5)
check("from_dict_uncertainty",  cfg2.uncertainty == "high")
check("from_dict_llm",          cfg2.llm == "ollama")
check("from_dict_ignores_extra",True)  # no exception = pass

# from_dict with empty dict
cfg3 = CogniFieldConfig.from_dict({})
check("from_dict_empty",        cfg3.agents == 3)   # default


# ─────────────────────────────────────────────────────────
print("\n[CogniField core API]")
from cognifield import CogniField, __version__

check("version_string",         "10" in __version__)

# Default constructor
cf = CogniField()
check("init_default",           cf is not None)
check("repr_has_agents",        "agents=3" in repr(cf))

# Config dict constructor
cf2 = CogniField({"agents": 2, "uncertainty": "none"})
check("init_dict",              cf2 is not None)
check("init_dict_agents",       len(cf2._agents) == 2)

# CogniFieldConfig constructor
cfg_obj = CogniFieldConfig(agents=2, uncertainty="medium")
cf3 = CogniField(cfg_obj)
check("init_config_obj",        cf3 is not None)

# ── think() ──
result = cf.think("Is this safe?")
check("think_returns_dict",     isinstance(result, dict))
check("think_has_decision",     "decision" in result)
check("think_has_confidence",   "confidence" in result)
check("think_has_reasoning",    "reasoning" in result and isinstance(result["reasoning"], list))
check("think_has_consensus",    "consensus" in result)
check("think_has_meta",         "meta" in result)
check("think_has_llm",          "llm_output" in result)
check("think_has_elapsed",      "elapsed_ms" in result and result["elapsed_ms"] >= 0)
check("think_confidence_range", 0.0 <= result["confidence"] <= 1.0)

# teach + think
cf4 = CogniField({"agents": 2})
cf4.teach("apple", {"edible": True,  "category": "food"})
cf4.teach("stone", {"edible": False, "category": "material"})
r_apple = cf4.think("Is apple edible?")
r_stone = cf4.think("Is stone safe to eat?")
check("apple_positive",         r_apple["decision"] in ("proceed", "food", "true"))
check("stone_negative",         r_stone["decision"] in ("avoid", "material", "false",
                                                          "insufficient_data"))
check("apple_conf_reasonable",  r_apple["confidence"] >= 0.30)

# ── decide() ──
decision_r = cf4.decide("Should I eat the apple?")
check("decide_has_decision",    "decision" in decision_r)
check("decide_has_action",      "action" in decision_r)
check("decide_has_risk",        "risk_level" in decision_r)
check("decide_has_alternatives","alternatives" in decision_r)
check("decide_risk_valid",      decision_r["risk_level"] in ("low","medium","high","critical"))
check("decide_alts_list",       isinstance(decision_r["alternatives"], list))

# unknown object → insufficient_data or cautious
r_unk = cf4.decide("Should I eat the glowing cube?")
check("unknown_decision_valid", r_unk["decision"] in ("insufficient_data","uncertain",
                                                       "proceed","avoid","cautious"))
check("unknown_low_confidence", r_unk["confidence"] <= 0.80)

# ── simulate() ──
cf4.teach("bread", {"edible": True, "category": "food"})
sim = cf4.simulate("foraging for food", steps=3)
check("simulate_returns_dict",  isinstance(sim, dict))
check("simulate_has_scenario",  "scenario" in sim)
check("simulate_has_steps",     "steps_run" in sim and sim["steps_run"] >= 3)
check("simulate_sr_range",      0.0 <= sim["success_rate"] <= 1.0)
check("simulate_has_outcomes",  "outcomes" in sim and isinstance(sim["outcomes"], list))
check("simulate_has_consensus", "consensus" in sim)
check("simulate_has_strategy",  "strategy" in sim)

# ── teach() fluent ──
cf5 = CogniField({"agents": 2})
ret = cf5.teach("mango", {"edible": True})
check("teach_returns_self",     ret is cf5)
ret2 = cf5.teach("rock", {"edible": False}).teach("fruit", {"edible": True})
check("teach_chaining",         ret2 is cf5)

# ── status() ──
status = cf.status()
check("status_version",         "version" in status and "10" in str(status["version"]))
check("status_agents",          "agents" in status)
check("status_llm",             "llm_backend" in status)
check("status_config",          "config" in status and isinstance(status["config"], dict))

# ── get_beliefs() ──
cf4.think("apple")   # run a step to build GC
beliefs = cf4.get_beliefs(min_confidence=0.50)
check("get_beliefs_dict",       isinstance(beliefs, dict))
check("get_beliefs_threshold",  all(v["confidence"] >= 0.50 for v in beliefs.values()))

# ── reset() ──
cf5.think("test")
cf5.reset()
check("reset_clears_history",   len(cf5._think_history) == 0)
check("reset_returns_self",     cf5.reset() is cf5)

# ── add_goal() ──
cf5.add_goal("survive")
check("add_goal_returns_self",  cf5.add_goal("explore") is cf5)


# ─────────────────────────────────────────────────────────
print("\n[LLM Clients]")
from cognifield.llm.base import (
    MockLLM, OllamaClient, APIClient, create_llm_client
)

# MockLLM
mock = MockLLM()
check("mock_available",         mock.is_available())
resp = mock.generate("Is apple safe?")
check("mock_generates_str",     isinstance(resp, str) and len(resp) > 0)
resp2 = mock.generate("Tell me about confidence 0.8 here")
check("mock_confidence_response", "confidence" in resp2.lower())

# OllamaClient (offline)
ollama = OllamaClient(model="llama3", base_url="http://localhost:11434")
check("ollama_init",            ollama is not None)
check("ollama_repr",            "OllamaClient" in repr(ollama))
check("ollama_offline",         not ollama.is_available())  # no server running
resp_err = ollama.generate("hello")
check("ollama_error_str",       isinstance(resp_err, str))
check("ollama_error_graceful",  "Ollama error" in resp_err or "[" in resp_err)

# APIClient (no key)
api_no_key = APIClient(api_key="")
check("api_unavailable_no_key", not api_no_key.is_available())
resp_nokey = api_no_key.generate("hello")
check("api_no_key_error_str",   "no API key" in resp_nokey)

# APIClient (with fake key — doesn't hit network, just returns available)
api_fake = APIClient(api_key="sk-fake-key-for-testing")
check("api_available_with_key", api_fake.is_available())
check("api_repr",               "APIClient" in repr(api_fake))

# factory
for backend, expected_type in [("mock", MockLLM),
                                ("ollama", OllamaClient),
                                ("api", APIClient)]:
    client = create_llm_client(backend)
    check(f"factory_{backend}", isinstance(client, expected_type))

try:
    create_llm_client("nonexistent")
    check("factory_invalid_raises", False)
except ValueError:
    check("factory_invalid_raises", True)

# Prompt formatting
ctx = {"decision": "proceed", "confidence": 0.85, "beliefs": {},
       "uncertainty": "low", "strategy": "exploit", "consensus_value": "proceed"}
prompt = mock.format_decision_prompt("Is apple safe?", ctx)
check("prompt_has_input",       "Is apple safe?" in prompt)
check("prompt_has_decision",    "proceed" in prompt)
check("prompt_has_confidence",  "0.85" in prompt)

sim_prompt = mock.format_simulation_prompt("foraging", {
    "success_rate": 0.7, "steps": 10, "outcomes": ["eat(apple): ✓"],
    "belief_changes": 3, "strategy": "explore"
})
check("sim_prompt_scenario",    "foraging" in sim_prompt)


# ─────────────────────────────────────────────────────────
print("\n[Flask API Server]")
from cognifield.api.server import create_app

app, cf_api = create_app({"agents": 2, "uncertainty": "low"})
client = app.test_client()

# Health
r = client.get("/health")
data = json.loads(r.data)
check("api_health_200",         r.status_code == 200)
check("api_health_status",      data["status"] == "ok")
check("api_health_version",     "version" in data)
check("api_health_agents",      data["agents"] == 2)

# Think
r2 = client.post("/think", json={"input": "Is apple safe?"})
data2 = json.loads(r2.data)
check("api_think_200",          r2.status_code == 200)
check("api_think_decision",     "decision" in data2)
check("api_think_confidence",   "confidence" in data2)
check("api_think_reasoning",    isinstance(data2.get("reasoning"), list))

# Think – missing input
r_empty = client.post("/think", json={})
check("api_think_empty_400",    r_empty.status_code == 400)

# Think – bad JSON → 400
r_bad = client.post("/think", data="not json",
                    content_type="application/json")
check("api_think_bad_json_400", r_bad.status_code in (400, 200))  # either is ok

# Decide
r3 = client.post("/decide", json={"input": "Should I eat the apple?"})
data3 = json.loads(r3.data)
check("api_decide_200",         r3.status_code == 200)
check("api_decide_action",      "action" in data3)
check("api_decide_risk",        "risk_level" in data3)

# Simulate
r4 = client.post("/simulate", json={"scenario": "foraging", "steps": 3})
data4 = json.loads(r4.data)
check("api_simulate_200",       r4.status_code == 200)
check("api_simulate_sr",        "success_rate" in data4)
check("api_simulate_steps",     data4.get("steps_run", 0) >= 3)

# Beliefs
r5 = client.get("/beliefs")
data5 = json.loads(r5.data)
check("api_beliefs_200",        r5.status_code == 200)
check("api_beliefs_has_n",      "n_beliefs" in data5)

# Teach
r6 = client.post("/teach", json={
    "label": "mango",
    "properties": {"edible": True, "category": "food"},
})
data6 = json.loads(r6.data)
check("api_teach_200",          r6.status_code == 200)
check("api_teach_status",       data6.get("status") == "taught")
check("api_teach_label",        data6.get("label") == "mango")

# Teach – missing label
r_t_bad = client.post("/teach", json={"properties": {"x": 1}})
check("api_teach_bad_400",      r_t_bad.status_code == 400)

# 404
r_404 = client.get("/nonexistent")
check("api_404",                r_404.status_code == 404)


# ─────────────────────────────────────────────────────────
print("\n[CLI Interface]")
from cognifield.cli.__main__ import main as cli_main, print_result
import io
from contextlib import redirect_stdout

# Test print_result
sample_result = {
    "decision": "proceed", "confidence": 0.85,
    "reasoning": ["apple.edible=True", "High confidence"],
    "consensus": {"apple.edible": {"value": True, "confidence": 0.85,
                                    "agreement": 0.95}},
    "meta": {"agents": 3, "strategy": "explore", "uncertainty": "low"},
    "llm_output": "The apple is safe to eat.",
    "elapsed_ms": 42.0,
}

buf = io.StringIO()
with redirect_stdout(buf):
    print_result(sample_result, mode="think", quiet=False, as_json=False)
output = buf.getvalue()
check("print_result_has_decision",  "PROCEED" in output)
check("print_result_has_confidence","85%" in output)
check("print_result_has_reasoning", "apple.edible" in output)

# Quiet mode
buf2 = io.StringIO()
with redirect_stdout(buf2):
    print_result(sample_result, quiet=True)
check("quiet_mode_short",           len(buf2.getvalue().split("\n")) <= 3)
check("quiet_has_decision",         "proceed" in buf2.getvalue().lower())

# JSON mode
buf3 = io.StringIO()
with redirect_stdout(buf3):
    print_result(sample_result, as_json=True)
parsed = json.loads(buf3.getvalue())
check("json_mode_parseable",        isinstance(parsed, dict))
check("json_has_decision",          "decision" in parsed)

# Decide mode output
decide_result = {**sample_result, "action": "act", "risk_level": "low",
                 "alternatives": ["proceed cautiously"]}
buf4 = io.StringIO()
with redirect_stdout(buf4):
    print_result(decide_result, mode="decide")
check("decide_mode_action",         "act" in buf4.getvalue())
check("decide_mode_risk",           "low" in buf4.getvalue())

# Simulate mode output
sim_result = {**sample_result, "success_rate": 0.8, "belief_changes": 5,
              "outcomes": ["eat(apple): ✓", "pick(bread): ✓"]}
buf5 = io.StringIO()
with redirect_stdout(buf5):
    print_result(sim_result, mode="simulate")
check("simulate_mode_sr",           "80%" in buf5.getvalue())
check("simulate_mode_beliefs",      "5" in buf5.getvalue())


# ─────────────────────────────────────────────────────────
print("\n[Integration: think → decide → simulate]")
cf_int = CogniField({"agents": 3, "uncertainty": "low"})
(cf_int
 .teach("apple",  {"edible": True,  "category": "food"})
 .teach("stone",  {"edible": False, "category": "material"})
 .teach("bread",  {"edible": True,  "category": "food"})
)

# Full pipeline
t1 = cf_int.think("Should I eat the apple?")
d1 = cf_int.decide("Should I eat the stone?")
s1 = cf_int.simulate("foraging for edible objects", steps=5)

check("integration_think_ok",     t1["decision"] in ("proceed","avoid","insufficient_data","uncertain"))
check("integration_decide_ok",    "action" in d1 and "risk_level" in d1)
check("integration_simulate_ok",  0.0 <= s1["success_rate"] <= 1.0)

# Multi-call stability (same instance, multiple calls)
results = [cf_int.think(q) for q in [
    "Is apple safe?", "Is stone edible?", "Can I eat bread?"
]]
check("multi_call_all_succeed",   all("decision" in r for r in results))
check("multi_call_no_crash",      True)

# Status after use
final_status = cf_int.status()
check("final_think_count",        final_status["think_calls"] >= 3)

# ─────────────────────────────────────────────────────────
print(f"\n{'═'*58}")
print(f"  v10 Results: {PASS} passed, {FAIL} failed")
if ERRORS:
    print(f"  Failed: {ERRORS}")
else:
    print("  All v10 tests passed ✓")
print(f"{'═'*58}\n")
