<div align="center">

# CogniField

**Self-Learning Embodied Cognitive Intelligence**

*Think deep. Act safely. Learn from experience.*

[![Tests](https://img.shields.io/badge/tests-1315%20passing-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)
[![Version](https://img.shields.io/badge/version-11.0.0-orange)](#)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#)

</div>

---

## What is CogniField?

CogniField is a **cognitive AI framework** — no LLMs, no deep learning.
It reasons from structured beliefs, acts through a virtual body, and
learns from what happens.

```
v11 full loop every step:
  THINK → SIMULATE → DECIDE → ACT → OBSERVE → LEARN
```

---

## Quick Start

```python
from cognifield import CogniField

cf = CogniField()
cf.teach("apple",  {"edible": True,  "category": "food"})
cf.teach("stone",  {"edible": False, "category": "material"})

# Think only
result = cf.think("Is this apple safe?")
print(result["decision"])       # "proceed"
print(result["confidence"])     # 0.87
print(result["thinking_steps"]) # 7

# Act
result = cf.act("eat", "apple")
print(result["effect"])   # "satisfied"
print(result["reward"])   # 0.5

# Full THINK → ACT → LEARN loop
step = cf.step("eat apple")
print(step["decision"])   # "proceed"
print(step["executed"])   # True
print(step["effect"])     # "satisfied"
print(step["reward"])     # 0.5

# Run an episode
episode = cf.run_episode([
    "inspect stone",
    "eat apple",
    "eat stone",        # blocked — known dangerous
    "eat mystery_orb",  # blocked — unknown safety rule
])
for s in episode:
    icon = "✓" if s["executed"] else "⊘"
    print(f"  {icon} {s['action']}({s['target']}) → {s['effect']}")
```

---

## Version History

| Ver | Focus | Tests |
|-----|-------|-------|
| v2 | Encoding, vector memory | 64 |
| v3 | World model, planner | 113 |
| v4 | Self-direction, meta-learning | 123 |
| v5 | Stable beliefs, risk engine | 144 |
| v6 | Multi-agent communication | 142 |
| v7 | Social intelligence, negotiation | 122 |
| v8 | Collective intelligence, EventBus | 90 |
| v9 | Adaptive strategies, meta-cognition | 131 |
| v10 | Production API, Flask, CLI, LLM | 115 |
| v11 | Deep reasoning, experience learning | 139 |
| **v11p2** | **Embodied intelligence, virtual body** | **132** |
| **Total** | | **1315 ✓** |

---

## Core API

### Reasoning

```python
cf = CogniField({
    "agents":            3,
    "thinking_mode":     "auto",   # fast / deep / auto
    "uncertainty":       "low",
    "unknown_safety_rule": True,
    "confidence_target": 0.65,
})

# Think
result = cf.think("Is this mushroom safe?")
# decision, confidence, thinking_steps, knowledge_state,
# reasoning[], world_model, consensus, meta

# Decide
decision = cf.decide("Should I eat the red berry?")
# + action, risk_level, alternatives

# Simulate
sim = cf.simulate("foraging in forest", steps=10)
# success_rate, outcomes, belief_changes
```

### Embodied actions

```python
# Single physical action
result = cf.act("eat", "apple")
result = cf.act("pick", "bread")
result = cf.act("inspect", "mushroom")
result = cf.act("move", "north")

# Full cognitive loop step
step = cf.step("eat apple")
# decision, executed, effect, reward, belief_updates, body{health, hunger}

# Episode
episode = cf.run_episode(["inspect apple", "eat apple", "eat stone"])

# Body state
body = cf.body_status()
# health, hunger, energy, position, inventory, alive
```

### Teaching and learning

```python
# Teach facts
cf.teach("apple",  {"edible": True,  "category": "food"})
cf.teach("stone",  {"edible": False, "category": "material"})

# Learn from a real outcome
cf.learn_from_outcome(
    "ate mushroom", "mushroom", "edible",
    prediction=True, actual=False,
    action="eat", reward=-0.4
)

# Self-correction + reflection
reflection = cf.self_reflect()
print(reflection["findings"])     # ["Self-corrected 2 beliefs"]
print(reflection["corrections"])
```

---

## Response Format

### `think()` / `step()`

```json
{
  "decision":        "proceed",
  "confidence":      0.87,
  "thinking_steps":  7,
  "thinking_mode":   "deep",
  "knowledge_state": "known",
  "safe":            true,
  "contradictions":  [],
  "reasoning": [
    "[knowledge_check] Found 'apple.edible': value=True, conf=0.87",
    "[risk_evaluation] Risk acceptable: conf=0.87 ≥ 0.40"
  ],
  "world_model": {"inferred_value": true, "inferred_conf": 0.68},
  "executed":        true,
  "effect":          "satisfied",
  "reward":          0.5,
  "belief_updates":  2,
  "body": {"health": 1.0, "hunger": 0.0, "inventory": []},
  "elapsed_ms":      14.2
}
```

---

## Safety Architecture

Three independent safety layers — **all three must pass**:

```
1. DeepThinker
   unknown knowledge_state  → decision = "avoid"
   risk evaluation fires    → decision = "avoid", safe = False

2. Decide phase
   decision == "avoid"      → should_act = False
   sim predicts harm + low  → should_act = False
   confidence < 0.35        → should_act = False

3. ActionSystem
   unknown + safety_rule    → BLOCKED
   known toxic/inedible     → BLOCKED
   body incapable           → BLOCKED (INVALID)
```

---

## New in v11 Part 2

### Virtual Body (`agents/body.py`)

```
Eyes   → LOOK, INSPECT
Hands  → PICK, DROP
Mouth  → EAT

Body state: health · hunger · energy · position · inventory
```

### Perception System (`agents/perception.py`)

Converts raw environment output to structured Observations with
classified signals (SUCCESS/FAILURE/RISK/DANGER/NOVEL) and inferred
belief updates.

### Action System (`agents/action_system.py`)

Validates every action against beliefs and safety rules before
execution. Logs all attempts including blocked ones.

### Interaction Loop (`core/interaction_loop.py`)

Orchestrates the 7-phase embodied loop. Handles intent parsing,
thinking, simulation, action, observation, and learning in one call.

---

## LLM Integration (Optional)

```python
# Ollama (local)
cf = CogniField({"llm": "ollama", "llm_model": "llama3"})

# OpenAI-compatible API
cf = CogniField({"llm": "api", "llm_api_key": "sk-..."})

# LLM is used only for natural language output — never for decisions
```

---

## CLI

```bash
python -m cognifield "Is this mushroom safe?"
python -m cognifield "Should I eat the berry?" --mode decide
python -m cognifield "Foraging in forest" --mode simulate --steps 10
python -m cognifield --status
```

## REST API

```bash
python -m cognifield.api.server

curl -X POST http://localhost:8000/think \
     -H "Content-Type: application/json" \
     -d '{"input": "Is this berry safe?"}'
```

---

## Installation

```bash
pip install numpy scipy scikit-learn Pillow
pip install flask   # optional: REST API
pip install -e .    # from source
```

---

## Test Coverage

```
test_all.py        :   64 /  64  ✓  (v2 core)
test_v3.py         :  113 / 113  ✓  (world model, planner)
test_v4.py         :  123 / 123  ✓  (self-direction)
test_v5.py         :  144 / 144  ✓  (beliefs, risk)
test_v6.py         :  142 / 142  ✓  (multi-agent)
test_v7.py         :  122 / 122  ✓  (social intelligence)
test_v8.py         :   90 /  90  ✓  (collective intelligence)
test_v9.py         :  131 / 131  ✓  (adaptive strategy)
test_v10.py        :  115 / 115  ✓  (production API)
test_v11.py        :  139 / 139  ✓  (deep reasoning)
test_v11_part2.py  :  132 / 132  ✓  (embodied intelligence)
───────────────────────────────────────────────
Total              : 1315 /1315  ✓  ALL PASSING
```

---

## Architecture Layers

```
v11p2: body + perception + action_system + interaction_loop
v11:   deep_thinker + experience_engine + world_model_v2
v10:   CogniField API + Flask REST + CLI + LLM integration
v9:    meta_cognition + uncertainty + strategy + self_evaluation
v8:    event_bus + global_consensus + group_mind
v7:    language_layer + negotiation + cooperation
v6:    communication + shared_memory + trust
v5:    beliefs + conflict_resolution + risk_engine
v4:    imagination + meta_learning + self_direction
v3:    world_model + planner + goals
v2:    encoding + vector_memory + reasoning
```

---

## Licence
MIT
