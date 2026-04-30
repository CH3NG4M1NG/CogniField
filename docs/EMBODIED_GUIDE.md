# CogniField v11 Part 2 — Embodied Intelligence Guide

## Overview

v11 Part 2 adds a complete **virtual body** to the cognitive system.
The agent no longer just reasons — it *acts* in a simulated world,
*observes* the results, and *learns* from what happened.

Full loop every step:

```
THINK → SIMULATE → DECIDE → ACT → OBSERVE → LEARN
```

---

## Quick Start

```python
from cognifield import CogniField

cf = CogniField()
cf.teach("apple",  {"edible": True,  "category": "food"})
cf.teach("stone",  {"edible": False, "category": "material"})

# Single action
result = cf.act("eat", "apple")
print(result["effect"])   # "satisfied"
print(result["reward"])   # 0.5

# Full cognitive loop step
step = cf.step("eat apple")
print(step["decision"])   # "proceed"
print(step["executed"])   # True
print(step["effect"])     # "satisfied"
print(step["body"]["health"])  # 1.0

# Run an episode
episode = cf.run_episode([
    "inspect apple",
    "eat apple",
    "eat stone",         # blocked — known dangerous
    "eat mystery_orb",   # blocked — unknown safety rule
])
for s in episode:
    icon = "✓" if s["executed"] else "⊘"
    print(f"  {icon} {s['action']}({s['target']}) → {s['effect']}")
```

---

## Methods

### `cf.act(action, object, force=False)` → `dict`

Execute one physical action directly.

```python
result = cf.act("eat", "apple")
# {
#   "action":      "eat",
#   "object":      "apple",
#   "status":      "success",
#   "effect":      "satisfied",
#   "reward":      0.5,
#   "reason":      "Mouth consumed apple: edible=True, ...",
#   "body_health": 1.0,
#   "body_hunger": 0.0,
#   "body_inventory": []
# }
```

**Safety rules:** actions on unknown objects are blocked by default.
Use `force=True` to bypass (use cautiously).

### `cf.step(query)` → `dict`

Run the full 7-phase loop on a natural language command.

```python
step = cf.step("eat apple")
# {
#   "step":       1,
#   "action":     "eat",
#   "target":     "apple",
#   "decision":   "proceed",        ← DeepThinker output
#   "confidence": 0.87,
#   "simulated":  "positive",       ← WorldModel prediction
#   "executed":   True,             ← did the action run?
#   "blocked":    None,             ← reason if blocked
#   "signal":     "success",        ← Perception output
#   "effect":     "satisfied",
#   "reward":     0.5,
#   "belief_updates": 2,            ← beliefs updated
#   "corrections":    0,            ← self-corrections
#   "body": {"health": 1.0, "hunger": 0.0, "inventory": []},
#   "elapsed_ms": 12.4
# }
```

### `cf.run_episode(queries)` → `list[dict]`

Run a sequence of steps. Returns list of step dicts.

```python
episode = cf.run_episode([
    "inspect mushroom",
    "eat mushroom",
    "inspect apple",
    "eat apple",
])
```

### `cf.body_status()` → `dict`

Current physical state of the body.

```python
cf.body_status()
# {
#   "health":    0.95,
#   "hunger":    0.15,
#   "energy":    0.88,
#   "position":  (0, 0),
#   "inventory": ["bread"],
#   "alive":     True,
#   "steps":     12,
#   "actions_taken": 8
# }
```

---

## Actions

| Action | Description | Safety check |
|--------|-------------|--------------|
| `eat`  | Consume an object | ✓ blocked if unknown or known-dangerous |
| `pick` | Pick up an object | ✓ blocked if hands full |
| `drop` | Release a held object | — |
| `inspect` | Examine carefully | Always allowed |
| `move` | Move north/south/east/west | Always allowed |
| `look` | Scan surroundings | Always allowed |
| `wait` | Rest (restore energy) | Always allowed |

---

## The 7-Phase Loop

Each `step()` call runs all 7 phases:

```
Phase 1: THINK
  DeepThinker runs N structured reasoning steps on (subject, predicate).
  Output: decision ("proceed" / "avoid" / ...) + confidence.

Phase 2: SIMULATE
  WorldModelV2 predicts the outcome before acting.
  Output: "positive" / "negative" / "uncertain" / effect_label.

Phase 3: DECIDE
  Combines think + simulate.
  Rules:
    - decision == "avoid"          → do NOT act
    - simulation == "negative"
      AND confidence < 0.55        → do NOT act
    - unknown safety rule active
      AND decision == "investigate" → do NOT act
    - confidence < 0.35 for
      consequential actions         → do NOT act

Phase 4: ACT
  ActionSystem validates + routes to VirtualBody.
  Validation checks:
    - object known? (unknown_safety_rule)
    - object toxic or inedible? (belief system)
    - body physically capable?
  Output: BodyActionResult with effect, reward, body_delta.

Phase 5: OBSERVE
  PerceptionSystem converts body result to Observation.
  Classifies signal: SUCCESS / FAILURE / RISK / DANGER / NOVEL.
  Infers edibility from eat outcomes (edible=True if success).

Phase 6: LEARN
  - Applies belief updates from perception
  - Updates WorldModelV2 with observed effect
  - Calls ExperienceEngine.learn_from_outcome()
  - Wrong confident predictions trigger belief correction

Phase 7: RETURN
  Bundles all phase outputs into EpisodeStep dict.
```

---

## Body State

The virtual body has four tracked quantities:

| State | Range | Changes |
|-------|-------|---------|
| `health` | 0–1 | eat(inedible) → -0.20; eat(edible) → +0.05 |
| `hunger` | 0–1 | eat(edible) → -0.35; move → +0.02; wait → +0.03 |
| `energy` | 0–1 | pick → -0.03; move → -0.04; wait → +0.05 |
| `position`| (x,y) | move(north) → (x, y+1) |

The agent dies when `health ≤ 0`. Episodes stop automatically.

---

## Learning Integration

After each action, the ExperienceEngine updates beliefs:

```python
# Correct prediction → small boost
eat(apple) → satisfied   # apple.edible confidence +0.04

# Wrong confident prediction → penalty
eat(stone) → damage      # stone.edible confidence -0.15
                         # (scaled by overconfidence)

# Pattern generalisation (after N consistent outcomes)
# 4× food objects edible → food.edible = True (category rule)
```

Check what was learned:

```python
cf.learn_from_outcome("ate red_berry", "red_berry", "edible",
                       prediction=True, actual=True,
                       action="eat", reward=0.4)

cf.self_reflect()
# findings: ["Rules derived: food.edible=True from 8 outcomes"]
```

---

## Safety Architecture

```
User calls step("eat mystery_substance")
         │
         ▼
DeepThinker: knowledge_state=unknown → decision="avoid"
         │
         ▼
Decide phase: decision==avoid → should_act=False
         │
         ▼
ActionSystem: also checks → unknown_safety_rule → BLOCKED
         │
         ▼
BodyActionResult: status=BLOCKED, effect=blocked_by_decision
         │
         ▼
EpisodeStep: executed=False, reward=0.0
```

Three independent safety layers:
1. **DeepThinker** — unknown or risky → returns "avoid"
2. **Decide phase** — translates decision to should_act
3. **ActionSystem** — validates belief system + safety rules

All three must pass for a consequential action to execute.
