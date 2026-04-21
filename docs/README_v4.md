# CogniField v4 🧠

**Autonomous Self-Improving Cognitive Architecture**

> *"Intelligence is not just learning about the world — it is learning how to learn about the world."*

CogniField v4 transforms the v3 goal-driven planning agent into a **fully autonomous, continuously self-improving system** that generates its own goals, imagines consequences before acting, abstracts general rules from experience, and adapts its own strategies over time.

---

## What's New in v4

| Module | v3 | v4 |
|--------|----|----|
| Goal system | External goals only | ✓ **Self-generated** from curiosity/failure/env/meta signals |
| Planning | Flat beam search | ✓ **Hierarchical** subgoal decomposition with learned templates |
| World model | Rules + transitions | ✓ + **Simulator** (imagination/counterfactual before acting) |
| Memory | Decay + clustering | ✓ + **Consolidation** (merge, prune, abstract on schedule) |
| Reasoning | Strategy portfolio | ✓ + **Abstraction engine** (categorical induction, generalisation) |
| Adaptation | Fixed parameters | ✓ **Meta-learning** (performance tracking → parameter updates) |
| Agent state | External only | ✓ **Internal state** (confidence, curiosity, fatigue, frustration) |
| Agent loop | 10 steps | ✓ **14-step** continuous autonomous loop |

---

## Architecture

```
cognifield/
│
├── agent/
│   ├── internal_state.py       ★ NEW: confidence, curiosity, fatigue, frustration
│   ├── goal_generator.py       ★ NEW: self-generates goals from 5 signal sources
│   ├── goals.py                goal lifecycle (from v3)
│   ├── agent_v3.py             v3 agent (still works)
│   └── agent_v4.py             ★ NEW: 14-step autonomous loop
│
├── memory/
│   ├── memory_store.py         vector similarity store (v2)
│   ├── relational_memory.py    typed concept graph (v3)
│   └── consolidation.py        ★ NEW: merge + prune + abstract (periodic)
│
├── reasoning/
│   ├── reasoning_engine.py     self-correcting retry loop (v2)
│   ├── abstraction.py          ★ NEW: rule induction (categorical, negative, temporal)
│   └── meta_learning.py        ★ NEW: performance tracking + strategy adaptation
│
├── world_model/
│   ├── transition_model.py     (state, action) → next_state rules (v3)
│   ├── causal_graph.py         symbolic cause-effect graph (v3)
│   └── simulator.py            ★ NEW: forward simulation + counterfactual
│
├── planning/
│   ├── planner.py              flat beam-search planner (v3)
│   └── hierarchical_planner.py ★ NEW: subgoal decomposition + learned templates
│
├── tests/
│   ├── test_all.py             v2:  64 tests
│   ├── test_v3.py              v3: 113 tests
│   └── test_v4.py              v4: 123 tests  ← NEW
│
└── examples/
    ├── demo.py                 v2 demo
    ├── demo_v3.py              v3 demo
    └── demo_v4.py              ★ NEW: 8-section autonomous agent demo
```

---

## How Self-Generated Goals Work

The `GoalGenerator` scans 5 independent signal sources every few steps:

```
1. CURIOSITY  → unknown concepts with open hypotheses
               "understand glowing_sphere" (priority=0.70)

2. FAILURE    → low success rates on specific actions
               "improve eat skill" (priority=0.48, because eat=20%)

3. ENVIRONMENT → low satiation, visible unknowns, low health
               "eat apple" (priority=0.90, satiation=0.25)

4. KNOWLEDGE GAP → objects with unknown edibility
               "learn object properties" (priority=0.50)

5. META       → high fatigue → consolidate; high frustration → revise
               "consolidate memory" (priority=0.61, fatigue=0.80)
```

Goal priorities scale with the internal state:
- High **curiosity** → curiosity goals get higher priority
- High **frustration** → failure-recovery goals get higher priority
- High **fatigue** → meta-goals (consolidate, rest) get higher priority

---

## How the World Simulator Works

Before executing any plan, the agent **imagines** the outcome:

```python
# Agent wants to eat something — simulate alternatives first
plans = [
    [("pick","apple"), ("eat","apple")],   # safe plan
    [("pick","stone"), ("eat","stone")],   # risky plan
]
results = simulator.evaluate_plans(current_state, plans, goal_vec)
# → pick(apple)→eat(apple) scores 0.613 (better)
# → pick(stone)→eat(stone) scores 0.455 (worse)
# Agent chooses the safe plan
```

The simulator also supports **counterfactual reasoning**:
```python
cf = simulator.counterfactual(state, ("eat","stone"), ("eat","apple"))
# → "better_choice: eat(apple), regret=0.198"
# Agent can learn: "I should have eaten the apple"
```

---

## How Abstraction Works

The `AbstractionEngine` induces general rules from specific experiences using three methods:

### 1. Categorical Induction
```
apple → edible=True,  is_a=food
bread → edible=True,  is_a=food
water → edible=True,  is_a=food
berry → edible=True,  is_a=food
                    ↓
food → edible=True  (confidence=0.95, support=4)
```

### 2. Negative Inference
```
stone  → edible=False, is_a=material
hammer → edible=False, is_a=material
                    ↓
material → edible=False  (confidence=0.80, support=2)
```

### 3. Temporal Pattern Detection
```
pick(apple) → eat(apple) → success  (3/4 times)
                    ↓
"pick_then_eat" is a reliable strategy  (confidence=0.75)
```

**Generalisation**: New object `mango → is_a=food` → agent infers `mango.edible=True` via the abstract rule, without ever tasting it.

---

## How Memory Consolidation Works

Runs periodically (every N steps) in 4 phases:

```
Phase 1: Decay all entries (activation -= decay_rate)
Phase 2: Strengthen frequently accessed entries (+activation × access_count)
Phase 3: Merge near-duplicates (40 apple experiences → 1 strong "apple" entry)
Phase 4: Abstract relational knowledge (2+ food items edible → food→edible=True)
Phase 5: Prune entries below activation threshold
```

Result: Memory stays lean and focused. Knowledge becomes increasingly abstract over time.

---

## How Meta-Learning Works

The `MetaLearner` tracks performance and adjusts agent parameters:

```
Observations over 60 steps:
  eat  success rate:  60%
  pick success rate:  90%
  Overall trend:     +40% improvement

Insights generated:
  "Action 'pick' is highly reliable (90%)"
  "Exploring novel things tends to succeed — explore more"

Parameter updates:
  novelty_threshold_adj: -0.050  (lower threshold → explore more)
  retry_budget:           1.000  (no change — success rate acceptable)
  confidence_scale:       1.000  (well calibrated)
```

---

## The 14-Step Autonomous Loop

```
Every step:
  1.  Observe environment / text input
  2.  Encode → latent vector
  3.  Update vector + relational memory
  4.  Detect novelty (threshold modulated by internal state)
  5.  Generate goals from all 5 signal sources (every 3 steps)
  6.  Select highest-priority active goal
  7.  Plan hierarchically (decompose → recurse → flat actions)
  8.  Simulate plan (imagination pre-screening)
  9.  Execute best action
  10. Receive environment feedback
  11. Update world model + causal graph + hypotheses
  12. Run abstraction engine (every 8 steps)
  13. Analyse performance + adapt parameters (every 5 steps)
  14. Consolidate memory (every 15 steps, if fatigued)
```

The agent **never stops** — it cycles indefinitely, always learning.

---

## Demo Highlights

```
Section 1 — Internal State:
  success×2 + novelty + failure×2 + goal_done
  conf: 0.50→0.58→0.67  frust: 0.10→0.07→0.00  cur: 0.60→0.69

Section 2 — Simulator:
  Best plan: pick(apple)→eat(apple)       score=0.613  reward=+0.417
  Worst plan: pick(stone)→eat(stone)       score=0.455
  Counterfactual: eat(stone) regret=0.198 → should have eaten apple

Section 3 — Abstraction:
  food  → edible=True   (conf=0.95, support=4) ✓ strong
  tool  → edible=False  (conf=0.80, support=2) ✓ strong
  mango: category=food → infers edible=True    ✓ generalisation works

Section 4 — Meta-Learning:
  Overall SR: 64%   Recent SR: 85%   Trend: +40% ↑ improving
  Insight: "Exploring novel things tends to succeed"
  Adapted: novelty_threshold_adj = -0.050

Section 5 — Consolidation:
  40 entries → 20  (merged=10, pruned=0, strengthened=5, abstract=2, 1ms)

Section 6 — Goal Generation:
  [environment] "eat apple"      priority=0.90  (satiation=0.25)
  [failure]     "avoid dangerous" priority=0.85  (2 dangerous actions)
  [curiosity]   "understand glowing_sphere" priority=0.70

Section 7 — Hierarchical Plan:
  "eat apple": [pick(apple), eat(apple)]  score=0.579
  "survive":   decomposed into [find food, avoid danger]
               recursively: [observe, move, pick, eat, inspect, distance]
```

---

## Installation & Running

```bash
# Requirements (same as v3)
pip install numpy scipy scikit-learn Pillow

# Run v4 demo
cd cognifield
PYTHONPATH=.. python examples/demo_v4.py

# Run all tests
PYTHONPATH=.. python tests/test_all.py   # 64 (v2)
PYTHONPATH=.. python tests/test_v3.py    # 113 (v3)
PYTHONPATH=.. python tests/test_v4.py    # 123 (v4)
```

---

## Test Coverage

```
v2 (test_all.py):  64 /  64 passed
v3 (test_v3.py):  113 / 113 passed
v4 (test_v4.py):  123 / 123 passed
─────────────────────────────────
Total:            300 / 300 passed
```

---

## Scaling Suggestions

| Dimension | Current | Next Step |
|-----------|---------|-----------|
| World model | Hand-crafted rules + transitions | Learn transition dynamics with lightweight regression |
| Abstraction | 3 fixed methods | Add analogy-based transfer (A:B :: C:D) |
| Planning | Depth-3 beam | Monte Carlo Tree Search for longer horizons |
| Meta-learning | Parameter adjustment | Strategy discovery (generate new strategy variants) |
| Internal state | 6 fixed dimensions | Learned state representation from experience |
| Autonomy | Episode-based | True online continuous learning without resets |

---

## Research Questions

1. Does self-generated goal diversity correlate with faster learning?
2. Can the abstraction engine discover rules not explicitly programmed?
3. Does imagination (simulation) reduce dangerous actions compared to reactive agents?
4. At what memory size does consolidation become necessary for performance?
5. Can the meta-learner converge to good parameters faster than random search?

---

## Licence
MIT — experimental research use.
