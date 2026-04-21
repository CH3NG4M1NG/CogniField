# CogniField v3 🧠⚡

**A Goal-Driven, Planning-Capable Cognitive Architecture**

> *From reactive agent to planning agent:  
> CogniField v3 introduces a world model, multi-step planner, goal system,  
> relational memory, and advanced curiosity with hypothesis testing.*

---

## What's New in v3

| Module | v2 | v3 |
|--------|----|----|
| World understanding | None | **TransitionModel** + **CausalGraph** |
| Planning | Reactive (one step) | **Multi-step forward planner** (beam search) |
| Goals | Implicit | **GoalSystem** with lifecycle + priorities |
| Memory | Vector similarity only | **RelationalMemory** (typed fact graph) |
| Curiosity | Novelty detection | **Hypothesis generation + testing** |
| Environment | SimpleEnv | **RichEnv** (fragile/heavy/unknown + partial observability) |
| Agent loop | 6 steps | **10-step** perceive→encode→memory→novelty→goal→plan→act→feedback→update→repeat |

---

## Architecture Overview

```
cognifield/
│
├── encoder/              (unchanged from v2)
│   ├── text_encoder.py
│   ├── image_encoder.py
│   └── audio_encoder.py
│
├── latent_space/         (unchanged)
│   └── frequency_space.py
│
├── memory/               (extended)
│   ├── memory_store.py         ← vector memory (v2)
│   └── relational_memory.py    ← NEW: typed concept graph
│
├── world_model/          ← NEW
│   ├── transition_model.py     ← learns (state,action)→next_state
│   └── causal_graph.py         ← eat(apple)→satisfied, apple→is_a→food
│
├── planning/             ← NEW
│   └── planner.py              ← depth-limited beam search
│
├── agent/
│   ├── agent.py                ← v2 agent (unchanged)
│   ├── agent_v3.py             ← NEW: full 10-step loop
│   └── goals.py                ← NEW: goal lifecycle management
│
├── curiosity/
│   ├── curiosity_engine.py     ← v2 basic novelty (unchanged)
│   └── advanced_curiosity.py  ← NEW: hypothesis generation + testing
│
├── environment/
│   ├── simple_env.py           ← v2 (unchanged)
│   └── rich_env.py             ← NEW: fragile/heavy/unknown + partial visibility
│
├── reasoning/            (unchanged)
├── language/             (unchanged)
├── loss/                 (unchanged)
│
├── examples/
│   ├── demo.py           ← v2 demo
│   └── demo_v3.py        ← NEW: 7-section v3 demo
│
└── tests/
    ├── test_all.py       ← 64 v2 tests
    └── test_v3.py        ← 113 v3 tests
```

---

## Installation

```bash
pip install numpy scipy scikit-learn Pillow
```

No transformers, no PyTorch required.

---

## Quick Start

```bash
PYTHONPATH=. python examples/demo_v3.py
PYTHONPATH=. python tests/test_v3.py
```

---

## How the World Model Works

The **TransitionModel** learns cause-effect relationships from environment experience:

```
(state_vector, action) → (next_state_vector, reward, success)
```

Two complementary representations:

**1. Vector transitions** (generalisation via similarity):
```python
# Agent eats apple 3 times → learns eat(food) is good
tm.record(state_before, "eat", state_after, reward=+0.5, success=True,
          object_name="apple", object_category="food")

# Later, predict outcome for a NEW food object (bread):
outcome, reward, conf = tm.predict_outcome("eat", "food")
# → ("success", +0.50, 1.00)  ← generalised from apple experience
```

**2. Symbolic rules** (WorldRule with Bayesian confidence):
```
eat(food)     → success, reward=+0.50, conf=1.00  ✓ reliable
eat(material) → failure, reward=-0.20, conf=0.00  ✓ reliable
pick(food)    → success, reward=+0.10, conf=0.50
```

Confidence is updated with each observation using hit/miss counting. A rule is *reliable* when confidence ≥ 0.6 and ≥ 2 observations.

The **CausalGraph** stores symbolic knowledge:
```python
eat(apple) → causes → satisfied    (weight=0.9)
eat(stone) → causes → damaged      (weight=0.9)
apple      → is_a   → food
apple      → edible → True
glass_jar  → fragile → True
```

---

## How Planning Works

The **Planner** uses depth-limited beam search to generate multi-step action sequences before acting:

```
Goal: "eat apple"
Current state: standing in world, apple visible

Step 1: Try all candidate actions:
  pick(apple) → score=0.64  (gets us closer to "apple in hand")
  eat(stone)  → score=0.12  (world model says: dangerous)
  observe     → score=0.40  (neutral)

Step 2: Expand best candidates (beam_width=3):
  [pick(apple)] + eat(apple) → score=0.70  ← BEST PLAN
  [pick(apple)] + pick(stone) → score=0.45
  ...

Final plan: [pick(apple), eat(apple)]  score=0.70
```

Step scoring formula:
```
step_score = 0.50 × goal_proximity(predicted_state, goal_vec)
           + 0.30 × expected_reward  (from world model)
           + 0.20 × rule_confidence
```

Safety check: plans containing steps with `expected_reward < -0.3` and `confidence ≥ 0.6` are flagged as unsafe.

**Symbolic fast path**: When the causal graph has enough knowledge, the planner uses direct rule lookup (faster than search). Falls back to beam search for unknown situations.

---

## How the Goal System Works

Goals drive the agent between steps, persisting across the perceive-act loop:

```python
# Define goals
agent.add_goal("eat apple",         GoalType.EAT_OBJECT, priority=0.85)
agent.add_goal("avoid eating stone", GoalType.AVOID,      priority=0.95)
agent.add_goal("explore purple_berry", GoalType.EXPLORE,  priority=0.50)

# Each step: select highest-priority active goal
goal = agent.goals.select_active_goal()
# → "avoid eating stone" (highest priority)

# After action: check if goal satisfied
agent.goals.check_goal_satisfied(goal, env_feedback)
agent.goals.mark_completed(goal)
```

Goal priority scoring (with aging):
```
effective_priority = base_priority + min(0.2, age_seconds/300) - attempts × 0.05
```

Old unresolved goals become more urgent. Repeated failures reduce urgency to prevent obsession.

---

## How Relational Memory Works

**RelationalMemory** stores typed facts as a directed concept graph:

```python
rm.add_fact("apple", "is_a", "food")
rm.add_fact("apple", "edible", True)
rm.add_fact("apple", "color", "red")
rm.add_fact("eat(apple)", "outcome", "satisfied")

# Queries
rm.find_edible()          # → ["apple", "bread", "water"]
rm.find_dangerous()       # → ["stone", "hammer", "glass_jar"]
rm.query("fragile", True) # → [("glass_jar", 1.0), ("water", 1.0)]
rm.what_is("apple")       # → "apple: is_a=food, edible=True, color=red"
rm.get_category("stone")  # → "material"
```

Facts are reinforced when re-observed (confidence increases) and can be queried by predicate+value or by subject.

---

## How Advanced Curiosity Works

When an unknown concept is encountered:

**1. Novelty detection:**
```
novelty = 0.7 × (1 - max_cosine_similarity_to_known)
        + 0.3 × symbolic_unknown_flag
```

**2. Hypothesis generation** (from vector similarity to known concepts):
```
purple_berry (novelty=0.65)
  → similar to 'bread' (sim=0.45)
  → hypothesis: edible=True  (conf=0.36, basis="similar to bread")
  → hypothesis: category=food (conf=0.36)
  → suggested action: "inspect"
```

**3. Hypothesis testing** (from environment observations):
```python
# After inspect(purple_berry) reveals edible=None:
curiosity.update_hypotheses("purple_berry", "edible", None)
# → hypothesis refuted (predicted True, observed None)

# After eating purple_berry reveals edible=True:
curiosity.update_hypotheses("purple_berry", "edible", True)
# → hypothesis confirmed!
```

Hypotheses progress through: `open → confirmed / refuted`

The `best_hypothesis_to_test()` method returns the most valuable unresolved hypothesis, prioritising important predicates (`edible`, `fragile`) with high uncertainty (confidence near 0.5).

---

## Demo Scenario: Eat Apple, Avoid Stone, Explore Unknown

```bash
PYTHONPATH=. python examples/demo_v3.py
```

Sections:
1. **World Model** — learn rules from simulated experience
2. **Planning** — generate `[pick(apple) → eat(apple)]` plan
3. **Relational Memory** — query `what_is()`, `find_edible()`, `query(fragile, True)`
4. **Advanced Curiosity** — form and test hypotheses about `purple_berry`
5. **Goal System** — lifecycle, priority, auto-inference
6. **Rich Environment** — partial observability, fragile objects, unknown edibility
7. **Full Agent Loop** — all modules integrated: phase A through F

---

## Key Results

```
World model rules (after 7 transitions):
  eat(food)     → success  reward=+0.50  conf=1.00  ✓ reliable
  eat(material) → failure  reward=-0.20  conf=0.00  ✓ reliable
  pick(food)    → success  reward=+0.10  conf=1.00  ✓ reliable

Relational memory:
  what_is('apple'):  edible=True, category=food, color=red, fragile=False
  find_edible():     ['apple', 'bread', 'water']
  find_dangerous():  ['stone', 'hammer', 'glass_jar']

Curiosity (purple_berry):
  Novelty: 0.535
  Hypothesis: edible=True  (conf=0.36, basis: similar to bread)
  Suggested action: eat
  After inspect → hypotheses updated

Plan for "eat apple":
  [pick(apple) → eat(apple)]  score=0.705  safe=True
```

---

## Running Tests

```bash
# v2 tests (64)
PYTHONPATH=. python tests/test_all.py

# v3 tests (113)
PYTHONPATH=. python tests/test_v3.py
```

Total: **177 tests, all passing.**

---

## Scaling Suggestions

| Horizon | Upgrade |
|---------|---------|
| Now | Increase `plan_depth` to 6-8; add more object types to `RichEnv` |
| Short | Add A* search as alternative planner for larger state spaces |
| Medium | Replace WorldRule Bayesian counting with a small neural predictor |
| Long | Multi-agent version: agents share relational memory, develop communication |
| Research | Does hypothesis-driven curiosity outperform random exploration on novel environments? |

---

## Architectural Philosophy

CogniField is **not** trying to build AGI.

It is an **experimental platform** for studying:

- How planning changes agent behaviour vs reactive systems
- Whether world-model generalisation (similar state → similar outcome) is sufficient for simple environments
- Whether curiosity + hypothesis testing can substitute for labelled supervision
- What is the minimum architecture needed for goal-directed embodied learning

Each module is **independently replaceable**: swap the symbolic WorldRule learner for a neural one without touching the planner. Replace the beam-search planner with Monte Carlo Tree Search. Replace RelationalMemory with a proper graph database. The interfaces are stable.

---

## Licence

MIT — experimental research use.
