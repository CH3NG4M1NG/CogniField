# CogniField v5 🧠

**Stable, Belief-Driven Cognitive Architecture**

> *"It is not enough to be intelligent — a reliable agent must know what it knows, doubt what it doubts, and never confuse the two."*

CogniField v5 transforms the autonomous v4 agent into a **stable, trustworthy system** that maintains consistent beliefs, handles uncertainty correctly, avoids dangerous actions, and improves reliably over time.

---

## What's New in v5

| Dimension | v4 | v5 |
|-----------|----|----|
| Knowledge | Raw rules, overwritten | ✓ **Bayesian belief system** (Beta dist, evidence aggregation) |
| Contradictions | Silently overwrite | ✓ **Conflict resolver** (5 strategies: evidence/recency/source/downgrade/experiment) |
| Consistency | No gate-keeping | ✓ **Consistency engine** (blocks/downgrades inconsistent updates, propagates implications) |
| Knowledge drift | Not detected | ✓ **Knowledge validator** (periodic re-verification against evidence) |
| Unknown objects | Act immediately | ✓ **Experiment engine** (safety ladder: observe→inspect→pick→eat) |
| Dangerous actions | Not prevented | ✓ **Risk engine** (probabilistic risk score, blocks unknown-object eating) |
| Memory | Vector only | ✓ **Episodic** (experience + importance) + **Procedural** (skill patterns) |
| Performance | Not measured | ✓ **Stability metrics** (belief_stability, consistency_score, grade A–F) |

---

## Architecture

```
cognifield/
│
├── world_model/
│   ├── belief_system.py        ★ NEW: Bayesian beliefs with evidence aggregation
│   ├── transition_model.py     state-action rules (v3)
│   ├── causal_graph.py         symbolic cause-effect (v3)
│   └── simulator.py            imagination (v4)
│
├── reasoning/
│   ├── conflict_resolver.py    ★ NEW: detects + resolves contradictions
│   ├── consistency_engine.py   ★ NEW: blocks inconsistent updates + propagates
│   ├── validation.py           ★ NEW: periodic belief re-verification
│   ├── abstraction.py          rule induction (v4)
│   ├── meta_learning.py        performance adaptation (v4)
│   └── reasoning_engine.py     self-correction (v2)
│
├── curiosity/
│   ├── experiment_engine.py    ★ NEW: structured safe experiments
│   └── advanced_curiosity.py   hypothesis generation (v3)
│
├── agent/
│   ├── risk_engine.py          ★ NEW: risk scoring + action blocking
│   ├── agent_v5.py             ★ NEW: 18-step stable loop
│   ├── agent_v4.py             v4 agent (still works)
│   ├── internal_state.py       cognitive state (v4)
│   ├── goal_generator.py       self-directed goals (v4)
│   └── goals.py                goal lifecycle (v3)
│
├── memory/
│   ├── episodic_memory.py      ★ NEW: time-tagged experiences + procedures
│   ├── memory_store.py         vector similarity (v2)
│   ├── relational_memory.py    concept graph (v3)
│   └── consolidation.py        merge/prune (v4)
│
├── evaluation/
│   └── metrics.py              ★ NEW: stability scoring (A–F grade)
│
├── tests/
│   ├── test_all.py              v2:  64 tests
│   ├── test_v3.py               v3: 113 tests
│   ├── test_v4.py               v4: 123 tests
│   └── test_v5.py               v5: 144 tests  ← NEW
│
└── examples/
    └── demo_v5.py              ★ NEW: 8-section stability demo
```

---

## The Belief System (Core Innovation)

Every fact is a **Belief** — not a raw value. Each belief tracks:

```python
Belief(key="apple.edible",
       value=True,
       confidence=0.828,   # Beta(4.8, 1.0) → prob 0.828
       pos_evidence=4.8,   # 4 direct observations + Laplace
       neg_evidence=1.0,
       source="direct_observation",
       last_updated=...)
```

**Update rule (Beta-Bayesian):**
```
New observation (direct_observation, weight=1.0):
  if agrees with current value: pos_evidence += 1.0
  if contradicts:               neg_evidence += 1.0
  confidence = pos_evidence / (pos_evidence + neg_evidence)
```

Source weights determine how much each update counts:
```
direct_observation: 1.0   ← strongest
inference:          0.7
abstraction:        0.6
simulation:         0.4
hypothesis:         0.2
prior:              0.1   ← weakest
```

**Convergence example:**
```
Update 1 (prior):               confidence = 0.550
Update 2 (hypothesis):          confidence = 0.587
Update 3 (simulation):          confidence = 0.646
Update 4 (direct_observation):  confidence = 0.738
Update 5 (direct_observation):  confidence = 0.793
Update 6 (direct_observation):  confidence = 0.828  ✓ reliable
```

---

## Conflict Resolution

When conflicting beliefs are detected, 5 resolution strategies are applied in priority order:

| Strategy | When Used | Outcome |
|----------|-----------|---------|
| `EVIDENCE_WINS` | One side has 2× more evidence | Higher-evidence belief wins |
| `SOURCE_PRIORITY` | High-trust source vs low-trust | Direct observation beats simulation |
| `RECENCY_WINS` | Old uncertain belief vs fresh evidence | Recent evidence wins |
| `DOWNGRADE_BOTH` | Evidence roughly tied | Both set to uncertain (0.5) |
| `REQUEST_EXPERIMENT` | No clear resolution | Flag for experimental testing |

---

## Consistency Engine

Before any belief is committed:
```
check_before_update("food.edible", False, source="simulation", weight=0.2)
→ Existing: food.edible=True (conf=0.960, certain)
→ Proposed: False (low-weight simulation)
→ Result: allowed=True, weight adjusted 0.20 → 0.06
→ Reason: "downgraded: conflicts with certain belief (conf=0.96)"
```

After committing, implications are propagated:
```
orange.is_a = food
→ food.edible = True (conf=0.96) is known
→ Infer: orange.edible = True (weight = 0.96 × 0.7 = 0.67)
→ Propagated automatically
```

---

## Experiment Engine — Safety Ladder

Before eating an unknown object:
```
prior_conf=0.10 → action=inspect  safety=SAFE      ✓
prior_conf=0.30 → action=pick     safety=LOW_RISK  ✓
prior_conf=0.50 → action=pick     safety=LOW_RISK  ✓
prior_conf=0.75 → action=eat      safety=HIGH_RISK ✓ (now confident enough)
```

The agent never eats an unknown object on first encounter. It follows the safety ladder, gathering evidence at each step before escalating.

---

## Risk Engine

```
observe(apple):        risk=0.000  → proceed
inspect(stone):        risk=0.000  → proceed
eat(apple, known edible): risk=0.000 → proceed
eat(purple_berry, unknown): risk=0.485 → CAUTION → suggest: combine
eat(stone, known inedible): risk=0.056 → proceed (known safe, low penalty)
```

---

## Stability Metrics

Grade A–F across 5 dimensions:
```
Dimension           Weight  Meaning
─────────────────── ──────  ──────────────────────────────────
success_rate         0.30   Action success rate
belief_stability     0.25   Std of belief confidences over time (low = stable)
consistency_score    0.20   How consistent same-action outcomes are
error_reduction      0.15   Whether mistakes are decreasing
conflict_rate        0.10   Fewer conflicts = more consistent beliefs
```

---

## Installation & Running

```bash
pip install numpy scipy scikit-learn Pillow

# v5 demo
PYTHONPATH=.. python examples/demo_v5.py

# All tests (444 total)
PYTHONPATH=.. python tests/test_all.py   #  64 (v2)
PYTHONPATH=.. python tests/test_v3.py    # 113 (v3)
PYTHONPATH=.. python tests/test_v4.py    # 123 (v4)
PYTHONPATH=.. python tests/test_v5.py    # 144 (v5)
```

---

## Test Coverage

```
v2 (test_all.py):   64 /  64  ✓
v3 (test_v3.py):   113 / 113  ✓
v4 (test_v4.py):   123 / 123  ✓
v5 (test_v5.py):   144 / 144  ✓
────────────────────────────────
Total:             444 / 444  ✓
```

---

## Licence
MIT — experimental research use.
