# CogniField v6 🧠🤝

**Multi-Agent Collaborative Cognitive Architecture**

> *"Intelligence distributed across minds can solve problems that no single mind can — if those minds can communicate, trust, and reach consensus."*

CogniField v6 extends the stable v5 single agent into a **fleet of collaborating agents** that share knowledge, build trust, reach consensus on contested beliefs, and learn socially from each other's direct experiences.

---

## What's New in v6

| Capability | v5 | v6 |
|-----------|----|----|
| Agents | 1 autonomous agent | ✓ N agents with shared infrastructure |
| Communication | None | ✓ Typed message bus (BELIEF/WARNING/OBSERVATION/QUESTION) |
| Shared knowledge | Private only | ✓ Community SharedMemory with versioning |
| Trust | N/A | ✓ Per-peer trust (accuracy × consistency × responsiveness) |
| Consensus | N/A | ✓ Multi-strategy voting (confidence/evidence/trust-weighted) |
| Roles | None | ✓ EXPLORER / ANALYST / RISK_MANAGER / PLANNER |
| Social learning | N/A | ✓ Agent B learns from A's observations without direct testing |
| Conflicts | Single-agent | ✓ Cross-agent conflict detection + resolution |
| Coordination | N/A | ✓ AgentManager: synchronised stepping + consensus rounds |

---

## Architecture

```
cognifield/
│
├── communication/
│   └── communication_module.py  ★ Message bus (CommunicationModule + Message)
│
├── memory/
│   └── shared_memory.py         ★ Community knowledge store (SharedMemory)
│
├── agents/
│   ├── trust_system.py          ★ Per-peer trust scoring
│   ├── agent_v6.py              ★ Multi-aware agent extending v5
│   └── agent_manager.py         ★ Fleet coordinator
│
├── reasoning/
│   └── consensus_engine.py      ★ Multi-agent belief aggregation
│
└── tests/
    └── test_v6.py               ★ 142 tests
```

---

## Message Types

| Type | Meaning | Example |
|------|---------|---------|
| `BELIEF` | "I believe X has property Y" | `apple.edible = True (conf=0.85)` |
| `OBSERVATION` | "I observed: action → outcome" | `eat(apple) → success, r=+0.5` |
| `WARNING` | "Danger! Do not do X" | `stone.edible = False (conf=0.92)` |
| `SUGGESTION` | "You should try this action" | `inspect(purple_berry)` |
| `QUESTION` | "What do you know about X?" | `purple_berry.edible?` |
| `CONSENSUS` | "We agreed: X is Y" | `food.edible = True (n=3 agents)` |

---

## Trust System

Each agent tracks three dimensions of trust per peer:

```
trust_score = 0.5 × accuracy       (how often their beliefs are correct)
            + 0.3 × consistency    (how non-contradictory they are)
            + 0.2 × responsiveness (how usefully they communicate)
```

Effects:
```python
# High-trust agent (accuracy=0.85)
message_weight("peer_A", confidence=0.8) → 0.566   # significant influence

# Low-trust agent (accuracy=0.30)
message_weight("peer_C", confidence=0.8) → 0.218   # minor influence
```

Trust decays toward neutral (0.5) between interactions — no agent gets permanent authority.

---

## Consensus Engine — 4 Strategies

```python
# Scenario: 3 agents vote on apple.edible
votes = [
    AgentVote("A", True,  conf=0.85, evidence=4.0, trust=0.85),
    AgentVote("B", True,  conf=0.78, evidence=3.0, trust=0.75),
    AgentVote("C", False, conf=0.55, evidence=1.5, trust=0.45),
]

# Trust-weighted: True wins 88% of weighted votes
# Evidence-weighted: True wins 77% of evidence
# Confidence-weighted: True wins 79% of confidence
# Supermajority (≥60%): True wins (88% > 60%)
→ Consensus: apple.edible = True (conf=0.783, agreement=88%)
```

If no strategy can reach supermajority: the key is marked **contested** and a collaborative experiment is scheduled.

---

## Social Learning

**Without social learning:**
```
Agent B wants to know if stone is edible.
B must eat stone → get hurt → learn from pain.
```

**With social learning:**
```
Agent A eats stone → gets hurt → broadcasts WARNING.
Agent B receives WARNING → weights by trust(A=0.71) → updates belief.
B's stone.edible confidence: 0.500 → 0.657 (without ever touching stone).
```

This is **observational learning** — one of the most powerful mechanisms in biological intelligence.

---

## Agent Roles

| Role | Novelty Threshold | Risk Tolerance | Share Frequency | Bias |
|------|------------------|----------------|-----------------|------|
| EXPLORER | 0.30 (low → curious) | 0.40 (bold) | Every 2 steps | Find unknowns |
| ANALYST | 0.45 | 0.30 (careful) | Every 3 steps | Verify beliefs |
| RISK_MANAGER | 0.50 (conservative) | 0.25 (very safe) | Every step | Warn about dangers |
| PLANNER | 0.40 | 0.35 | Every 4 steps | Execute goals |

---

## Full 17-Step Agent Loop (v6)

```
 1. Observe environment
 2. Observe peers (incoming messages)
 3. Update private memory
 4. Validate knowledge (periodic)
 5. Detect novelty
 6. Receive + evaluate messages (trust-weighted)
 7. Update beliefs from peer messages
 8. Share relevant knowledge (periodic)
 9. Generate goals (including social goals)
10. Select highest-priority goal
11. Plan hierarchically
12. Simulate outcomes
13. Risk check
14. Act
15. Receive feedback
16. Update beliefs + trust scores
17. Participate in consensus (periodic)
```

---

## Demo: A discovers, B confirms, C uses without testing

```
Step 1 — A (Explorer) encounters purple_berry
         A experiments: inspect → pick → eat
         A broadcasts: "purple_berry.edible=True (conf=0.82)"

Step 2 — B (Analyst) receives A's message
         B checks: trust(A) = 0.71 → message weight = 0.58
         B updates: purple_berry.edible → conf 0.50 → 0.66
         B confirms independently: inspect(purple_berry)
         B broadcasts: "purple_berry.edible=True (conf=0.78)"

Step 3 — Consensus round
         A: True(0.82), B: True(0.78) → agreement=100%
         Consensus: purple_berry.edible=True (conf=0.783)
         Written to SharedMemory for all to read

Step 4 — C (Planner) reads from SharedMemory
         Sees purple_berry.edible=True from consensus
         C skips all experiments → directly eats purple_berry
         C learned from the community in 0 direct experiments
         (vs A's 3 experiments, B's 1 experiment)
```

---

## Installation & Running

```bash
pip install numpy scipy scikit-learn Pillow

# v6 demo
cd cognifield
PYTHONPATH=.. python examples/demo_v6.py

# All tests (586 total)
PYTHONPATH=.. python tests/test_all.py   #  64 (v2)
PYTHONPATH=.. python tests/test_v3.py    # 113 (v3)
PYTHONPATH=.. python tests/test_v4.py    # 123 (v4)
PYTHONPATH=.. python tests/test_v5.py    # 144 (v5)
PYTHONPATH=.. python tests/test_v6.py    # 142 (v6)
```

---

## Test Coverage

```
v2 (test_all.py):    64 /  64  ✓
v3 (test_v3.py):    113 / 113  ✓
v4 (test_v4.py):    123 / 123  ✓
v5 (test_v5.py):    144 / 144  ✓
v6 (test_v6.py):    142 / 142  ✓
────────────────────────────────
Total:              586 / 586  ✓
```

---

## Licence
MIT — experimental research use.
