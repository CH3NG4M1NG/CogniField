"""
examples/basic_usage.py
========================
CogniField v10 — Basic Usage Guide

Run: python examples/basic_usage.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cognifield import CogniField

print("=" * 60)
print("  CogniField v10 — Basic Usage")
print("=" * 60)


# ──────────────────────────────────────────────
# 1. Simplest usage
# ──────────────────────────────────────────────
print("\n1. Simplest usage:")
print("-" * 40)

cf = CogniField()
result = cf.think("Is this berry safe to eat?")

print(f"  Question:   'Is this berry safe to eat?'")
print(f"  Decision:   {result['decision']}")
print(f"  Confidence: {result['confidence']:.0%}")
print(f"  Reasoning:  {result['reasoning'][0]}")


# ──────────────────────────────────────────────
# 2. Teach facts, then think
# ──────────────────────────────────────────────
print("\n2. Teach facts, then reason:")
print("-" * 40)

cf2 = CogniField()
(cf2
 .teach("apple",        {"edible": True,  "category": "food", "color": "red"})
 .teach("stone",        {"edible": False, "category": "material", "heavy": True})
 .teach("purple_berry", {"edible": True,  "category": "food"})
 .teach("mushroom_x",   {"edible": False, "category": "plant", "toxic": True})
)

questions = [
    "Can I eat the apple?",
    "Is the stone safe to eat?",
    "What about the purple berry?",
    "Should I eat mushroom_x?",
]

for q in questions:
    r = cf2.think(q)
    icon = "✓" if r["decision"] == "proceed" else ("✗" if r["decision"] == "avoid" else "?")
    print(f"  {icon} {q:45s} → {r['decision']:20s} ({r['confidence']:.0%})")


# ──────────────────────────────────────────────
# 3. Decision mode with risk assessment
# ──────────────────────────────────────────────
print("\n3. Decision mode with risk assessment:")
print("-" * 40)

cf3 = CogniField()
cf3.teach("red_fruit", {"edible": True, "category": "food"})

decision = cf3.decide("Should I eat the red fruit?")
print(f"  Decision:     {decision['decision']}")
print(f"  Action:       {decision['action']}")
print(f"  Risk level:   {decision['risk_level']}")
print(f"  Alternatives: {decision['alternatives']}")


# ──────────────────────────────────────────────
# 4. Configuration options
# ──────────────────────────────────────────────
print("\n4. Custom configuration:")
print("-" * 40)

configs = [
    {"agents": 1, "uncertainty": "none",    "label": "minimal  (1 agent, no noise)"},
    {"agents": 3, "uncertainty": "medium",  "label": "default  (3 agents, medium noise)"},
    {"agents": 5, "uncertainty": "high",    "label": "robust   (5 agents, high noise)"},
]
for cfg in configs:
    label = cfg.pop("label")
    cf_c  = CogniField(cfg)
    r     = cf_c.think("Is food edible?")
    print(f"  [{label}] → decision={r['decision']}, conf={r['confidence']:.0%}")


# ──────────────────────────────────────────────
# 5. Simulation
# ──────────────────────────────────────────────
print("\n5. Scenario simulation (5 steps):")
print("-" * 40)

cf4 = CogniField({"agents": 3})
cf4.teach("apple", {"edible": True,  "category": "food"})
cf4.teach("stone", {"edible": False, "category": "material"})

sim = cf4.simulate("foraging for food in an unfamiliar environment", steps=5)
print(f"  Scenario:        {sim['scenario']}")
print(f"  Success rate:    {sim['success_rate']:.0%}")
print(f"  Beliefs updated: {sim['belief_changes']}")
print(f"  Strategy used:   {sim['strategy']}")
if sim["outcomes"]:
    print(f"  Sample outcomes: {', '.join(sim['outcomes'][:3])}")


# ──────────────────────────────────────────────
# 6. System status
# ──────────────────────────────────────────────
print("\n6. System status:")
print("-" * 40)
status = cf4.status()
for k, v in status.items():
    if k != "config":
        print(f"  {k:20s}: {v}")

beliefs = cf4.get_beliefs(min_confidence=0.55)
print(f"\n  Known beliefs ({len(beliefs)}):")
for key, info in list(beliefs.items())[:4]:
    print(f"    {key:25s} = {str(info['value']):6s} "
          f"(conf={info['confidence']:.2f})")

print("\n✓ Basic usage complete.")
