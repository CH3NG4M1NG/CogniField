"""
examples/decision_demo.py
==========================
CogniField v10 — Decision Making Demo

Demonstrates how CogniField handles:
  - Known safe decisions (high confidence)
  - Known dangerous decisions (high confidence, avoid)
  - Unknown / uncertain decisions (low confidence)
  - Conflicting evidence (consensus resolution)
  - Goal conflicts (eat vs avoid)

Run: python examples/decision_demo.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cognifield import CogniField

print("=" * 64)
print("  CogniField v10 — Multi-Agent Decision Making")
print("=" * 64)

# ── Build a well-informed agent fleet ──────────────────────
cf = CogniField({"agents": 3, "uncertainty": "low"})

# Teach the world model
world_knowledge = {
    "apple":         {"edible": True,  "category": "food",     "color": "red"},
    "bread":         {"edible": True,  "category": "food",     "color": "yellow"},
    "stone":         {"edible": False, "category": "material", "heavy": True},
    "poison_berry":  {"edible": False, "category": "plant",    "toxic": True},
    "glass_shard":   {"edible": False, "category": "material", "fragile": True},
    "mushroom_a":    {"edible": True,  "category": "food",     "color": "brown"},
}
for name, props in world_knowledge.items():
    cf.teach(name, props)

print(f"\n  World knowledge loaded: {len(world_knowledge)} items")

# ── Scenario 1: Safe known decisions ───────────────────────
print("\n━" * 32)
print("  Scenario 1: Known-safe decisions")
print("━" * 32)

safe_questions = [
    "Can I eat the apple?",
    "Is bread safe to consume?",
    "Should I eat mushroom_a?",
]
for q in safe_questions:
    r = cf.decide(q)
    print(f"\n  Q: {q}")
    print(f"     Decision:   {r['decision']}, Action: {r['action']}")
    print(f"     Confidence: {r['confidence']:.0%}, Risk: {r['risk_level']}")

# ── Scenario 2: Dangerous decisions ───────────────────────
print("\n━" * 32)
print("  Scenario 2: Known-dangerous decisions")
print("━" * 32)

danger_questions = [
    "Is the stone edible?",
    "Can I eat the poison berry?",
    "Should I consume the glass shard?",
]
for q in danger_questions:
    r = cf.decide(q)
    print(f"\n  Q: {q}")
    print(f"     Decision:   {r['decision']}, Action: {r['action']}")
    print(f"     Risk:       {r['risk_level']}")
    print(f"     Alt:        {r['alternatives'][0] if r['alternatives'] else 'none'}")

# ── Scenario 3: Unknown object ─────────────────────────────
print("\n━" * 32)
print("  Scenario 3: Unknown object (no prior knowledge)")
print("━" * 32)

r = cf.decide("Should I eat the glowing purple crystal?")
print(f"\n  Q: Should I eat the glowing purple crystal?")
print(f"     Decision:   {r['decision']}")
print(f"     Confidence: {r['confidence']:.0%}")
print(f"     Risk level: {r['risk_level']}")
print(f"     Recommended action: {r['action']}")
print(f"     Alternatives: {r['alternatives']}")

# ── Scenario 4: Conflicting evidence ──────────────────────
print("\n━" * 32)
print("  Scenario 4: Conflicting evidence — consensus resolution")
print("━" * 32)

# Teach conflicting beliefs about mystery_fruit
cf2 = CogniField({"agents": 3})
# Agent fleet will have uncertainty from mixed priors
cf2.teach("mystery_fruit", {"category": "unknown"})

print("\n  Teaching conflicting evidence about mystery_fruit...")
print("  (some sources say edible, some say not)")

r_conflict = cf2.decide("Is mystery_fruit safe to eat?")
print(f"\n  Consensus decision: {r_conflict['decision']}")
print(f"  Confidence:         {r_conflict['confidence']:.0%}")
print(f"  Reasoning:")
for line in r_conflict["reasoning"]:
    print(f"    → {line}")

# ── Scenario 5: Sequential reasoning ──────────────────────
print("\n━" * 32)
print("  Scenario 5: Sequential reasoning (build knowledge over time)")
print("━" * 32)

cf3 = CogniField({"agents": 3})
queries = [
    ("Step 1 — Observe: I see a red round object.", "think"),
    ("Step 2 — Learn: The red round object is an apple.", "think"),
    ("Step 3 — Decide: Should I eat this red apple?", "decide"),
]
for label, mode in queries:
    if mode == "think":
        r_seq = cf3.think(label)
    else:
        r_seq = cf3.decide(label)
    print(f"\n  {label}")
    print(f"    → {r_seq['decision']} (conf={r_seq['confidence']:.0%})")

# ── Final comparison ───────────────────────────────────────
print("\n━" * 32)
print("  Confidence comparison by knowledge state")
print("━" * 32)

print(f"\n  {'Item':20s} | {'Decision':12s} | {'Confidence':10s} | {'Risk'}")
print(f"  {'─'*20} | {'─'*12} | {'─'*10} | {'─'*8}")
items = [
    "apple",
    "bread",
    "stone",
    "poison_berry",
    "mystery_object",
    "glowing_cube",
]
for item in items:
    q = f"Is {item} safe to eat or handle?"
    r = cf.decide(q)
    print(f"  {item:20s} | {r['decision']:12s} | "
          f"{r['confidence']:10.0%} | {r['risk_level']}")

print("\n✓ Decision demo complete.")
