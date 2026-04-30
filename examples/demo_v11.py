"""
examples/demo_v11.py
=====================
CogniField v11 — Self-Learning Deep Reasoning Demo

Demonstrates all v11 capabilities:
  1. Learning-first pipeline     (checks knowledge before deciding)
  2. Deep thinking mode          (multi-step structured deliberation)
  3. Unknown safety rule         (avoid when input is unknown)
  4. Experience engine           (learns from outcomes automatically)
  5. World model v2              (object → category → property → effect)
  6. Self-correction             (detects and fixes wrong beliefs)
  7. Pre-decision simulation     (validates before committing)
  8. Full scenario               (unknown → learns → improves over time)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

def sep(t): print(f"\n{'═'*64}\n  {t}\n{'═'*64}")
def sec(t): print(f"\n  {'─'*60}\n  {t}\n  {'─'*60}")


# ════════════════════════════════════════════════════════
# 1. Learning-First Pipeline
# ════════════════════════════════════════════════════════
def demo_learning_first():
    sep("1. Learning-First Pipeline")
    from cognifield.cognifield_v11 import CogniFieldV11

    cf = CogniFieldV11({"agents": 2, "learning_first": True, "thinking_mode": "deep"})
    cf.teach("apple", {"edible": True,  "category": "food"})
    cf.teach("stone", {"edible": False, "category": "material"})

    sec("Known object — direct knowledge path")
    r = cf.think("Is apple safe to eat?")
    print(f"  Decision:       {r['decision']} ({r['confidence']:.0%})")
    print(f"  Knowledge state:{r['knowledge_state']}")
    print(f"  Thinking steps: {r['thinking_steps']}")
    print(f"  Reasoning (first 3 steps):")
    for line in r["reasoning"][:3]:
        print(f"    → {line[:70]}")

    sec("Unknown object — learning probe triggered")
    r2 = cf.think("Is purple_crystal safe to eat?")
    print(f"  Decision:       {r2['decision']} ({r2['confidence']:.0%})")
    print(f"  Knowledge state:{r2['knowledge_state']}")
    print(f"  Reasoning (learning mode):")
    for line in r2["reasoning"][:4]:
        print(f"    → {line[:70]}")

    sec("Partial knowledge — category inference")
    # Teach category but not direct edibility
    cf.teach("mango", {"category": "food"})  # no explicit edible=True
    r3 = cf.think("Can I eat mango?")
    print(f"  Decision:       {r3['decision']} ({r3['confidence']:.0%})")
    print(f"  Knowledge state:{r3['knowledge_state']}")
    if r3.get("world_model", {}).get("inferred_value") is not None:
        wm = r3["world_model"]
        print(f"  World model inferred: edible={wm['inferred_value']} "
              f"(conf={wm['inferred_conf']:.2f})")


# ════════════════════════════════════════════════════════
# 2. Deep Thinking Mode
# ════════════════════════════════════════════════════════
def demo_deep_thinking():
    sep("2. Deep Thinking Mode — Multi-Step Deliberation")
    from cognifield.cognifield_v11 import CogniFieldV11
    from cognifield.core.deep_thinker import DeepThinker, ThinkingMode

    cf_fast = CogniFieldV11({"agents": 2, "thinking_mode": "fast"})
    cf_deep = CogniFieldV11({"agents": 2, "thinking_mode": "deep"})
    cf_auto = CogniFieldV11({"agents": 2, "thinking_mode": "auto"})

    for cf_inst in [cf_fast, cf_deep, cf_auto]:
        cf_inst.teach("red_berry", {"edible": True, "category": "food"})
        cf_inst.teach("stone",     {"edible": False,"category": "material"})

    sec("Comparing reasoning depth by mode")
    for cf_inst, label in [(cf_fast,"FAST"), (cf_deep,"DEEP"), (cf_auto,"AUTO")]:
        r = cf_inst.think("Is red_berry safe to eat?")
        print(f"\n  [{label}] Decision={r['decision']:22s} "
              f"conf={r['confidence']:.0%} steps={r['thinking_steps']}")
        if r["reasoning"]:
            print(f"    First step: {r['reasoning'][0][:65]}")
            if len(r["reasoning"]) > 1:
                print(f"    Last step:  {r['reasoning'][-1][:65]}")

    sec("Deep thinking trace for dangerous object")
    dt = DeepThinker(mode=ThinkingMode.DEEP, min_steps=5, confidence_target=0.70)
    from cognifield.world_model.belief_system import BeliefSystem
    bs = BeliefSystem()
    for _ in range(3):
        bs.update("glass_shard.edible",   False, "direct_observation")
        bs.update("glass_shard.dangerous",True,  "direct_observation")
        bs.update("glass_shard.category", "material","direct_observation")

    result = dt.think("glass_shard", "edible", bs)
    print(f"\n  Input: glass_shard.edible")
    print(f"  Final: {result.decision} ({result.confidence:.0%}) "
          f"safe={result.safe} steps={result.n_steps}")
    print(f"\n  Full reasoning chain:")
    for i, t in enumerate(result.thoughts[:7]):
        print(f"    Step {i+1} [{t.step.value:20s}] "
              f"conf={t.confidence:.3f} delta={t.delta:+.3f}")
        print(f"           {t.finding[:62]}")


# ════════════════════════════════════════════════════════
# 3. Unknown Safety Rule
# ════════════════════════════════════════════════════════
def demo_unknown_safety():
    sep("3. Unknown Safety Rule")
    from cognifield.cognifield_v11 import CogniFieldV11

    cf_safe   = CogniFieldV11({"agents": 2, "unknown_safety_rule": True})
    cf_unsafe = CogniFieldV11({"agents": 2, "unknown_safety_rule": False})

    unknowns = [
        "glowing_orb", "purple_crystal", "mystery_substance",
        "unknown_fungus", "alien_plant"
    ]
    sec("With unknown_safety_rule=True vs False")
    print(f"\n  {'Input':22s} | {'SAFE rule':16s} | {'NO rule':16s}")
    print(f"  {'─'*22} | {'─'*16} | {'─'*16}")
    for obj in unknowns:
        r_safe   = cf_safe.think(f"Is {obj} edible?")
        r_unsafe = cf_unsafe.think(f"Is {obj} edible?")
        safe_dec   = f"{r_safe['decision']} ({r_safe['confidence']:.0%})"
        unsafe_dec = f"{r_unsafe['decision']} ({r_unsafe['confidence']:.0%})"
        print(f"  {obj:22s} | {safe_dec:16s} | {unsafe_dec}")

    sec("After teaching facts — safety rule relaxes")
    cf_safe.teach("glowing_orb", {"edible": True, "category": "food"})
    r_known = cf_safe.think("Is glowing_orb edible?")
    print(f"\n  After teaching glowing_orb=food:")
    print(f"  decision={r_known['decision']} ({r_known['confidence']:.0%}) "
          f"state={r_known['knowledge_state']}")


# ════════════════════════════════════════════════════════
# 4. Experience Engine
# ════════════════════════════════════════════════════════
def demo_experience_engine():
    sep("4. Experience Engine — Learning From Outcomes")
    from cognifield.cognifield_v11 import CogniFieldV11

    cf = CogniFieldV11({"agents": 2, "thinking_mode": "deep", "uncertainty": "low"})
    cf.teach("test_berry", {"category": "food"})  # unknown edibility

    sec("Before learning — uncertain decision")
    r_before = cf.think("Is test_berry edible?")
    print(f"  Before: {r_before['decision']} ({r_before['confidence']:.0%})")

    sec("Simulating repeated outcomes")
    outcomes = [
        (True,  True,  "eat", +0.50),   # ate it, was safe
        (True,  True,  "eat", +0.50),
        (True,  True,  "eat", +0.45),
        (True,  True,  "eat", +0.50),
        (True,  True,  "eat", +0.50),
    ]
    for i, (pred, actual, action, reward) in enumerate(outcomes):
        lo = cf.learn_from_outcome(
            f"eat test_berry attempt {i+1}",
            "test_berry", "edible",
            pred, actual, action, reward
        )
        print(f"  Outcome {i+1}: correct={pred==actual} "
              f"corrections={lo['corrections_made']}")

    sec("After learning — belief updated")
    r_after = cf.think("Is test_berry edible?")
    print(f"  After:  {r_after['decision']} ({r_after['confidence']:.0%})")
    print(f"  Rules derived: {cf._exp_engine.derived_rules()}")

    sec("Learning from a WRONG confident prediction")
    cf2 = CogniFieldV11({"agents": 2})
    cf2.teach("poison_berry", {"edible": True, "category": "food"})  # WRONG!
    r_wrong_before = cf2.think("Is poison_berry edible?")
    print(f"\n  Before: {r_wrong_before['decision']} ({r_wrong_before['confidence']:.0%})")

    # Three wrong outcomes
    for i in range(3):
        lo = cf2.learn_from_outcome(
            "eat poison_berry", "poison_berry", "edible",
            prediction=True, actual=False, action="eat", reward=-0.8
        )
        if lo["corrections_made"] > 0:
            print(f"  Correction {i+1}: {lo['details'][0][:65]}")

    r_wrong_after = cf2.think("Is poison_berry edible?")
    print(f"  After:  {r_wrong_after['decision']} ({r_wrong_after['confidence']:.0%})")


# ════════════════════════════════════════════════════════
# 5. World Model v2
# ════════════════════════════════════════════════════════
def demo_world_model():
    sep("5. World Model v2 — Object → Category → Property → Effect")
    from cognifield.core.world_model_v2 import WorldModelV2

    wm = WorldModelV2()

    sec("Pre-seeded category knowledge")
    for cat in ["food", "material", "tool"]:
        e = wm.get_entity(cat)
        if e:
            props = {k: (v, round(e.get_confidence(k), 2))
                     for k, v in list(e.properties.items())[:4]}
            print(f"  {cat}: {props}")

    sec("Adding entities with category inheritance")
    wm.add_entity("apple",  "food",     {"color": "red",    "size": "small"})
    wm.add_entity("stone",  "material", {"color": "grey",   "heavy": True})
    wm.add_entity("hammer", "tool",     {"material": "metal"})
    wm.add_entity("mango",  "food",     {"color": "yellow"})

    sec("Property inference via hierarchy")
    tests = [
        ("apple",  "edible"),
        ("stone",  "edible"),
        ("mango",  "edible"),
        ("hammer", "edible"),
        ("apple",  "digestible"),
        ("stone",  "digestible"),
    ]
    print(f"\n  {'Entity':10s} | {'Property':12s} | {'Inferred':8s} | {'Confidence':10s}")
    print(f"  {'─'*10} | {'─'*12} | {'─'*8} | {'─'*10}")
    for entity, prop in tests:
        val, conf = wm.infer_property(entity, prop)
        print(f"  {entity:10s} | {prop:12s} | {str(val):8s} | {conf:.3f}")

    sec("Effect inference")
    effects = [("eat","apple"), ("eat","stone"), ("pick","apple"),
               ("drop","hammer"), ("eat","mango")]
    print(f"\n  {'Action(Target)':20s} | {'Effect':12s} | {'Reward':6s} | {'Conf'}")
    print(f"  {'─'*20} | {'─'*12} | {'─'*6} | {'─'*5}")
    for action, target in effects:
        effect, reward, conf = wm.infer_effect(action, target)
        print(f"  {action}({target}):".ljust(22) +
              f"  {effect:12s} | {reward:+.2f}   | {conf:.2f}")

    sec("Causal chains for deep thinker")
    chains = wm.causal_chains("apple", "edible")
    print(f"  apple.edible causal chains: {chains}")

    sec("World knowledge query")
    from cognifield.cognifield_v11 import CogniFieldV11
    cf = CogniFieldV11({"agents": 2})
    cf.teach("apple", {"edible": True, "category": "food", "color": "red"})
    wk = cf.world_knowledge("apple")
    print(f"\n  world_knowledge('apple'):")
    print(f"    category:   {wk['category']}")
    print(f"    properties: {list(wk['properties'].keys())}")
    print(f"    eat effect: reward={wk['eat_effect']['reward']:+.2f} "
          f"({wk['eat_effect']['effect']})")

    print(f"\n  World model summary: {wm.summary()}")


# ════════════════════════════════════════════════════════
# 6. Self-Correction
# ════════════════════════════════════════════════════════
def demo_self_correction():
    sep("6. Self-Correction — Detecting and Fixing Wrong Beliefs")
    from cognifield.cognifield_v11 import CogniFieldV11

    cf = CogniFieldV11({"agents": 2, "self_correction": True, "correction_interval": 5})
    cf.teach("mushroom_x", {"edible": True, "category": "food"})  # wrong!

    sec("Initial wrong belief")
    r1 = cf.think("Is mushroom_x edible?")
    print(f"  Initial: {r1['decision']} ({r1['confidence']:.0%})")

    sec("Feeding wrong outcomes")
    for i in range(4):
        lo = cf.learn_from_outcome(
            "eat mushroom_x", "mushroom_x", "edible",
            prediction=True, actual=False, action="eat", reward=-0.6
        )
        print(f"  Outcome {i+1}: correct=False "
              f"corrections={lo['corrections_made']}")

    sec("Self-reflect + audit")
    reflection = cf.self_reflect()
    print(f"\n  Findings:    {reflection['findings']}")
    print(f"  Corrections: {len(reflection['corrections'])} beliefs fixed")
    if reflection['corrections']:
        for c in reflection['corrections'][:2]:
            print(f"    → {c[:65]}")

    sec("After self-correction")
    r_after = cf.think("Is mushroom_x edible?")
    print(f"  After correction: {r_after['decision']} ({r_after['confidence']:.0%})")


# ════════════════════════════════════════════════════════
# 7. Full v11 Scenario — Unknown to Known Over Time
# ════════════════════════════════════════════════════════
def demo_full_scenario():
    sep("7. Full Scenario — Unknown → Learns → Improves")
    from cognifield.cognifield_v11 import CogniFieldV11

    cf = CogniFieldV11({
        "agents":            3,
        "thinking_mode":     "auto",
        "uncertainty":       "low",
        "learning_first":    True,
        "unknown_safety_rule":True,
        "self_correction":   True,
        "sim_before_decide": True,
        "confidence_target": 0.65,
    })

    # Background knowledge
    cf.teach("apple",  {"edible": True,  "category": "food"})
    cf.teach("stone",  {"edible": False, "category": "material"})
    cf.teach("bread",  {"edible": True,  "category": "food"})

    print(f"\n  System: {cf}")
    print(f"  Config: mode={cf._v11_cfg.thinking_mode}, "
          f"min_steps={cf._v11_cfg.min_thinking_steps}, "
          f"confidence_target={cf._v11_cfg.confidence_target}")

    sec("Phase A: First encounter with jungle_berry (unknown)")
    r1 = cf.think("Is jungle_berry safe to eat?")
    print(f"  Step 1 — First encounter:")
    print(f"    Decision:  {r1['decision']} ({r1['confidence']:.0%})")
    print(f"    State:     {r1['knowledge_state']}")
    print(f"    Steps:     {r1['thinking_steps']}")

    sec("Phase B: Teaching category then re-evaluating")
    cf.teach("jungle_berry", {"category": "food"})  # tells it the category
    r2 = cf.think("Is jungle_berry safe to eat?")
    print(f"  Step 2 — After knowing it's food:")
    print(f"    Decision:  {r2['decision']} ({r2['confidence']:.0%})")
    print(f"    State:     {r2['knowledge_state']}")
    wm = r2.get("world_model", {})
    if wm.get("inferred_value"):
        print(f"    Inferred:  edible={wm['inferred_value']} "
              f"(conf={wm['inferred_conf']:.2f})")

    sec("Phase C: Learning from real outcomes (3 successes)")
    for i in range(3):
        lo = cf.learn_from_outcome(
            "eat jungle_berry", "jungle_berry", "edible",
            prediction=True, actual=True, action="eat", reward=+0.45
        )
    r3 = cf.think("Is jungle_berry safe to eat?")
    print(f"  Step 3 — After 3 successful eat outcomes:")
    print(f"    Decision:  {r3['decision']} ({r3['confidence']:.0%})")
    print(f"    State:     {r3['knowledge_state']}")

    sec("Phase D: Contradictory evidence (1 failure)")
    lo4 = cf.learn_from_outcome(
        "eat jungle_berry", "jungle_berry", "edible",
        prediction=True, actual=False, action="eat", reward=-0.3
    )
    r4 = cf.think("Is jungle_berry safe to eat?")
    print(f"  Step 4 — After 1 failure:")
    print(f"    Decision:  {r4['decision']} ({r4['confidence']:.0%})")
    if r4.get("contradictions"):
        print(f"    Contradictions: {r4['contradictions'][:1]}")

    sec("Phase E: Confidence progression chart")
    confs = [r1["confidence"], r2["confidence"], r3["confidence"], r4["confidence"]]
    labels = ["unknown", "+category", "+3 successes", "+1 failure"]
    print(f"\n  jungle_berry.edible confidence over time:")
    for label, conf in zip(labels, confs):
        bar = "█" * int(conf * 30) + "░" * (30 - int(conf * 30))
        print(f"    {label:15s} [{bar}] {conf:.0%}")

    sec("Phase F: Self-reflect at end")
    reflection = cf.self_reflect()
    v11_status = cf.status()
    print(f"\n  Think calls:    {v11_status['think_calls']}")
    print(f"  Thinking stats: {v11_status['v11']['thinking']}")
    print(f"  Experience:     {v11_status['v11']['experience']}")
    print(f"  World model:    {v11_status['v11']['world_model']}")
    if reflection["findings"]:
        print(f"  Findings:       {reflection['findings']}")
    print(f"  Rules derived:  {cf._exp_engine.derived_rules()}")

    sec("Final decision quality comparison")
    scenarios = [
        ("known-safe",    "Is apple edible?",            ("proceed", "proceed_with_caution")),
        ("known-danger",  "Is stone edible?",            ("avoid",)),
        ("unknown",       "Is alien_fruit edible?",      ("avoid", "investigate")),
        ("learned",       "Is jungle_berry safe to eat?",("proceed", "proceed_with_caution", "avoid")),
        ("partial",       "Can I eat mango?",             ("proceed", "proceed_with_caution", "investigate")),
    ]
    cf.teach("mango", {"category": "food"})
    print(f"\n  {'Scenario':15s} | {'Decision':22s} | {'Conf':5s} | {'Steps':5s} | {'State'}")
    print(f"  {'─'*15} | {'─'*22} | {'─'*5} | {'─'*5} | {'─'*8}")
    for label, q, _ in scenarios:
        r = cf.think(q)
        print(f"  {label:15s} | {r['decision']:22s} | "
              f"{r['confidence']:5.0%} | {r['thinking_steps']:5d} | "
              f"{r['knowledge_state']}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   CogniField v11 — Self-Learning Deep Reasoning Demo     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    demo_learning_first()
    demo_deep_thinking()
    demo_unknown_safety()
    demo_experience_engine()
    demo_world_model()
    demo_self_correction()
    demo_full_scenario()
    print("\n" + "═"*64 + "\n  v11 Demo complete.\n" + "═"*64)
