"""
examples/demo_v11_part2.py
===========================
CogniField v11 Part 2 — Embodied Intelligence Demo

Full loop: THINK → SIMULATE → DECIDE → ACT → OBSERVE → LEARN

Scenarios:
  1. Agent eats apple → success → satisfied
  2. Agent eats stone → damage → learns → avoids next time
  3. Unknown berry → avoids → learns when taught → proceeds
  4. Full episode with body state tracking
  5. Learning progression over repeated trials
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def sep(t):  print(f"\n{'═'*64}\n  {t}\n{'═'*64}")
def sec(t):  print(f"\n  {'─'*60}\n  {t}\n  {'─'*60}")
def body_bar(v, label, width=20):
    filled = int(v * width)
    bar = "█" * filled + "░" * (width - filled)
    print(f"    {label:12s} [{bar}] {v:.0%}")


# ════════════════════════════════════════════════════════
# Scenario 1: Eat apple → success
# ════════════════════════════════════════════════════════
def scenario_1():
    sep("Scenario 1: Eat Apple → Satisfied")
    from cognifield import CogniField

    cf = CogniField({"agents": 2, "thinking_mode": "auto"})
    cf.teach("apple", {"edible": True, "category": "food"})

    print("\n  Agent knows about apple. Running eat apple...")

    # Think first
    r_think = cf.think("Is apple safe to eat?")
    print(f"\n  Think result:")
    print(f"    Decision:   {r_think['decision']} ({r_think['confidence']:.0%})")
    print(f"    Steps:      {r_think['thinking_steps']}")
    print(f"    State:      {r_think['knowledge_state']}")

    # Act
    r_act = cf.act("eat", "apple")
    print(f"\n  Act result:")
    print(f"    Status:     {r_act['status']}")
    print(f"    Effect:     {r_act['effect']}")
    print(f"    Reward:     {r_act['reward']:+.2f}")
    print(f"    Reason:     {r_act['reason'][:60]}")

    # Body state
    print(f"\n  Body after eating apple:")
    body = r_act
    body_bar(body["body_health"], "Health")
    body_bar(1 - body["body_hunger"], "Fullness")


# ════════════════════════════════════════════════════════
# Scenario 2: Eat stone → damage → learns → avoids
# ════════════════════════════════════════════════════════
def scenario_2():
    sep("Scenario 2: Eat Stone → Damage → Learns → Avoids")
    from cognifield import CogniField

    cf = CogniField({"agents": 2, "thinking_mode": "auto"})
    # Intentionally teach stone as food (wrong!)
    cf.teach("stone", {"category": "material"})  # no edible property

    sec("Step 1: Agent with no edibility knowledge tries to eat stone")
    r1 = cf.step("eat stone")
    print(f"  Decision:  {r1['decision']}")
    print(f"  Executed:  {r1['executed']}")
    print(f"  Effect:    {r1['effect']}")
    print(f"  Blocked?   {r1['blocked'] or 'No'}")
    if not r1["executed"]:
        print(f"  ✓ Unknown safety rule prevented harmful action")

    sec("Teaching that stone is NOT edible (forced lesson)")
    cf.teach("stone", {"edible": False, "category": "material"})

    r2 = cf.step("eat stone")
    print(f"  Decision:  {r2['decision']}")
    print(f"  Executed:  {r2['executed']}")
    print(f"  Effect:    {r2['effect']}")
    print(f"  ✓ Known dangerous — blocked by action system")

    sec("Force eating stone to observe damage (force=True)")
    r_forced = cf.act("eat", "stone", force=True)
    print(f"  Status:    {r_forced['status']}")
    print(f"  Effect:    {r_forced['effect']}")
    print(f"  Reward:    {r_forced['reward']:+.2f}")
    print(f"  Health:    {r_forced['body_health']:.0%}")

    sec("After damage — agent learns from outcome")
    lo = cf.learn_from_outcome(
        "ate stone", "stone", "edible",
        prediction=False, actual=False, action="eat", reward=-0.3
    )
    print(f"  Corrections: {lo['corrections_made']}")

    sec("Final check — stone now firmly avoided")
    r_final = cf.think("Should I eat stone?")
    print(f"  Decision:   {r_final['decision']} ({r_final['confidence']:.0%})")
    r_act = cf.step("eat stone")
    print(f"  Step result: executed={r_act['executed']}, "
          f"effect={r_act['effect']}")
    if not r_act["executed"]:
        print(f"  ✓ Agent now consistently avoids stone")


# ════════════════════════════════════════════════════════
# Scenario 3: Unknown berry → avoids → learns → proceeds
# ════════════════════════════════════════════════════════
def scenario_3():
    sep("Scenario 3: Unknown Berry → Avoid → Learn → Proceed")
    from cognifield import CogniField

    cf = CogniField({"agents": 2, "unknown_safety_rule": True})

    sec("Step 1: Encounter unknown berry")
    r1 = cf.think("Is jungle_berry safe?")
    print(f"  Decision:  {r1['decision']}")
    print(f"  State:     {r1['knowledge_state']}")
    print(f"  Confidence:{r1['confidence']:.0%}")
    print(f"  ✓ Unknown safety rule: {r1['decision'] == 'avoid'}")

    r_act1 = cf.step("eat jungle_berry")
    print(f"\n  eat jungle_berry: executed={r_act1['executed']} "
          f"effect={r_act1['effect']}")

    sec("Step 2: Inspect first — observe it's a food")
    r_inspect = cf.step("inspect jungle_berry")
    print(f"  inspect: executed={r_inspect['executed']} "
          f"signal={r_inspect['signal']}")

    sec("Step 3: Teach category (partial knowledge)")
    cf.teach("jungle_berry", {"category": "food"})

    r2 = cf.think("Is jungle_berry safe?")
    print(f"  After teaching category:")
    print(f"  Decision:  {r2['decision']}")
    print(f"  State:     {r2['knowledge_state']}")
    wm = r2.get("world_model", {})
    if wm.get("inferred_value"):
        print(f"  Inferred:  edible={wm['inferred_value']} "
              f"(conf={wm['inferred_conf']:.2f})")

    sec("Step 4: Teach explicit edibility")
    cf.teach("jungle_berry", {"edible": True, "category": "food"})

    r3 = cf.think("Is jungle_berry safe?")
    print(f"  After full knowledge:")
    print(f"  Decision:  {r3['decision']} ({r3['confidence']:.0%})")

    r_eat = cf.step("eat jungle_berry")
    print(f"\n  eat jungle_berry:")
    print(f"  Executed:  {r_eat['executed']}")
    print(f"  Effect:    {r_eat['effect']}")
    print(f"  Reward:    {r_eat['reward']:+.2f}")

    sec("Step 5: Reinforce from 3 successful outcomes")
    for i in range(3):
        cf.learn_from_outcome("eat jungle_berry", "jungle_berry", "edible",
                               True, True, "eat", 0.4)
    r_final = cf.think("Is jungle_berry safe?")
    print(f"  After 3 successes:")
    print(f"  Decision:   {r_final['decision']} ({r_final['confidence']:.0%})")
    print(f"  Knowledge:  {r_final['knowledge_state']}")


# ════════════════════════════════════════════════════════
# Scenario 4: Full episode with body state tracking
# ════════════════════════════════════════════════════════
def scenario_4():
    sep("Scenario 4: Full Episode — Body State Tracking")
    from cognifield import CogniField

    cf = CogniField({"agents": 2, "thinking_mode": "auto", "verbose": False})
    cf.teach("apple",  {"edible": True,  "category": "food"})
    cf.teach("bread",  {"edible": True,  "category": "food"})
    cf.teach("stone",  {"edible": False, "category": "material"})
    cf.teach("mushroom_unknown", {"category": "plant"})

    queries = [
        "inspect apple",
        "inspect stone",
        "eat apple",
        "inspect mushroom_unknown",
        "eat stone",            # should be blocked
        "eat bread",
        "eat mushroom_unknown", # should be cautious
        "inspect apple",
    ]

    print(f"\n  {'#':3s} | {'Query':28s} | {'Action':8s} | {'Target':18s} | "
          f"{'Result':22s} | {'Health':6s} | {'Hunger':6s}")
    print(f"  {'─'*3} | {'─'*28} | {'─'*8} | {'─'*18} | {'─'*22} | "
          f"{'─'*6} | {'─'*6}")

    for q in queries:
        s = cf.step(q)
        exec_icon = "✓" if s["executed"] else "⊘"
        effect    = s["effect"][:20] if s["effect"] else "–"
        print(f"  {s['step']:3d} | {q:28s} | {s['action']:8s} | "
              f"{s['target']:18s} | {exec_icon} {effect:20s} | "
              f"{s['body']['health']:.0%}   | {s['body']['hunger']:.0%}")

    body = cf.body_status()
    print(f"\n  Final body state:")
    body_bar(body["health"],       "Health")
    body_bar(1 - body["hunger"],   "Fullness")
    body_bar(body["energy"],       "Energy")
    print(f"    Position:    {body['position']}")
    print(f"    Inventory:   {body['inventory']}")


# ════════════════════════════════════════════════════════
# Scenario 5: Learning progression over repeated trials
# ════════════════════════════════════════════════════════
def scenario_5():
    sep("Scenario 5: Learning Progression Over Time")
    from cognifield import CogniField

    cf = CogniField({"agents": 2, "thinking_mode": "auto"})

    # Start with only category knowledge
    cf.teach("red_berry", {"category": "food"})

    print(f"\n  Tracking confidence on red_berry.edible over time:")
    print(f"\n  {'Phase':30s} | {'Decision':22s} | {'Confidence':10s}")
    print(f"  {'─'*30} | {'─'*22} | {'─'*10}")

    # Phase 1: before any experience
    r = cf.think("Is red_berry safe?")
    print(f"  {'1. Category only':30s} | {r['decision']:22s} | {r['confidence']:.0%}")

    # Phase 2: after 3 successes
    for _ in range(3):
        cf.learn_from_outcome("eat red_berry","red_berry","edible",True,True,"eat",0.4)
    r2 = cf.think("Is red_berry safe?")
    print(f"  {'2. After 3 successes':30s} | {r2['decision']:22s} | {r2['confidence']:.0%}")

    # Phase 3: after a failure
    cf.learn_from_outcome("eat red_berry","red_berry","edible",True,False,"eat",-0.2)
    r3 = cf.think("Is red_berry safe?")
    print(f"  {'3. After 1 failure':30s} | {r3['decision']:22s} | {r3['confidence']:.0%}")

    # Phase 4: after teaching explicit edibility
    cf.teach("red_berry", {"edible": True, "category": "food"})
    r4 = cf.think("Is red_berry safe?")
    print(f"  {'4. After explicit teach':30s} | {r4['decision']:22s} | {r4['confidence']:.0%}")

    # Phase 5: after 5 more successes
    for _ in range(5):
        cf.learn_from_outcome("eat red_berry","red_berry","edible",True,True,"eat",0.4)
    r5 = cf.think("Is red_berry safe?")
    print(f"  {'5. After 5 more successes':30s} | {r5['decision']:22s} | {r5['confidence']:.0%}")

    # Confidence progression visual
    confs = [r["confidence"], r2["confidence"], r3["confidence"],
             r4["confidence"], r5["confidence"]]
    labels = ["category_only", "+3_successes", "+1_failure",
              "+explicit_teach", "+5_successes"]

    print(f"\n  red_berry.edible confidence progression:")
    for label, conf in zip(labels, confs):
        bar  = "█" * int(conf * 30) + "░" * (30 - int(conf * 30))
        print(f"    {label:20s} [{bar}] {conf:.0%}")

    # Reflection
    reflection = cf.self_reflect()
    print(f"\n  Self-reflect findings: {reflection['findings'] or 'none'}")
    print(f"  Rules derived:         {cf._exp_engine.derived_rules() or 'none yet'}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  CogniField v11 Part 2 — Embodied Intelligence Demo      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    scenario_1()
    scenario_2()
    scenario_3()
    scenario_4()
    scenario_5()
    print("\n" + "═"*64 + "\n  Demo complete.\n" + "═"*64)
