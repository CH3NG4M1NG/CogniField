"""
examples/demo_v11_part3.py
===========================
CogniField v11 Part 3 — Game Interaction Demo

Scenarios:
  1. Java adapter  — exploration + health/hunger
  2. Bedrock       — partial observations
  3. Mobile        — ADB command translation
  4. Vision system — health/hunger bar detection
  5. Survival goals — automatic objective management
  6. Language learner — Minecraft ID → world model
  7. Full game episode — survival loop with learning
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def sep(t):  print(f"\n{'═'*64}\n  {t}\n{'═'*64}")
def sec(t):  print(f"\n  {'─'*60}\n  {t}\n  {'─'*60}")


def demo_java():
    sep("1. Java Adapter — Simulation")
    from cognifield.game import JavaAdapter
    ja = JavaAdapter(simulation=True, seed=42)
    ja.connect()
    obs = ja.get_observation()
    print(f"  health={obs.health:.1f}/20  hunger={obs.hunger:.1f}/20  biome={obs.biome}")
    print(f"  blocks={[b.block_id for b in obs.visible_blocks[:3]]}")
    print(f"  inventory={[(i.item_id,i.count) for i in obs.inventory]}")
    ja.move("north", steps=2)
    obs2 = ja.get_observation()
    print(f"  After move north: pos={obs2.position}")
    ja.sim_set_hunger(6.0)
    ja.sim_spawn_item("minecraft:bread", 2)
    ate = ja.eat("minecraft:bread")
    obs3 = ja.get_observation()
    print(f"  After eating bread: hunger={obs3.hunger:.1f}/20  success={ate}")
    print(f"  Summary: {ja.summary()}")


def demo_bedrock():
    sep("2. Bedrock Adapter — Partial Observations")
    from cognifield.game import BedrockAdapter
    ba = BedrockAdapter(simulation=True, seed=42)
    ba.connect()
    obs = ba.get_observation()
    print(f"  partial={obs.partial}")
    print(f"  missing_fields={obs.missing_fields}")
    print(f"  available: health={obs.health:.1f}  hunger={obs.hunger:.1f}")
    print(f"  inventory={[(i.item_id,i.count) for i in obs.inventory]}")
    ok = ba.send_action({"type":"move","direction":"south"})
    print(f"  move south: ok={ok}")
    ba.sim_set_hunger(5.0)
    ba.sim_add_item("minecraft:apple",1)
    ok2 = ba.send_action({"type":"use","target":"minecraft:apple"})
    obs2 = ba.get_observation()
    print(f"  after eat apple: hunger={obs2.hunger:.1f}/20  ok={ok2}")
    cmd = ba._action_to_command({"type":"command","command":"/give @s minecraft:bread 5"})
    print(f"  command → '{cmd}'")


def demo_mobile():
    sep("3. Mobile Adapter — ADB Commands (dry-run)")
    from cognifield.game import MobileAdapter
    ma = MobileAdapter(dry_run=True, resolution="1080x1920")
    ma.connect()
    actions = [
        {"type":"move","direction":"forward"},
        {"type":"jump"},
        {"type":"use"},
        {"type":"attack"},
        {"type":"inventory"},
        {"type":"tap","x":540,"y":960},
        {"type":"swipe","x1":200,"y1":1600,"x2":200,"y2":1450,"duration_ms":300},
        {"type":"keyevent","keycode":82},
    ]
    for a in actions:
        ma.send_action(a)
    log = ma.get_adb_log()
    print(f"  Generated {len(log)} ADB commands:")
    for cmd in log[:5]:
        print(f"    $ {cmd}")
    obs = ma.get_observation()
    print(f"  Stub obs: partial={obs.partial}  missing={len(obs.missing_fields)} fields")
    ma.update_from_vision(health=16.0, hunger=12.0)
    obs2 = ma.get_observation()
    print(f"  After vision inject: health={obs2.health:.1f}  hunger={obs2.hunger:.1f}")


def demo_vision():
    sep("4. Vision System — Health/Hunger Detection")
    from cognifield.vision import VisionSystem
    vs = VisionSystem(simulation=True, seed=42)
    print(f"\n  {'Step':4s} | {'Health':20s} | {'Hunger':20s} | Danger | Food")
    print(f"  {'─'*4} | {'─'*20} | {'─'*20} | {'─'*6} | {'─'*4}")
    for i in range(10):
        r = vs.analyze()
        hb = "█"*int(r.health_pct*10)+"░"*(10-int(r.health_pct*10))
        fb = "█"*int(r.hunger_pct*10)+"░"*(10-int(r.hunger_pct*10))
        print(f"  {i+1:4d} | [{hb}] {r.health_pct:.0%} | [{fb}] {r.hunger_pct:.0%} | "
              f"{'yes' if r.danger_detected else 'no ':3s}    | {'yes' if r.food_visible else 'no'}")
    print(f"\n  Summary: {vs.summary()}")
    print(f"  Last blocks: {vs.last_reading().detected_blocks}")


def demo_survival_goals():
    sep("5. Survival Goals — Automatic Objectives")
    from cognifield.game import JavaAdapter, SurvivalGoalManager
    from cognifield.game.base_adapter import EntityInfo
    ja = JavaAdapter(simulation=True, seed=42)
    ja.connect()
    sg = SurvivalGoalManager()
    scenarios = [
        ("Safe & full",         20.0, 20.0, False),
        ("Hungry",              20.0,  6.0, False),
        ("Critical hunger",     20.0,  2.0, False),
        ("Low health",           3.0, 20.0, False),
        ("Hostile mob nearby",  15.0, 15.0, True),
    ]
    for label, hp, hung, hostile in scenarios:
        ja.sim_set_health(hp)
        ja.sim_set_hunger(hung)
        obs = ja.get_observation()
        if hostile:
            obs.entities = [EntityInfo("zombie",(5.0,64.0,5.0),20.0,True)]
        goals = sg.update(obs)
        top   = sg.top_goal()
        print(f"\n  [{label}]")
        print(f"    hp={hp:.0f} hung={hung:.0f} danger={obs.is_in_danger}")
        print(f"    Top: {top.name!r} [{top.priority.value}] → '{top.to_query()}'")
        print(f"    All: {[g.name for g in goals[:4]]}")
        ja.sim_set_health(20.0); ja.sim_set_hunger(20.0)


def demo_language_learner():
    sep("6. Language Learner — Minecraft → CogniField")
    from cognifield.game import LanguageLearner
    from cognifield import CogniField
    ll = LanguageLearner()
    test_ids = [
        "minecraft:apple","minecraft:stone","minecraft:oak_log",
        "minecraft:iron_ore","minecraft:sweet_berries","minecraft:bread",
        "minecraft:diamond_sword","minecraft:crafting_table","minecraft:sand",
    ]
    print(f"\n  {'ID':35s} | {'Name':18s} | {'Category':12s} | Edible")
    print(f"  {'─'*35} | {'─'*18} | {'─'*12} | ──────")
    for mc_id in test_ids:
        c = ll.process_id(mc_id)
        if c:
            ed = str(c.properties.get("edible","?"))
            print(f"  {mc_id:35s} | {c.name:18s} | {c.category:12s} | {ed}")
    for etype, hostile in [("zombie",True),("pig",False),("creeper",True),("cow",False)]:
        c = ll._process_entity(etype, hostile)
        print(f"  entity:{etype:10s} → {c.category:14s} dangerous={c.properties.get('dangerous')} edible={c.properties.get('edible')}")

    sec("Integration with CogniField")
    cf = CogniField({"agents":2})
    ll2 = LanguageLearner(world_model=cf._world_model_v2,
                          belief_system=cf._agents[0].beliefs)
    for mc_id in ["minecraft:apple","minecraft:bread","minecraft:stone"]:
        for _ in range(3):
            ll2.process_id(mc_id)
    print(f"\n  Beliefs from language learner:")
    for key in ["apple.edible","bread.edible","stone.edible"]:
        b = cf._agents[0].beliefs.get(key)
        if b:
            print(f"    {key:20s} = {b.value} (conf={b.confidence:.2f})")
    print(f"  Summary: {ll2.summary()}")


def demo_full_episode():
    sep("7. Full Game Episode — Survival Loop")
    from cognifield import CogniField
    from cognifield.game import JavaAdapter
    cf = CogniField({"agents":2,"thinking_mode":"auto","unknown_safety_rule":True})
    cf.teach("apple",  {"edible":True,  "category":"food"})
    cf.teach("bread",  {"edible":True,  "category":"food"})
    cf.teach("stone",  {"edible":False, "category":"material"})
    ja = JavaAdapter(simulation=True, seed=99)
    ja.connect()
    ja.sim_set_hunger(5.0)
    ja.sim_spawn_item("minecraft:bread", 3)
    loop = cf.create_game_loop(adapter=ja, verbose=False)
    steps = loop.run_episode(n_steps=10)
    print(f"\n  {'#':3s}|{'Goal':14s}|{'Decision':20s}|{'Effect':20s}|{'HP':5s}|{'HG':5s}|{'New':3s}")
    print(f"  {'─'*3}|{'─'*14}|{'─'*20}|{'─'*20}|{'─'*5}|{'─'*5}|{'─'*3}")
    for s in steps:
        lr   = s.loop_result
        dec  = (lr.thinking_decision if lr else "none")[:18]
        eff  = (lr.effect if lr else "–")[:18]
        goal = (s.active_goal.name if s.active_goal else "–")[:12]
        print(f"  {s.step:3d}|{goal:14s}|{dec:20s}|{eff:20s}|"
              f"{s.observation.health:5.1f}|{s.observation.hunger:5.1f}|{s.new_concepts:3d}")
    summ = loop.summary()
    print(f"\n  Steps={summ['total_steps']}  "
          f"mean_health={summ['mean_health']:.1f}  "
          f"mean_hunger={summ['mean_hunger']:.1f}  "
          f"survival={summ['survival_rate']:.0%}")
    lang = summ["lang_summary"]
    print(f"  Concepts learned={lang.get('known_concepts',0)}  "
          f"foods={lang.get('known_foods',[])}  "
          f"dangers={lang.get('known_dangers',[])}")
    r = cf.self_reflect()
    print(f"  Reflection: {r['findings'] or 'stable'}  "
          f"rules={cf._exp_engine.derived_rules()}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  CogniField v11 Part 3 — Game Interaction Demo           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    demo_java()
    demo_bedrock()
    demo_mobile()
    demo_vision()
    demo_survival_goals()
    demo_language_learner()
    demo_full_episode()
    print("\n"+"═"*64+"\n  Part 3 Demo complete.\n"+"═"*64)
