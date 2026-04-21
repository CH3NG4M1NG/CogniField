"""
examples/demo_v3.py
====================
CogniField v3 — Full Demo Scenario

Scenario:
  Agent in world with: apple (edible), stone (not edible),
  bread (edible), glass_jar (fragile), purple_berry (unknown)

Tasks demonstrated:
  1.  World model learning — agent learns eat rules from experience
  2.  Relational memory    — stores apple→edible=True, stone→edible=False
  3.  Curiosity            — generates hypotheses about purple_berry
  4.  Planning             — plans pick→eat to satisfy "eat apple" goal
  5.  Goal execution       — executes plan step-by-step
  6.  Danger avoidance     — refuses to eat stone after learning
  7.  Unknown exploration  — tests purple_berry with inspect then eat
  8.  World model query    — shows learned rules and causal graph
  9.  Relational queries   — "what can I eat?", "what is apple?"
  10. Full agent summary
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def sep(title: str) -> None:
    print(f"\n{'═'*58}")
    print(f"  {title}")
    print(f"{'═'*58}")


def section(title: str) -> None:
    print(f"\n  {'─'*52}")
    print(f"  {title}")
    print(f"  {'─'*52}")


# ═══════════════════════════════════════════════════════════
# 1. World Model: learning cause-effect
# ═══════════════════════════════════════════════════════════
def demo_world_model():
    sep("1. World Model — Learning Cause-Effect Rules")

    from cognifield.world_model.transition_model import TransitionModel
    from cognifield.world_model.causal_graph import CausalGraph
    from cognifield.latent_space.frequency_space import FrequencySpace

    space = FrequencySpace(dim=64)
    tm    = TransitionModel(space=space, dim=64)
    cg    = CausalGraph()
    rng   = np.random.default_rng(42)

    def make_state(name: str) -> np.ndarray:
        v = rng.standard_normal(64).astype(np.float32)
        return space.l2(v)

    states = {n: make_state(n) for n in
              ["holding_apple", "eating_apple", "after_apple",
               "holding_stone", "eating_stone", "after_stone",
               "exploring"]}

    # Simulate experiences
    experiences = [
        ("eat", "apple", "food",  True,  +0.5),
        ("eat", "apple", "food",  True,  +0.5),
        ("eat", "stone", "material", False, -0.2),
        ("eat", "stone", "material", False, -0.2),
        ("pick", "apple", "food",  True, +0.1),
        ("pick", "stone", "material", True, +0.05),
        ("inspect", "apple", "food", True, +0.05),
        ("inspect", "purple_berry", "unknown", True, +0.05),
    ]

    for action, obj, cat, success, reward in experiences:
        s   = states.get(f"holding_{obj}", states["exploring"])
        ns  = states.get(f"after_{obj}", states["exploring"])
        tm.record(s, action, ns, reward, success, obj, cat)
        cg.ingest_feedback(action, obj, {"edible": success and "eat" in action,
                                          "category": cat}, success, reward)

    print("\n  Learned World Rules:")
    print(f"  {'action':8s} | {'category':12s} | {'outcome':10s} | reward | conf | reliable")
    print(f"  {'─'*8} | {'─'*12} | {'─'*10} | {'─'*6} | {'─'*4} | {'─'*8}")
    for rule in tm.rule_summary():
        r_mark = "✓" if rule["reliable"] else "–"
        print(f"  {rule['action']:8s} | {rule['category']:12s} | "
              f"{rule['outcome']:10s} | {rule['reward']:+.3f} | "
              f"{rule['confidence']:.2f} | {r_mark}")

    print("\n  Predictions:")
    for action, cat, obj in [("eat","food","apple"), ("eat","material","stone"),
                               ("pick","food","apple")]:
        outcome, reward, conf = tm.predict_outcome(action, cat, obj)
        print(f"    {action}({obj}) → outcome={outcome}, reward={reward:+.2f}, conf={conf:.2f}")

    print("\n  Causal Graph:")
    for obj in ["apple", "stone", "purple_berry"]:
        effects = cg.get_effects(f"eat({obj})")
        if effects:
            eff_str = ", ".join(f"{e}({w:.1f})" for e, w in effects)
            print(f"    eat({obj}) → {eff_str}")

    print(f"\n  Edible from causal graph: {cg.find_edible_objects()}")
    print(f"  Can eat food? {tm.can_do('eat','food')}  |  Can eat material? {tm.can_do('eat','material')}")


# ═══════════════════════════════════════════════════════════
# 2. Planning Engine
# ═══════════════════════════════════════════════════════════
def demo_planning():
    sep("2. Planning Engine — Multi-Step Action Sequences")

    from cognifield.world_model.transition_model import TransitionModel
    from cognifield.world_model.causal_graph import CausalGraph
    from cognifield.planning.planner import Planner
    from cognifield.latent_space.frequency_space import FrequencySpace
    from cognifield.encoder.text_encoder import TextEncoder

    space = FrequencySpace(dim=64)
    enc   = TextEncoder(dim=64); enc.fit()
    tm    = TransitionModel(space=space, dim=64)
    cg    = CausalGraph()

    # Pre-teach the world model
    rng = np.random.default_rng(7)
    sv  = lambda: space.l2(rng.standard_normal(64).astype(np.float32))

    teach_data = [
        ("eat",  "apple",  "food",     True,  +0.5),
        ("eat",  "bread",  "food",     True,  +0.5),
        ("eat",  "stone",  "material", False, -0.2),
        ("pick", "apple",  "food",     True,  +0.1),
        ("pick", "bread",  "food",     True,  +0.1),
        ("pick", "stone",  "material", True,  +0.05),
        ("inspect", "purple_berry", "unknown", True, +0.05),
    ]
    for action, obj, cat, success, reward in teach_data:
        tm.record(sv(), action, sv(), reward, success, obj, cat)
        cg.ingest_feedback(action, obj, {"edible": "food" in cat,
                                          "category": cat}, success, reward)
        cg.add_property(obj, "edible", "food" in cat)
        if cat not in ("unknown",):
            cg.add_is_a(obj, cat)

    planner = Planner(tm, cg, space, max_depth=4, beam_width=3, dim=64)

    goal_text = "eat apple"
    goal_vec  = enc.encode(goal_text)
    state_vec = enc.encode("standing in world, apple nearby")
    available = [("apple","food"), ("stone","material"),
                 ("bread","food"), ("purple_berry","unknown")]

    print(f"\n  Goal: '{goal_text}'")
    print(f"  Available objects: {[n for n,_ in available]}")

    section("Symbolic Plan (rule-based)")
    plan = planner.plan(goal_text, goal_vec, state_vec, available,
                        inventory=[])
    print(f"  {plan.describe()}")
    print(f"\n  Action sequence:")
    for i, (a, o) in enumerate(plan.action_sequence):
        step = plan.steps[i]
        print(f"    {i+1}. {a}({o or '–'})  score={step.score:.3f}  "
              f"reward_exp={step.expected_reward:+.3f}  conf={step.confidence:.2f}")

    section("Plan safety check")
    print(f"  Plan safe? {planner.is_safe(plan)}")

    # Try planning a dangerous action
    bad_plan = planner.plan("eat stone", enc.encode("eat stone"),
                            state_vec, available, inventory=["stone"])
    print(f"\n  Dangerous plan (eat stone): {bad_plan.describe()}")
    print(f"  Safe? {planner.is_safe(bad_plan)}")


# ═══════════════════════════════════════════════════════════
# 3. Relational Memory
# ═══════════════════════════════════════════════════════════
def demo_relational_memory():
    sep("3. Relational Memory — Concept Knowledge Graph")

    from cognifield.memory.relational_memory import RelationalMemory
    from cognifield.encoder.text_encoder import TextEncoder

    enc = TextEncoder(dim=64); enc.fit()
    mem = RelationalMemory(dim=64)

    # Load object knowledge
    objects = {
        "apple":  {"is_a": "food", "edible": True,  "color": "red",    "fragile": False},
        "stone":  {"is_a": "material", "edible": False, "color": "grey",  "heavy": True},
        "bread":  {"is_a": "food", "edible": True,  "color": "yellow", "fragile": False},
        "water":  {"is_a": "food", "edible": True,  "color": "clear",  "fragile": True},
        "hammer": {"is_a": "tool", "edible": False, "heavy": True,     "fragile": False},
        "glass_jar": {"is_a": "tool", "edible": False, "fragile": True, "color": "clear"},
    }

    for name, props in objects.items():
        vec = enc.encode(name)
        mem.add_object_properties(name, props, vector=vec)

    print("\n  Concept facts:")
    for obj in ["apple", "stone", "glass_jar"]:
        print(f"    {mem.what_is(obj)}")

    print("\n  Query: what can I eat?")
    edible = mem.find_edible()
    print(f"    → {edible}")

    print("\n  Query: what is dangerous?")
    dangerous = mem.find_dangerous()
    print(f"    → {dangerous}")

    print("\n  Query: what objects are fragile?")
    fragile = mem.query("fragile", True)
    print(f"    → {[(s, round(c,2)) for s,c in fragile]}")

    print("\n  Query: what is the category of stone?")
    print(f"    → {mem.get_category('stone')}")

    print("\n  Vector recall (query='red fruit'):")
    q_vec = enc.encode("red fruit")
    results = mem.recall_similar(q_vec, k=3)
    for sim, entry in results:
        print(f"    '{entry.label}' (sim={sim:.3f})")


# ═══════════════════════════════════════════════════════════
# 4. Advanced Curiosity — Hypothesis Testing
# ═══════════════════════════════════════════════════════════
def demo_advanced_curiosity():
    sep("4. Advanced Curiosity — Hypothesis Generation & Testing")

    from cognifield.curiosity.advanced_curiosity import AdvancedCuriosityEngine
    from cognifield.memory.relational_memory import RelationalMemory
    from cognifield.memory.memory_store import MemoryStore
    from cognifield.encoder.text_encoder import TextEncoder
    from cognifield.latent_space.frequency_space import FrequencySpace

    space      = FrequencySpace(dim=64)
    enc        = TextEncoder(dim=64); enc.fit()
    rel_mem    = RelationalMemory(dim=64, space=space)
    vec_mem    = MemoryStore(dim=64)
    curiosity  = AdvancedCuriosityEngine(space, rel_mem, vec_mem,
                                          novelty_threshold=0.35, dim=64)

    # Load known concepts
    known = {
        "apple": {"edible": True,  "category": "food",     "color": "red"},
        "stone": {"edible": False, "category": "material", "heavy": True},
        "bread": {"edible": True,  "category": "food",     "color": "yellow"},
    }
    for name, props in known.items():
        vec = enc.encode(name + " " + " ".join(f"{k} {v}" for k,v in props.items()))
        rel_mem.add_object_properties(name, props, vector=vec)
        vec_mem.store(vec, label=name, modality="concept", allow_duplicate=True)

    # Unknown objects to explore
    unknowns = [
        ("purple_berry", "small purple round berry"),
        ("glowing_cube",  "small blue glowing cube"),
        ("mystery_powder", "strange white powder"),
    ]

    print("\n  Known concepts loaded: apple, stone, bread\n")

    for concept, description in unknowns:
        unknown_vec = enc.encode(description)
        novelty     = curiosity.detect_novelty(unknown_vec, concept)
        report      = curiosity.explore(concept, unknown_vec)

        print(f"  Unknown: '{concept}'  (novelty={novelty:.3f})")
        print(f"    Description: '{description}'")
        print(f"    Priority: {report['priority']:.3f}")
        print(f"    Suggested action: {report['suggested_action']}")
        print(f"    Hypotheses ({report['n_hypotheses']}):")
        for pred, val, conf, basis in report["hypotheses"][:3]:
            print(f"      {pred}={val}  conf={conf:.2f}  basis: {basis}")
        print()

    # Test hypothesis update
    print("  --- Hypothesis Update ---")
    print("  Simulating: inspect(purple_berry) reveals edible=True")
    affected = curiosity.update_hypotheses("purple_berry", "edible", True)
    for h in affected:
        print(f"    {h.subject}.{h.predicate} → {h.status} (conf={h.confidence:.2f})")

    best = curiosity.best_hypothesis_to_test()
    if best:
        print(f"\n  Best hypothesis to test next: "
              f"{best.subject}.{best.predicate}={best.predicted}  "
              f"→ action: {best.test_action}")

    print(f"\n  Curiosity summary: {curiosity.summary()}")


# ═══════════════════════════════════════════════════════════
# 5. Goal System
# ═══════════════════════════════════════════════════════════
def demo_goals():
    sep("5. Goal System — Persistent Objectives")

    from cognifield.agent.goals import GoalSystem, GoalType

    gs = GoalSystem()

    # Add various goals
    g1 = gs.add_eat_goal("apple",  priority=0.8)
    g2 = gs.add_avoid_goal("stone", priority=0.9)
    g3 = gs.add_explore_goal("investigate purple_berry", priority=0.5)
    g4 = gs.add_acquire_goal("bread", priority=0.6)

    print(f"\n  Goals added: {gs.active_count} active")
    print(f"  Summary: {gs.summary()}")

    # Select goal
    selected = gs.select_active_goal()
    print(f"\n  Selected goal: '{selected.label}' (priority={selected.priority})")

    # Simulate completion
    gs.mark_completed(g2)   # avoid stone (always active → satisfied when we don't eat it)
    print(f"\n  After completing 'avoid stone': {gs.summary()}")

    # Auto-infer goals from world state
    new_goals = gs.infer_goals_from_context(
        known_edible=["apple", "bread"],
        unknown_objects=["glowing_cube"],
        inventory=[],
    )
    print(f"\n  Auto-inferred {len(new_goals)} new goals:")
    for g in new_goals:
        print(f"    '{g.label}' (type={g.goal_type.value}, priority={g.priority})")


# ═══════════════════════════════════════════════════════════
# 6. Rich Environment
# ═══════════════════════════════════════════════════════════
def demo_rich_env():
    sep("6. Rich Environment — Partial Observability")

    from cognifield.environment.rich_env import RichEnv

    env = RichEnv(seed=42)
    print(f"\n  World: {', '.join(env.object_names)}")
    print(f"  Visibility radius: {env.VISIBILITY_RADIUS} cells")
    print(f"  Agent start: {env._agent_pos}\n")

    steps = [
        ("observe",  ()),
        ("move",     (3, 2)),
        ("observe",  ()),
        ("inspect",  ("apple",)),
        ("inspect",  ("stone",)),
        ("pick",     ("apple",)),
        ("eat",      ("apple",)),
        ("move",     (6, 6)),
        ("observe",  ()),
        ("inspect",  ("purple_berry",)),
    ]

    total_reward = 0.0
    for action, args in steps:
        fb = env.step(action, *args)
        total_reward += fb.get("reward", 0)
        status = "✓" if fb["success"] else "✗"
        r_str  = f"r={fb['reward']:+.2f}" if fb.get("reward") is not None else ""
        print(f"  {status} {action:10s}{str(args):20s} {r_str:8s} {fb['message'][:55]}")

    print(f"\n  Total reward: {total_reward:+.3f}")
    print(f"  Final state:  {env}")
    print(f"  Stats:        {env.stats()}")


# ═══════════════════════════════════════════════════════════
# 7. Full v3 Agent: eat apple / avoid stone / explore unknown
# ═══════════════════════════════════════════════════════════
def demo_full_agent_v3():
    sep("7. Full v3 Agent — Eat Apple, Avoid Stone, Explore Unknown")

    from cognifield.agent.agent_v3 import CogniFieldAgentV3, AgentV3Config
    from cognifield.environment.rich_env import RichEnv
    from cognifield.agent.goals import GoalType

    env   = RichEnv(seed=77)
    agent = CogniFieldAgentV3(
        config=AgentV3Config(
            dim=64,
            plan_depth=3,
            plan_beam=3,
            novelty_threshold=0.35,
            verbose=False,   # we control output below
        ),
        env=env,
    )

    # ── Phase A: Teach the agent about the world ──
    section("Phase A: Teaching world knowledge")

    agent.teach("apple red fruit food", "apple",
                {"edible": True, "category": "food", "color": "red"})
    agent.teach("stone grey heavy rock", "stone",
                {"edible": False, "category": "material", "color": "grey"})
    agent.teach("bread yellow food soft", "bread",
                {"edible": True, "category": "food", "color": "yellow"})
    agent.teach("glass jar fragile clear", "glass_jar",
                {"edible": False, "category": "tool", "fragile": True})

    # Teach from simulated experience
    rng = np.random.default_rng(0)
    sv  = lambda: agent.space.l2(rng.standard_normal(agent.cfg.dim).astype(np.float32))
    for action, obj, cat, success, reward in [
        ("eat","apple","food",True,0.5), ("eat","apple","food",True,0.5),
        ("eat","stone","material",False,-0.2), ("eat","stone","material",False,-0.2),
        ("pick","apple","food",True,0.1), ("pick","bread","food",True,0.1),
        ("inspect","purple_berry","unknown",True,0.05),
    ]:
        agent.world_model.record(sv(),action,sv(),reward,success,obj,cat)
        agent.causal_graph.ingest_feedback(action,obj,
            {"edible":"food" in cat,"category":cat},success,reward)
        agent.causal_graph.add_property(obj,"edible","food" in cat)

    print(f"  Memory: {len(agent.vec_memory)} vectors, "
          f"{agent.rel_memory.n_facts()} relational facts")
    print(f"  World model: {agent.world_model.n_rules} rules")
    print(f"  Edible objects known: {agent.what_can_i_eat()}")
    print(f"  Dangerous: {agent.what_is_dangerous()}")

    # ── Phase B: Set goals ──
    section("Phase B: Setting goals")

    g_eat   = agent.add_goal("eat apple", GoalType.EAT_OBJECT, target="apple", priority=0.85)
    g_avoid = agent.add_goal("avoid eating stone", GoalType.AVOID, target="stone", priority=0.95)
    g_expl  = agent.add_goal("explore purple_berry", GoalType.EXPLORE,
                              target="purple_berry", priority=0.5)

    print(f"  Active goals: {[g.label for g in agent.goals._goals]}")

    # ── Phase C: Run agent loop with verbosity ──
    section("Phase C: Agent loop — planning and acting")

    print(f"\n  {'Step':4s} | {'Action':12s} | {'Object':15s} | {'Goal':25s} | {'Reward':6s} | {'Novel'}")
    print(f"  {'─'*4} | {'─'*12} | {'─'*15} | {'─'*25} | {'─'*6} | {'─'*5}")

    for i in range(15):
        # Auto-infer goals from world state
        if agent.goals.active_count == 0:
            edible = agent.rel_memory.find_edible()
            unknown = [o.name for o in env.visible_objects()
                       if o.category == "unknown" or o.edible is None]
            agent.goals.infer_goals_from_context(edible, unknown, env.inventory)

        s = agent.step(verbose=False)

        action_str = (s.action_taken or "–")[:12]
        obj_str    = (s.action_obj or "–")[:15]
        goal_str   = (s.active_goal or "–")[:25]
        reward_str = f"{s.env_reward:+.2f}" if s.env_reward is not None else "  N/A"
        novel_str  = "⚡" if s.novelty >= 0.35 else "–"
        print(f"  {s.step:4d} | {action_str:12s} | {obj_str:15s} | "
              f"{goal_str:25s} | {reward_str:6s} | {novel_str}")

    # ── Phase D: Encounter unknown object ──
    section("Phase D: Encountering purple_berry (unknown)")

    # Move agent near purple_berry if it exists
    if "purple_berry" in env.object_names:
        pb = env.get_object("purple_berry")
        if pb:
            env._agent_pos = (pb.position[0], pb.position[1])

    print("  Agent sees purple_berry. Triggering curiosity...")
    pb_vec    = agent.enc.encode("purple berry unknown small")
    novelty   = agent.curiosity.detect_novelty(pb_vec, "purple_berry")
    report    = agent.curiosity.explore("purple_berry", pb_vec)

    print(f"  Novelty: {novelty:.3f}")
    print(f"  Hypotheses generated: {report['n_hypotheses']}")
    for pred, val, conf, basis in report["hypotheses"][:3]:
        print(f"    • {pred}={val} (conf={conf:.2f}) — {basis}")
    print(f"  Suggested test action: {report['suggested_action']}")

    # Test via environment inspect
    fb = env.step("inspect", "purple_berry")
    if fb["success"]:
        props = fb.get("object_props", {})
        print(f"\n  Inspect result: {fb['message'][:70]}")
        # Update hypotheses
        for prop, val in props.items():
            affected = agent.curiosity.update_hypotheses("purple_berry", prop, val)
            if affected:
                for h in affected:
                    print(f"  Hypothesis {h.subject}.{h.predicate}: {h.status}")

    # ── Phase E: Query the world model ──
    section("Phase E: World model knowledge")

    print("  Learned rules:")
    for rule in agent.world_model.rule_summary():
        r_str = "✓" if rule["reliable"] else "–"
        print(f"    {rule['action']:6s}({rule['category']:10s}) → "
              f"{rule['outcome']:10s} reward={rule['reward']:+.3f} "
              f"conf={rule['confidence']:.2f} {r_str}")

    print(f"\n  Causal graph summary: {agent.causal_graph.summary()}")

    # ── Phase F: Relational queries ──
    section("Phase F: Relational memory queries")

    queries = ["apple", "stone", "purple_berry", "glass_jar"]
    for q in queries:
        print(f"  what_is('{q}'):  {agent.what_is(q)}")

    print(f"\n  What can I eat?     {agent.what_can_i_eat()}")
    print(f"  What is dangerous?  {agent.what_is_dangerous()}")
    print(f"\n  Memory recall ('red fruit'):")
    for sim, label in agent.recall("red fruit", k=3):
        print(f"    '{label}' (sim={sim:.3f})")

    # ── Final Summary ──
    section("Final Summary")
    summ = agent.summary()
    print(f"  Steps run:              {summ['steps']}")
    print(f"  Vector memory:          {summ['vector_memory']}")
    print(f"  Relational facts:       {summ['relational_facts']}")
    print(f"  World model rules:      {summ['world_model_rules']}")
    print(f"  World model transitions:{summ['world_transitions']}")
    print(f"  Goals completed:        {summ['goals']['completed']}")
    print(f"  Active goals:           {summ['goals']['active_goals']}")
    print(f"  Curiosity explorations: {summ['curiosity']['explorations']}")
    print(f"  Hypotheses open:        {summ['curiosity']['hypotheses_open']}")
    print(f"  Hypotheses confirmed:   {summ['curiosity']['hypotheses_confirmed']}")
    if summ["env"]:
        print(f"  Env reward:             {summ['env']['total_reward']}")
        print(f"  Health:                 {summ['env']['health']}")
        print(f"  Satiation:              {summ['env']['satiation']}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     CogniField v3 — Goal-Driven Planning Agent Demo     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_world_model()
    demo_planning()
    demo_relational_memory()
    demo_advanced_curiosity()
    demo_goals()
    demo_rich_env()
    demo_full_agent_v3()

    print("\n" + "═"*58)
    print("  v3 Demo complete.")
    print("═"*58)
