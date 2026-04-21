"""
examples/demo_v4.py
====================
CogniField v4 — Autonomous Self-Improving Agent Demo

Demonstrates:
  1. InternalState — cognitive state modulation
  2. WorldSimulator — imagination before acting
  3. AbstractionEngine — rule induction from experience
  4. MetaLearner — performance analysis + strategy adaptation
  5. MemoryConsolidator — periodic memory compression
  6. GoalGenerator — self-directed goal creation
  7. HierarchicalPlanner — subgoal decomposition
  8. Continuous autonomous loop — agent runs without instructions
  9. Generalisation — applies learned rules to new objects
  10. Self-improvement — fewer mistakes over time
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def sep(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def section(title: str) -> None:
    print(f"\n  {'─'*54}")
    print(f"  {title}")
    print(f"  {'─'*54}")


def make_vec(dim=64, seed=None):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# ═══════════════════════════════════════════════════════════
# 1. Internal State
# ═══════════════════════════════════════════════════════════
def demo_internal_state():
    sep("1. Internal State — Cognitive Control Signals")

    from cognifield.agent.internal_state import InternalState

    state = InternalState()
    print(f"\n  Initial state: {state}")

    # Simulate a series of events
    events = [
        ("success (reward=0.5)",  lambda: state.on_success(0.5)),
        ("success (reward=0.5)",  lambda: state.on_success(0.5)),
        ("novel input",           lambda: state.on_novel_input(0.7)),
        ("failure (penalty=0.3)", lambda: state.on_failure(0.3)),
        ("failure (penalty=0.3)", lambda: state.on_failure(0.3)),
        ("goal completed",        lambda: state.on_goal_completed()),
        ("exploration",           lambda: state.on_exploration()),
        ("tick (time step)",      lambda: state.tick()),
    ]

    print(f"\n  {'Event':30s} | {'conf':5s} | {'unc':5s} | {'cur':5s} | "
          f"{'fat':5s} | {'frust':5s}")
    print(f"  {'─'*30} | {'─'*5} | {'─'*5} | {'─'*5} | {'─'*5} | {'─'*5}")

    for label, fn in events:
        fn()
        s = state
        print(f"  {label:30s} | {s.confidence:.3f} | {s.uncertainty:.3f} | "
              f"{s.curiosity:.3f} | {s.fatigue:.3f} | {s.frustration:.3f}")

    print(f"\n  Decision signals:")
    print(f"    explore_weight:    {state.exploration_weight():.3f}")
    print(f"    risk_tolerance:    {state.risk_tolerance():.3f}")
    print(f"    should_consolidate:{state.should_consolidate()}")
    print(f"    should_meta_learn: {state.should_meta_learn()}")
    print(f"    should_explore_boldly: {state.should_explore_boldly()}")
    print(f"    eff_novelty_threshold: "
          f"{state.effective_novelty_threshold(0.4):.3f}")


# ═══════════════════════════════════════════════════════════
# 2. World Simulator
# ═══════════════════════════════════════════════════════════
def demo_simulator():
    sep("2. World Simulator — Imagination Before Acting")

    from cognifield.world_model.transition_model import TransitionModel
    from cognifield.world_model.causal_graph import CausalGraph
    from cognifield.world_model.simulator import WorldSimulator
    from cognifield.latent_space.frequency_space import FrequencySpace

    space = FrequencySpace(dim=64)
    tm    = TransitionModel(space=space, dim=64)
    cg    = CausalGraph()

    # Teach world model
    _seed = [0]
    def sv():
        _seed[0] += 1
        return make_vec(64, _seed[0])
    for action, obj, cat, success, reward in [
        ("eat","apple","food",True,0.5), ("eat","apple","food",True,0.5),
        ("eat","stone","material",False,-0.2), ("eat","stone","material",False,-0.2),
        ("pick","apple","food",True,0.1), ("pick","bread","food",True,0.1),
        ("inspect","purple_berry","unknown",True,0.05),
    ]:
        tm.record(sv(),action,sv(),reward,success,obj,cat)
        cg.ingest_feedback(action,obj,{"edible":cat=="food","category":cat},success,reward)
        cg.add_property(obj,"edible",cat=="food")
        if cat!="unknown": cg.add_is_a(obj,cat)

    sim = WorldSimulator(tm, cg, space, dim=64)

    state   = make_vec(64, seed=0)
    goal    = make_vec(64, seed=1)

    section("Simulating alternative plans")
    plans = [
        [("pick","apple"), ("eat","apple")],
        [("pick","stone"), ("eat","stone")],
        [("inspect","apple"), ("pick","apple"), ("eat","apple")],
        [("observe",""), ("pick","apple")],
    ]

    results = sim.evaluate_plans(state, plans, goal)
    print(f"\n  {'Plan':42s} | score | reward | reached")
    print(f"  {'─'*42} | {'─'*5} | {'─'*6} | {'─'*7}")
    for score, result in results:
        seq_str = " → ".join(
            f"{a}({o})" if o else a for a, o in result.action_sequence
        )
        print(f"  {seq_str:42s} | {score:.3f} | {result.total_reward:+.3f} | "
              f"{result.goal_reached}")

    section("Hypothesis testing via simulation")
    test_cases = [
        ("eat", "apple"),
        ("eat", "stone"),
        ("inspect", "purple_berry"),
    ]
    for action, obj in test_cases:
        result = sim.test_hypothesis(action, obj, state)
        rec = result["recommendation"]
        print(f"  {action}({obj}): "
              f"outcome={result['predicted_outcome']}  "
              f"success_rate={result['success_rate']:.0%}  "
              f"reward={result['mean_reward']:+.3f}  → {rec}")

    section("Counterfactual reasoning")
    cf = sim.counterfactual(
        state,
        taken_action=("eat", "stone"),
        alternative=("eat", "apple"),
        goal_vec=goal
    )
    print(f"  Took:        {cf['taken_action']}  score={cf['taken_score']:.3f}")
    print(f"  Alternative: {cf['alternative']}   score={cf['alt_score']:.3f}")
    print(f"  Better choice: {cf['better_choice']}  regret={cf['regret']:.3f}")


# ═══════════════════════════════════════════════════════════
# 3. Abstraction Engine
# ═══════════════════════════════════════════════════════════
def demo_abstraction():
    sep("3. Abstraction Engine — Rule Induction from Experience")

    from cognifield.reasoning.abstraction import AbstractionEngine
    from cognifield.memory.relational_memory import RelationalMemory
    from cognifield.world_model.transition_model import TransitionModel
    from cognifield.world_model.causal_graph import CausalGraph
    from cognifield.latent_space.frequency_space import FrequencySpace

    space = FrequencySpace(dim=64)
    rm    = RelationalMemory(dim=64, space=space)
    tm    = TransitionModel(space=space, dim=64)
    cg    = CausalGraph()

    # Specific experiences to abstract from
    food_objects = {
        "apple":  {"edible": True,  "category": "food",     "color": "red"},
        "bread":  {"edible": True,  "category": "food",     "color": "yellow"},
        "water":  {"edible": True,  "category": "food",     "color": "clear"},
        "berry":  {"edible": True,  "category": "food",     "color": "blue"},
    }
    danger_objects = {
        "stone":  {"edible": False, "category": "material", "heavy": True},
        "hammer": {"edible": False, "category": "tool",     "heavy": True},
        "glass":  {"edible": False, "category": "tool",     "fragile": True},
    }

    for name, props in {**food_objects, **danger_objects}.items():
        rm.add_object_properties(name, props)
        tm.record(make_vec(),
                  "eat", make_vec(),
                  0.5 if props["edible"] else -0.2,
                  props["edible"], name, props["category"])
        cg.ingest_feedback("eat", name, props, props["edible"],
                           0.5 if props["edible"] else -0.2)
        cg.add_is_a(name, props["category"])

    ae = AbstractionEngine(rm, tm, cg, space, min_support=2, min_confidence=0.6)

    print(f"\n  Loaded {len(food_objects)} food objects and "
          f"{len(danger_objects)} dangerous objects")
    print(f"  Running abstraction engine...\n")

    rules = ae.run(verbose=True)

    print(f"\n  All abstract rules extracted ({len(rules)}):")
    print(f"  {'Subject':20s} | {'Predicate':15s} | {'Value':8s} | "
          f"{'Conf':5s} | {'Support':7s} | {'Method'}")
    print(f"  {'─'*20} | {'─'*15} | {'─'*8} | {'─'*5} | {'─'*7} | {'─'*25}")
    for rule in rules:
        print(f"  {rule.subject:20s} | {rule.predicate:15s} | "
              f"{str(rule.value):8s} | {rule.confidence:.3f} | "
              f"{rule.support:7d} | {rule.method}")

    print(f"\n  Summary: {ae.summary()}")

    # Test generalisation
    section("Generalisation to new object")
    new_food = "mango"
    rm.add_fact(new_food, "is_a", "food")
    rules2 = ae.run()
    mango_edible = rm.get_value("food", "edible")
    print(f"  After adding mango→is_a→food:")
    print(f"    food.edible (abstract rule) = {mango_edible}")
    print(f"    Can infer mango.edible = True via category rule!")


# ═══════════════════════════════════════════════════════════
# 4. Meta-Learning
# ═══════════════════════════════════════════════════════════
def demo_meta_learning():
    sep("4. Meta-Learning — Performance Analysis & Strategy Adaptation")

    from cognifield.reasoning.meta_learning import MetaLearner

    ml = MetaLearner(history_window=50, adapt_rate=0.1)

    # Simulate agent history: first bad, then improving
    import random
    rng = random.Random(42)

    print("\n  Simulating 60 steps of agent history...")
    print(f"  (Phase 1: poor performance, Phase 2: improving)\n")

    phases = [
        # (n_steps, success_rate, action_set)
        (20, 0.3, ["eat", "eat", "eat"]),   # bad: trying to eat wrong things
        (20, 0.6, ["pick", "eat", "inspect"]),  # improving
        (20, 0.8, ["pick", "eat", "pick"]),     # good
    ]

    step = 0
    for n, sr, actions in phases:
        for _ in range(n):
            step += 1
            action  = rng.choice(actions)
            success = rng.random() < sr
            reward  = rng.uniform(0.3, 0.5) if success else rng.uniform(-0.3, -0.1)
            ml.record(
                step=step, action=action,
                success=success, reward=reward,
                goal_type="eat_object" if "eat" in action else "acquire",
                plan_depth=rng.randint(1, 3),
                novelty=rng.uniform(0.1, 0.6),
                confidence=0.3 + sr * 0.5,
            )

    analysis = ml.analyse()

    print(f"  Analysis after {step} steps:")
    print(f"    Overall success rate: {analysis['overall_sr']:.1%}")
    print(f"    Recent success rate:  {analysis['recent_sr']:.1%}")
    print(f"    Trend:                {analysis['trend']:+.3f} "
          f"({'improving' if analysis['trend'] > 0 else 'declining'})")
    print(f"\n  Action success rates:")
    for action, sr in analysis['action_sr'].items():
        bar = "█" * int(sr * 20)
        print(f"    {action:8s}: {sr:.1%}  {bar}")
    print(f"\n  Insights:")
    for ins in analysis['insights']:
        print(f"    • {ins}")
    print(f"\n  Adapted parameters:")
    for k, v in analysis['params'].items():
        print(f"    {k:30s}: {v:.3f}")
    print(f"\n  Strategy ranking: {ml.strategy_ranking()[:3]}")


# ═══════════════════════════════════════════════════════════
# 5. Memory Consolidation
# ═══════════════════════════════════════════════════════════
def demo_consolidation():
    sep("5. Memory Consolidation — Merging & Pruning")

    from cognifield.memory.consolidation import MemoryConsolidator
    from cognifield.memory.memory_store import MemoryStore
    from cognifield.memory.relational_memory import RelationalMemory
    from cognifield.latent_space.frequency_space import FrequencySpace

    space = FrequencySpace(dim=64)
    vm    = MemoryStore(dim=64)
    rm    = RelationalMemory(dim=64, space=space)
    cons  = MemoryConsolidator(vm, rm, space, merge_threshold=0.92)

    # Store many near-duplicate entries (simulating repeated experiences)
    base = make_vec(64, seed=0)
    rng  = np.random.default_rng(1)
    for i in range(30):
        noise  = rng.standard_normal(64).astype(np.float32) * 0.02
        v      = space.l2(base + noise)
        vm.store(v, label=f"apple_experience_{i}", modality="text",
                 allow_duplicate=True)

    # Add diverse entries
    for i in range(10):
        v = make_vec(64, seed=i + 50)
        vm.store(v, label=f"diverse_concept_{i}", modality="text",
                 allow_duplicate=True)

    # Some high-access entries
    for entry in vm._entries[:5]:
        entry.access_count = 10

    # Some low-activation entries
    for entry in vm._entries[5:10]:
        entry.activation = 0.04  # below prune threshold

    print(f"\n  Memory before consolidation: {len(vm)} entries")

    # Load food knowledge for abstraction
    for name, props in [
        ("apple",{"edible":True,"is_a":"food"}),
        ("bread",{"edible":True,"is_a":"food"}),
        ("stone",{"edible":False,"is_a":"material"}),
        ("hammer",{"edible":False,"is_a":"material"}),
    ]:
        rm.add_object_properties(name, props)

    report = cons.consolidate(verbose=True)

    print(f"\n  Consolidation report:")
    print(f"    Merged:      {report.merged}")
    print(f"    Pruned:      {report.pruned}")
    print(f"    Strengthened:{report.strengthened}")
    print(f"    Abstractions:{report.abstractions}")
    print(f"    Size change: {report.before_size} → {report.after_size}")
    print(f"    Time: {report.elapsed_ms:.1f}ms")
    print(f"\n  Memory after consolidation: {len(vm)} entries")
    print(f"  Abstracted: food→edible=True "
          f"({rm.get_value('food','edible')})")


# ═══════════════════════════════════════════════════════════
# 6. Goal Generator
# ═══════════════════════════════════════════════════════════
def demo_goal_generator():
    sep("6. Goal Generator — Self-Directed Goal Creation")

    from cognifield.agent.goal_generator import GoalGenerator
    from cognifield.agent.goals import GoalSystem
    from cognifield.agent.internal_state import InternalState
    from cognifield.curiosity.advanced_curiosity import AdvancedCuriosityEngine
    from cognifield.memory.relational_memory import RelationalMemory
    from cognifield.memory.memory_store import MemoryStore
    from cognifield.world_model.transition_model import TransitionModel
    from cognifield.latent_space.frequency_space import FrequencySpace
    from cognifield.encoder.text_encoder import TextEncoder

    space = FrequencySpace(dim=64)
    enc   = TextEncoder(dim=64); enc.fit()
    rm    = RelationalMemory(dim=64, space=space)
    vm    = MemoryStore(dim=64)
    tm    = TransitionModel(space=space, dim=64)
    cur   = AdvancedCuriosityEngine(space, rm, vm, dim=64)
    gs    = GoalSystem()

    gg = GoalGenerator(gs, rm, vm, cur, tm, space, enc.encode, max_active_goals=6)

    # Simulate world state
    rm.add_object_properties("apple", {"edible": True, "category": "food"})
    rm.add_object_properties("stone", {"edible": False, "category": "material"})

    # Explore an unknown object to fill curiosity log
    unk_vec = make_vec(64, seed=99)
    cur.explore("glowing_sphere", unk_vec)

    # Add a poor-performing rule to world model
    for _ in range(3):
        tm.record(make_vec(),"eat",make_vec(),-0.2,False,"stone","material")

    # Internal state: agent is curious and a bit frustrated
    is_ = InternalState()
    is_.on_failure(0.3)
    is_.on_failure(0.3)
    is_.on_novel_input(0.7)

    print(f"\n  Internal state: {is_}")
    print(f"  Curiosity explorations: {cur.n_explorations}")

    # Environment observation: low satiation, unknown objects visible
    env_obs = {
        "satiation": 0.25,
        "unknown_objects": ["glowing_sphere", "mystery_dust"],
        "health": 0.9,
    }
    perf = {
        "overall_success_rate": 0.35,
        "action_success": {"eat": 0.2, "pick": 0.7},
        "recent_danger_count": 2,
    }

    goals = gg.generate(is_, env_obs, perf, max_new_goals=5)

    print(f"\n  Generated {len(goals)} goals:")
    for g in goals:
        src = g.metadata.get("source", "?")
        rat = g.metadata.get("rationale", "")[:60]
        print(f"    [{src:12s}] '{g.label}'  priority={g.priority:.3f}")
        print(f"              → {rat}")

    print(f"\n  Goal generator summary: {gg.summary()}")


# ═══════════════════════════════════════════════════════════
# 7. Hierarchical Planner
# ═══════════════════════════════════════════════════════════
def demo_hierarchical_planner():
    sep("7. Hierarchical Planner — Subgoal Decomposition")

    from cognifield.planning.planner import Planner
    from cognifield.planning.hierarchical_planner import HierarchicalPlanner
    from cognifield.world_model.transition_model import TransitionModel
    from cognifield.world_model.causal_graph import CausalGraph
    from cognifield.world_model.simulator import WorldSimulator
    from cognifield.latent_space.frequency_space import FrequencySpace
    from cognifield.encoder.text_encoder import TextEncoder

    space = FrequencySpace(dim=64)
    enc   = TextEncoder(dim=64); enc.fit()
    tm    = TransitionModel(space=space, dim=64)
    cg    = CausalGraph()

    # Train world model
    for action, obj, cat, success, reward in [
        ("eat","apple","food",True,0.5), ("eat","bread","food",True,0.5),
        ("eat","stone","material",False,-0.2),
        ("pick","apple","food",True,0.1), ("pick","bread","food",True,0.1),
        ("inspect","purple_berry","unknown",True,0.05),
        ("observe","","unknown",True,0.0),
    ]:
        tm.record(make_vec(),action,make_vec(),reward,success,obj,cat)
        cg.ingest_feedback(action,obj,{"edible":cat=="food","category":cat},success,reward)
        cg.add_property(obj,"edible",cat=="food")
        if cat!="unknown": cg.add_is_a(obj,cat)

    sim     = WorldSimulator(tm, cg, space, dim=64)
    flat    = Planner(tm, cg, space, max_depth=3, beam_width=3, dim=64)
    h_plan  = HierarchicalPlanner(flat, sim, space, max_depth=2, dim=64)

    state_vec = enc.encode("standing in world")
    available = [("apple","food"),("stone","material"),
                 ("bread","food"),("purple_berry","unknown")]

    goals_to_plan = [
        ("eat apple",           "eat an apple from the world"),
        ("explore",             "explore environment"),
        ("understand purple_berry", "understand purple berry"),
        ("survive",             "survive in the environment"),
    ]

    for goal_label, desc in goals_to_plan:
        goal_vec = enc.encode(desc)
        hp = h_plan.plan_hierarchical(
            goal_label, goal_vec, state_vec, available, inventory=[]
        )
        print(f"\n  Goal: '{goal_label}'  (score={hp.total_score:.3f})")
        print(hp.describe())
        if hp.flat_actions:
            print(f"  Flat action sequence: "
                  f"{' → '.join(f'{a}({o})' if o else a for a,o in hp.flat_actions)}")

    # Simulate and score the hierarchical plans
    section("Simulating hierarchical plan")
    goal_vec = enc.encode("eat apple")
    hp       = h_plan.plan_hierarchical(
        "eat apple", goal_vec, state_vec, available, inventory=[]
    )
    if hp.flat_actions:
        sim_res = sim.simulate(state_vec, hp.flat_actions, goal_vec)
        print(f"  Simulation result: {sim_res.describe()}")
        score = sim._score_simulation(sim_res, goal_vec)
        print(f"  Plan quality score: {score:.3f}")


# ═══════════════════════════════════════════════════════════
# 8. Full Autonomous Agent
# ═══════════════════════════════════════════════════════════
def demo_full_v4():
    sep("8. Full Autonomous v4 Agent — Self-Directed Learning")

    from cognifield.agent.agent_v4 import CogniFieldAgentV4, AgentV4Config
    from cognifield.environment.rich_env import RichEnv
    from cognifield.agent.goals import GoalType

    env   = RichEnv(seed=42)
    agent = CogniFieldAgentV4(
        config=AgentV4Config(
            dim=64,
            plan_depth=3,
            plan_beam=3,
            novelty_threshold=0.35,
            consolidation_interval=15,
            abstraction_interval=10,
            meta_analysis_interval=8,
            verbose=False,
        ),
        env=env,
    )

    # ── Phase A: Teach world knowledge ──
    section("Phase A: Grounding world knowledge")

    for name, props in [
        ("apple",   {"edible":True,  "category":"food",    "color":"red"}),
        ("bread",   {"edible":True,  "category":"food",    "color":"yellow"}),
        ("stone",   {"edible":False, "category":"material","heavy":True}),
        ("glass_jar",{"edible":False,"category":"tool",    "fragile":True}),
    ]:
        agent.teach(f"{name} {' '.join(f'{k} {v}' for k,v in props.items())}",
                    name, props)
        for action,success,reward in [
            ("eat", name in ("apple","bread"), 0.5 if name in ("apple","bread") else -0.2),
            ("pick", True, 0.1),
        ]:
            agent.world_model.record(
                make_vec(), action, make_vec(),
                reward, success, name, props["category"]
            )
            agent.causal_graph.ingest_feedback(
                action, name, props, success, reward
            )

    print(f"  Taught {agent.rel_memory.n_facts()} facts about "
          f"{len(agent.rel_memory._graph)} objects")
    print(f"  World model rules: {agent.world_model.n_rules}")
    print(f"  Edible known: {agent.what_can_i_eat()}")
    print(f"  Dangerous: {agent.what_is_dangerous()}")

    # ── Phase B: Autonomous run ──
    section("Phase B: Autonomous run (30 steps)")

    agent.add_goal("eat apple", GoalType.EAT_OBJECT, target="apple", priority=0.85)
    agent.add_goal("avoid eating stone", GoalType.AVOID, target="stone", priority=0.95)

    log = agent.run_autonomous(n_steps=30, verbose=True, stop_on_goal="eat apple")

    # ── Phase C: Unknown object encounter ──
    section("Phase C: Encounter unknown — purple_berry")

    # Move near purple_berry
    pb = env.get_object("purple_berry")
    if pb:
        env._agent_pos = pb.position

    print("\n  Triggering curiosity about purple_berry...")
    pb_vec  = agent.enc.encode("purple berry unknown mysterious")
    novelty = agent.curiosity.detect_novelty(pb_vec, "purple_berry")
    report  = agent.curiosity.explore("purple_berry", pb_vec)

    print(f"  Novelty: {novelty:.3f}")
    print(f"  Hypotheses ({report['n_hypotheses']}):")
    for pred, val, conf, basis in report["hypotheses"][:3]:
        print(f"    • {pred}={val} (conf={conf:.2f}) — {basis}")
    print(f"  Suggested test: {report['suggested_action']}")

    # Simulate eating it first (before actually doing it)
    sim_result = agent.simulate_action("eat", "purple_berry")
    print(f"\n  Simulation of eat(purple_berry):")
    print(f"    predicted: {sim_result['predicted_outcome']}  "
          f"(rate={sim_result['success_rate']:.0%})  → {sim_result['recommendation']}")

    # Actually inspect it
    print("\n  Executing: inspect(purple_berry)...")
    s = agent.step(force_action=("inspect","purple_berry"), verbose=True)

    # ── Phase D: Abstraction check ──
    section("Phase D: Knowledge abstraction after experience")

    new_rules = agent.abstraction.run(verbose=True)
    print(f"\n  Abstraction summary: {agent.abstraction.summary()}")
    print(f"  Abstract rule — food.edible: "
          f"{agent.rel_memory.get_value('food','edible')}")
    print(f"  Abstract rule — material.edible: "
          f"{agent.rel_memory.get_value('material','edible')}")

    # ── Phase E: Generalisation ──
    section("Phase E: Generalising to new object (mango)")

    agent.teach("mango yellow tropical fruit", "mango",
                {"category": "food", "color": "yellow"})

    # Can agent infer mango is edible via category rule?
    cat = agent.rel_memory.get_category("mango")
    cat_edible = agent.rel_memory.get_value(cat or "food", "edible")
    direct_edible = agent.rel_memory.get_value("mango", "edible")

    print(f"  mango.category    = {cat}")
    print(f"  {cat or 'food'}.edible    = {cat_edible}  (abstract rule)")
    print(f"  mango.edible (direct) = {direct_edible}")
    print(f"  Generalisation works: {cat_edible == True}")

    # ── Phase F: Self-improvement over time ──
    section("Phase F: Self-improvement — success rate over time")

    # Show improvement by comparing early vs late step success rates
    if len(log) >= 10:
        early_steps = [s for s in log[:10] if s.env_success is not None]
        late_steps  = [s for s in log[-10:] if s.env_success is not None]
        early_sr = sum(s.env_success for s in early_steps) / max(len(early_steps), 1)
        late_sr  = sum(s.env_success for s in late_steps)  / max(len(late_steps),  1)
        print(f"  Early steps success rate: {early_sr:.1%}")
        print(f"  Late  steps success rate: {late_sr:.1%}")
        direction = "↑ improved" if late_sr >= early_sr else "↓ declined"
        print(f"  Trend: {direction}")

    # ── Final summary ──
    section("Final Agent Summary")
    summ = agent.summary()
    for k, v in summ.items():
        if k not in ("internal_state", "env"):
            print(f"  {k:30s}: {v}")
    print(f"\n  Internal state:")
    for k, v in summ["internal_state"].items():
        print(f"    {k:15s}: {v}")
    if summ["env"]:
        print(f"\n  Environment:")
        for k, v in summ["env"].items():
            print(f"    {k:15s}: {v}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   CogniField v4 — Autonomous Self-Improving Agent Demo  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_internal_state()
    demo_simulator()
    demo_abstraction()
    demo_meta_learning()
    demo_consolidation()
    demo_goal_generator()
    demo_hierarchical_planner()
    demo_full_v4()

    print("\n" + "═"*60)
    print("  v4 Demo complete.")
    print("═"*60)
