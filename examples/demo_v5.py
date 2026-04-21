"""
examples/demo_v5.py
====================
CogniField v5 — Stability Demo

Demonstrates:
  1. Belief System     — Bayesian evidence aggregation
  2. Conflict Resolver — handling contradictory observations
  3. Consistency Engine — blocking/downgrading inconsistent updates
  4. Knowledge Validator — periodic re-verification
  5. Experiment Engine — structured safe hypothesis testing
  6. Risk Engine       — blocking dangerous low-confidence actions
  7. Episodic Memory   — experience tracking with importance
  8. Metrics           — stability scoring over time
  9. Full Agent Loop   — all v5 modules integrated
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import time


def sep(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def section(title: str) -> None:
    print(f"\n  {'─'*56}")
    print(f"  {title}")
    print(f"  {'─'*56}")


# ═══════════════════════════════════════════════════════════
# 1. Belief System
# ═══════════════════════════════════════════════════════════
def demo_belief_system():
    sep("1. Belief System — Bayesian Evidence Aggregation")
    from cognifield.world_model.belief_system import BeliefSystem

    bs = BeliefSystem()
    print("\n  Aggregating evidence for apple.edible:")
    print(f"  {'Update':4s} | {'Source':20s} | {'Confidence':10s} | {'Evidence'}")
    print(f"  {'─'*4} | {'─'*20} | {'─'*10} | {'─'*20}")

    def show(n, src, value):
        b = bs.update("apple.edible", value, source=src)
        print(f"  {n:4d} | {src:20s} | {b.confidence:.6f}  | "
              f"+{b.pos_evidence:.1f} / -{b.neg_evidence:.1f}")

    show(1, "prior",              True)
    show(2, "hypothesis",         True)
    show(3, "simulation",         True)
    show(4, "direct_observation", True)
    show(5, "direct_observation", True)
    show(6, "direct_observation", True)

    b = bs.get("apple.edible")
    print(f"\n  After 6 updates: {b}")
    print(f"  Certain? {b.certainty_label}  |  Reliable? {b.is_reliable}")

    # Contrasting: stone is NOT edible
    for _ in range(4):
        bs.update("stone.edible", False, source="direct_observation", weight=1.0)
    stone = bs.get("stone.edible")
    print(f"\n  stone.edible: {stone}")

    # Unknown: purple_berry
    bs.update("purple_berry.edible", True, source="hypothesis", weight=0.3)
    pb = bs.get("purple_berry.edible")
    print(f"  purple_berry.edible: {pb}")
    print(f"  needs_verification: {pb.needs_verification}")

    print(f"\n  Belief System summary: {bs.summary()}")


# ═══════════════════════════════════════════════════════════
# 2. Conflict Resolver
# ═══════════════════════════════════════════════════════════
def demo_conflict_resolver():
    sep("2. Conflict Resolver — Handling Contradictions")
    from cognifield.world_model.belief_system import BeliefSystem
    from cognifield.reasoning.conflict_resolver import ConflictResolver

    bs = BeliefSystem()
    cr = ConflictResolver()

    # Plant a high-confidence belief, then contradict it
    for _ in range(4):
        bs.update("stone.edible", False, source="direct_observation")

    stone_b = bs.get("stone.edible")
    print(f"\n  Established: stone.edible={stone_b.value} (conf={stone_b.confidence:.3f})")

    # Now inject contradicting evidence (e.g. from a glitchy sensor)
    bs.update("stone.edible", True, source="simulation", weight=0.5)
    print(f"  Injected contradiction: stone.edible=True (from simulation)")

    # Run conflict detection
    records = cr.scan(bs)
    print(f"\n  Conflicts detected: {len(records)}")
    for r in records:
        print(f"    key={r.key}  strategy={r.strategy.value}")
        print(f"    {r.value_a}({r.conf_a:.2f}) vs {r.value_b}({r.conf_b:.2f})")
        print(f"    → resolved to: {r.resolved_to}  notes: {r.notes[:60]}")

    # Direct resolution
    print(f"\n  Direct conflict resolution:")
    rec = cr.resolve_direct(bs, "mystery.edible",
                             val_a=True,  conf_a=0.4,
                             val_b=False, conf_b=0.75)
    print(f"    {rec.strategy.value}: resolved_to={rec.resolved_to} "
          f"(conf={rec.resolved_conf:.3f})")
    print(f"    notes: {rec.notes}")

    print(f"\n  Resolver summary: {cr.summary()}")


# ═══════════════════════════════════════════════════════════
# 3. Consistency Engine
# ═══════════════════════════════════════════════════════════
def demo_consistency_engine():
    sep("3. Consistency Engine — Blocking Contradictions")
    from cognifield.world_model.belief_system import BeliefSystem
    from cognifield.reasoning.consistency_engine import ConsistencyEngine

    bs = BeliefSystem()
    ce = ConsistencyEngine(bs, strict_mode=False)  # downgrade, don't block

    # Establish a certain belief
    for _ in range(6):
        bs.update("food.edible", True, source="direct_observation")
    bs.update("apple.is_a", "food", source="direct_observation")
    bs.update("bread.is_a", "food", source="direct_observation")

    print(f"\n  Established: food.edible=True (conf={bs.get_confidence('food.edible'):.3f})")
    print(f"  apple is_a food, bread is_a food")

    # Try to contradict
    test_cases = [
        ("apple.edible", False,  "hypothesis",         0.3,  "contradicts food rule"),
        ("stone.edible", False,  "direct_observation", 1.0,  "new knowledge, no conflict"),
        ("food.edible",  False,  "simulation",         0.2,  "contradicts certain belief"),
        ("bread.edible", True,   "direct_observation", 1.0,  "consistent with food rule"),
    ]

    print(f"\n  {'Key':20s} | {'Value':6s} | {'Source':20s} | {'Allowed':7s} | Note")
    print(f"  {'─'*20} | {'─'*6} | {'─'*20} | {'─'*7} | {'─'*30}")
    for key, val, source, weight, note in test_cases:
        allowed, adj_wt, reason = ce.check_before_update(key, val, source, weight)
        ok = "✓ yes" if allowed else "✗ no"
        adj = f"w={adj_wt:.2f}" if adj_wt != weight else "ok"
        print(f"  {key:20s} | {str(val):6s} | {source:20s} | {ok:7s} | {note}")
        if adj != "ok":
            print(f"    ↳ weight adjusted: {weight:.2f}→{adj_wt:.2f} ({reason[:40]})")

    # Propagation
    print(f"\n  Propagation: setting bread.is_a=food propagates edibility...")
    propagated = ce.propagate("bread.is_a")
    print(f"  Propagated beliefs: {propagated}")
    if propagated:
        for key in propagated:
            b = bs.get(key)
            if b:
                print(f"    {key} = {b.value} (conf={b.confidence:.3f})")

    audit = ce.audit()
    print(f"\n  Audit: consistent={audit['consistent']}, "
          f"violations={audit['n_violations']}, "
          f"downgraded={audit['downgraded']}")


# ═══════════════════════════════════════════════════════════
# 4. Knowledge Validator
# ═══════════════════════════════════════════════════════════
def demo_knowledge_validator():
    sep("4. Knowledge Validator — Periodic Re-Verification")
    from cognifield.world_model.belief_system import BeliefSystem
    from cognifield.world_model.transition_model import TransitionModel
    from cognifield.memory.relational_memory import RelationalMemory
    from cognifield.reasoning.validation import KnowledgeValidator
    from cognifield.latent_space.frequency_space import FrequencySpace

    space = FrequencySpace(dim=64)
    bs    = BeliefSystem()
    rm    = RelationalMemory(dim=64, space=space)
    tm    = TransitionModel(space=space, dim=64)

    # Set up some beliefs
    for name, edible, cat in [
        ("apple", True,  "food"),
        ("bread", True,  "food"),
        ("stone", False, "material"),
        ("hammer",False, "tool"),
    ]:
        bs.update(f"{name}.edible", edible, source="direct_observation")
        bs.update(f"{name}.is_a",   cat,    source="direct_observation")
        rm.add_object_properties(name, {"edible": edible, "is_a": cat})
        for _ in range(3):
            v = np.random.randn(64).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-8
            tm.record(v, "eat", v, 0.5 if edible else -0.2, edible, name, cat)

    # Artificially inflate one belief to test downgrade
    apple_b = bs.get("apple.edible")
    apple_b.confidence   = 0.99
    apple_b.pos_evidence = 50.0

    # Another one: over-pessimistic about bread
    bread_b = bs.get("bread.edible")
    bread_b.confidence   = 0.20
    bread_b.pos_evidence = 0.3

    print(f"\n  Before validation:")
    for name in ["apple","bread","stone"]:
        b = bs.get(f"{name}.edible")
        if b:
            print(f"    {name}.edible = {b.value} (conf={b.confidence:.3f})")

    kv = KnowledgeValidator(bs, rm, tm, validation_interval=0, max_drift=0.20)
    results = kv.validate_all(verbose=True)

    print(f"\n  After validation:")
    for name in ["apple","bread","stone"]:
        b = bs.get(f"{name}.edible")
        if b:
            print(f"    {name}.edible = {b.value} (conf={b.confidence:.3f})")

    print(f"\n  Validation summary: {kv.summary()}")


# ═══════════════════════════════════════════════════════════
# 5. Experiment Engine
# ═══════════════════════════════════════════════════════════
def demo_experiment_engine():
    sep("5. Experiment Engine — Structured Safe Testing")
    from cognifield.world_model.belief_system import BeliefSystem
    from cognifield.world_model.transition_model import TransitionModel
    from cognifield.world_model.causal_graph import CausalGraph
    from cognifield.world_model.simulator import WorldSimulator
    from cognifield.memory.relational_memory import RelationalMemory
    from cognifield.curiosity.advanced_curiosity import AdvancedCuriosityEngine
    from cognifield.curiosity.experiment_engine import (
        ExperimentEngine, SAFETY_LADDER, MIN_CONF_FOR_RISK
    )
    from cognifield.memory.memory_store import MemoryStore
    from cognifield.latent_space.frequency_space import FrequencySpace

    space = FrequencySpace(dim=64)
    bs    = BeliefSystem()
    rm    = RelationalMemory(dim=64, space=space)
    vm    = MemoryStore(dim=64)
    tm    = TransitionModel(space=space, dim=64)
    cg    = CausalGraph()
    sim   = WorldSimulator(tm, cg, space, dim=64)
    cur   = AdvancedCuriosityEngine(space, rm, vm, dim=64)
    ee    = ExperimentEngine(bs, sim, cur, min_conf_to_act=0.70)

    # Pre-load some world model experience
    for action, obj, cat, success, reward in [
        ("eat","apple","food",True,0.5), ("eat","apple","food",True,0.5),
        ("eat","stone","material",False,-0.2),
        ("inspect","purple_berry","unknown",True,0.05),
    ]:
        v = np.random.randn(64).astype(np.float32); v /= np.linalg.norm(v) + 1e-8
        tm.record(v, action, v, reward, success, obj, cat)

    # Set up hypotheses
    cur.generate_hypotheses("purple_berry",
                            np.random.randn(64).astype(np.float32),
                            known_props={})

    print(f"\n  Safety Ladder for 'edible' property:")
    prior_conf = [0.10, 0.25, 0.50, 0.65, 0.80]
    for pc in prior_conf:
        bs_temp = BeliefSystem()
        bs_temp.update("purple_berry.edible", True, source="hypothesis",
                        weight=pc * 0.8)
        exp = ExperimentEngine(bs_temp, sim, cur)
        e   = exp.design("purple_berry", "edible")
        safe, reason = exp.is_safe_to_execute(e)
        icon = "✓" if safe else "✗"
        print(f"    prior_conf={pc:.2f} → action={e.action:8s} "
              f"safety={e.safety_level.name:10s} {icon} {'' if safe else '→ ' + reason[:30]}")

    # Full experiment workflow
    section("Full Experiment Protocol: purple_berry")
    bs.update("purple_berry.edible", True, source="hypothesis", weight=0.2)
    state_vec = np.zeros(64, dtype=np.float32); state_vec[0] = 1.0

    exp = ee.design("purple_berry", "edible", state_vec=state_vec)
    print(f"\n  Designed experiment:")
    print(f"    target:          {exp.target}")
    print(f"    property:        {exp.property}")
    print(f"    prior_conf:      {exp.prior_confidence:.3f}")
    print(f"    action:          {exp.action}")
    print(f"    safety_level:    {exp.safety_level.name}")
    print(f"    sim_prediction:  {exp.sim_prediction}")
    print(f"    simulation_conf: {exp.sim_confidence:.3f}")

    safe, reason = ee.is_safe_to_execute(exp)
    print(f"\n  Safe to execute? {safe} — {reason}")

    # Simulate positive result (inspect reveals edible=True)
    fake_feedback = {
        "success":      True,
        "reward":       0.05,
        "action":       exp.action,
        "object_name":  "purple_berry",
        "object_props": {"edible": True, "fragile": False, "category": "food"},
        "learned":      "purple_berry is edible",
    }
    exp.status = "designed"  # mark ready for processing
    result = ee.process_result(exp, fake_feedback)

    print(f"\n  Experiment result: {result.insight}")
    print(f"  New belief confidence: {exp.post_confidence:.3f}")
    print(f"  Belief updates: {result.belief_update}")
    print(f"\n  Engine summary: {ee.summary()}")


# ═══════════════════════════════════════════════════════════
# 6. Risk Engine
# ═══════════════════════════════════════════════════════════
def demo_risk_engine():
    sep("6. Risk Engine — Blocking Dangerous Actions")
    from cognifield.world_model.belief_system import BeliefSystem
    from cognifield.agent.risk_engine import RiskEngine

    bs = BeliefSystem()
    re = RiskEngine(bs, risk_tolerance=0.35)

    # Load beliefs
    for _ in range(4): bs.update("apple.edible",  True,  source="direct_observation")
    for _ in range(4): bs.update("stone.edible",  False, source="direct_observation")

    test_actions = [
        ("observe",  "apple",        "safe observation"),
        ("inspect",  "stone",        "safe inspection"),
        ("pick",     "apple",        "pick up apple"),
        ("eat",      "apple",        "eat known edible"),
        ("eat",      "stone",        "eat known NOT edible"),
        ("eat",      "purple_berry", "eat UNKNOWN object"),
        ("eat",      "glowing_cube", "eat UNKNOWN object"),
        ("drop",     "apple",        "drop apple (not fragile)"),
    ]

    print(f"\n  {'Action':8s} | {'Target':14s} | {'Risk':6s} | {'Decision':8s} | {'Reason'}")
    print(f"  {'─'*8} | {'─'*14} | {'─'*6} | {'─'*8} | {'─'*35}")
    for action, target, note in test_actions:
        ra = re.assess(action, target, agent_confidence=0.6)
        icon = "✓" if ra.decision == "proceed" else ("⚠" if ra.decision == "caution" else "🛑")
        alt  = f" → try: {ra.safer_alternative}" if ra.safer_alternative and ra.decision != "proceed" else ""
        print(f"  {action:8s} | {target:14s} | {ra.risk_score:.4f} | "
              f"{icon} {ra.decision:7s} | {note}{alt}")

    # Safest from candidates
    candidates = [("eat","apple"), ("eat","stone"), ("inspect","purple_berry"),
                  ("pick","apple"), ("eat","purple_berry")]
    safest, ra = re.safest_action(candidates)
    print(f"\n  Safest action from candidates: {safest} (risk={ra.risk_score:.4f})")

    # Filter safe
    safe_actions = re.filter_safe(candidates)
    print(f"  Safe actions: {[(a,o) for (a,o),_ in safe_actions]}")
    print(f"\n  Risk profile: {re.risk_profile()}")


# ═══════════════════════════════════════════════════════════
# 7. Metrics
# ═══════════════════════════════════════════════════════════
def demo_metrics():
    sep("7. Stability Metrics — Performance Over Time")
    from cognifield.evaluation.metrics import AgentMetrics
    import random

    m = AgentMetrics(window=30)
    rng = random.Random(42)

    phases = [
        (15, 0.30, 0.45, "unstable start"),
        (15, 0.65, 0.70, "learning phase"),
        (15, 0.85, 0.88, "stable operation"),
    ]

    print(f"\n  {'Phase':20s} | {'SR':5s} | {'Stab':5s} | {'Cons':5s} | {'ERR':5s} | Grade")
    print(f"  {'─'*20} | {'─'*5} | {'─'*5} | {'─'*5} | {'─'*5} | {'─'*5}")

    for (n_steps_phase, sr, belief_conf, desc) in phases:
        for step_i in range(n_steps_phase):
            success = rng.random() < sr
            m.record(
                step=int(m._n_total + 1),
                success=success,
                reward=rng.uniform(0.3, 0.5) if success else rng.uniform(-0.3,-0.1),
                belief_confidence=belief_conf + rng.uniform(-0.05, 0.05),
                n_conflicts=0 if belief_conf > 0.7 else rng.randint(0, 2),
                n_blocks=rng.randint(0, 1),
                novelty=rng.uniform(0.1, 0.5),
                action=rng.choice(["eat","pick","inspect"]),
            )
            # Snapshot beliefs for stability tracking
            m.snapshot_beliefs({
                "apple.edible": belief_conf + rng.uniform(-0.03, 0.03),
                "stone.edible": 1 - belief_conf + rng.uniform(-0.03, 0.03),
            })
        r = m.report()
        print(f"  {desc:20s} | {r['success_rate']:.3f} | "
              f"{r['belief_stability']:.3f} | {r['consistency_score']:.3f} | "
              f"{r['error_reduction']:+.3f} | {m.stability_grade()}")

    print(f"\n  Final comprehensive report:")
    r = m.report()
    for k, v in r.items():
        print(f"    {k:25s}: {v}")


# ═══════════════════════════════════════════════════════════
# 8. Full v5 Agent Demo
# ═══════════════════════════════════════════════════════════
def demo_full_v5():
    sep("8. Full v5 Agent — Stable Belief-Driven Operation")
    from cognifield.agent.agent_v5 import CogniFieldAgentV5, AgentV5Config
    from cognifield.environment.rich_env import RichEnv
    from cognifield.agent.goals import GoalType

    env   = RichEnv(seed=42)
    agent = CogniFieldAgentV5(
        config=AgentV5Config(
            dim=64, risk_tolerance=0.35,
            plan_depth=3, plan_beam=3,
            novelty_threshold=0.35,
            consolidation_interval=20,
            abstraction_interval=15,
            meta_analysis_interval=10,
            validation_interval=8,
            verbose=False,
        ),
        env=env,
    )

    # ── Phase A: Teach world ──
    section("Phase A: Teaching world knowledge")
    for name, props in [
        ("apple",    {"edible":True,  "category":"food",     "color":"red"}),
        ("bread",    {"edible":True,  "category":"food",     "color":"yellow"}),
        ("stone",    {"edible":False, "category":"material", "heavy":True}),
        ("glass_jar",{"edible":False, "category":"tool",     "fragile":True}),
    ]:
        agent.teach(f"{name} {' '.join(f'{k} {v}' for k,v in props.items())}",
                    name, props)
        for action, success, reward in [
            ("eat",  name in ("apple","bread"), 0.5 if name in ("apple","bread") else -0.2),
            ("pick", True, 0.1),
        ]:
            v = np.random.randn(agent.cfg.dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-8
            agent.world_model.record(v, action, v, reward, success, name, props["category"])

    print(f"  Beliefs loaded: {len(agent.beliefs)}")
    print(f"  Edible known: {agent.what_can_i_eat()}")
    print(f"  Dangerous: {agent.what_is_dangerous()}")

    # ── Phase B: Unknown object — structured experiment ──
    section("Phase B: Encountering purple_berry (unknown)")
    print("  Agent encounters purple_berry — scheduling experiment...")
    agent.schedule_experiment("purple_berry")

    pb_vec = agent.enc.encode("purple berry unknown")
    agent.curiosity.explore("purple_berry", pb_vec)
    agent.beliefs.update("purple_berry.edible", True,
                          source="hypothesis", weight=0.2)

    print(f"  purple_berry.edible confidence: "
          f"{agent.how_confident('purple_berry','edible'):.3f}")
    print(f"  Risk of eat(purple_berry): ", end="")
    ra = agent.risk_engine.assess("eat", "purple_berry", agent_confidence=0.5)
    print(f"{ra.decision} (risk={ra.risk_score:.3f})")

    # ── Phase C: Set goals + run ──
    section("Phase C: Autonomous run — safe, belief-driven")
    agent.add_goal("eat apple", GoalType.EAT_OBJECT, target="apple", priority=0.85)
    agent.add_goal("avoid eating stone", GoalType.AVOID, target="stone", priority=0.95)

    log = agent.run_autonomous(n_steps=30, verbose=True, stop_on_goal="eat apple")

    # ── Phase D: Belief stability check ──
    section("Phase D: Belief Stability After 30 Steps")
    for name in ["apple", "stone", "bread", "glass_jar", "purple_berry"]:
        b_edible = agent.beliefs.get(f"{name}.edible")
        if b_edible:
            label = b_edible.certainty_label
            print(f"  {name:14s}.edible = {str(b_edible.value):6s} "
                  f"conf={b_edible.confidence:.3f}  [{label}]  "
                  f"ev={b_edible.total_evidence:.1f}")

    # ── Phase E: Conflict test ──
    section("Phase E: Injecting Conflicting Evidence")
    print("  Injecting: stone.edible=True (from low-quality sensor)...")
    agent.beliefs.update("stone.edible", True, source="simulation", weight=0.2)
    conflicts = agent.conflict_resolver.scan(agent.beliefs)
    print(f"  Conflicts detected: {len(conflicts)}")
    for c in conflicts:
        print(f"    {c.key}: {c.strategy.value} → {c.notes[:60]}")

    stone_b = agent.beliefs.get("stone.edible")
    print(f"  stone.edible after resolution: {stone_b.value} "
          f"(conf={stone_b.confidence:.3f})")

    # ── Phase F: Generalisation via abstraction ──
    section("Phase F: Abstraction — Generalising to New Object")
    new_rules = agent.abstraction.run(verbose=True)
    agent.teach("mango tropical fruit", "mango", {"category": "food"})

    # Can agent infer mango edibility from food→edible rule?
    cat_edible = agent.beliefs.get_value("food.edible")
    prop_conf  = agent.how_confident("food", "edible")
    print(f"\n  food.edible = {cat_edible} (conf={prop_conf:.3f})")
    print(f"  mango.category = food → can infer mango.edible={cat_edible}")

    # ── Phase G: Long-run stability (100 more steps) ──
    section("Phase G: Long-Run Stability Test (50 more steps)")
    log2 = agent.run_autonomous(n_steps=50, verbose=False)
    r    = agent.metrics.report()
    print(f"  After {agent._step_count} total steps:")
    print(f"  Success rate:      {r['success_rate']:.1%}")
    print(f"  Belief stability:  {r['belief_stability']:.3f}")
    print(f"  Consistency score: {r['consistency_score']:.3f}")
    print(f"  Error reduction:   {r['error_reduction']:+.3f}")
    print(f"  Conflict rate:     {r['conflict_rate']:.3f}")
    print(f"  Stability grade:   {agent.metrics.stability_grade()}")

    # ── Final Summary ──
    section("Final Agent Summary")
    summ = agent.summary()
    key_fields = ["steps","beliefs","reliable_beliefs","relational_facts",
                  "world_model_rules","abstract_rules","goals_completed",
                  "conflicts_resolved","stability_grade","success_rate"]
    for k in key_fields:
        print(f"  {k:30s}: {summ[k]}")
    print(f"\n  Edible known:    {summ['edible_known']}")
    print(f"  Dangerous:       {summ['dangerous_known']}")
    print(f"  Risk profile:    {summ['risk_profile']}")
    print(f"\n  Experiment stats: {summ['experiments']}")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   CogniField v5 — Stable, Belief-Driven Agent Demo     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_belief_system()
    demo_conflict_resolver()
    demo_consistency_engine()
    demo_knowledge_validator()
    demo_experiment_engine()
    demo_risk_engine()
    demo_metrics()
    demo_full_v5()

    print("\n" + "═"*60)
    print("  v5 Demo complete.")
    print("═"*60)
