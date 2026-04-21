"""
examples/demo_v9.py
====================
CogniField v9 — Adaptive Self-Reflective Intelligence Demo

Demonstrates all 6 new v9 modules:
  1. MetaCognition   — overconfidence detection, bias detection, calibration
  2. UncertaintyEngine — noise injection, confidence decay, partial observability
  3. GoalConflictResolver — competing goals, trade-off decisions
  4. StrategyManager — explore/exploit/verify/recover switching
  5. TemporalMemory  — long-term patterns, drift tracking, recurrence
  6. SelfEvaluator   — graded performance report + weakness list

And the full v9 agent scenario:
  7. Full demo: conflicting goals → uncertainty → adaptation → improvement
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import time


def sep(title): print(f"\n{'═'*64}\n  {title}\n{'═'*64}")
def section(title): print(f"\n  {'─'*60}\n  {title}\n  {'─'*60}")


# ═══════════════════════════════════════════════════════════
# 1. MetaCognition
# ═══════════════════════════════════════════════════════════
def demo_meta_cognition():
    sep("1. MetaCognition — Self-Analysis & Bias Detection")
    from cognifield.core.meta_cognition import MetaCognitionEngine
    import random

    mc  = MetaCognitionEngine(overconf_threshold=0.15, reflection_interval=5)
    rng = random.Random(42)

    print(f"\n  Simulating overconfident agent (conf=0.85, actual_sr≈0.40):")
    print(f"\n  {'Step':4s} | {'Conf':5s} | {'Correct':7s} | {'Domain'}")
    print(f"  {'─'*4} | {'─'*5} | {'─'*7} | {'─'*10}")

    for step in range(1, 26):
        # Agent is overconfident: claims 0.85 but only right 40% of the time
        conf    = 0.85 + rng.uniform(-0.05, 0.05)
        correct = rng.random() < 0.40
        domain  = rng.choice(["edible", "edible", "fragile"])
        mc.record_step(step, correct, reward=0.5 if correct else -0.2,
                       mean_conf=conf, predicate=domain)
        mc.record_outcome(conf, True, correct, domain, step)
        if step % 5 == 0:
            print(f"  {step:4d} | {conf:.3f} | {'✓' if correct else '✗':7s} | {domain}")

    section("Reflection at step 25")
    log = []
    def adjust_fn(issue, magnitude):
        log.append(f"{issue}: magnitude={magnitude:.3f}")

    reflections = mc.reflect(25, adjust_fn=adjust_fn)
    for r in reflections:
        print(f"\n  Finding:  {r.finding}")
        print(f"  Action:   {r.action_taken}")
        print(f"  Metric:   {r.metric}, before={r.before:.3f}, after={r.after:.3f}")

    print(f"\n  Adjustments triggered: {log}")
    print(f"\n  Summary: {mc.summary()}")

    section("Calibration curve")
    cal_pts = [(k, cp) for k, cp in mc._calibration.items() if cp.n_samples >= 2]
    print(f"\n  {'Conf bucket':12s} | {'Predicted':9s} | {'Empirical':9s} | {'Error':6s}")
    print(f"  {'─'*12} | {'─'*9} | {'─'*9} | {'─'*6}")
    for key, cp in sorted(cal_pts):
        print(f"  {key:12s} | {cp.conf_bucket:.3f}     | {cp.empirical_rate:.3f}     | "
              f"{cp.calibration_error:.3f}")


# ═══════════════════════════════════════════════════════════
# 2. UncertaintyEngine
# ═══════════════════════════════════════════════════════════
def demo_uncertainty_engine():
    sep("2. UncertaintyEngine — Noise, Decay & Partial Observability")
    from cognifield.core.uncertainty_engine import (
        UncertaintyEngine, UncertaintyLevel
    )
    from cognifield.world_model.belief_system import BeliefSystem

    section("Noise injection across uncertainty levels")
    print(f"\n  {'Level':8s} | {'Noise%':6s} | {'n_corrupt/10':12s} | {'Sample outcome'}")
    print(f"  {'─'*8} | {'─'*6} | {'─'*12} | {'─'*20}")

    for level in UncertaintyLevel:
        ue  = UncertaintyEngine(level=level, seed=42)
        n_corrupt = 0
        for _ in range(10):
            obs = ue.corrupt(True, confidence=0.85, predicate="edible")
            if obs.was_corrupted:
                n_corrupt += 1
        sample = ue.corrupt(True, confidence=0.85, predicate="edible")
        print(f"  {level.value:8s} | {ue.summary()['noise_rate']:6.0%} | "
              f"{n_corrupt:12d} | observed={sample.observed_value}, "
              f"corrupt={sample.was_corrupted}")

    section("Belief confidence decay under uncertainty")
    bs = BeliefSystem()
    for _ in range(5): bs.update("apple.edible", True, "direct_observation")
    conf_before = bs.get("apple.edible").confidence
    print(f"\n  Initial apple.edible confidence: {conf_before:.4f}")

    for level in [UncertaintyLevel.LOW, UncertaintyLevel.MEDIUM,
                  UncertaintyLevel.HIGH, UncertaintyLevel.CHAOTIC]:
        bs_test = BeliefSystem()
        for _ in range(5): bs_test.update("apple.edible", True, "direct_observation")
        ue_test = UncertaintyEngine(level=level, seed=42)
        for _ in range(10):
            ue_test.decay_all_beliefs(bs_test, steps=1)
        after = bs_test.get("apple.edible").confidence
        floor = ue_test.summary().get("decay_rate", 0)
        print(f"  {level.value:8s}: {conf_before:.4f} → {after:.4f} "
              f"(decay_rate={ue_test.summary()['decay_rate']:.4f})")

    section("Partial observability")
    ue = UncertaintyEngine(level=UncertaintyLevel.MEDIUM, seed=42)
    ue.hide_property("heavy")
    ue.hide_property("temperature")
    print(f"\n  Visible props: {ue.summary()['visible_props']}")
    print(f"  Hidden props:  {ue.summary()['hidden_props']}")

    for prop in ["edible", "heavy", "temperature", "fragile"]:
        obs = ue.corrupt(True, 0.8, predicate=prop)
        obs_str = "HIDDEN" if not ue.is_observable(prop) else str(obs.observed_value)
        print(f"  observe({prop}): {obs_str} "
              f"(conf_weight={obs.confidence_weight:.2f})")

    section("Auto-detection of uncertainty level")
    ue2 = UncertaintyEngine(level=UncertaintyLevel.LOW, seed=42)
    # Simulate chaotic rewards
    rng2 = np.random.default_rng(42)
    for _ in range(20):
        reward = rng2.choice([-0.5, -0.3, 0.1, 0.5, 0.8])
        ue2.record_outcome_variance(float(reward))
    detected = ue2.auto_detect_level()
    print(f"\n  Started as: LOW")
    print(f"  After chaotic rewards: auto-detected as: {detected.value}")

    section("Consensus slowdown under uncertainty")
    for level in UncertaintyLevel:
        ue3 = UncertaintyEngine(level=level)
        sm  = ue3.consensus_supermajority(base=0.55)
        print(f"  {level.value:8s}: base=0.55 → required_supermajority={sm:.3f}")


# ═══════════════════════════════════════════════════════════
# 3. GoalConflictResolver
# ═══════════════════════════════════════════════════════════
def demo_goal_conflict_resolver():
    sep("3. GoalConflictResolver — Competing Goals & Trade-offs")
    from cognifield.agents.goal_conflict_resolver import (
        GoalConflictResolver, ResolutionStrategy, ConflictType
    )
    from cognifield.agents.goals import GoalSystem, GoalType

    section("Building conflicting goal set")
    gs = GoalSystem(max_active=8)
    gs.add_eat_goal("apple",        priority=0.80)
    gs.add_eat_goal("bread",        priority=0.70)
    gs.add_avoid_goal("stone",      priority=0.95)
    gs.add_explore_goal("explore",  priority=0.50)
    # Manually create an avoid-explore conflict
    from cognifield.agents.goals import Goal
    cautious = gs.add_goal(
        label="avoid unknown objects",
        goal_type=GoalType.AVOID,
        target="unknown", priority=0.75
    )

    all_goals = [g for g in gs._goals]
    print(f"\n  Active goals ({len(all_goals)}):")
    for g in all_goals:
        print(f"    [{g.goal_type.value:15s}] '{g.label}' (p={g.priority:.2f})")

    gcr = GoalConflictResolver(strategy=ResolutionStrategy.UTILITY_MAXIMISATION)
    conflicts = gcr.detect_conflicts(all_goals)
    print(f"\n  Detected conflicts ({len(conflicts)}):")
    for c in conflicts:
        print(f"    [{c.conflict_type.value:10s}] {c.description[:60]}")
        print(f"    severity={c.severity:.2f}")

    section("Resolution strategies comparison")
    for strategy in [ResolutionStrategy.PRIORITY_ORDER,
                     ResolutionStrategy.UTILITY_MAXIMISATION,
                     ResolutionStrategy.SATISFICING]:
        gcr_s = GoalConflictResolver(strategy=strategy, sat_threshold=0.65)
        decision = gcr_s.resolve(all_goals)
        print(f"\n  Strategy: {strategy.value}")
        print(f"    Chosen ({len(decision.chosen_goals)}): "
              f"{[g.label for g in decision.chosen_goals]}")
        print(f"    Dropped ({len(decision.dropped_goals)}): "
              f"{[g.label for g in decision.dropped_goals]}")
        print(f"    Utility: {decision.utility_score:.3f}")

    print(f"\n  Summary: {gcr.summary()}")


# ═══════════════════════════════════════════════════════════
# 4. StrategyManager
# ═══════════════════════════════════════════════════════════
def demo_strategy_manager():
    sep("4. StrategyManager — Dynamic Explore/Exploit Switching")
    from cognifield.agents.strategy_manager import StrategyManager, Strategy
    import random

    sm  = StrategyManager(
        initial_strategy=Strategy.EXPLORE,
        eval_freq=8,
        fail_threshold=0.25,
        win_threshold=0.70,
        max_consec_fails=4,
    )

    rng = random.Random(42)

    global_step = [0]
    def simulate_phase(n_steps, sr, novelty, label):
        print(f"\n  Phase: {label} (n={n_steps}, sr={sr:.0%}, novelty={novelty:.1f})")
        switches = []
        for i in range(n_steps):
            global_step[0] += 1
            step    = global_step[0]
            success = rng.random() < sr
            sm.record_step(step, success, reward=0.4 if success else -0.15,
                           novelty=novelty)
            event = sm.evaluate(step, peer_agreement=0.6)
            if event:
                switches.append(f"step {step}: {event.from_strat.value}"
                                 f" → {event.to_strat.value} ({event.reason[:40]})")
        for s in switches:
            print(f"    Switch: {s}")
        print(f"    Current strategy: {sm.current.value}")

    simulate_phase(12, 0.65, 0.45, "Good exploration phase")
    simulate_phase(8,  0.80, 0.12, "High success, low novelty")
    simulate_phase(10, 0.18, 0.25, "Struggling phase")
    simulate_phase(3,  0.10, 0.20, "Consecutive failures")
    simulate_phase(10, 0.60, 0.35, "Recovery + improvement")

    print(f"\n  Strategy history (last 5):")
    for entry in sm.summary()["switch_history"]:
        print(f"    step={entry['step']}: {entry['from']} → {entry['to']}: {entry['reason'][:50]}")

    print(f"\n  Time per strategy: {sm.summary()['time_per_strategy']}")
    print(f"  Total switches: {sm.switches()}")


# ═══════════════════════════════════════════════════════════
# 5. TemporalMemory
# ═══════════════════════════════════════════════════════════
def demo_temporal_memory():
    sep("5. TemporalMemory — Long-Term Patterns & Drift")
    from cognifield.memory.temporal_memory import TemporalMemory
    import random

    tm  = TemporalMemory(window=12)
    rng = random.Random(42)

    # Simulate: apple always edible, stone never, purple_berry unstable
    scenarios = [
        ("eat", "apple",       lambda: True,                  0.50),
        ("eat", "stone",       lambda: False,                 -0.20),
        ("eat", "purple_berry",lambda: rng.random() < 0.55,  0.30),
        ("pick","rock",        lambda: rng.random() < 0.70,  0.10),
    ]

    for step in range(1, 31):
        for action, target, outcome_fn, base_reward in scenarios:
            success = outcome_fn()
            reward  = base_reward if success else -abs(base_reward)
            tm.record_outcome(action, target, success, reward, step=step,
                              context={"strategy": "explore" if step < 15 else "exploit"})

        # Snapshot belief confidence (simulated drift)
        for key, trend in [("apple.edible", 0.02), ("stone.edible", -0.015),
                            ("purple_berry.edible", 0.0)]:
            conf = 0.5 + trend * step + rng.uniform(-0.02, 0.02)
            tm.record_belief_snapshot(key, float(np.clip(conf, 0.1, 0.95)), step=step)

    section("Pattern detection")
    print(f"\n  {'Key':25s} | {'Type':10s} | {'SR':5s} | {'n':3s} | {'Conf':5s}")
    print(f"  {'─'*25} | {'─'*10} | {'─'*5} | {'─'*3} | {'─'*5}")
    patterns = tm.detect_all_patterns()
    for p in patterns:
        print(f"  {p.key:25s} | {p.pattern_type:10s} | "
              f"{p.success_rate:.3f} | {p.n_samples:3d} | {p.confidence:.3f}")

    section("Belief drift analysis")
    for key in ["apple.edible", "stone.edible", "purple_berry.edible"]:
        drift = tm.belief_drift(key)
        mc    = tm.mean_confidence(key)
        print(f"  {key:25s}: drift={drift:12s}, mean_conf={mc:.3f}")

    section("Recurrence detection")
    # Simulate 4 consecutive failures on eat(purple_berry)
    for _ in range(4):
        tm.record_outcome("eat", "mystery_obj", False, -0.3, step=31)
    print(f"\n  eat(mystery_obj) stuck: {tm.is_stuck('eat','mystery_obj',threshold=3)}")
    print(f"  eat(apple) stuck:       {tm.is_stuck('eat','apple',threshold=3)}")

    section("Context correlation")
    best_strat = tm.best_strategy_for_context()
    print(f"\n  Best strategy from context history: {best_strat}")
    print(f"\n  Summary: {tm.summary()}")


# ═══════════════════════════════════════════════════════════
# 6. SelfEvaluator
# ═══════════════════════════════════════════════════════════
def demo_self_evaluator():
    sep("6. SelfEvaluator — Graded Performance Reports")
    from cognifield.agents.self_evaluator import SelfEvaluator, DIMENSIONS
    from cognifield.agents.agent_v9 import CogniFieldAgentV9, AgentV9Config
    from cognifield.environment.rich_env import RichEnv

    # Build a minimal agent for eval
    agent = CogniFieldAgentV9(
        config=AgentV9Config(
            agent_id="eval_test", dim=64, verbose=False, seed=42
        ),
        env=RichEnv(seed=42),
    )
    agent.teach("apple food edible", "apple", {"edible":True,"category":"food"})
    agent.teach("stone material", "stone", {"edible":False,"category":"material"})

    # Run 15 steps to generate data
    for _ in range(15):
        agent.step(verbose=False)

    se = SelfEvaluator(eval_freq=1, weakness_threshold=0.50)
    report = se.evaluate(step=15, agent=agent)

    print(f"\n  Self-evaluation at step 15:")
    print(f"\n  {'Dimension':20s} | {'Score':6s} | {'Status'}")
    print(f"  {'─'*20} | {'─'*6} | {'─'*15}")
    for dim, score in report.scores.items():
        status = "✓ OK" if score >= 0.50 else "⚠ WEAK"
        print(f"  {dim:20s} | {score:.4f} | {status}")

    print(f"\n  Overall: {report.overall:.3f}  →  Grade: {report.grade}")
    print(f"  Weaknesses: {report.weaknesses}")
    print(f"\n  Improvement suggestions:")
    for dim, suggestion in report.suggestions.items():
        print(f"    [{dim}] {suggestion}")


# ═══════════════════════════════════════════════════════════
# 7. Full v9 Agent Scenario
# ═══════════════════════════════════════════════════════════
def demo_full_v9():
    sep("7. Full v9 System — Adaptive Self-Reflective Intelligence")
    from cognifield.agents.agent_v9 import CogniFieldAgentV9, AgentV9Config
    from cognifield.agents.agent_v7 import AgentRole
    from cognifield.agents.strategy_manager import Strategy
    from cognifield.core.uncertainty_engine import UncertaintyLevel
    from cognifield.agents.goals import GoalType
    from cognifield.communication.communication_module import CommunicationModule
    from cognifield.memory.shared_memory import SharedMemory
    from cognifield.agents.group_mind import GroupMind
    from cognifield.reasoning.global_consensus import GlobalConsensus
    from cognifield.core.event_bus import EventBus
    from cognifield.planning.cooperation_engine import CooperationEngine
    from cognifield.environment.rich_env import RichEnv

    env  = RichEnv(seed=42)
    bus  = CommunicationModule()
    sm   = SharedMemory()
    eb   = EventBus()
    gm   = GroupMind(event_bus=eb)
    ce   = CooperationEngine()
    gc   = GlobalConsensus(sm, bus, eb, supermajority=0.60)

    agents = []
    roles  = [AgentRole.EXPLORER, AgentRole.ANALYST, AgentRole.RISK_MANAGER]
    for i, role in enumerate(roles):
        cfg = AgentV9Config(
            agent_id=f"v9_{i}", role=role,
            dim=64, seed=42+i, verbose=False,
            uncertainty_level="medium",
        )
        a = CogniFieldAgentV9(
            config=cfg, env=env, comm_bus=bus, shared_mem=sm,
            group_mind=gm, global_cons=gc, event_bus=eb, coop_engine=ce,
        )
        agents.append(a)
        ce.register_agent(a.agent_id, role.value)
        for b_agent in agents[:-1]:
            a.register_for_negotiation(b_agent)
            b_agent.register_for_negotiation(a)

    section("Phase A: World + Conflicting Goals")
    for name, props in [
        ("apple",  {"edible":True,  "category":"food"}),
        ("stone",  {"edible":False, "category":"material"}),
        ("bread",  {"edible":True,  "category":"food"}),
    ]:
        for a in agents:
            a.teach(f"{name} {' '.join(f'{k} {v}' for k,v in props.items())}",
                    name, props)

    # Inject conflicting goals into Explorer
    explorer = agents[0]
    explorer.add_goal("eat apple",            GoalType.EAT_OBJECT, "apple",    0.80)
    explorer.add_goal("eat bread",            GoalType.EAT_OBJECT, "bread",    0.70)
    explorer.add_goal("avoid unknown objects",GoalType.AVOID,      "unknown",  0.75)
    explorer.add_goal("explore surroundings", GoalType.EXPLORE,    "world",    0.55)

    active_goals = [g for g in explorer.goal_system._goals if g.is_active]
    print(f"\n  Explorer has {len(active_goals)} competing goals:")
    for g in active_goals:
        print(f"    [{g.goal_type.value:14s}] '{g.label}' (p={g.priority:.2f})")

    conflicts = explorer.goal_resolver.detect_conflicts(active_goals)
    print(f"\n  Detected {len(conflicts)} conflicts:")
    for c in conflicts:
        print(f"    [{c.conflict_type.value}] {c.description[:65]}")

    decision = explorer.goal_resolver.resolve(
        active_goals, belief_system=explorer.beliefs,
        internal_state=explorer.internal_state
    )
    print(f"\n  Resolution (utility_maximisation):")
    print(f"    Chosen:  {[g.label for g in decision.chosen_goals]}")
    print(f"    Dropped: {[g.label for g in decision.dropped_goals]}")
    print(f"    Utility: {decision.utility_score:.3f}")

    section("Phase B: Uncertain Environment")
    print(f"\n  Uncertainty levels over 3 phases:")
    for level_name in ["low", "medium", "high"]:
        ue  = agents[0].uncertainty
        ue.level = {"low":    UncertaintyLevel.LOW,
                    "medium": UncertaintyLevel.MEDIUM,
                    "high":   UncertaintyLevel.HIGH}[level_name]
        n_corrupt = 0
        for _ in range(10):
            obs = ue.corrupt(True, 0.85, "edible")
            if obs.was_corrupted: n_corrupt += 1
        print(f"    {level_name:6s}: {n_corrupt}/10 observations corrupted")

    section("Phase C: 20-Round Adaptive Run")
    gm.set_primary_goal("eat apple")
    for a in agents:
        a.add_goal("eat apple", GoalType.EAT_OBJECT, target="apple", priority=0.85)

    print(f"\n  {'Rnd':3s}|{'Agent':8s}|{'Role':14s}|{'Strat':12s}|"
          f"{'Unc':8s}|{'Rew':5s}|{'Switch':6s}|"
          f"{'Conflicts':9s}|{'Grade'}")
    print(f"  {'─'*3}|{'─'*8}|{'─'*14}|{'─'*12}|"
          f"{'─'*8}|{'─'*5}|{'─'*6}|{'─'*9}|{'─'*5}")

    for rnd in range(20):
        # Periodic global consensus
        if rnd % 5 == 4:
            tm_cross = {a.agent_id: float(np.mean(
                [a2.effective_trust(a.agent_id) for a2 in agents
                 if a2.agent_id != a.agent_id])) for a in agents}
            ab = {a.agent_id: a.beliefs for a in agents}
            gc.run_round(ab, tm_cross)
            gc.apply_to_all(ab)

        for a in agents:
            a.ensure_bidirectional_comm()
            s = a.step(verbose=False)
            if rnd % 4 == 3:
                rew = f"{s.env_reward:+.2f}" if s.env_reward else "  N/A"
                sw  = "✓" if s.strategy_switched else " "
                grade = s.self_eval_grade or "–"
                print(f"  {rnd+1:3d}|{a.agent_id:8s}|{a.role.value:14s}|"
                      f"{s.strategy:12s}|{s.uncertainty_level:8s}|"
                      f"{rew:5s}|{sw:6s}|{s.goal_conflicts:9d}|{grade:5s}")

    section("Phase D: Self-Reflection Logs")
    for a in agents:
        log = a.get_reflection_log(3)
        if log:
            print(f"\n  {a.agent_id} ({a.role.value}) reflections:")
            for entry in log:
                print(f"    → {entry[:70]}")

    section("Phase E: Temporal Patterns")
    for a in agents:
        patterns = a.temporal_mem.detect_all_patterns()
        if patterns:
            print(f"\n  {a.agent_id} patterns:")
            for p in patterns[:3]:
                print(f"    {p.key:25s}: {p.pattern_type:10s} "
                      f"(sr={p.success_rate:.2f}, n={p.n_samples})")

    section("Phase F: Strategy Distribution")
    for a in agents:
        tps = a.strategy_mgr.summary()["time_per_strategy"]
        sw  = a.strategy_mgr.switches()
        print(f"\n  {a.agent_id}: switches={sw}, "
              f"time={tps}")

    section("Final Agent Summary")
    for a in agents:
        summ = a.v9_summary()
        mc   = summ["meta_cognition"]
        se   = summ["self_evaluation"]
        unc  = summ["uncertainty"]
        strat= summ["strategy"]
        print(f"\n  {a.agent_id} ({a.role.value}):")
        print(f"    Beliefs: {summ['beliefs']}, Grade: {summ['stability_grade']}")
        print(f"    Strategy: {strat['current_strategy']} "
              f"(switches={strat['switches']})")
        print(f"    Uncertainty: {unc['level']} "
              f"(corrupted={unc['n_corrupted']}/{unc['n_total']})")
        print(f"    Calibration: {mc['calibration_score']:.3f}, "
              f"Overconfident: {mc['overconfident']}")
        if se.get("n_reports", 0) > 0:
            print(f"    Self-eval: grade={se['latest_grade']}, "
                  f"improvement={se['improvement']:+.3f}")
            if se.get("weaknesses"):
                print(f"    Weaknesses: {se['weaknesses']}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   CogniField v9 — Adaptive Self-Reflective Demo         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    demo_meta_cognition()
    demo_uncertainty_engine()
    demo_goal_conflict_resolver()
    demo_strategy_manager()
    demo_temporal_memory()
    demo_self_evaluator()
    demo_full_v9()
    print("\n" + "═"*64 + "\n  v9 Demo complete.\n" + "═"*64)
