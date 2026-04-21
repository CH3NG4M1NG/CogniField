"""
tests/test_v9.py
================
CogniField v9 Test Suite — 122 tests

Modules:
  MetaCognitionEngine · UncertaintyEngine · GoalConflictResolver
  StrategyManager · TemporalMemory · SelfEvaluator · AgentV9

Key checks:
  - different behaviours under same scenario
  - confidence fluctuates realistically under uncertainty
  - agents change strategy after repeated failure
  - goal conflicts are visible and resolved
  - self-reflection generates actionable findings

Run: PYTHONPATH=.. python tests/test_v9.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import random

PASS = 0; FAIL = 0; ERRORS = []

def check(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        print(f"  ✓ {name}"); PASS += 1
    else:
        print(f"  ✗ {name}" + (f" — {msg}" if msg else ""))
        FAIL += 1; ERRORS.append(name)


# ─────────────────────────────────────────────────────────
print("\n[MetaCognitionEngine]")
from cognifield.core.meta_cognition import MetaCognitionEngine

mc = MetaCognitionEngine(overconf_threshold=0.15, reflection_interval=5)
rng = random.Random(42)

check("mc_init",                  mc._adjustments == 0)

# Record outcomes
for step in range(1, 21):
    conf    = 0.85
    correct = rng.random() < 0.35   # overconfident: claims 0.85, right only 35%
    mc.record_step(step, correct, 0.5 if correct else -0.2, conf, "eat", "edible")
    mc.record_outcome(conf, True, correct, "edible", step)

over, mc_conf, asr = mc.detect_overconfidence(window=15)
check("overconfidence_detected",  over, f"gap={mc_conf - asr:.3f}")
check("mean_conf_range",          0.0 <= mc_conf <= 1.0)
check("actual_sr_range",          0.0 <= asr <= 1.0)
check("gap_positive",             mc_conf > asr)

# Bias detection
biases = mc.detect_biases()
check("bias_dict",                isinstance(biases, dict))
check("bias_detected_edible",     "edible" in biases, f"biases={biases}")
check("bias_rate_range",          all(0 <= v <= 1 for v in biases.values()))

# Reflection
log = []
def adj_fn(issue, mag): log.append((issue, mag))

reflections = mc.reflect(20, adjust_fn=adj_fn)
check("reflect_list",             isinstance(reflections, list))
check("reflect_has_entries",      len(reflections) >= 1)
check("reflect_finding_str",      all(isinstance(r.finding, str) for r in reflections))
check("adjustment_callback",      len(log) >= 1)
check("adjustments_counted",      mc._adjustments >= 1)

# Calibration
cal = mc.calibration_score()
check("calibration_score_range",  0.0 <= cal <= 1.0)

trend = mc.performance_trend()
check("trend_valid",              trend in ("improving","declining","stable","unknown"))

summ = mc.summary()
check("mc_summary_keys",          all(k in summ for k in
                                       ["overconfident","calibration_score","biases"]))
check("mc_repr",                  "MetaCognition" in repr(mc))

# Different behaviour: low-conf agent should NOT be flagged overconfident
mc2 = MetaCognitionEngine(overconf_threshold=0.15)
for i in range(15):
    mc2.record_step(i+1, True, 0.5, 0.60)  # accurate
over2, _, _ = mc2.detect_overconfidence()
check("accurate_not_overconfident", not over2)


# ─────────────────────────────────────────────────────────
print("\n[UncertaintyEngine]")
from cognifield.core.uncertainty_engine import (
    UncertaintyEngine, UncertaintyLevel, NoisyObservation
)
from cognifield.world_model.belief_system import BeliefSystem

ue_none   = UncertaintyEngine(UncertaintyLevel.NONE,   seed=0)
ue_medium = UncertaintyEngine(UncertaintyLevel.MEDIUM, seed=0)
ue_chaotic= UncertaintyEngine(UncertaintyLevel.CHAOTIC,seed=0)

# Noise injection
obs_none   = ue_none.corrupt(True, 0.85)
obs_chaotic= ue_chaotic.corrupt(True, 0.85)
check("no_noise_clean",           not obs_none.was_corrupted)
check("none_value_unchanged",     obs_none.observed_value == True)
check("noisy_obs_type",           isinstance(obs_chaotic, NoisyObservation))

# With many trials, chaotic corrupts more than none
ue_n_test = UncertaintyEngine(UncertaintyLevel.NONE,   seed=42)
ue_c_test = UncertaintyEngine(UncertaintyLevel.CHAOTIC,seed=42)
n_corrupt_none    = sum(1 for _ in range(50) if ue_n_test.corrupt(True,0.85).was_corrupted)
n_corrupt_chaotic = sum(1 for _ in range(50) if ue_c_test.corrupt(True,0.85).was_corrupted)
check("chaotic_more_corrupt",     n_corrupt_chaotic > n_corrupt_none)

# Confidence decay
bs_test = BeliefSystem()
for _ in range(5): bs_test.update("a.e", True, "direct_observation")
conf_before = bs_test.get("a.e").confidence
UncertaintyEngine(UncertaintyLevel.HIGH, seed=0).decay_all_beliefs(bs_test, steps=5)
conf_after = bs_test.get("a.e").confidence
check("decay_reduces_confidence", conf_after < conf_before)
check("decay_not_below_floor",    conf_after >= 0.10)

# Different levels decay at different rates
bs_low  = BeliefSystem()
bs_high = BeliefSystem()
for bs in [bs_low, bs_high]:
    for _ in range(5): bs.update("x.y", True, "direct_observation")
UncertaintyEngine(UncertaintyLevel.LOW,  seed=0).decay_all_beliefs(bs_low,  steps=5)
UncertaintyEngine(UncertaintyLevel.HIGH, seed=0).decay_all_beliefs(bs_high, steps=5)
check("high_decays_faster",       bs_high.get("x.y").confidence < bs_low.get("x.y").confidence)

# Partial observability
ue_po = UncertaintyEngine(UncertaintyLevel.MEDIUM, seed=0)
ue_po.hide_property("heavy")
check("observable_true",          ue_po.is_observable("edible"))
check("observable_false",         not ue_po.is_observable("heavy"))
obs_hidden = ue_po.corrupt(True, 0.9, "heavy")
check("hidden_obs_none",          obs_hidden.observed_value is None)
check("hidden_weight_zero",       obs_hidden.confidence_weight == 0.0)
ue_po.reveal_property("heavy")
check("revealed_observable",      ue_po.is_observable("heavy"))

# Vector noise
vec = np.ones(8, dtype=np.float32) / np.sqrt(8)
noisy = ue_medium.add_vector_noise(vec)
check("vector_noise_shape",       noisy.shape == vec.shape)
check("vector_noise_changed",     not np.allclose(noisy, vec))

# Auto-detection
ue_auto = UncertaintyEngine(UncertaintyLevel.LOW, seed=0)
for _ in range(20): ue_auto.record_outcome_variance(float(np.random.choice([-0.5,0.8])))
detected = ue_auto.auto_detect_level()
check("auto_detect_not_none",     detected is not None)
check("auto_detect_changed",      detected != UncertaintyLevel.LOW)

# Consensus slowdown
sm_none   = ue_none.consensus_supermajority(0.55)
sm_chaotic= ue_chaotic.consensus_supermajority(0.55)
check("chaotic_higher_sm",        sm_chaotic > sm_none)
check("sm_range",                 0.55 <= sm_chaotic <= 0.95)

summ_ue = ue_medium.summary()
check("ue_summary_level",         "level" in summ_ue)
check("ue_repr",                  "UncertaintyEngine" in repr(ue_medium))


# ─────────────────────────────────────────────────────────
print("\n[GoalConflictResolver]")
from cognifield.agents.goal_conflict_resolver import (
    GoalConflictResolver, ConflictType, ResolutionStrategy
)
from cognifield.agents.goals import GoalSystem, GoalType

gs = GoalSystem(max_active=8)
g_eat_a = gs.add_eat_goal("apple",  priority=0.80)
g_eat_b = gs.add_eat_goal("bread",  priority=0.70)
g_avoid = gs.add_avoid_goal("stone",priority=0.95)
g_expl  = gs.add_explore_goal("explore", priority=0.50)
all_goals = list(gs._goals)

gcr = GoalConflictResolver(strategy=ResolutionStrategy.UTILITY_MAXIMISATION)

# Conflict detection
conflicts = gcr.detect_conflicts(all_goals)
check("conflicts_list",           isinstance(conflicts, list))
check("conflicts_detected",       len(conflicts) >= 1)

# Check conflict types present
ctypes = {c.conflict_type for c in conflicts}
check("resource_conflict",        ConflictType.RESOURCE in ctypes,
      f"types found: {ctypes}")

# Resolution strategies
for strat in [ResolutionStrategy.PRIORITY_ORDER,
              ResolutionStrategy.UTILITY_MAXIMISATION,
              ResolutionStrategy.SATISFICING]:
    gcr_s = GoalConflictResolver(strategy=strat, sat_threshold=0.65)
    decision = gcr_s.resolve(all_goals)
    check(f"resolve_{strat.value}_chosen",  len(decision.chosen_goals) >= 1)
    check(f"resolve_{strat.value}_utility", 0.0 <= decision.utility_score <= 1.0)
    check(f"resolve_{strat.value}_strat",   decision.strategy_used == strat)

# No-conflict path
single_goal = [g_eat_a]
gcr_nc = GoalConflictResolver()
dec_nc = gcr_nc.resolve(single_goal)
check("no_conflict_all_chosen",   len(dec_nc.chosen_goals) == 1)
check("no_conflict_zero_dropped", len(dec_nc.dropped_goals) == 0)

# Summary
summ_gcr = gcr.summary()
check("gcr_summary_keys",         "total_conflicts" in summ_gcr and "by_type" in summ_gcr)
check("gcr_repr",                 "GoalConflictResolver" in repr(gcr))


# ─────────────────────────────────────────────────────────
print("\n[StrategyManager]")
from cognifield.agents.strategy_manager import (
    StrategyManager, Strategy, STRATEGY_CONFIGS
)

sm = StrategyManager(
    initial_strategy=Strategy.EXPLORE,
    eval_freq=6, fail_threshold=0.25,
    win_threshold=0.70, max_consec_fails=4,
)
rng2 = random.Random(7)

check("sm_initial",               sm.current == Strategy.EXPLORE)
check("sm_n_configs",             len(STRATEGY_CONFIGS) == len(Strategy))

# Record struggling performance
step = 0
for i in range(24):
    step += 1
    success = rng2.random() < 0.20   # very low sr
    sm.record_step(step, success, -0.2, 0.3)
    sm.evaluate(step)

# After enough failures, should have switched
check("strategy_switched",        sm.switches() >= 1, f"switches={sm.switches()}")
check("switched_to_recover",      sm.current in [Strategy.RECOVER, Strategy.VERIFY,
                                                   Strategy.COOPERATIVE])

# Record good performance → should switch to EXPLOIT
sm2 = StrategyManager(initial_strategy=Strategy.EXPLORE,
                       eval_freq=6, win_threshold=0.70)
for i in range(1, 19):
    sm2.record_step(i, rng2.random() < 0.85, 0.5, 0.05)  # high sr, low novelty
    sm2.evaluate(i)
check("exploit_switch",           sm2.current == Strategy.EXPLOIT or sm2.switches() >= 1)

# Consecutive failures threshold
sm3 = StrategyManager(max_consec_fails=3, eval_freq=4, fail_threshold=0.10)
for i in range(1, 5):
    sm3.record_step(i, False, -0.3, 0.2)   # 4 consecutive failures
    sm3.evaluate(i)
check("consec_fail_recover",      sm3.current == Strategy.RECOVER)

# Config access
cfg = sm.get_config()
check("config_has_novt",          hasattr(cfg, "novelty_threshold"))
check("config_has_risk",          hasattr(cfg, "risk_tolerance"))

summ_sm = sm.summary()
check("sm_summary_current",       "current_strategy" in summ_sm)
check("sm_summary_switches",      "switches" in summ_sm)
check("sm_repr",                  "StrategyManager" in repr(sm))

# Apply to mock agent
class MockCfg:
    novelty_threshold = 0.4
    risk_tolerance    = 0.35

class MockAgent:
    cfg = MockCfg()
    class risk_engine:
        risk_tolerance = 0.35

ma = MockAgent()
sm.apply_to_agent(ma)
check("apply_changes_threshold",  ma.cfg.novelty_threshold != 0.4
                                   or ma.cfg.risk_tolerance != 0.35)


# ─────────────────────────────────────────────────────────
print("\n[TemporalMemory]")
from cognifield.memory.temporal_memory import TemporalMemory, TemporalPattern

tm = TemporalMemory(window=10)
rng3 = random.Random(13)

# Record outcomes
for step in range(1, 25):
    tm.record_outcome("eat", "apple",        True,              0.5, step)
    tm.record_outcome("eat", "stone",        False,            -0.2, step)
    tm.record_outcome("eat", "purple_berry", rng3.random()<0.5, 0.3, step,
                      context={"strategy":"explore"})

# Pattern detection
p_apple = tm.detect_pattern("eat", "apple")
check("pattern_apple_exists",    p_apple is not None)
check("pattern_apple_stable",    p_apple.pattern_type == "stable")
check("pattern_apple_sr_high",   p_apple.success_rate >= 0.95)

p_stone = tm.detect_pattern("eat", "stone")
check("pattern_stone_declining", p_stone.pattern_type == "declining")
check("pattern_stone_sr_zero",   p_stone.success_rate == 0.0)

p_pb = tm.detect_pattern("eat", "purple_berry")
check("pattern_pb_exists",       p_pb is not None)
check("pattern_pb_type_valid",   p_pb.pattern_type in ["unstable","improving","declining","stable"])

all_patterns = tm.detect_all_patterns()
check("detect_all_list",         isinstance(all_patterns, list))
check("detect_all_count",        len(all_patterns) >= 3)

# Recurrence detection
tm2 = TemporalMemory()
for _ in range(4):
    tm2.record_outcome("eat","unknown",False,-0.3,1)
check("stuck_detected",          tm2.is_stuck("eat","unknown",threshold=3))
check("not_stuck_apple",         not tm2.is_stuck("eat","apple",threshold=3))

# Belief drift
for step in range(1, 15):
    tm.record_belief_snapshot("apple.edible", min(0.95, 0.5 + step*0.03), step)
    tm.record_belief_snapshot("stone.edible", max(0.05, 0.9 - step*0.04), step)

drift_apple = tm.belief_drift("apple.edible")
drift_stone = tm.belief_drift("stone.edible")
check("drift_apple_rising",      drift_apple in ["rising", "stable"])
check("drift_stone_falling",     drift_stone in ["falling", "stable"])
check("drift_unknown",           tm.belief_drift("nonexistent") == "unknown")

# Success rates
sr_apple = tm.success_rate_for("eat","apple")
sr_stone  = tm.success_rate_for("eat","stone")
check("sr_apple_high",           sr_apple > 0.8)
check("sr_stone_zero",           sr_stone == 0.0)
check("sr_unknown_50",           tm.success_rate_for("eat","ghost") == 0.5)

summ_tm = tm.summary()
check("tm_summary_keys",         "tracked_keys" in summ_tm and "n_patterns" in summ_tm)
check("tm_repr",                 "TemporalMemory" in repr(tm))


# ─────────────────────────────────────────────────────────
print("\n[SelfEvaluator]")
from cognifield.agents.self_evaluator import (
    SelfEvaluator, EvalReport, DIMENSIONS
)
from cognifield.agents.agent_v9 import CogniFieldAgentV9, AgentV9Config
from cognifield.environment.rich_env import RichEnv

se = SelfEvaluator(eval_freq=1, weakness_threshold=0.50)

# Create minimal agent
agent_se = CogniFieldAgentV9(
    config=AgentV9Config(agent_id="se_test", dim=64, verbose=False, seed=0),
    env=RichEnv(seed=0),
)
agent_se.teach("apple food edible","apple",{"edible":True,"category":"food"})

# Run steps to generate data
for _ in range(10): agent_se.step(verbose=False)

report = se.evaluate(step=10, agent=agent_se)
check("eval_returns_report",      isinstance(report, EvalReport))
check("eval_has_scores",          isinstance(report.scores, dict))
check("eval_all_dimensions",      all(d in report.scores for d in DIMENSIONS))
check("eval_scores_range",        all(0.0 <= s <= 1.0 for s in report.scores.values()))
check("eval_overall_range",       0.0 <= report.overall <= 1.0)
check("eval_grade_valid",         report.grade in "ABCDF")
check("eval_weaknesses_list",     isinstance(report.weaknesses, list))
check("eval_suggestions_dict",    isinstance(report.suggestions, dict))

# Suggestions match weaknesses
check("suggestions_match_weak",   all(w in report.suggestions for w in report.weaknesses))

# is_excellent and needs_improvement
check("excellent_check",          isinstance(report.is_excellent(), bool))
check("needs_improvement_check",  isinstance(report.needs_improvement(), bool))
check("worst_dimension",          report.worst_dimension() in DIMENSIONS)

# Multiple evaluations
for _ in range(5): agent_se.step(verbose=False)
report2 = se.evaluate(step=15, agent=agent_se)
check("second_eval_works",        report2 is not None)

# Improvement tracking
imp = se.improvement_over_time()
check("improvement_float",        isinstance(imp, float))

latest = se.latest_report()
check("latest_report",            latest is not None)
check("latest_is_most_recent",    latest.step == 15)

summ_se = se.summary()
check("se_summary_n_reports",     summ_se["n_reports"] >= 2)
check("se_repr",                  "SelfEvaluator" in repr(se))


# ─────────────────────────────────────────────────────────
print("\n[AgentV9 Integration]")
from cognifield.agents.agent_v9 import CogniFieldAgentV9, AgentV9Config, V9Step
from cognifield.agents.agent_v7 import AgentRole
from cognifield.agents.strategy_manager import Strategy
from cognifield.core.uncertainty_engine import UncertaintyLevel

agent_v9 = CogniFieldAgentV9(
    config=AgentV9Config(
        agent_id="test_v9", role=AgentRole.EXPLORER,
        dim=64, verbose=False, seed=42,
        uncertainty_level="medium",
    ),
    env=RichEnv(seed=42),
)

check("v9_has_metacog",           agent_v9.meta_cog is not None)
check("v9_has_uncertainty",       agent_v9.uncertainty is not None)
check("v9_has_goal_resolver",     agent_v9.goal_resolver is not None)
check("v9_has_strategy_mgr",      agent_v9.strategy_mgr is not None)
check("v9_has_temporal_mem",      agent_v9.temporal_mem is not None)
check("v9_has_self_eval",         agent_v9.self_eval is not None)

# Initial state
check("v9_initial_strategy",      agent_v9.strategy_mgr.current == Strategy.EXPLORE)
check("v9_uncertainty_medium",    agent_v9.uncertainty.level == UncertaintyLevel.MEDIUM)

# Step
agent_v9.teach("apple food edible","apple",{"edible":True,"category":"food"})
s = agent_v9.step(verbose=False)
check("v9_step_type",             isinstance(s, V9Step))
check("v9_step_uncertainty_lvl",  isinstance(s.uncertainty_level, str))
check("v9_step_strategy",         isinstance(s.strategy, str))
check("v9_step_switched",         isinstance(s.strategy_switched, bool))
check("v9_step_goal_conflicts",   s.goal_conflicts >= 0)
check("v9_step_reflection",       s.reflection_items >= 0)
check("v9_step_decay",            s.decay_applied >= 0)

# Force strategy
agent_v9.force_strategy(Strategy.RECOVER)
check("force_strategy_works",     agent_v9.strategy_mgr.current == Strategy.RECOVER)

# Set uncertainty
agent_v9.set_uncertainty_level(UncertaintyLevel.HIGH)
check("set_uncertainty_works",    agent_v9.uncertainty.level == UncertaintyLevel.HIGH)

# Reflection log
log = agent_v9.get_reflection_log(3)
check("reflection_log_list",      isinstance(log, list))

# Is stuck
check("is_stuck_false",           not agent_v9.is_stuck("eat","apple"))

# Run multiple steps to build history
for _ in range(15):
    agent_v9.step(verbose=False)
check("steps_accumulated",        agent_v9._step_count >= 16)

# v9_summary
summ_v9 = agent_v9.v9_summary()
check("v9_summary_metacog",       "meta_cognition" in summ_v9)
check("v9_summary_uncertainty",   "uncertainty" in summ_v9)
check("v9_summary_goal_conf",     "goal_conflicts" in summ_v9)
check("v9_summary_strategy",      "strategy" in summ_v9)
check("v9_summary_temporal",      "temporal_memory" in summ_v9)
check("v9_summary_selfeval",      "self_evaluation" in summ_v9)

check("v9_repr",                  "AgentV9" in repr(agent_v9))

# Behaviour differs under different uncertainty
agent_low  = CogniFieldAgentV9(
    config=AgentV9Config(agent_id="low",  uncertainty_level="none",   dim=64, seed=1),
    env=RichEnv(seed=1)
)
agent_high = CogniFieldAgentV9(
    config=AgentV9Config(agent_id="high", uncertainty_level="chaotic",dim=64, seed=1),
    env=RichEnv(seed=1)
)
for a in [agent_low, agent_high]:
    a.teach("apple food","apple",{"edible":True,"category":"food"})
    for _ in range(5): a.step(verbose=False)

# High uncertainty should have more noise corruptions
low_corrupt  = agent_low.uncertainty._n_corrupted
high_corrupt = agent_high.uncertainty._n_corrupted
check("high_unc_more_corruptions", high_corrupt >= low_corrupt)


# ─────────────────────────────────────────────────────────
print(f"\n{'═'*58}")
print(f"  v9 Results: {PASS} passed, {FAIL} failed")
if ERRORS:
    print(f"  Failed: {ERRORS}")
else:
    print("  All v9 tests passed ✓")
print(f"{'═'*58}\n")
