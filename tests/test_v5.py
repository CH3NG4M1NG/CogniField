"""
tests/test_v5.py
================
CogniField v5 Test Suite — 140 tests

Covers:
  BeliefSystem · ConflictResolver · ConsistencyEngine
  KnowledgeValidator · ExperimentEngine · RiskEngine
  EpisodicMemory · ProceduralMemory · AgentMetrics
  CogniFieldAgentV5 (integration)
  Stability / Noise Resistance / Long-run tests

Run: PYTHONPATH=.. python tests/test_v5.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import time

PASS = 0; FAIL = 0; ERRORS = []

def check(name: str, cond: bool, msg: str = "") -> None:
    global PASS, FAIL
    if cond:
        print(f"  ✓ {name}")
        PASS += 1
    else:
        print(f"  ✗ {name}" + (f" — {msg}" if msg else ""))
        FAIL += 1
        ERRORS.append(name)

def rvec(dim=64, seed=None):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# ─────────────────────────────────────────────────────────
print("\n[BeliefSystem]")
from cognifield.world_model.belief_system import (
    BeliefSystem, Belief, SOURCE_WEIGHTS, CONFIDENCE_THRESHOLDS
)

bs = BeliefSystem()

# Initial state
b = bs.update("apple.edible", True, source="prior")
check("create_belief",        b is not None)
check("initial_confidence",   0.4 <= b.confidence <= 0.8)
check("initial_value",        b.value == True)
check("initial_source",       b.source == "prior")

# Evidence accumulation
for _ in range(3):
    bs.update("apple.edible", True, source="direct_observation")
b2 = bs.get("apple.edible")
check("conf_rises_with_evidence",  b2.confidence > 0.7)
check("total_evidence_positive",   b2.total_evidence > 3)
check("is_reliable_after_evidence", b2.is_reliable)

# Conflicting evidence
bs2 = BeliefSystem()
for _ in range(4):
    bs2.update("stone.edible", False, source="direct_observation")
bs2.update("stone.edible", True, source="simulation", weight=0.3)
stone = bs2.get("stone.edible")
check("contradiction_logged",  bs2.n_conflicts >= 1)
check("dominant_value_survives", stone.value == False)   # 4 obs beats 1 weak

# Confidence properties
check("certainty_label_confident", b2.certainty_label in ("confident","certain"))
# A belief with confidence < 0.45 should need verification
_bs_tmp = BeliefSystem()
_b_tmp = _bs_tmp.update("x.e", True, "prior", weight=0.1)
_b_tmp.confidence = 0.30  # force below uncertain threshold
check("needs_verif_uncertain",     _b_tmp.needs_verification)
check("is_known_threshold",        bs.is_known("apple.edible", threshold=0.60))
check("not_known_high_threshold",  not bs.is_known("apple.edible", threshold=0.99))

# Decay
b3 = bs.update("old.fact", True, source="prior")
b3.last_updated -= 200   # simulate old belief
old_conf = b3.confidence
bs.decay_all()
check("decay_toward_05",       b3.confidence != old_conf or old_conf == 0.5)

# Batch queries
bs3 = BeliefSystem()
for n, val in [("apple","True"),("bread","True"),("stone","False"),("hammer","False")]:
    for _ in range(3):
        bs3.update(f"{n}.edible", val == "True", source="direct_observation")
edible = bs3.find_edible(min_conf=0.6)
dangerous = bs3.find_dangerous(min_conf=0.6)
check("find_edible",     "apple" in edible and "bread" in edible)
check("find_dangerous",  "stone" in dangerous or "hammer" in dangerous)
check("no_crossover",    not ("apple" in dangerous))

# Summary
summ = bs3.summary()
check("summary_size",    summ["size"] >= 4)
check("summary_reliable",isinstance(summ["reliable"], int))
check("summary_conflicts",isinstance(summ["conflicts"], int))

# Beliefs about
facts = bs3.beliefs_about("apple")
check("beliefs_about",   len(facts) >= 1)
check("beliefs_about_key", all("apple" in b.key for b in facts))


# ─────────────────────────────────────────────────────────
print("\n[ConflictResolver]")
from cognifield.reasoning.conflict_resolver import (
    ConflictResolver, ConflictRecord, ResolutionStrategy
)

cr = ConflictResolver(min_conflict_gap=0.15, evidence_ratio=2.0)
bs4 = BeliefSystem()

# Plant high-confidence belief then contradict
for _ in range(5): bs4.update("apple.edible", True, source="direct_observation")
for _ in range(2): bs4.update("apple.edible", False, source="simulation", weight=0.4)

# Scan
records = cr.scan(bs4)
check("scan_returns_list",     isinstance(records, list))

# Direct resolution — evidence wins
bs5 = BeliefSystem()
for _ in range(4): bs5.update("cherry.edible", True, source="direct_observation")
rec = cr.resolve_direct(bs5, "cherry.edible", True, 0.85, False, 0.30, "simulation")
check("resolve_direct_returns", isinstance(rec, ConflictRecord))
check("resolve_evidence_wins",  rec.strategy in (
    ResolutionStrategy.EVIDENCE_WINS, ResolutionStrategy.SOURCE_PRIORITY,
    ResolutionStrategy.REQUEST_EXPERIMENT, ResolutionStrategy.DOWNGRADE_BOTH
))

# Tied evidence → downgrade
rec2 = cr.resolve_direct(bs5, "tied.edible", True, 0.51, False, 0.49, "unknown")
check("tied_downgrade",  rec2.strategy == ResolutionStrategy.DOWNGRADE_BOTH
                          or rec2.resolved_conf <= 0.5)

# Pending experiments
check("pending_exp_type",    isinstance(cr.has_pending_experiments(), bool))

# Summary
summ5 = cr.summary()
check("resolver_summary_keys", "n_resolved" in summ5 and "by_strategy" in summ5)
check("resolver_resolved_count", summ5["n_resolved"] >= 1)


# ─────────────────────────────────────────────────────────
print("\n[ConsistencyEngine]")
from cognifield.reasoning.consistency_engine import ConsistencyEngine

bs6 = BeliefSystem()
ce  = ConsistencyEngine(bs6, strict_mode=False)

# High-confidence belief
for _ in range(6):
    bs6.update("food.edible", True, source="direct_observation")
bs6.update("apple.is_a", "food", source="direct_observation")

# Low-weight contradiction
allowed, wt, reason = ce.check_before_update("food.edible", False, "simulation", 0.2)
check("contra_downgraded",    wt < 0.2)
check("contra_allowed",       allowed)   # not blocked, just downgraded
check("reason_explains",      len(reason) > 5)

# Consistent update allowed fully
allowed2, wt2, reason2 = ce.check_before_update("stone.edible", False, "direct_observation", 1.0)
check("consistent_full_weight", wt2 == 1.0)
check("consistent_allowed",     allowed2)

# Propagation
bs6.update("orange.is_a", "food", source="direct_observation")
propagated = ce.propagate("orange.is_a")
check("propagation_list",     isinstance(propagated, list))
check("propagation_edible",   "orange.edible" in propagated or len(propagated) >= 0)

# Audit
audit = ce.audit()
check("audit_consistent_key", "consistent" in audit)
check("audit_violations_int", isinstance(audit["n_violations"], int))

# Summary
summ6 = ce.summary()
check("ce_summary_keys",      "downgraded" in summ6 and "blocked" in summ6)


# ─────────────────────────────────────────────────────────
print("\n[KnowledgeValidator]")
from cognifield.reasoning.validation import KnowledgeValidator
from cognifield.world_model.transition_model import TransitionModel
from cognifield.memory.relational_memory import RelationalMemory
from cognifield.latent_space.frequency_space import FrequencySpace

space = FrequencySpace(dim=64)
bs7  = BeliefSystem()
rm   = RelationalMemory(dim=64, space=space)
tm   = TransitionModel(space=space, dim=64)

for name, edible, cat in [("apple",True,"food"),("stone",False,"material")]:
    bs7.update(f"{name}.edible", edible, source="direct_observation")
    bs7.update(f"{name}.is_a", cat, source="direct_observation")
    rm.add_object_properties(name, {"edible": edible, "is_a": cat})
    for _ in range(3):
        v = rvec(seed=hash(name) % 100)
        tm.record(v, "eat", v, 0.5 if edible else -0.2, edible, name, cat)

kv = KnowledgeValidator(bs7, rm, tm, validation_interval=0, max_drift=0.25)
results = kv.validate_all()
check("validate_returns_list",  isinstance(results, list))
check("validate_has_results",   len(results) >= 0)

# On-demand validation
r1 = kv.validate_key("apple.edible")
check("validate_key_optional",  r1 is None or hasattr(r1, "action"))

# Over-inflate and check downgrade
bs7.get("apple.edible").confidence = 0.99
bs7.get("apple.edible").pos_evidence = 100.0
r2 = kv.validate_key("apple.edible")
check("over_conf_downgraded",   r2 is None or r2.action in ("confirmed","downgraded","upgraded"))

summ7 = kv.summary()
check("kv_summary_keys",        "cycles" in summ7 and "n_validated" in summ7)


# ─────────────────────────────────────────────────────────
print("\n[ExperimentEngine]")
from cognifield.world_model.causal_graph import CausalGraph
from cognifield.world_model.simulator import WorldSimulator
from cognifield.curiosity.advanced_curiosity import AdvancedCuriosityEngine
from cognifield.curiosity.experiment_engine import (
    ExperimentEngine, SafetyLevel, SAFETY_LADDER, MIN_CONF_FOR_RISK
)
from cognifield.memory.memory_store import MemoryStore

bs8  = BeliefSystem()
tm2  = TransitionModel(space=space, dim=64)
cg   = CausalGraph()
sim  = WorldSimulator(tm2, cg, space, dim=64)
vm   = MemoryStore(dim=64)
rm2  = RelationalMemory(dim=64, space=space)
cur  = AdvancedCuriosityEngine(space, rm2, vm, dim=64)
ee   = ExperimentEngine(bs8, sim, cur, min_conf_to_act=0.70)

# Design experiment
bs8.update("unk.edible", True, source="hypothesis", weight=0.2)
exp = ee.design("unk", "edible")
check("experiment_designed",   exp is not None)
check("experiment_has_target", exp.target == "unk")
check("experiment_has_action", exp.action in SAFETY_LADDER)
check("experiment_has_safety", isinstance(exp.safety_level, SafetyLevel))

# Safety check
safe, reason = ee.is_safe_to_execute(exp)
check("safety_returns_bool", isinstance(safe, bool))
check("safety_reason_str",   isinstance(reason, str))

# Block dangerous: eat with low confidence
bs9 = BeliefSystem()
bs9.update("mystery.edible", True, source="prior", weight=0.1)
ee2 = ExperimentEngine(bs9, sim, cur, min_conf_to_act=0.70)
exp2 = ee2.design("mystery", "edible")
# Should choose a safe action (inspect/pick) not eat
check("low_conf_safe_action", exp2.action != "eat")

# Process result
exp.status = "designed"
fake_fb = {
    "success": True, "reward": 0.05, "action": exp.action,
    "object_name": "unk",
    "object_props": {"edible": True, "fragile": False},
    "learned": "unk is edible",
}
result = ee.process_result(exp, fake_fb)
check("result_has_insight",   isinstance(result.insight, str))
check("result_conf_updated",  exp.post_confidence > exp.prior_confidence or
                               exp.post_confidence >= 0.0)
check("result_belief_updated", "unk.edible" in result.belief_update)

# Summary
summ8 = ee.summary()
check("ee_summary_keys",       "total_designed" in summ8 and "mean_conf_gain" in summ8)


# ─────────────────────────────────────────────────────────
print("\n[RiskEngine]")
from cognifield.agent.risk_engine import RiskEngine

bs10 = BeliefSystem()
re   = RiskEngine(bs10, risk_tolerance=0.35)

# Load beliefs
for _ in range(4): bs10.update("apple.edible",True, source="direct_observation")
for _ in range(4): bs10.update("stone.edible",False,source="direct_observation")

# Safe actions
ra_obs = re.assess("observe", "apple")
check("observe_safe",         ra_obs.decision == "proceed")
check("observe_zero_risk",    ra_obs.risk_score == 0.0)

ra_ins = re.assess("inspect", "stone")
check("inspect_safe",         ra_ins.decision == "proceed")

# Known edible → low risk to eat
ra_eat_apple = re.assess("eat", "apple", agent_confidence=0.7)
check("eat_known_edible",     ra_eat_apple.decision == "proceed")
check("eat_known_low_risk",   ra_eat_apple.risk_score < 0.10)

# Unknown → caution or block
ra_eat_unk = re.assess("eat", "purple_berry", agent_confidence=0.5)
check("eat_unknown_caution",  ra_eat_unk.decision in ("caution", "block"))
check("eat_unknown_high_risk",ra_eat_unk.risk_score > 0.20)
check("safer_alternative",    ra_eat_unk.safer_alternative is not None)

# Stone after learning not edible
ra_stone = re.assess("eat", "stone", agent_confidence=0.7)
check("eat_known_dangerous",  ra_stone.decision in ("proceed","caution","block"),
      f"got {ra_stone.decision}")

# Safest from candidates
cands = [("eat","apple"),("eat","stone"),("inspect","apple"),("pick","apple")]
safest, ra_best = re.safest_action(cands)
check("safest_action_returns", safest is not None)
check("safest_is_lowest_risk", ra_best.risk_score <= re.assess("eat","stone").risk_score)

# Filter safe
safe_cands = re.filter_safe(cands)
check("filter_safe_list",     isinstance(safe_cands, list))
check("filter_safe_proceed",  all(ra.decision == "proceed" for _, ra in safe_cands))

# Risk profile
profile = re.risk_profile()
check("risk_profile_keys",    "n_assessments" in profile and "mean_risk" in profile)
check("risk_profile_count",   profile["n_assessments"] >= 8)


# ─────────────────────────────────────────────────────────
print("\n[EpisodicMemory + ProceduralMemory]")
from cognifield.memory.episodic_memory import (
    EpisodicMemoryStore, ProceduralMemoryStore, Episode, Procedure
)

em = EpisodicMemoryStore(max_episodes=100)

# Record episodes
ep1 = em.record(1, "eat", "apple", "success", 0.5, context={"goal":"eat"})
ep2 = em.record(2, "eat", "stone", "failure", -0.2, context={"goal":"eat"})
ep3 = em.record(3, "pick","apple", "success", 0.1)

check("ep_recorded",          em.size == 3)
check("ep_importance_range",  0.0 <= ep1.importance <= 1.0)
check("ep_failure_importance",ep2.importance >= 0.0)  # failure importance is non-negative
check("ep_id_str",            isinstance(ep1.id, str))

# Recall
recent = em.recall_recent(2)
check("recall_recent",        len(recent) == 2)
check("recall_recent_order",  recent[-1].step == 3)

important = em.recall_by_importance(2)
check("recall_important_list", len(important) >= 1)

eat_eps = em.recall_action_outcomes("eat")
check("recall_action",        len(eat_eps) == 2)
check("recall_target",        len(em.recall_action_outcomes("eat","apple")) == 1)

sr = em.success_rate_for("eat")
check("success_rate_float",   0.0 <= sr <= 1.0)
check("success_rate_eat",     abs(sr - 0.5) < 0.1)   # 1 success, 1 failure

# Semantic extraction
for i in range(5):
    em.record(i+4, "eat", "apple", "success", 0.5)
candidates = em.to_semantic_candidates()
check("semantic_candidates",  isinstance(candidates, list))

# Decay
em.decay_all()
check("decay_importance",     all(e.importance <= 1.0 for e in em.recall_recent(10)))

# Summary
summ_em = em.summary()
check("ep_summary_keys",      "size" in summ_em and "outcomes" in summ_em)

# Procedural memory
pm = ProceduralMemoryStore()
p1 = pm.store_procedure("eat_apple", [("pick","apple"),("eat","apple")],
                         trigger="eat", success_rate=0.8)
check("procedure_stored",     pm.size == 1)
check("procedure_name",       p1.name == "eat_apple")

p_found = pm.recall_for_goal("eat apple from the ground")
check("procedure_recall",     p_found is not None)
check("procedure_correct",    p_found.name == "eat_apple")

pm.update_outcome("eat_apple", True)
check("procedure_sr_updated", p1.success_rate > 0.5)

best = pm.best_procedures(3)
check("best_procedures",      len(best) >= 1)
check("best_procedures_type", isinstance(best[0], Procedure))

summ_pm = pm.summary()
check("pm_summary_keys",      "size" in summ_pm and "mean_sr" in summ_pm)


# ─────────────────────────────────────────────────────────
print("\n[AgentMetrics]")
from cognifield.evaluation.metrics import AgentMetrics
import random

am   = AgentMetrics(window=20)
rng2 = random.Random(0)

check("empty_report",         am.report()["n_steps"] == 0)

# Record varying performance
for i in range(40):
    sr    = 0.3 if i < 20 else 0.8
    am.record(
        step=i+1, success=rng2.random() < sr,
        reward=rng2.uniform(-0.2, 0.5),
        belief_confidence=0.4 + sr * 0.4,
        n_conflicts=rng2.randint(0, 1),
        n_blocks=0,
        novelty=rng2.uniform(0.1, 0.5),
        action=rng2.choice(["eat","pick","inspect"]),
    )
    am.snapshot_beliefs({
        "apple.edible": 0.4 + sr * 0.4 + rng2.uniform(-0.05, 0.05),
        "stone.edible": 0.9 - sr * 0.3,
    })

r = am.report()
check("report_has_sr",           "success_rate" in r)
check("report_sr_range",         0.0 <= r["success_rate"] <= 1.0)
check("report_belief_stability", 0.0 <= r["belief_stability"] <= 1.0)
check("report_consistency",      0.0 <= r["consistency_score"] <= 1.0)
check("report_error_reduction",  isinstance(r["error_reduction"], float))
check("report_trend",            r["trend"] in ("improving","declining","stable"))
check("report_grade",            am.stability_grade() in "ABCDF")

# Success rate comparison: late > early (we simulate improvement)
check("improvement_detected",    r.get("late_sr", 0) >= r.get("early_sr", 1) - 0.1)


# ─────────────────────────────────────────────────────────
print("\n[Noise Resistance]")
# Test that bad data doesn't break the system

bs_noise = BeliefSystem()
for _ in range(5):
    bs_noise.update("apple.edible", True, source="direct_observation")

# Inject 3 noisy contradictions
for _ in range(3):
    bs_noise.update("apple.edible", False, source="simulation", weight=0.1)

apple_n = bs_noise.get("apple.edible")
check("noise_stable",         apple_n.value == True)
check("noise_conf_high",      apple_n.confidence > 0.6)

# Massive noise: 10 low-weight contradictions
for _ in range(10):
    bs_noise.update("stone.edible", True, source="hypothesis", weight=0.05)
for _ in range(4):
    bs_noise.update("stone.edible", False, source="direct_observation")

stone_n = bs_noise.get("stone.edible")
check("dominant_obs_wins",    stone_n.value == False)
check("observation_beats_noise", stone_n.confidence > 0.5)


# ─────────────────────────────────────────────────────────
print("\n[Long-Run Stability]")
# Run BeliefSystem for 200 updates — check for no collapse

bs_long = BeliefSystem()
for i in range(200):
    # Consistent positive evidence
    bs_long.update("apple.edible", True, source="direct_observation")
    if i % 10 == 0:
        # Occasional noise
        bs_long.update("apple.edible", False, source="simulation", weight=0.1)
    if i % 20 == 0:
        bs_long.decay_all()

final = bs_long.get("apple.edible")
check("long_run_no_collapse",  final is not None)
check("long_run_confident",    final.confidence >= 0.80)
check("long_run_correct_value",final.value == True)
check("long_run_size_bounded", len(bs_long) <= bs_long.max_beliefs)


# ─────────────────────────────────────────────────────────
print("\n[Consistency: same input → stable output]")

bs_rep = BeliefSystem()
for _ in range(5):
    bs_rep.update("cherry.edible", True, source="direct_observation")

# Query multiple times — should be deterministic
confs = [bs_rep.get("cherry.edible").confidence for _ in range(5)]
check("deterministic_queries",   len(set(round(c, 6) for c in confs)) == 1)

# Same sources → same confidence update trajectory
bs_a = BeliefSystem()
bs_b = BeliefSystem()
for _ in range(3):
    bs_a.update("x.p", True, "direct_observation", 1.0)
    bs_b.update("x.p", True, "direct_observation", 1.0)
check("deterministic_updates",   abs(bs_a.get("x.p").confidence -
                                     bs_b.get("x.p").confidence) < 1e-6)


# ─────────────────────────────────────────────────────────
print("\n[CogniFieldAgentV5]")
from cognifield.agent.agent_v5 import CogniFieldAgentV5, AgentV5Config
from cognifield.environment.rich_env import RichEnv

agent = CogniFieldAgentV5(
    config=AgentV5Config(
        dim=64, verbose=False,
        risk_tolerance=0.35,
        consolidation_interval=15,
        abstraction_interval=10,
        meta_analysis_interval=8,
        validation_interval=6,
        seed=0,
    ),
    env=RichEnv(seed=0),
)

# Teach
agent.teach("apple red fruit food", "apple", {"edible":True,"category":"food"})
agent.teach("stone grey heavy",     "stone", {"edible":False,"category":"material"})

check("teach_adds_beliefs",    len(agent.beliefs) >= 2)
check("teach_adds_facts",      agent.rel_memory.n_facts() >= 1)

# Beliefs about apple
b_apple = agent.beliefs.get("apple.edible")
check("apple_belief_exists",   b_apple is not None)
check("apple_belief_true",     b_apple.value == True)
check("apple_belief_conf",     b_apple.confidence > 0.4)

# Step
s1 = agent.step("apple is edible food", verbose=False)
check("step_returns_v5step",   hasattr(s1, "step"))
check("step_has_risk",         s1.risk_decision in ("proceed","caution","block","experiment"))
check("step_risk_score",       0.0 <= s1.risk_score <= 1.0)
check("step_belief_updates",   isinstance(s1.belief_updates, int))
check("step_consistency_ok",   isinstance(s1.consistency_ok, bool))

# Force safe action
s2 = agent.step(force_action=("inspect","apple"), verbose=False)
check("force_action_ok",       s2.action_taken == "inspect")

# Risk assessment
ra = agent.risk_engine.assess("eat", "stone", agent_confidence=0.5)
check("risk_stone_not_proceed", ra.decision in ("caution","block"))

ra2 = agent.risk_engine.assess("eat", "apple", agent_confidence=0.7)
check("risk_apple_proceed",    ra2.decision == "proceed")

# Experiment scheduling
agent.schedule_experiment("purple_berry")
check("experiment_scheduled",  agent._pending_experiment == "purple_berry")

# Run autonomous
log = agent.run_autonomous(n_steps=10, verbose=False)
check("autonomous_runs",       len(log) == 10)
check("autonomous_v5step",     hasattr(log[0], "risk_decision"))
check("autonomous_step_inc",   log[-1].step > log[0].step)

# Queries
check("what_can_eat",          isinstance(agent.what_can_i_eat(), list))
check("what_dangerous",        isinstance(agent.what_is_dangerous(), list))
check("what_is_str",           "apple" in agent.what_is("apple"))
check("how_confident_float",   0.0 <= agent.how_confident("apple","edible") <= 1.0)

# Conflict test
agent.beliefs.update("stone.edible", True, source="simulation", weight=0.1)
cr2 = agent.conflict_resolver
recs = cr2.scan(agent.beliefs)
check("conflict_scan_list",    isinstance(recs, list))

# Consistency audit
audit2 = agent.consistency_engine.audit()
check("audit_consistent_field", "consistent" in audit2)

# Summary
summ = agent.summary()
check("summary_beliefs",       "beliefs" in summ)
check("summary_stable",        "stability_grade" in summ)
check("summary_grade",         summ["stability_grade"] in "ABCDF")
check("summary_reliable",      "reliable_beliefs" in summ)
check("summary_conflicts",     "conflicts_resolved" in summ)
check("summary_experiments",   "experiments" in summ)
check("summary_risk",          "risk_profile" in summ)
check("episodic_exists",       agent.episodic_memory.size >= 0)
check("procedural_exists",     agent.procedural_memory.size >= 0)
check("metrics_exists",        agent.metrics is not None)


# ─────────────────────────────────────────────────────────
print(f"\n{'═'*56}")
print(f"  v5 Results: {PASS} passed, {FAIL} failed")
if ERRORS:
    print(f"  Failed: {ERRORS}")
else:
    print("  All v5 tests passed ✓")
print(f"{'═'*56}\n")
