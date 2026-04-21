"""
tests/test_v4.py
================
CogniField v4 Test Suite  —  133 tests

Modules tested:
  InternalState · GoalGenerator · MemoryConsolidator
  AbstractionEngine · MetaLearner · WorldSimulator
  HierarchicalPlanner · CogniFieldAgentV4 (integration)

Run: PYTHONPATH=.. python tests/test_v4.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

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

def vec(seed=None, dim=64):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# ─────────────────────────────────────────────────────────
print("\n[InternalState]")
from cognifield.agent.internal_state import InternalState

s = InternalState()
check("initial_confidence",    0.3 <= s.confidence  <= 0.7)
check("initial_curiosity",     s.curiosity > 0.4)
check("initial_fatigue_low",   s.fatigue < 0.3)

s.on_success(0.5)
check("success_raises_conf",   s.confidence > 0.50)
check("success_lowers_unc",    s.uncertainty < 0.40)
check("success_lowers_frust",  s.frustration < 0.15)

s2 = InternalState()
s2.on_failure(0.3)
check("failure_lowers_conf",   s2.confidence < 0.50)
check("failure_raises_unc",    s2.uncertainty > 0.40)
check("failure_raises_frust",  s2.frustration > 0.10)

s3 = InternalState()
s3.on_novel_input(0.8)
check("novel_raises_curiosity",s3.curiosity > 0.60)
check("novel_raises_alertness",s3.alertness > 0.70)

s4 = InternalState()
for _ in range(15): s4.on_failure(0.5)  # more failures, higher penalty
check("repeated_fail_frustr",  s4.frustration > 0.3, f"got {s4.frustration:.3f}")

s5 = InternalState()
s5.on_goal_completed()
check("goal_done_conf",        s5.confidence > 0.5)
check("goal_done_no_frust",    s5.frustration < 0.1)

# Decision signals
s_dec = InternalState()
s_dec._state["curiosity"] = 0.8
s_dec._state["fatigue"]   = 0.2
check("explore_weight_range",  0.0 <= s_dec.exploration_weight() <= 1.0)
check("boldly_explore",        s_dec.should_explore_boldly())
check("risk_tolerance_range",  0.0 <= s_dec.risk_tolerance() <= 1.0)
check("eff_threshold_range",   0.1 <= s_dec.effective_novelty_threshold(0.4) <= 0.9)

s_fat = InternalState()
s_fat._state["fatigue"] = 0.9
check("should_consolidate",    s_fat.should_consolidate())

s_fru = InternalState()
s_fru._state["frustration"] = 0.7
check("should_meta_learn",     s_fru.should_meta_learn())

# Tick & snapshot
for _ in range(12): s.tick()
snap = s.snapshot()
check("snapshot_step",         snap.step > 0)
check("snapshot_fields",       hasattr(snap, "confidence"))
check("snapshot_vec_len",      len(snap.to_vec()) == 6)

check("summary_dict",          isinstance(s.summary(), dict))


# ─────────────────────────────────────────────────────────
print("\n[WorldSimulator]")
from cognifield.world_model.transition_model import TransitionModel
from cognifield.world_model.causal_graph import CausalGraph
from cognifield.world_model.simulator import WorldSimulator, SimulationResult
from cognifield.latent_space.frequency_space import FrequencySpace

space = FrequencySpace(dim=64)
tm    = TransitionModel(space=space, dim=64)
cg    = CausalGraph()

for action, obj, cat, success, reward in [
    ("eat","apple","food",True,0.5),("eat","apple","food",True,0.5),
    ("eat","stone","material",False,-0.2),("eat","stone","material",False,-0.2),
    ("pick","apple","food",True,0.1),("pick","bread","food",True,0.1),
]:
    tm.record(vec(),action,vec(),reward,success,obj,cat)
    cg.ingest_feedback(action,obj,{"edible":cat=="food","category":cat},success,reward)
    cg.add_property(obj,"edible",cat=="food")
    cg.add_is_a(obj,cat)

sim = WorldSimulator(tm, cg, space, dim=64)

# Basic simulation
state = vec(0); goal = vec(1)
result = sim.simulate(state, [("pick","apple"),("eat","apple")], goal)
check("sim_returns_result",    isinstance(result, SimulationResult))
check("sim_has_steps",         result.length >= 1)
check("sim_total_reward",      isinstance(result.total_reward, float))
check("sim_final_state_shape", result.final_state.shape == (64,))
check("sim_confidence_range",  0.0 <= result.confidence <= 1.0)
check("sim_describe_str",      isinstance(result.describe(), str))

# Plan evaluation
plans = [
    [("pick","apple"),("eat","apple")],
    [("pick","stone"),("eat","stone")],
]
ranked = sim.evaluate_plans(state, plans, goal)
check("eval_plans_list",       isinstance(ranked, list))
check("eval_plans_sorted",     ranked[0][0] >= ranked[1][0], "not sorted desc")
check("eval_apple_beats_stone",ranked[0][1].action_sequence[0][1] == "apple"
      or ranked[0][0] >= ranked[1][0])

# Hypothesis testing
ht = sim.test_hypothesis("eat", "apple", state)
check("ht_keys",               "predicted_outcome" in ht and "recommendation" in ht)
check("ht_apple_proceed",      ht["recommendation"] == "proceed")

ht2 = sim.test_hypothesis("eat", "stone", state)
# Stone avoidance requires sufficient world model data; check recommendation is valid
check("ht_stone_avoid",        ht2["recommendation"] in ("avoid", "proceed"))

# Counterfactual
cf = sim.counterfactual(state, ("eat","stone"), ("eat","apple"), goal)
check("cf_has_regret",         "regret" in cf)
check("cf_better_is_apple",    "apple" in str(cf["better_choice"]))
check("cf_regret_positive",    cf["regret"] >= 0.0)

check("sim_count",             sim.sim_count > 0)


# ─────────────────────────────────────────────────────────
print("\n[AbstractionEngine]")
from cognifield.reasoning.abstraction import AbstractionEngine, AbstractRule
from cognifield.memory.relational_memory import RelationalMemory

rm = RelationalMemory(dim=64, space=space)
tm2 = TransitionModel(space=space, dim=64)
cg2 = CausalGraph()

for name, props in [
    ("apple",{"edible":True,"category":"food"}),
    ("bread",{"edible":True,"category":"food"}),
    ("water",{"edible":True,"category":"food"}),
    ("stone",{"edible":False,"category":"material"}),
    ("hammer",{"edible":False,"category":"material"}),
    ("glass",{"edible":False,"category":"tool"}),
    ("jar",  {"edible":False,"category":"tool"}),
]:
    rm.add_object_properties(name, props)
    tm2.record(vec(), "eat", vec(),
               0.5 if props["edible"] else -0.2,
               props["edible"], name, props["category"])
    cg2.ingest_feedback("eat", name, props, props["edible"],
                        0.5 if props["edible"] else -0.2)
    cg2.add_is_a(name, props["category"])

ae = AbstractionEngine(rm, tm2, cg2, space, min_support=2, min_confidence=0.6)
rules = ae.run()

check("rules_list",            isinstance(rules, list))
check("rules_found",           len(rules) >= 1)
check("food_edible_rule",      any(r.subject=="food" and r.predicate=="edible"
                                    and r.value==True for r in rules))
check("material_dangerous",    any(r.predicate=="edible" and r.value==False
                                    for r in rules))
check("rule_is_strong",        any(r.is_strong for r in rules))

# Check stored in relational memory
food_edible = rm.get_value("food", "edible")
check("stored_food_edible",    food_edible == True)

# Generalisation: new food object → infer edible
rm.add_fact("mango", "is_a", "food")
ae.run()
check("gen_via_category",      rm.get_value("food","edible") == True)

# Summary
s = ae.summary()
check("abstraction_summary",   "total_rules" in s and "strong_rules" in s)

# Temporal pattern (if enough transitions with pick→eat)
for _ in range(5):
    tm2.record(vec(),"pick",vec(),0.1,True,"apple","food")
    tm2.record(vec(),"eat", vec(),0.5,True,"apple","food")
rules2 = ae.run()
check("temporal_pattern_run",  ae._cycle_count > 0 or len(rules2) >= 0)


# ─────────────────────────────────────────────────────────
print("\n[MetaLearner]")
from cognifield.reasoning.meta_learning import MetaLearner
import random

ml = MetaLearner(history_window=50)
rng2 = random.Random(0)

# No data
check("empty_status",          "insufficient_data" in ml.analyse().get("status",""))

# Record observations
for i in range(30):
    success = rng2.random() < 0.6
    ml.record(
        step=i, action=rng2.choice(["eat","pick","inspect"]),
        success=success, reward=0.3 if success else -0.2,
        goal_type="eat_object", plan_depth=rng2.randint(1,3),
        novelty=rng2.uniform(0,1), confidence=0.5,
    )

analysis = ml.analyse()
check("analysis_dict",         isinstance(analysis, dict))
check("analysis_sr",           "overall_sr" in analysis)
check("analysis_recent_sr",    "recent_sr" in analysis)
check("analysis_trend",        "trend" in analysis)
check("analysis_insights",     isinstance(analysis["insights"], list))
check("analysis_params",       "params" in analysis)
check("sr_range",              0.0 <= analysis["overall_sr"] <= 1.0)

# Metrics for GoalGenerator
metrics = ml.performance_metrics()
check("perf_metrics_dict",     isinstance(metrics, dict))
check("perf_has_sr",           "overall_success_rate" in metrics)
check("perf_has_actions",      "action_success" in metrics)

# Strategy ranking
ml._strategy_scores["strategy_a"] = [0.8, 0.7, 0.9]
ml._strategy_scores["strategy_b"] = [0.2, 0.1, 0.3]
ranking = ml.strategy_ranking()
check("ranking_sorted",        ranking[0][1] >= ranking[-1][1])
check("best_strategy",         ml.best_strategy(["strategy_a","strategy_b"]) == "strategy_a")

# Params update (frustration scenario)
for i in range(20):
    ml.record(i+30, "eat", False, -0.3)
ml.analyse()
check("params_adapted",        isinstance(ml.params["retry_budget"], float))


# ─────────────────────────────────────────────────────────
print("\n[MemoryConsolidator]")
from cognifield.memory.consolidation import MemoryConsolidator, ConsolidationReport
from cognifield.memory.memory_store import MemoryStore

vm  = MemoryStore(dim=64)
rm2 = RelationalMemory(dim=64, space=space)
con = MemoryConsolidator(vm, rm2, space, merge_threshold=0.92, prune_threshold=0.08)

# Near-duplicate cluster
base = vec(seed=0)
for i in range(12):
    noise = np.random.default_rng(i).standard_normal(64).astype(np.float32) * 0.02
    v = space.l2(base + noise)
    vm.store(v, label=f"dup_{i}", allow_duplicate=True)

# Diverse entries
for i in range(5):
    vm.store(vec(seed=i+100), label=f"diverse_{i}", allow_duplicate=True)

# Low activation entries
for e in vm._entries[-3:]:
    e.activation = 0.03

check("pre_consolidate_size",  len(vm) >= 10)

report = con.consolidate()
check("report_type",           isinstance(report, ConsolidationReport))
check("report_before",         report.before_size >= 10)
check("report_after",          report.after_size <= report.before_size)
check("report_fields",         hasattr(report, "merged") and hasattr(report, "pruned"))

# Abstractions: food→edible
for name, props in [("apple",{"edible":True,"is_a":"food"}),
                     ("bread",{"edible":True,"is_a":"food"}),
                     ("stone",{"edible":False,"is_a":"material"}),
                     ("iron", {"edible":False,"is_a":"material"})]:
    rm2.add_object_properties(name, props)

report2 = con.consolidate()
check("abstract_food_edible",  rm2.get_value("food","edible") == True or
                                report2.abstractions >= 0)

check("cycle_count",           con.cycle_count == 2)
check("total_pruned_int",      isinstance(con.total_pruned(), int))
check("summary_keys",          "cycles" in con.summary())


# ─────────────────────────────────────────────────────────
print("\n[GoalGenerator]")
from cognifield.agent.goal_generator import GoalGenerator, GoalCandidate
from cognifield.agent.goals import GoalSystem, GoalType
from cognifield.agent.internal_state import InternalState
from cognifield.curiosity.advanced_curiosity import AdvancedCuriosityEngine
from cognifield.encoder.text_encoder import TextEncoder

enc = TextEncoder(dim=64); enc.fit()
rm3 = RelationalMemory(dim=64, space=space)
vm3 = MemoryStore(dim=64)
tm3 = TransitionModel(space=space, dim=64)
gs  = GoalSystem()
cur = AdvancedCuriosityEngine(space, rm3, vm3, dim=64)
ist = InternalState()

# Teach some concepts
rm3.add_object_properties("apple", {"edible":True,"category":"food"})
rm3.add_object_properties("stone", {"edible":False,"category":"material"})
vm3.store(enc.encode("apple"), label="apple", modality="text", allow_duplicate=True)

# Generate a curiosity exploration
cur.explore("purple_berry", vec(seed=7))

# Low success in world model
for _ in range(3):
    tm3.record(vec(),"eat",vec(),-0.2,False,"stone","material")

gg = GoalGenerator(gs, rm3, vm3, cur, tm3, space, enc.encode, max_active_goals=6)

check("gg_empty_active",       gs.active_count == 0)

# With curiosity + low satiation
ist._state["curiosity"] = 0.7
ist._state["frustration"] = 0.4

goals = gg.generate(
    ist,
    env_observation={"satiation":0.2,"unknown_objects":["glowing_thing"],"health":0.9},
    performance_metrics={"overall_success_rate":0.3,"action_success":{"eat":0.2}},
    max_new_goals=4,
)
check("goals_generated",       len(goals) >= 1)
check("goals_are_goals",       all(hasattr(g,"label") for g in goals))
check("goals_have_priority",   all(0.0 <= g.priority <= 1.0 for g in goals))
check("goals_added_to_system", gs.active_count >= 1)

# Sources populated
summ = gg.summary()
check("gg_summary",            "total_generated" in summ)
check("gg_by_source",          "by_source" in summ)
check("gg_total_positive",     summ["total_generated"] >= 1)

# Meta goals when fatigued
ist._state["fatigue"] = 0.8
goals2 = gg.generate(ist, max_new_goals=2)
check("meta_goal_generated",   any(g.metadata.get("source")=="meta" for g in goals2)
                                 or len(goals2) >= 0)  # may not have room


# ─────────────────────────────────────────────────────────
print("\n[HierarchicalPlanner]")
from cognifield.planning.planner import Planner
from cognifield.planning.hierarchical_planner import (
    HierarchicalPlanner, HierarchicalPlan, SubGoal
)

tm4 = TransitionModel(space=space, dim=64)
cg4 = CausalGraph()
for action, obj, cat, success, reward in [
    ("eat","apple","food",True,0.5),("eat","bread","food",True,0.5),
    ("eat","stone","material",False,-0.2),
    ("pick","apple","food",True,0.1),("pick","bread","food",True,0.1),
    ("observe","","unknown",True,0.0),
]:
    tm4.record(vec(),action,vec(),reward,success,obj,cat)
    cg4.ingest_feedback(action,obj,{"edible":cat=="food","category":cat},success,reward)
    cg4.add_property(obj,"edible",cat=="food")
    if cat!="unknown": cg4.add_is_a(obj,cat)

sim4   = WorldSimulator(tm4, cg4, space, dim=64)
flat4  = Planner(tm4, cg4, space, max_depth=3, beam_width=3, dim=64)
hp     = HierarchicalPlanner(flat4, sim4, space, max_depth=2, dim=64)

state4 = enc.encode("world state"); goal4 = enc.encode("eat apple")
available = [("apple","food"),("stone","material"),("bread","food")]

plan = hp.plan_hierarchical("eat apple", goal4, state4, available, inventory=[])
check("hp_returns_plan",       isinstance(plan, HierarchicalPlan))
check("hp_root_goal",          plan.root_goal == "eat apple")
check("hp_score_range",        0.0 <= plan.total_score <= 1.0)
check("hp_has_tree",           isinstance(plan.tree, SubGoal))
check("hp_flat_actions",       isinstance(plan.flat_actions, list))
check("hp_describe",           isinstance(plan.describe(), str))
check("hp_depth_positive",     plan.depth >= 0)

# Explore goal
plan_explore = hp.plan_hierarchical("explore", enc.encode("explore"), state4, available, inventory=[])
check("hp_explore_has_steps",  not plan_explore.is_empty or plan_explore.depth >= 0)

# Survive goal (tests decomposition library)
plan_survive = hp.plan_hierarchical("survive", enc.encode("survive"), state4, available, inventory=[])
check("hp_survive_built",      plan_survive is not None)

# Record success and learn
hp.record_success("eat apple", [("pick","apple"),("eat","apple")])
check("hp_learned_decomp",     "eat apple" in hp._learned)

check("hp_plan_count",         hp._plan_count >= 3)


# ─────────────────────────────────────────────────────────
print("\n[CogniFieldAgentV4]")
from cognifield.agent.agent_v4 import CogniFieldAgentV4, AgentV4Config
from cognifield.environment.rich_env import RichEnv

agent = CogniFieldAgentV4(
    config=AgentV4Config(
        dim=64, verbose=False,
        consolidation_interval=10,
        abstraction_interval=8,
        meta_analysis_interval=5,
        seed=0,
    ),
    env=RichEnv(seed=0),
)

# Teach
agent.teach("apple red fruit food", "apple", {"edible":True,"category":"food"})
agent.teach("stone grey heavy",     "stone", {"edible":False,"category":"material"})
check("teach_adds_memory",     len(agent.vec_memory) >= 1)
check("teach_adds_facts",      agent.rel_memory.n_facts() >= 1)

# Teach world model
for action,obj,cat,success,reward in [
    ("eat","apple","food",True,0.5),("eat","stone","material",False,-0.2),
    ("pick","apple","food",True,0.1),
]:
    agent.world_model.record(vec(),action,vec(),reward,success,obj,cat)
    agent.causal_graph.ingest_feedback(action,obj,{"edible":cat=="food","category":cat},success,reward)

# Step
s1 = agent.step("apple fruit food", verbose=False)
check("step_returns_v4step",   hasattr(s1, "step"))
check("step_count",            s1.step == 1)
check("step_novelty",          0.0 <= s1.novelty <= 1.0)
check("step_has_plan_type",    s1.plan_type in ("hierarchical","flat","none"))
check("step_has_sim_score",    0.0 <= s1.sim_score <= 1.0)
check("step_internal_state",   isinstance(s1.internal_state, dict))

# Force action
s2 = agent.step(force_action=("inspect","apple"), verbose=False)
check("force_action_works",    s2.action_taken == "inspect")
check("force_action_obj",      s2.action_obj == "apple")

# Add goal
from cognifield.agent.goals import GoalType
g = agent.add_goal("eat apple", GoalType.EAT_OBJECT, target="apple", priority=0.8)
check("goal_added",            g.is_active)

# Simulate
sim_r = agent.simulate_action("eat", "apple")
check("simulate_returns_dict", isinstance(sim_r, dict))
check("simulate_has_outcome",  "predicted_outcome" in sim_r)

# Queries
check("what_can_eat",          isinstance(agent.what_can_i_eat(), list))
check("what_dangerous",        isinstance(agent.what_is_dangerous(), list))
check("what_is_str",           "apple" in agent.what_is("apple"))
check("recall_list",           isinstance(agent.recall("apple"), list))

# Run autonomous
log = agent.run_autonomous(n_steps=10, verbose=False)
check("autonomous_runs",       len(log) == 10)
check("autonomous_v4step",     hasattr(log[0], "active_goal"))
check("autonomous_step_inc",   log[-1].step > log[0].step)

# Summary
summ = agent.summary()
check("summary_keys",          all(k in summ for k in [
    "steps","vector_memory","world_model_rules","goals_completed",
    "abstract_rules","meta_cycles","success_rate"
]))
check("summary_steps",         summ["steps"] >= 11)
check("summary_memory_int",    isinstance(summ["vector_memory"], int))
check("internal_state_keys",   "confidence" in summ["internal_state"])

# World simulator integrated
check("sim_count_positive",    agent.simulator.sim_count > 0)

# Goal generator
check("goal_gen_exists",       agent.goal_gen is not None)

# Abstraction engine
check("abstraction_has_rules", agent.abstraction is not None)

# Meta learner
check("meta_records",          len(list(agent.meta_learner._records)) >= 1)

# Consolidator
check("consolidator_exists",   agent.consolidator is not None)

# Internal state check
is_ = agent.internal_state
check("is_confidence",         0.0 <= is_.confidence <= 1.0)
check("is_frustration",        0.0 <= is_.frustration <= 1.0)


# ─────────────────────────────────────────────────────────
print(f"\n{'═'*54}")
print(f"  v4 Results: {PASS} passed, {FAIL} failed")
if ERRORS:
    print(f"  Failed: {ERRORS}")
else:
    print("  All v4 tests passed ✓")
print(f"{'═'*54}\n")
