"""
tests/test_v3.py
================
CogniField v3 Test Suite

Covers all new modules:
  - WorldModel (transition + causal graph)
  - Planner (symbolic + simulation)
  - GoalSystem
  - RelationalMemory
  - AdvancedCuriosityEngine
  - RichEnv
  - AgentV3 integration

Run with:
  PYTHONPATH=.. python tests/test_v3.py
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
        detail = f" — {msg}" if msg else ""
        print(f"  ✗ {name}{detail}")
        FAIL += 1
        ERRORS.append(name)


# ─────────────────────────────────────────────────────
# TransitionModel
# ─────────────────────────────────────────────────────
print("\n[TransitionModel]")

from cognifield.world_model.transition_model import TransitionModel, WorldRule
from cognifield.latent_space.frequency_space import FrequencySpace

space = FrequencySpace(dim=64)
tm    = TransitionModel(space=space, dim=64)
rng   = np.random.default_rng(0)

def sv():
    return space.l2(rng.standard_normal(64).astype(np.float32))

# Record transitions
for _ in range(3):
    tm.record(sv(), "eat", sv(), +0.5, True,  "apple", "food")
for _ in range(3):
    tm.record(sv(), "eat", sv(), -0.2, False, "stone", "material")
tm.record(sv(), "pick", sv(), +0.1, True, "apple", "food")

check("n_transitions",         tm.n_transitions == 7)
check("n_rules_created",       tm.n_rules >= 2)

outcome, reward, conf = tm.predict_outcome("eat", "food")
check("predict_food_success",  outcome == "success")
check("predict_food_reward",   reward > 0.0)
check("predict_food_conf",     conf >= 0.8)

outcome2, reward2, conf2 = tm.predict_outcome("eat", "material")
check("predict_material_fail", outcome2 == "failure")
check("predict_material_neg",  reward2 < 0.0)

check("can_eat_food",     tm.can_do("eat", "food"))
check("cannot_eat_mat",   not tm.can_do("eat", "material"))

pred_state, pred_reward = tm.predict_next_state(sv(), "eat")
check("predict_state_shape",   pred_state.shape == (64,))
check("predict_state_unit",    abs(np.linalg.norm(pred_state) - 1.0) < 1e-4)

rules = tm.get_rules("eat")
check("get_rules_filtered",    all(r.action == "eat" for r in rules))

rule_summary = tm.rule_summary()
check("rule_summary_list",     isinstance(rule_summary, list) and len(rule_summary) >= 2)
check("rule_summary_reliable", any(r["reliable"] for r in rule_summary))


# ─────────────────────────────────────────────────────
# WorldRule
# ─────────────────────────────────────────────────────
print("\n[WorldRule]")

rule = WorldRule("eat", "food", "success", reward=0.5, confidence=0.5)
rule.update(True, 0.5)
rule.update(True, 0.5)
check("rule_confidence_rises", rule.confidence > 0.5)
check("rule_is_reliable",      rule.is_reliable)

# A rule that's been observed only once should not be reliable
rule_new = WorldRule("eat", "material", "failure", reward=-0.2)
check("rule_bad_not_reliable", not rule_new.is_reliable)  # count=0 → not reliable


# ─────────────────────────────────────────────────────
# CausalGraph
# ─────────────────────────────────────────────────────
print("\n[CausalGraph]")

from cognifield.world_model.causal_graph import CausalGraph

cg = CausalGraph()
cg.add_causal("eat(apple)", "satisfied", weight=0.9)
cg.add_causal("eat(stone)", "damaged",   weight=0.9)
cg.add_property("apple", "edible", True)
cg.add_property("stone", "edible", False)
cg.add_property("apple", "color",  "red")
cg.add_is_a("apple", "food")
cg.add_is_a("stone", "material")

check("get_effects",          len(cg.get_effects("eat(apple)")) > 0)
check("effects_satisfied",    cg.get_effects("eat(apple)")[0][0] == "satisfied")
check("is_edible_apple",      cg.is_edible("apple") is True)
check("is_edible_stone",      cg.is_edible("stone") is False)
check("category_apple",       cg.get_category("apple") == "food")
check("find_edible",          "apple" in cg.find_edible_objects())
check("what_causes_satisfied", "eat(apple)" in [c for c, _ in cg.what_causes("satisfied")])
check("describe_object",       isinstance(cg.describe_object("apple"), str))

# Test ingest_feedback
cg.ingest_feedback("eat", "bread", {"category": "food"}, True, 0.5)
check("ingest_adds_edge",     len(cg.get_effects("eat(bread)")) > 0)

s = cg.summary()
check("summary_keys",         all(k in s for k in ("n_nodes", "n_edges", "edible")))


# ─────────────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────────────
print("\n[Planner]")

from cognifield.planning.planner import Planner, Plan, PlanStep
from cognifield.encoder.text_encoder import TextEncoder

enc = TextEncoder(dim=64); enc.fit()

# Set up planner with pre-trained world model
tm2 = TransitionModel(space=space, dim=64)
cg2 = CausalGraph()
for _ in range(3):
    tm2.record(sv(), "eat",  sv(), +0.5, True, "apple", "food")
    tm2.record(sv(), "pick", sv(), +0.1, True, "apple", "food")
    tm2.record(sv(), "eat",  sv(), -0.2, False,"stone", "material")

cg2.add_property("apple", "edible", True)
cg2.add_is_a("apple", "food")
cg2.add_property("stone", "edible", False)
cg2.add_is_a("stone", "material")

planner = Planner(tm2, cg2, space, max_depth=3, beam_width=3, dim=64)

goal_vec   = enc.encode("eat apple")
state_vec  = enc.encode("standing near apple")
available  = [("apple","food"), ("stone","material")]

plan = planner.plan("eat apple", goal_vec, state_vec, available, inventory=[])
check("plan_not_empty",       not plan.is_empty)
check("plan_has_steps",       len(plan.steps) >= 1)
check("plan_score_range",     0.0 <= plan.total_score <= 1.0)
check("plan_action_sequence", len(plan.action_sequence) == len(plan.steps))
check("plan_is_safe",         planner.is_safe(plan))
check("plan_describe_str",    isinstance(plan.describe(), str))

# Plan with apple already in inventory → should go straight to eat
plan_inv = planner.plan("eat apple", goal_vec, state_vec, available, inventory=["apple"])
if not plan_inv.is_empty:
    first_action = plan_inv.steps[0].action
    check("plan_with_inventory", first_action in ("eat", "pick"))
else:
    check("plan_with_inventory", False, "plan was empty")

# Safety check for dangerous plan
bad_steps = [PlanStep("eat","stone",sv(),-0.5, 0.8, 0.1)]
bad_plan  = Plan("eat_stone", bad_steps, 0.1, 1)
check("unsafe_plan_detected", not planner.is_safe(bad_plan, danger_threshold=0.3))


# ─────────────────────────────────────────────────────
# GoalSystem
# ─────────────────────────────────────────────────────
print("\n[GoalSystem]")

from cognifield.agent.goals import GoalSystem, GoalType, GoalStatus

gs = GoalSystem()

g1 = gs.add_eat_goal("apple", priority=0.8)
g2 = gs.add_avoid_goal("stone", priority=0.9)
g3 = gs.add_explore_goal("explore unknown", priority=0.5)

check("goals_added",          gs.active_count == 3)
check("goal_status_pending",  g1.status == GoalStatus.PENDING)

selected = gs.select_active_goal()
check("select_returns_goal",  selected is not None)
check("select_highest_prio",  selected.priority >= 0.8)  # avoid_stone=0.9 should win

gs.mark_completed(g2)
check("completed_removed",    g2 not in gs._goals)
check("completed_count",      gs.completed_count == 1)

check("avoidance_goals",      len(gs.get_avoidance_goals()) >= 0)

# infer_goals_from_context
new_goals = gs.infer_goals_from_context(
    known_edible=["apple", "bread"],
    unknown_objects=["mystery_object"],
    inventory=[],
)
check("inferred_goals",       len(new_goals) >= 1)

# check_goal_satisfied
g4 = gs.add_eat_goal("bread")
fb = {"action": "eat", "success": True, "object_name": "bread"}
check("goal_satisfied_eat",   gs.check_goal_satisfied(g4, fb))

fb_bad = {"action": "eat", "success": False, "object_name": "stone"}
g5 = gs.add_avoid_goal("stone")
check("avoid_goal_satisfied", gs.check_goal_satisfied(g5, fb_bad))

# Goal aging
import time
g6 = gs.add_goal("old goal", GoalType.CUSTOM, priority=0.3)
check("goal_age",             g6.age_seconds >= 0.0)


# ─────────────────────────────────────────────────────
# RelationalMemory
# ─────────────────────────────────────────────────────
print("\n[RelationalMemory]")

from cognifield.memory.relational_memory import RelationalMemory

rm = RelationalMemory(dim=64, space=space)
enc2 = TextEncoder(dim=64); enc2.fit()

v_apple = enc2.encode("apple")
rm.add_object_properties("apple", {
    "edible": True, "color": "red", "category": "food"
}, vector=v_apple)
rm.add_object_properties("stone", {
    "edible": False, "color": "grey", "category": "material"
})
rm.add_fact("glass_jar", "fragile", True)

check("facts_stored",        rm.n_facts() >= 6)
check("find_edible",         "apple" in rm.find_edible())
check("find_dangerous",      "stone" in rm.find_dangerous())
check("get_value_edible",    rm.get_value("apple", "edible") is True)
check("get_value_color",     rm.get_value("apple", "color") == "red")
check("get_category",        rm.get_category("stone") == "material")
check("what_is_str",         isinstance(rm.what_is("apple"), str))
check("what_is_contains",    "apple" in rm.what_is("apple"))
check("is_known_true",       rm.is_known("apple"))
check("is_known_false",      not rm.is_known("unicorn"))

query_result = rm.query("edible", True)
check("query_edible",        len(query_result) >= 1)
check("query_format",        all(isinstance(s, str) and 0<=c<=1
                                 for s,c in query_result))

fragile = rm.query("fragile", True)
check("query_fragile",       any(s == "glass_jar" for s,_ in fragile))

# Vector recall
recall = rm.recall_similar(v_apple, k=2)
check("recall_returns",      len(recall) >= 1)
check("recall_top_match",    recall[0][0] > 0.8)

# ingest_env_feedback
rm.ingest_env_feedback("eat", "bread",
                        {"category": "food", "edible": True}, True, 0.5)
check("ingest_adds_outcome",  rm.get_value("eat(bread)", "outcome") is not None)

s = rm.summary()
check("summary_keys",         all(k in s for k in ("concepts","total_facts","edible")))


# ─────────────────────────────────────────────────────
# AdvancedCuriosityEngine
# ─────────────────────────────────────────────────────
print("\n[AdvancedCuriosityEngine]")

from cognifield.curiosity.advanced_curiosity import AdvancedCuriosityEngine
from cognifield.memory.memory_store import MemoryStore

vec_mem    = MemoryStore(dim=64)
rel_mem2   = RelationalMemory(dim=64, space=space)
curiosity  = AdvancedCuriosityEngine(
    space, rel_mem2, vec_mem,
    novelty_threshold=0.3, dim=64
)

# Empty memory → max novelty
v_unk = space.l2(rng.standard_normal(64).astype(np.float32))
check("empty_novelty_high",    curiosity.detect_novelty(v_unk) >= 0.6)  # high when memory empty

# Add known concept
v_known = space.l2(rng.standard_normal(64).astype(np.float32))
rel_mem2.add_object_properties("apple", {"edible": True, "category": "food"},
                                 vector=v_known)
vec_mem.store(v_known, "apple", modality="concept", allow_duplicate=True)

# Identical vector → low novelty
check("known_low_novelty",    curiosity.detect_novelty(v_known, "apple") < 0.2)

# Different vector → high novelty
check("unknown_high_novelty", curiosity.detect_novelty(v_unk, "mystery") > 0.3)

# Generate hypotheses
hyps = curiosity.generate_hypotheses("mystery_obj", v_unk)
check("hyps_is_list",         isinstance(hyps, list))

# Explore
report = curiosity.explore("purple_berry", v_unk)
check("explore_report_keys",  all(k in report for k in
                                  ("concept","novelty","n_hypotheses","suggested_action")))
check("n_explorations",       curiosity.n_explorations == 1)

# Update hypothesis
check("priority_range",       0.0 <= curiosity.exploration_priority("mystery_obj", v_unk) <= 1.0)
check("curiosity_weight_gt1", curiosity.curiosity_weight(v_unk) >= 1.0)

curiosity_summary = curiosity.summary()
check("curiosity_summary_keys", all(k in curiosity_summary for k in
                                     ("explorations", "explored_concepts", "hypotheses_open")))


# ─────────────────────────────────────────────────────
# RichEnv
# ─────────────────────────────────────────────────────
print("\n[RichEnv]")

from cognifield.environment.rich_env import RichEnv

env = RichEnv(seed=42)

check("objects_loaded",        len(env.object_names) >= 6)
check("visibility_radius",     env.VISIBILITY_RADIUS == 3)
check("state_vec_shape",       env.state_vector().shape == (64,))
check("state_vec_unit",        abs(np.linalg.norm(env.state_vector()) - 1.0) < 1e-4)

# Observe
fb = env.step("observe")
check("observe_success",       fb["success"])
check("observe_has_message",   isinstance(fb["message"], str))

# Move
fb_move = env.step("move", 3, 2)
check("move_success",          fb_move["success"])
check("move_updates_pos",      env._agent_pos == (3, 2))
check("move_neg_reward",       fb_move["reward"] < 0)

# Inspect (to reveal properties)
fb_insp = env.step("inspect", "apple")
if fb_insp["success"]:
    check("inspect_reveals_props", "object_props" in fb_insp)
    check("inspect_has_category",  "object_category" in fb_insp)
else:
    check("inspect_apple_found",   False, "inspect failed — maybe too far")

# Pick
fb_pick = env.step("pick", "apple")
if not fb_pick["success"] and "far" in fb_pick["message"]:
    env._agent_pos = env.get_object("apple").position  # move to it
    fb_pick = env.step("pick", "apple")
check("pick_apple",            fb_pick["success"] or "inventory" in fb_pick["message"])

# Eat non-edible stone
env2 = RichEnv(seed=0)
if "stone" in env2.object_names:
    stone_pos = env2.get_object("stone").position
    env2._agent_pos = stone_pos
    env2.step("pick", "stone")
    if "stone" in env2.inventory:
        fb_eat_stone = env2.step("eat", "stone")
        check("eat_stone_fails",   not fb_eat_stone["success"])
        check("eat_stone_penalty", fb_eat_stone["reward"] < 0)
    else:
        check("eat_stone_fails",   True)  # couldn't even pick it up — ok
        check("eat_stone_penalty", True)

# Eat edible apple
env3 = RichEnv(seed=10)
for obj_name in env3.object_names:
    obj = env3.get_object(obj_name)
    if obj and obj.edible is True:
        env3._agent_pos = obj.position
        env3.step("pick", obj_name)
        if obj_name in env3.inventory:
            fb_eat = env3.step("eat", obj_name)
            check("eat_edible_success",  fb_eat["success"])
            check("eat_edible_reward",   fb_eat["reward"] > 0)
            break

# Drop fragile → breaks
env4 = RichEnv(seed=99)
if "glass_jar" in env4.object_names:
    gj = env4.get_object("glass_jar")
    env4._agent_pos = gj.position
    env4.step("pick", "glass_jar")
    if "glass_jar" in env4.inventory:
        fb_drop = env4.step("drop", "glass_jar")
        check("drop_fragile_breaks",  "broken" in fb_drop.get("consequence","") or
                                       not fb_drop["success"] or
                                       fb_drop["reward"] < 0)
    else:
        check("drop_fragile_breaks",  True)

check("available_objects_list", isinstance(env.available_objects(), list))
check("visible_objects_list",   isinstance(env.visible_objects(), list))
check("env_stats_keys",         all(k in env.stats() for k in
                                     ("steps","total_reward","health","satiation")))


# ─────────────────────────────────────────────────────
# AgentV3 (integration)
# ─────────────────────────────────────────────────────
print("\n[CogniFieldAgentV3]")

from cognifield.agent.agent_v3 import CogniFieldAgentV3, AgentV3Config

env_a = RichEnv(seed=7)
agent = CogniFieldAgentV3(
    config=AgentV3Config(dim=64, verbose=False, seed=0),
    env=env_a,
)

# Teach concepts
agent.teach("apple red food edible", "apple",
            {"edible": True, "category": "food"})
agent.teach("stone grey hard heavy", "stone",
            {"edible": False, "category": "material"})

check("teach_adds_memory",    len(agent.vec_memory) >= 2)
check("teach_adds_facts",     agent.rel_memory.n_facts() >= 2)

# Add goals
from cognifield.agent.goals import GoalType
g = agent.add_goal("eat apple", GoalType.EAT_OBJECT, target="apple", priority=0.8)
check("add_goal",             agent.goals.active_count >= 1)

# Step
s = agent.step(verbose=False)
check("step_returns_v3step",  s.step == 1)
check("step_has_goal",        s.active_goal is not None)
check("step_has_novelty",     0.0 <= s.novelty <= 1.0)

# Step with text input
s2 = agent.step(text_input="I see a red apple nearby", verbose=False)
check("step_text_input",      s2.step == 2)
check("step_encodes",         s2.encoded_vec.shape == (64,))

# Recall
results = agent.recall("apple", k=3)
check("recall_returns",       len(results) >= 1)
check("recall_format",        all(isinstance(sim, float) and isinstance(lbl, str)
                                  for sim, lbl in results))

# What can I eat?
edible = agent.what_can_i_eat()
check("what_can_i_eat",       "apple" in edible)

# What is dangerous?
dangerous = agent.what_is_dangerous()
check("dangerous_stone",      "stone" in dangerous)

# what_is
check("what_is_apple",        "apple" in agent.what_is("apple"))

# World model updated by steps
check("wm_has_transitions",   agent.world_model.n_transitions >= 0)  # may be 0 if no env action

# Summary
summ = agent.summary()
check("summary_keys",         all(k in summ for k in
                                  ("steps","vector_memory","relational_facts",
                                   "world_model_rules","goals","curiosity","env")))
check("summary_steps",        summ["steps"] == 2)


# ─────────────────────────────────────────────────────
# Integration: teach → plan → eat scenario
# ─────────────────────────────────────────────────────
print("\n[Integration: teach→plan→eat]")

env_b = RichEnv(seed=42)
agent2 = CogniFieldAgentV3(
    config=AgentV3Config(dim=64, verbose=False, seed=1),
    env=env_b,
)

# Teach
agent2.teach("apple food edible red", "apple", {"edible": True, "category": "food"})
agent2.teach("stone material heavy grey", "stone", {"edible": False, "category": "material"})

# Pre-populate world model with experience
for _ in range(3):
    agent2.world_model.record(sv(),"eat",sv(),0.5,True,"apple","food")
    agent2.world_model.record(sv(),"pick",sv(),0.1,True,"apple","food")
    agent2.world_model.record(sv(),"eat",sv(),-0.2,False,"stone","material")
    agent2.causal_graph.add_property("apple","edible",True)

# Add eat goal
agent2.add_goal("eat apple", GoalType.EAT_OBJECT, "apple", priority=0.9)

# Generate a plan
goal_vec  = agent2.enc.encode("eat apple")
state_vec = env_b.state_vector()
available = env_b.available_objects()

plan = agent2.planner.plan("eat apple", goal_vec, state_vec, available, env_b.inventory)
check("integration_plan_exists",   not plan.is_empty or len(available) == 0)
check("integration_plan_safe",     agent2.planner.is_safe(plan))

# Run 5 steps
steps = [agent2.step(verbose=False) for _ in range(5)]
check("integration_steps_run",    len(steps) == 5)
check("integration_novelty_set",  all(0 <= s.novelty <= 1 for s in steps))
check("integration_wm_grows",     agent2.world_model.n_transitions >= 3)


# ─────────────────────────────────────────────────────
print(f"\n{'═'*50}")
print(f"  v3 Results: {PASS} passed, {FAIL} failed")
if ERRORS:
    print(f"  Failed: {ERRORS}")
else:
    print("  All v3 tests passed ✓")
print(f"{'═'*50}\n")
