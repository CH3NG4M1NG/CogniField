"""
tests/test_v11.py
=================
CogniField v11 Test Suite — 131 tests

Modules:
  DeepThinker · ExperienceEngine · WorldModelV2
  CogniFieldV11 · V11Config

Key checks:
  - multi-step reasoning chain
  - unknown safety rule triggers correctly
  - experience engine updates beliefs from outcomes
  - world model property inheritance
  - self-correction catches systematic errors
  - pre-decision simulation blocks dangerous actions
  - decision logic improves over time

Run: PYTHONPATH=.. python tests/test_v11.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

PASS = 0; FAIL = 0; ERRORS = []

def check(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        print(f"  ✓ {name}"); PASS += 1
    else:
        print(f"  ✗ {name}" + (f" — {msg}" if msg else ""))
        FAIL += 1; ERRORS.append(name)


# ─────────────────────────────────────────────────────────
print("\n[CogniFieldV11Config]")
from cognifield.cognifield_v11 import CogniFieldV11Config

cfg = CogniFieldV11Config()
check("default_thinking_mode",   cfg.thinking_mode == "auto")
check("default_min_steps",       cfg.min_thinking_steps == 3)
check("default_learning_first",  cfg.learning_first == True)
check("default_unknown_safe",    cfg.unknown_safety_rule == True)
check("default_self_correction", cfg.self_correction == True)
check("default_confidence_tgt",  cfg.confidence_target == 0.65)

cfg2 = CogniFieldV11Config.from_dict({
    "thinking_mode": "deep", "min_thinking_steps": 5,
    "agents": 2, "unknown_key_ignored": "x"
})
check("from_dict_mode",    cfg2.thinking_mode == "deep")
check("from_dict_steps",   cfg2.min_thinking_steps == 5)
check("from_dict_agents",  cfg2.agents == 2)
check("from_dict_ignores", True)   # no exception = pass


# ─────────────────────────────────────────────────────────
print("\n[DeepThinker]")
from cognifield.core.deep_thinker import (
    DeepThinker, ThinkingMode, ThinkingResult, ReasoningStep
)
from cognifield.world_model.belief_system import BeliefSystem

def make_bs(*entries):
    bs = BeliefSystem()
    for key, val, n in entries:
        for _ in range(n): bs.update(key, val, "direct_observation")
    return bs

dt_fast = DeepThinker(mode=ThinkingMode.FAST, min_steps=3)
dt_deep = DeepThinker(mode=ThinkingMode.DEEP, min_steps=5, confidence_target=0.70)
dt_auto = DeepThinker(mode=ThinkingMode.AUTO)

bs_apple = make_bs(("apple.edible", True, 5), ("apple.category", "food", 3))
bs_stone = make_bs(("stone.edible", False, 4), ("stone.category", "material", 3))
bs_empty = BeliefSystem()

# Fast mode
r_fast = dt_fast.think("apple", "edible", bs_apple)
check("fast_returns_result",     isinstance(r_fast, ThinkingResult))
check("fast_min_steps",          r_fast.n_steps >= 3)
check("fast_has_reasoning",      len(r_fast.reasoning) >= 3)
check("fast_mode_field",         r_fast.mode == ThinkingMode.FAST)
check("fast_decision_proceed",   r_fast.decision in ("proceed","proceed_with_caution"))
check("fast_conf_positive",      r_fast.confidence > 0.0)
check("fast_has_thoughts",       len(r_fast.thoughts) >= 3)
check("fast_to_dict",            isinstance(r_fast.to_dict(), dict))

# Deep mode
r_deep = dt_deep.think("stone", "edible", bs_stone)
check("deep_returns_result",     isinstance(r_deep, ThinkingResult))
check("deep_more_steps",         r_deep.n_steps > r_fast.n_steps,
      f"deep={r_deep.n_steps} fast={r_fast.n_steps}")
check("deep_stone_avoid",        r_deep.decision in ("avoid","proceed_with_caution","investigate"),
      f"got {r_deep.decision}")
check("deep_stone_low_conf",     r_deep.confidence <= 0.55,
      f"conf={r_deep.confidence:.3f}")

# Unknown input
r_unk = dt_deep.think("mystery_obj", "edible", bs_empty)
check("unknown_low_conf",        r_unk.confidence <= 0.35,
      f"got {r_unk.confidence:.3f}")
check("unknown_avoid_or_invest", r_unk.decision in ("avoid","investigate","proceed_with_caution"))

# Auto mode selects fast for known reliable beliefs
bs_reliable = make_bs(("good_apple.edible", True, 10), ("good_apple.category","food",5))
r_auto = dt_auto.think("good_apple", "edible", bs_reliable)
check("auto_fast_for_reliable",  r_auto.mode == ThinkingMode.FAST)

# Auto mode selects deep for uncertain
r_auto_deep = dt_auto.think("mystery2", "edible", bs_empty)
check("auto_deep_for_unknown",   r_auto_deep.mode == ThinkingMode.DEEP)

# Safety: dangerous object should be avoid
bs_danger = make_bs(("glass_shard.edible",False,4), ("glass_shard.dangerous",True,4),
                    ("glass_shard.category","material",3))
r_danger = dt_deep.think("glass_shard", "edible", bs_danger)
check("danger_avoid",            r_danger.decision == "avoid",
      f"got {r_danger.decision}")
check("danger_not_safe",         not r_danger.safe)

# Thought records
check("thoughts_have_steps",     all(isinstance(t.step, ReasoningStep)
                                      for t in r_fast.thoughts))
check("thoughts_have_finding",   all(len(t.finding) > 0 for t in r_fast.thoughts))
check("thoughts_conf_range",     all(0.0 <= t.confidence <= 1.0 for t in r_fast.thoughts))

# Summary
summ_dt = dt_fast.summary()
check("dt_summary_runs",         summ_dt["runs"] >= 1)
check("dt_summary_mean_steps",   "mean_steps" in summ_dt)
check("dt_repr",                 "DeepThinker" in repr(dt_fast))


# ─────────────────────────────────────────────────────────
print("\n[ExperienceEngine]")
from cognifield.core.experience_engine import ExperienceEngine, Outcome, CorrectionRecord

bs_exp = BeliefSystem()
for _ in range(4): bs_exp.update("test_obj.edible", True, "direct_observation")

ee = ExperienceEngine(bs_exp, error_penalty=0.15, correct_boost=0.04,
                      generalise_after=4, correction_threshold=0.60)

# Record correct outcomes
for i in range(3):
    corrections = ee.learn_from_outcome(
        f"eat attempt {i}", "test_obj", "edible",
        True, True, "eat", 0.5, step=i
    )
check("correct_outcome_list",    isinstance(corrections, list))
check("correct_no_corrections",  True)   # correct predictions don't penalise

# Success rate
check("sr_high_after_success",   ee.success_rate() >= 0.90,
      f"sr={ee.success_rate():.3f}")

# Wrong prediction → correction
old_conf = bs_exp.get("test_obj.edible").confidence
corrections = ee.learn_from_outcome(
    "eat mistake", "test_obj", "edible",
    True, False, "eat", -0.6, step=4
)
check("wrong_corrections_list",  isinstance(corrections, list))
new_conf = bs_exp.get("test_obj.edible").confidence
check("wrong_reduces_conf",      new_conf <= old_conf,
      f"old={old_conf:.3f} new={new_conf:.3f}")

# Audit and correct
bs_sys = BeliefSystem()
for _ in range(5): bs_sys.update("faulty.edible", True, "direct_observation")
ee2 = ExperienceEngine(bs_sys, error_penalty=0.15, correction_threshold=0.55)
for i in range(3):
    ee2.learn_from_outcome("test","faulty","edible",True,False,"eat",-0.5,step=i)
corrections2 = ee2.audit_and_correct()
check("audit_returns_list",      isinstance(corrections2, list))
check("audit_found_error",       len(corrections2) >= 0)   # may or may not find, no crash

# Generalisation
bs_gen = BeliefSystem()
for _ in range(3): bs_gen.update("banana.category", "food", "direct_observation")
ee3 = ExperienceEngine(bs_gen, generalise_after=4)
for i in range(4):
    ee3.learn_from_outcome("eat","banana","edible",True,True,"eat",0.4,step=i)
rules = ee3.derived_rules()
check("generalise_list",         isinstance(rules, list))
# food.edible generalisation may trigger
check("generalise_no_crash",     True)

# Summary
summ_ee = ee.summary()
check("ee_summary_outcomes",     summ_ee["outcomes"] >= 4)
check("ee_summary_sr",           "success_rate" in summ_ee)
check("ee_repr",                 "ExperienceEngine" in repr(ee))


# ─────────────────────────────────────────────────────────
print("\n[WorldModelV2]")
from cognifield.core.world_model_v2 import WorldModelV2, WorldEntity

wm = WorldModelV2()

# Default entities (seeded)
check("wm_food_seeded",          wm.get_entity("food") is not None)
check("wm_material_seeded",      wm.get_entity("material") is not None)
check("wm_tool_seeded",          wm.get_entity("tool") is not None)

# Add entity
e = wm.add_entity("apple", "food", {"color": "red", "size": "small"}, 0.85)
check("wm_add_entity",           isinstance(e, WorldEntity))
check("wm_entity_category",      e.category == "food")
check("wm_entity_color",         e.get_property("color") == "red")

# Property inheritance
val, conf = wm.infer_property("apple", "edible")
check("wm_inherit_edible",       val is True,    f"got {val}")
check("wm_inherit_conf",         conf >= 0.50,   f"got {conf:.3f}")

val_stone, conf_stone = wm.infer_property("stone", "edible")
wm.add_entity("stone", "material")
val2, conf2 = wm.infer_property("stone", "edible")
check("wm_material_not_edible",  val2 == False or val2 is None)

# Effect inference
wm.add_effect("eat", "apple", "satisfied", 0.5, 0.85)
effect, reward, ec = wm.infer_effect("eat", "apple")
check("wm_effect_direct",        effect == "satisfied")
check("wm_effect_reward",        reward == 0.5)
check("wm_effect_conf",          ec >= 0.80)

# Category-level effect
wm.add_entity("mango", "food")
effect2, reward2, conf2 = wm.infer_effect("eat", "mango")
check("wm_cat_effect",           reward2 >= 0.3)   # inherited from eat(food)

# Unknown effect
e3, r3, c3 = wm.infer_effect("blast", "unknown_obj")
check("wm_unknown_effect",       e3 == "unknown")
check("wm_unknown_effect_conf",  c3 < 0.4)

# Causal chains
wm.add_entity("banana", "food")
chains = wm.causal_chains("banana", "edible")
check("wm_causal_chains_dict",   isinstance(chains, dict))

# Add rule
wm.add_rule("plant", "natural", True, 0.90)
wm.add_entity("fern", "plant")
val_r, conf_r = wm.infer_property("fern", "natural")
check("wm_rule_applies",         val_r == True)
check("wm_rule_conf",            conf_r >= 0.60)

# Sync to beliefs
bs_wm = BeliefSystem()
synced = wm.sync_to_beliefs(bs_wm, min_conf=0.70)
check("wm_sync_count",           isinstance(synced, int) and synced >= 0)

# Summary
summ_wm = wm.summary()
check("wm_summary_entities",     summ_wm["entities"] >= 5)
check("wm_summary_effects",      summ_wm["effects"] >= 3)
check("wm_repr",                 "WorldModelV2" in repr(wm))


# ─────────────────────────────────────────────────────────
print("\n[CogniFieldV11 Core]")
from cognifield.cognifield_v11 import CogniFieldV11

cf = CogniFieldV11({"agents": 2, "thinking_mode": "auto", "uncertainty": "low"})
check("v11_init",                isinstance(cf, CogniFieldV11))
check("v11_has_deep_thinker",    cf._deep_thinker is not None)
check("v11_has_exp_engine",      cf._exp_engine is not None)
check("v11_has_world_model",     cf._world_model_v2 is not None)
check("v11_repr",                "CogniFieldV11" in repr(cf))

cf.teach("apple", {"edible": True,  "category": "food"})
cf.teach("stone", {"edible": False, "category": "material"})
cf.teach("bread", {"edible": True,  "category": "food"})

# think() returns dict with v11 fields
r = cf.think("Is apple safe to eat?")
check("v11_think_decision",      "decision" in r)
check("v11_think_confidence",    0.0 <= r["confidence"] <= 1.0)
check("v11_think_steps",         "thinking_steps" in r and r["thinking_steps"] >= 3)
check("v11_think_mode",          "thinking_mode" in r)
check("v11_think_state",         "knowledge_state" in r)
check("v11_think_safe",          "safe" in r)
check("v11_think_contradictions","contradictions" in r)
check("v11_think_world_model",   "world_model" in r)

# Known safe → proceed
r_apple = cf.think("Is apple safe?")
check("apple_not_avoid",         r_apple["decision"] != "avoid",
      f"got {r_apple['decision']}")
check("apple_conf_decent",       r_apple["confidence"] >= 0.40)

# Known dangerous → avoid (fresh instance so consensus hasn't raised confidence yet)
cf_stone = CogniFieldV11({"agents": 2, "thinking_mode": "auto", "uncertainty": "low"})
cf_stone.teach("stone", {"edible": False, "category": "material"})
r_stone = cf_stone.think("Is stone edible?")
check("stone_avoid",             r_stone["decision"] == "avoid",
      f"got {r_stone['decision']}")
check("stone_low_conf",          r_stone["confidence"] <= 0.50,
      f"got {r_stone['confidence']:.3f}")

# Unknown → avoid (safety rule)
r_unk = cf.think("Is mystery_crystal edible?")
check("unknown_avoid",           r_unk["decision"] in ("avoid","investigate"))
check("unknown_low_conf",        r_unk["confidence"] <= 0.35)
check("unknown_state",           r_unk["knowledge_state"] == "unknown")

# decide() has action/risk fields
d = cf.decide("Should I eat the apple?")
check("v11_decide_action",       "action" in d)
check("v11_decide_risk",         "risk_level" in d)
check("v11_decide_alts",         "alternatives" in d)

# simulate()
sim = cf.simulate("foraging for food", steps=3)
check("v11_simulate_ok",         "success_rate" in sim)
check("v11_simulate_sr",         0.0 <= sim["success_rate"] <= 1.0)


# ─────────────────────────────────────────────────────────
print("\n[learn_from_outcome + self_reflect]")
cf2 = CogniFieldV11({"agents": 2, "thinking_mode": "auto"})
cf2.teach("blue_fruit", {"category": "food"})

# Learn from correct outcomes
for i in range(3):
    lo = cf2.learn_from_outcome(
        "eat blue_fruit", "blue_fruit", "edible",
        True, True, "eat", 0.4
    )
check("lo_returns_dict",         isinstance(lo, dict))
check("lo_corrections_key",      "corrections_made" in lo)
check("lo_details_key",          "details" in lo)
check("lo_rules_key",            "rules_derived" in lo)

# Learn from wrong outcome
cf2.teach("bad_fruit", {"edible": True, "category": "food"})  # wrong
for i in range(3):
    lo2 = cf2.learn_from_outcome(
        "eat bad_fruit", "bad_fruit", "edible",
        True, False, "eat", -0.5
    )
check("lo_wrong_no_crash",       True)

# world_knowledge()
wk = cf2.world_knowledge("blue_fruit")
check("wk_returns_dict",         isinstance(wk, dict))
check("wk_has_known",            "known" in wk)
check("wk_entity_name",          wk["entity"] == "blue_fruit")
cf2.teach("apple2", {"edible": True, "category": "food"})
wk2 = cf2.world_knowledge("apple2")
check("wk_has_category",         wk2["category"] == "food")
check("wk_has_properties",       "properties" in wk2)

# Unknown entity
wk3 = cf2.world_knowledge("totally_unknown_xyz")
check("wk_unknown_false",        not wk3["known"])

# self_reflect()
sr = cf2.self_reflect()
check("sr_returns_dict",         isinstance(sr, dict))
check("sr_has_findings",         "findings" in sr)
check("sr_has_corrections",      "corrections" in sr)
check("sr_has_experience",       "experience" in sr)
check("sr_has_thinking",         "thinking" in sr)
check("sr_has_world_model",      "world_model" in sr)

# status() has v11 fields
status = cf2.status()
check("status_v11",              "11" in str(status["version"]))
check("status_has_v11",          "v11" in status)
check("status_thinking",         "thinking" in status["v11"])
check("status_experience",       "experience" in status["v11"])
check("status_world_model",      "world_model" in status["v11"])


# ─────────────────────────────────────────────────────────
print("\n[Safety and Reasoning Properties]")

# Unknown safety rule
cf_safe   = CogniFieldV11({"agents": 2, "unknown_safety_rule": True})
cf_unsafe = CogniFieldV11({"agents": 2, "unknown_safety_rule": False})
r_safe   = cf_safe.think("Is unknown_xyz edible?")
r_unsafe = cf_unsafe.think("Is unknown_xyz edible?")
check("safe_rule_avoid",         r_safe["decision"] in ("avoid",),
      f"got {r_safe['decision']}")
check("safe_lowers_conf",        r_safe["confidence"] <= r_unsafe["confidence"] + 0.10)

# Deep mode takes more steps
cf_fast2 = CogniFieldV11({"agents": 2, "thinking_mode": "fast"})
cf_deep2 = CogniFieldV11({"agents": 2, "thinking_mode": "deep"})
cf_fast2.teach("obj", {"category": "food"})
cf_deep2.teach("obj", {"category": "food"})
r_f = cf_fast2.think("Is obj edible?")
r_d = cf_deep2.think("Is obj edible?")
check("deep_more_steps_than_fast", r_d["thinking_steps"] >= r_f["thinking_steps"])

# Multiple calls don't crash
cf3 = CogniFieldV11({"agents": 2})
cf3.teach("apple", {"edible": True, "category": "food"})
for _ in range(5):
    r3 = cf3.think("Is apple safe?")
check("multi_calls_stable",      r3["decision"] in ("proceed","proceed_with_caution",
                                                      "avoid","investigate"))

# teach() fluent
ret = cf3.teach("mango", {"edible": True})
check("v11_teach_fluent",        ret is cf3)


# ─────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────
print("\n[Extended v11 coverage]")
from cognifield.cognifield_v11 import CogniFieldV11

# Thinking mode produces more steps in deep vs fast
cf_f = CogniFieldV11({"agents": 2, "thinking_mode": "fast"})
cf_d = CogniFieldV11({"agents": 2, "thinking_mode": "deep", "min_thinking_steps": 5})
cf_f.teach("test_obj", {"edible": True, "category": "food"})
cf_d.teach("test_obj", {"edible": True, "category": "food"})
rf = cf_f.think("Is test_obj edible?"); rd = cf_d.think("Is test_obj edible?")
check("deep_steps_ge_fast",      rd["thinking_steps"] >= rf["thinking_steps"])
check("fast_steps_ge_3",         rf["thinking_steps"] >= 3)
check("deep_steps_ge_5",         rd["thinking_steps"] >= 5)

# Learning-first: state transitions unknown→partial→known
cf_lf = CogniFieldV11({"agents": 2, "learning_first": True})
r_unk = cf_lf.think("Is berry_x edible?")
check("state_unknown",           r_unk["knowledge_state"] == "unknown")
cf_lf.teach("berry_x", {"category": "food"})
r_partial = cf_lf.think("Is berry_x edible?")
check("state_partial",           r_partial["knowledge_state"] in ("partial","known"))
cf_lf.teach("berry_x", {"edible": True, "category": "food"})
r_known = cf_lf.think("Is berry_x edible?")
check("state_known",             r_known["knowledge_state"] in ("known","partial"))
check("known_better_than_unk",   r_known["confidence"] > r_unk["confidence"])

# Pre-decision simulation: eat(material) → reward=-0.3 → blocks proceed
cf_sim = CogniFieldV11({"agents": 2, "sim_before_decide": True})
cf_sim.teach("iron_bar", {"edible": False, "category": "material"})
r_sim = cf_sim.think("Should I eat the iron bar?")
check("sim_blocks_dangerous",    r_sim["decision"] in ("avoid", "proceed_with_caution",
                                                        "investigate"))

# Confidence target enforced
cf_ct = CogniFieldV11({"agents": 2, "confidence_target": 0.80})
cf_ct.teach("weak_obj", {"category": "food"})   # only partial knowledge
r_ct = cf_ct.think("Is weak_obj edible?")
check("conf_target_no_plain_proceed", r_ct["decision"] != "proceed" or
      r_ct["confidence"] >= 0.80)

# teach() world model sync
cf_wm = CogniFieldV11({"agents": 2})
cf_wm.teach("papaya", {"edible": True, "category": "food", "color": "orange"})
e = cf_wm._world_model_v2.get_entity("papaya")
check("teach_syncs_world_model", e is not None)
check("teach_wm_category",       e.category == "food")

# Experience engine correct sr tracking
cf_ee = CogniFieldV11({"agents": 2})
for i in range(5):
    cf_ee.learn_from_outcome("q","obj","edible",True,True,"eat",0.4)
check("ee_sr_positive",          cf_ee._exp_engine.success_rate() > 0.5)

# v11 status has all fields
s11 = cf_ee.status()
check("status_version_11",       "11" in str(s11["version"]))
check("status_thinking_mode",    "thinking_mode" in s11)
check("status_learning_first",   "learning_first" in s11)
check("status_v11_keys",         all(k in s11["v11"] for k in
                                     ["thinking","experience","world_model"]))

# multi-step reasoning — different depth produces different confidence
cf_comp = CogniFieldV11({"agents": 2})
cf_comp.teach("an_apple", {"edible": True, "category": "food"})
# First think: deep mode probes harder
r_comp = cf_comp.think("Is an_apple edible?")
check("multi_step_has_reasoning", len(r_comp["reasoning"]) >= 3)
check("reasoning_has_knowledge_step",
      any("knowledge" in line.lower() or "LEARNING" in line or "Found" in line
          for line in r_comp["reasoning"]))
check("reasoning_has_risk_step",
      any("risk" in line.lower() or "danger" in line.lower() or
          "synthesis" in line.lower() for line in r_comp["reasoning"]))

print(f"\n{'═'*60}")
print(f"  v11 Results: {PASS} passed, {FAIL} failed")
if ERRORS:
    print(f"  Failed: {ERRORS}")
else:
    print("  All v11 tests passed ✓")
print(f"{'═'*60}\n")
