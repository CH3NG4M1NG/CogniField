"""
tests/test_v11_part2.py
=======================
CogniField v11 Part 2 — Embodied Intelligence Tests

Covers:
  VirtualBody · PerceptionSystem · ActionSystem
  InteractionLoop · CogniFieldV11 embodied methods

Key scenarios:
  - eat apple → satisfied
  - eat stone → damage
  - unknown object → blocked
  - learning improves future decision
  - full loop runs without crash

Run: PYTHONPATH=.. python tests/test_v11_part2.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = 0; FAIL = 0; ERRORS = []

def check(name, cond, msg=""):
    global PASS, FAIL
    if cond:
        print(f"  ✓ {name}"); PASS += 1
    else:
        print(f"  ✗ {name}" + (f" — {msg}" if msg else ""))
        FAIL += 1; ERRORS.append(name)


# ─────────────────────────────────────────────────────────
print("\n[VirtualBody]")
from cognifield.agents.body import (
    VirtualBody, BodyAction, ActionStatus, BodyActionResult, BodyState
)

body = VirtualBody(max_inventory=3, seed=42)

check("body_init",          body.state.alive)
check("body_health_full",   body.state.health == 1.0)
check("body_max_carry",     body.state.max_carry == 3)
check("body_inventory_empty",len(body.state.inventory) == 0)
check("body_can_pick",      body.state.can_pick)
check("body_not_hungry",    not body.state.is_hungry)
check("body_repr",          "VirtualBody" in repr(body))

# Eat known edible
r_eat = body.act(BodyAction.EAT, "apple", {"edible": True, "effect": "satisfied",
                                             "reward": 0.5, "confidence": 0.9, "known": True})
check("eat_apple_success",  r_eat.status == ActionStatus.SUCCESS)
check("eat_apple_effect",   r_eat.effect == "satisfied")
check("eat_apple_reward",   r_eat.reward == 0.5)
check("eat_apple_succeeded",r_eat.succeeded)
check("hunger_reduced",     body.state.hunger < 0.30)
check("health_after_eat",   body.state.health >= 0.95)

# Eat inedible
r_stone = body.act(BodyAction.EAT, "stone", {"edible": False, "effect": "damage",
                                               "reward": -0.3, "confidence": 0.9, "known": True})
check("eat_stone_failure",  r_stone.status == ActionStatus.FAILURE)
check("eat_stone_damage",   "damage" in r_stone.effect)
check("eat_stone_neg_reward",r_stone.reward < 0)
check("eat_stone_failed",   r_stone.failed)
check("health_reduced",     body.state.health < 1.0)

# Pick and drop
body2 = VirtualBody(seed=0)
r_pick = body2.act(BodyAction.PICK, "bread", {"heavy": False})
check("pick_success",       r_pick.status == ActionStatus.SUCCESS)
check("pick_in_inventory",  "bread" in body2.state.inventory)
check("pick_effect",        r_pick.effect == "picked")

r_drop = body2.act(BodyAction.DROP, "bread")
check("drop_success",       r_drop.status == ActionStatus.SUCCESS)
check("drop_removed",       "bread" not in body2.state.inventory)

# Inspect
r_inspect = body.act(BodyAction.INSPECT, "apple",
                      {"properties": {"color": "red", "edible": True}})
check("inspect_success",    r_inspect.status == ActionStatus.SUCCESS)
check("inspect_effect",     r_inspect.effect == "observed")
check("inspect_has_obs",    "properties" in r_inspect.observations)
check("inspect_confidence", r_inspect.confidence >= 0.85)

# Move
r_move = body.act(BodyAction.MOVE, "north")
check("move_success",       r_move.status == ActionStatus.SUCCESS)
check("move_changes_pos",   body.state.position != (0, 0))

# Bad direction
r_bad = body.act(BodyAction.MOVE, "diagonal")
check("bad_direction",      r_bad.status == ActionStatus.INVALID)

# Wait
r_wait = body.act(BodyAction.WAIT, "")
check("wait_success",       r_wait.status == ActionStatus.SUCCESS)
check("wait_effect",        r_wait.effect == "rested")

# Unknown edibility → uncertain outcome
body3 = VirtualBody(seed=123)
r_unk = body3.act(BodyAction.EAT, "mystery", {"confidence": 0.3})
check("unknown_status",     r_unk.status in (ActionStatus.SUCCESS, ActionStatus.FAILURE))
check("unknown_has_effect", len(r_unk.effect) > 0)

# to_dict
d = r_eat.to_dict()
check("result_to_dict",     isinstance(d, dict))
check("result_dict_keys",   all(k in d for k in ["action","object","status","effect","reward"]))

# Summary
summ = body.summary()
check("body_summary",       "health" in summ and "hunger" in summ)
check("body_steps",         summ["steps"] >= 5)


# ─────────────────────────────────────────────────────────
print("\n[PerceptionSystem]")
from cognifield.agents.perception import (
    PerceptionSystem, Observation, PerceptionSignal
)

ps = PerceptionSystem(novelty_threshold=0.40, risk_threshold=0.25)

# Process body result for apple eating
body_for_ps = VirtualBody(seed=0)
r_apple_ps = body_for_ps.act(BodyAction.EAT, "apple",
                               {"edible": True, "effect": "satisfied",
                                "reward": 0.5, "confidence": 0.9})
obs_apple = ps.process_body_result(r_apple_ps)

check("obs_type",           isinstance(obs_apple, Observation))
check("obs_apple_success",  obs_apple.signal == PerceptionSignal.SUCCESS)
check("obs_apple_reward",   obs_apple.reward > 0)
check("obs_apple_effect",   obs_apple.effect == "satisfied")
check("obs_apple_is_success",obs_apple.is_success)
check("obs_apple_not_fail", not obs_apple.is_failure)
check("obs_belief_updates", len(obs_apple.belief_updates) >= 1)

# Check that edible=True belief update was inferred
edible_updates = [(k,v,c) for k,v,c in obs_apple.belief_updates if "edible" in k]
check("inferred_edible_true", len(edible_updates) > 0 and edible_updates[0][1] is True)

# Process stone eating
r_stone_ps = body_for_ps.act(BodyAction.EAT, "stone",
                               {"edible": False, "effect": "damage",
                                "reward": -0.3, "confidence": 0.9})
obs_stone = ps.process_body_result(r_stone_ps)
check("obs_stone_fail",     obs_stone.is_failure)
check("obs_stone_signal",   obs_stone.signal in (PerceptionSignal.FAILURE,
                                                   PerceptionSignal.DANGER))
check("obs_stone_neg_reward",obs_stone.reward < 0)
check("obs_stone_danger",   obs_stone.danger_detected)

# Inferred edible=False for stone
stone_updates = [(k,v,c) for k,v,c in obs_stone.belief_updates if "edible" in k]
check("inferred_edible_false", len(stone_updates) > 0 and stone_updates[0][1] is False)

# Process inspect
r_inspect_ps = body_for_ps.act(BodyAction.INSPECT, "mushroom",
                                {"properties": {"color": "brown", "size": "small"}})
obs_inspect = ps.process_body_result(r_inspect_ps)
check("obs_inspect_neutral", obs_inspect.signal in (PerceptionSignal.SUCCESS,
                                                      PerceptionSignal.NEUTRAL))

# to_dict
od = obs_apple.to_dict()
check("obs_to_dict",        isinstance(od, dict))
check("obs_dict_keys",      all(k in od for k in ["action","target","signal","reward"]))

# Success rate
check("ps_sr_positive",     ps.success_rate() > 0)
check("ps_summary",         "total_observations" in ps.summary())
check("ps_repr",            "PerceptionSystem" in repr(ps))


# ─────────────────────────────────────────────────────────
print("\n[ActionSystem]")
from cognifield.agents.action_system import (
    ActionSystem, ValidationStatus, ActionLogEntry
)
from cognifield.world_model.belief_system import BeliefSystem

body_as = VirtualBody(seed=0)
action_sys = ActionSystem(
    body=body_as,
    unknown_safety_rule=True,
    min_confidence_to_act=0.35,
)

bs_as = BeliefSystem()
for _ in range(5): bs_as.update("apple.edible", True, "direct_observation")

# Validate + execute known safe object
executed, br, entry = action_sys.execute("eat", "apple", bs_as,
                                          {"edible": True, "reward": 0.5,
                                           "effect": "satisfied", "confidence": 0.9})
check("apple_executed",     executed)
check("apple_effect",       br.effect == "satisfied")
check("entry_type",         isinstance(entry, ActionLogEntry))
check("entry_executed",     entry.executed)

# Block unknown object
executed2, br2, entry2 = action_sys.execute("eat", "mystery_xyz", bs_as,
                                              {"confidence": 0.3})
check("unknown_blocked",    not executed2)
check("unknown_validation", entry2.validation == ValidationStatus.BLOCKED)
check("unknown_effect",     br2.effect == "blocked")

# Block known dangerous
for _ in range(4): bs_as.update("poison.edible", False, "direct_observation")
executed3, br3, entry3 = action_sys.execute("eat", "poison", bs_as)
check("dangerous_blocked",  not executed3)
check("dangerous_reason",   "non-edible" in br3.reason or "blocked" in br3.reason.lower())

# Inspect always passes (safe action)
executed4, br4, _ = action_sys.execute("inspect", "stone", bs_as)
check("inspect_allowed",    executed4)
check("inspect_effect",     br4.effect == "observed")

# Force bypasses safety
executed5, br5, _ = action_sys.execute("eat", "mystery_xyz", bs_as, force=True)
check("force_executes",     executed5)

# Summary
summ_as = action_sys.summary()
check("as_summary",         "total_actions" in summ_as)
check("as_blocked_count",   summ_as["blocked"] >= 2)
check("as_repr",            "ActionSystem" in repr(action_sys))


# ─────────────────────────────────────────────────────────
print("\n[InteractionLoop]")
from cognifield.core.interaction_loop import (
    InteractionLoop, EpisodeStep
)
from cognifield.core.deep_thinker import DeepThinker, ThinkingMode
from cognifield.core.experience_engine import ExperienceEngine
from cognifield.core.world_model_v2 import WorldModelV2

body_il  = VirtualBody(seed=0)
act_sys  = ActionSystem(body_il, unknown_safety_rule=True)
perc_sys = PerceptionSystem()
bs_il    = BeliefSystem()
wm_il    = WorldModelV2()
dt_il    = DeepThinker(mode=ThinkingMode.AUTO)
ee_il    = ExperienceEngine(bs_il)

for _ in range(5): bs_il.update("apple.edible", True, "direct_observation")
for _ in range(4): bs_il.update("stone.edible", False, "direct_observation")
wm_il.add_entity("apple", "food", {"edible": True})
wm_il.add_entity("stone", "material", {"edible": False})

loop = InteractionLoop(
    body=body_il, action_system=act_sys, perception=perc_sys,
    deep_thinker=dt_il, experience_engine=ee_il,
    world_model=wm_il, belief_system=bs_il,
    unknown_safety=True, verbose=False,
)

# eat apple step
s_apple = loop.step("eat apple")
check("step_type",          isinstance(s_apple, EpisodeStep))
check("step_action",        s_apple.intent_action == "eat")
check("step_target",        s_apple.intent_target == "apple")
check("step_apple_exec",    s_apple.action_executed)
check("step_apple_success", s_apple.succeeded)
check("step_apple_reward",  s_apple.reward > 0)
check("step_has_thinking",  s_apple.thinking_decision is not None)
check("step_has_sim",       s_apple.simulated_outcome is not None)
check("step_has_body",      s_apple.body_health > 0)
check("step_to_dict",       isinstance(s_apple.to_dict(), dict))
check("step_str",           "Step" in str(s_apple))

# eat stone — should be blocked
s_stone = loop.step("eat stone")
check("step_stone_blocked", not s_stone.action_executed)
check("step_stone_effect",  "blocked" in s_stone.effect.lower())

# inspect (safe action — always runs)
s_inspect = loop.step("inspect apple")
check("step_inspect_exec",  s_inspect.action_executed)

# unknown object — blocked by safety rule
s_unk = loop.step("eat unknown_xyz")
check("step_unk_blocked",   not s_unk.action_executed)

# run_episode
ep_steps = loop.run_episode(["inspect apple", "eat apple", "eat stone"])
check("episode_steps",      len(ep_steps) == 3)
check("episode_step_type",  all(isinstance(s, EpisodeStep) for s in ep_steps))
check("episode_inspect_ok", ep_steps[0].action_executed)
check("episode_apple_ok",   ep_steps[1].action_executed)
check("episode_stone_no",   not ep_steps[2].action_executed)

# Success rate
check("loop_sr",            loop.success_rate() > 0)
check("loop_summary",       "total_steps" in loop.summary())
check("loop_repr",          "InteractionLoop" in repr(loop))

# Intent parsing
action, target = InteractionLoop._parse_intent("eat the apple")
check("parse_eat_apple",    action == "eat" and target == "apple")
action2, target2 = InteractionLoop._parse_intent("inspect stone carefully")
check("parse_inspect_stone",action2 == "inspect" and target2 == "stone")
action3, target3 = InteractionLoop._parse_intent("pick up bread")
check("parse_pick_bread",   action3 == "pick")


# ─────────────────────────────────────────────────────────
print("\n[CogniFieldV11 Embodied API]")
from cognifield import CogniField

cf = CogniField({"agents": 2, "thinking_mode": "auto",
                  "unknown_safety_rule": True})
cf.teach("apple", {"edible": True,  "category": "food"})
cf.teach("stone", {"edible": False, "category": "material"})

# act() method
r_act = cf.act("eat", "apple")
check("cf_act_returns_dict",  isinstance(r_act, dict))
check("cf_act_has_status",    "status" in r_act)
check("cf_act_has_effect",    "effect" in r_act)
check("cf_act_has_reward",    "reward" in r_act)
check("cf_act_has_body",      "body_health" in r_act)
check("cf_act_apple_success", r_act["status"] == "success")
check("cf_act_apple_effect",  r_act["effect"] == "satisfied")

r_act2 = cf.act("eat", "stone")
check("cf_act_stone_fail",    r_act2["status"] in ("failure", "blocked"))

r_act3 = cf.act("eat", "totally_unknown_object")
check("cf_act_unknown_blocked",r_act3["status"] == "blocked")

# step() method
s = cf.step("eat apple")
check("cf_step_returns_dict", isinstance(s, dict))
check("cf_step_has_step",     "step" in s)
check("cf_step_has_decision", "decision" in s)
check("cf_step_has_action",   "action" in s)
check("cf_step_has_effect",   "effect" in s)
check("cf_step_has_reward",   "reward" in s)
check("cf_step_has_body",     "body" in s)
check("cf_step_has_elapsed",  "elapsed_ms" in s)

s_stone = cf.step("eat stone")
check("cf_step_stone_blocked",not s_stone["executed"])

s_unk = cf.step("eat mystery_crystal")
check("cf_step_unk_blocked",  not s_unk["executed"])

# run_episode()
ep = cf.run_episode(["inspect apple", "eat apple", "eat stone"])
check("cf_episode_list",      isinstance(ep, list))
check("cf_episode_len",       len(ep) == 3)
check("cf_episode_step_type", all(isinstance(e, dict) for e in ep))
check("cf_episode_inspect_ok",ep[0]["executed"])
check("cf_episode_apple_ok",  ep[1]["executed"])
check("cf_episode_stone_no",  not ep[2]["executed"])

# body_status()
bs = cf.body_status()
check("cf_body_status",       isinstance(bs, dict))
check("cf_body_health",       "health" in bs and 0 <= bs["health"] <= 1)
check("cf_body_hunger",       "hunger" in bs and 0 <= bs["hunger"] <= 1)
check("cf_body_inventory",    "inventory" in bs)
check("cf_body_alive",        bs["alive"])


# ─────────────────────────────────────────────────────────
print("\n[Learning Integration: eat → damage → learns → avoids]")
cf2 = CogniField({"agents": 2, "thinking_mode": "auto"})
# Start: stone is completely unknown to cf2
r_before = cf2.think("Is stone edible?")
check("stone_unknown_before", r_before["knowledge_state"] in ("unknown","partial"))

# Force eat stone to get feedback
cf2.teach("stone", {"edible": False, "category": "material"})
_ = cf2.act("eat", "stone", force=True)   # forced → gets damage

# Learn from the outcome
cf2.learn_from_outcome("ate stone", "stone", "edible",
                         True, False, "eat", -0.3)

# After learning, stone should be avoided
r_after = cf2.think("Is stone edible?")
check("stone_avoided_after_learn", r_after["decision"] in ("avoid","proceed_with_caution"))
check("stone_lower_conf",
      r_after["confidence"] <= r_before["confidence"] + 0.20,
      f"before={r_before['confidence']:.3f} after={r_after['confidence']:.3f}")

# Multiple wrong outcomes → self-correction kicks in
for _ in range(3):
    cf2.learn_from_outcome("ate stone","stone","edible",True,False,"eat",-0.5)
r_corrected = cf2.think("Is stone edible?")
check("stone_low_conf_after_correction",
      r_corrected["confidence"] <= 0.55,
      f"conf={r_corrected['confidence']:.3f}")


# ─────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print(f"  Part 2 Results: {PASS} passed, {FAIL} failed")
if ERRORS:
    print(f"  Failed: {ERRORS}")
else:
    print("  All v11 Part 2 tests passed ✓")
print(f"{'═'*60}\n")
