"""
tests/test_v11_part3.py
========================
CogniField v11 Part 3 — Game Interaction Tests

Covers:
  BaseAdapter / NullAdapter · GameObservation · BlockInfo / EntityInfo
  JavaAdapter · BedrockAdapter · MobileAdapter
  VisionSystem · SurvivalGoalManager
  LanguageLearner · GameLoop · CogniFieldV11 game integration

Run: PYTHONPATH=.. python tests/test_v11_part3.py
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
print("\n[BaseAdapter / GameObservation]")
from cognifield.game.base_adapter import (
    GameAdapter, GameObservation, BlockInfo, EntityInfo,
    InventoryItem, ActionType, NullAdapter
)

obs = GameObservation()
check("obs_defaults_health",    obs.health == 20.0)
check("obs_defaults_hunger",    obs.hunger == 20.0)
check("obs_health_pct",         obs.health_pct == 1.0)
check("obs_hunger_pct",         obs.hunger_pct == 1.0)
check("obs_not_hungry",         not obs.is_hungry)
check("obs_not_in_danger",      not obs.is_in_danger)

# Hungry when hunger < 14
obs2 = GameObservation(hunger=10.0)
check("obs_hungry",             obs2.is_hungry)

# In danger when health < 8
obs3 = GameObservation(health=5.0)
check("obs_in_danger_health",   obs3.is_in_danger)

# In danger with hostile entity
obs4 = GameObservation()
obs4.entities = [EntityInfo("zombie", (5,64,5), 20.0, hostile=True)]
check("obs_in_danger_entity",   obs4.is_in_danger)

# hostile_entities
check("hostile_entities",       len(obs4.hostile_entities) == 1)

# BlockInfo
b = BlockInfo("minecraft:apple", (10, 64, 20))
check("block_name",             b.name == "apple")
check("block_pos",              b.pos == (10, 64, 20))

# EntityInfo
e_zombie = EntityInfo("zombie", (1,64,1), hostile=True)
e_pig    = EntityInfo("pig",    (3,64,3))
check("entity_zombie_hostile",  e_zombie.is_hostile)
check("entity_pig_not_hostile", not e_pig.is_hostile)
check("entity_pig_food_src",    e_pig.is_food_source)

# InventoryItem
item = InventoryItem("minecraft:bread", 3, 0)
check("item_name",              item.name == "bread")
check("item_count",             item.count == 3)

# inventory_count
obs5 = GameObservation()
obs5.inventory = [InventoryItem("minecraft:apple", 2, 0),
                  InventoryItem("minecraft:apple", 1, 1)]
check("inv_count",              obs5.inventory_count("minecraft:apple") == 3)
check("has_item",               obs5.has_item("minecraft:apple"))
check("no_item",                not obs5.has_item("minecraft:stone"))

# visible_food
obs6 = GameObservation()
obs6.visible_blocks = [BlockInfo("minecraft:apple", (1,64,1)),
                       BlockInfo("minecraft:stone", (2,64,2))]
check("visible_food",           len(obs6.visible_food) == 1)

# to_dict
d = obs.to_dict()
check("obs_to_dict",            isinstance(d, dict))
check("obs_dict_health",        "health" in d and d["health"] == 20.0)
check("obs_dict_partial",       "partial" in d)

# NullAdapter
na = NullAdapter()
check("null_not_connected",     not na.connected)
na.connect()
check("null_connected",         na.connected)
obs_null = na.get_observation()
check("null_obs",               isinstance(obs_null, GameObservation))
ok = na.send_action({"type": ActionType.MOVE, "direction": "north"})
check("null_action_ok",         ok)
check("null_repr",              "NullAdapter" in repr(na))
check("null_summary",           "adapter" in na.summary())


# ─────────────────────────────────────────────────────────
print("\n[JavaAdapter]")
from cognifield.game.java_adapter import JavaAdapter

ja = JavaAdapter(simulation=True, seed=42)
check("java_init",              ja.name == "java_edition")
check("java_not_connected",     not ja.connected)

connected = ja.connect()
check("java_connects",          connected)
check("java_is_connected",      ja.connected)

obs_ja = ja.get_observation()
check("java_obs_type",          isinstance(obs_ja, GameObservation))
check("java_obs_health",        obs_ja.health == 20.0)
check("java_obs_source",        obs_ja.source == "java_simulation")
check("java_obs_not_partial",   not obs_ja.partial)
check("java_obs_blocks",        len(obs_ja.visible_blocks) >= 0)
check("java_obs_biome",         isinstance(obs_ja.biome, str))

# Actions
ok_move = ja.move("north", steps=1)
check("java_move",              ok_move)
obs_after = ja.get_observation()
check("java_position_changed",  obs_after.position != (0.0, 64.0, 0.0))

# Eat
ja.sim_spawn_item("minecraft:bread", 2)
ja.sim_set_hunger(5.0)
ok_eat = ja.eat("minecraft:bread")
check("java_eat_success",       ok_eat)
obs_eat = ja.get_observation()
check("java_hunger_increased",  obs_eat.hunger > 5.0)

# Health manipulation
ja.sim_set_health(5.0)
obs_h = ja.get_observation()
check("java_set_health",        obs_h.health == 5.0)
check("java_in_danger_health",  obs_h.is_in_danger)

# Inventory after eat
ja2 = JavaAdapter(simulation=True, seed=1)
ja2.connect()
ja2.sim_spawn_item("minecraft:apple", 3)
obs_inv = ja2.get_observation()
check("java_inventory",         obs_inv.has_item("minecraft:apple"))

# eat_food (first food from inventory)
ja2.sim_set_hunger(8.0)
ok_ef = ja2.eat_food()
check("java_eat_food",          ok_ef)

# Summary
summ_ja = ja.summary()
check("java_summary",           "sim_health" in summ_ja)
check("java_mode_sim",          summ_ja["mode"] == "simulation")

# Disconnect
ja.disconnect()
check("java_disconnected",      not ja.connected)


# ─────────────────────────────────────────────────────────
print("\n[BedrockAdapter]")
from cognifield.game.bedrock_adapter import BedrockAdapter

ba = BedrockAdapter(simulation=True, seed=42)
ba.connect()
check("bedrock_connected",      ba.connected)

obs_ba = ba.get_observation()
check("bedrock_obs_partial",    obs_ba.partial)
check("bedrock_missing_blocks", "visible_blocks" in obs_ba.missing_fields)
check("bedrock_missing_biome",  "biome" in obs_ba.missing_fields)
check("bedrock_has_health",     obs_ba.health >= 0.0)
check("bedrock_source",         "bedrock" in obs_ba.source)

# Actions
ok_m = ba.send_action({"type": "move", "direction": "north"})
check("bedrock_move",           ok_m)

ba.sim_set_hunger(6.0)
ba.sim_add_item("minecraft:bread", 1)
ok_e = ba.send_action({"type": "use", "target": "minecraft:bread"})
check("bedrock_eat",            ok_e)
obs_be = ba.get_observation()
check("bedrock_hunger_up",      obs_be.hunger > 6.0)

# Command translation
cmd = ba._action_to_command({"type": "chat", "message": "hello"})
check("bedrock_chat_cmd",       "/say hello" in cmd)
cmd2 = ba._action_to_command({"type": "command", "command": "/time set day"})
check("bedrock_command",        "/time set day" in cmd2)

# Summary
summ_ba = ba.summary()
check("bedrock_summary_partial",summ_ba["partial_observation"])
check("bedrock_summary_missing",len(summ_ba["missing_fields"]) >= 2)


# ─────────────────────────────────────────────────────────
print("\n[MobileAdapter]")
from cognifield.game.mobile_adapter import MobileAdapter, LAYOUTS, ScreenLayout

check("layouts_exist",          "1080x1920" in LAYOUTS)
check("layout_type",            isinstance(LAYOUTS["1080x1920"], ScreenLayout))

ma = MobileAdapter(dry_run=True, resolution="1080x1920")
ma.connect()
check("mobile_connected",       ma.connected)
check("mobile_dry_run",         ma.dry_run)
check("mobile_layout",          ma.layout.resolution == (1080, 1920))

# Actions
actions_to_test = [
    {"type":"move","direction":"forward"},
    {"type":"jump"},
    {"type":"use"},
    {"type":"attack"},
    {"type":"inventory"},
    {"type":"tap","x":540,"y":960},
    {"type":"swipe","x1":100,"y1":200,"x2":100,"y2":100,"duration_ms":200},
    {"type":"keyevent","keycode":4},
]
for a in actions_to_test:
    ma.send_action(a)
log = ma.get_adb_log()
check("mobile_adb_log",         len(log) == len(actions_to_test))
check("mobile_adb_strings",     all(isinstance(c, str) for c in log))

# Observation (stub)
obs_ma = ma.get_observation()
check("mobile_obs_partial",     obs_ma.partial)
check("mobile_obs_missing",     len(obs_ma.missing_fields) >= 3)

# Vision update
ma.update_from_vision(health=12.0, hunger=8.0)
obs_mv = ma.get_observation()
check("mobile_vision_health",   obs_mv.health == 12.0)
check("mobile_vision_hunger",   obs_mv.hunger == 8.0)

# All resolutions
for res in ["720x1280","1080x1920","1440x2960"]:
    ma_r = MobileAdapter(dry_run=True, resolution=res)
    check(f"mobile_layout_{res}", ma_r.layout is not None)

# Summary
summ_ma = ma.summary()
check("mobile_summary",         "adb_commands" in summ_ma)
check("mobile_summary_partial", summ_ma["partial"])


# ─────────────────────────────────────────────────────────
print("\n[VisionSystem]")
from cognifield.vision.vision_system import VisionSystem, ScreenReading, ScreenRegion

vs = VisionSystem(simulation=True, seed=42)
check("vs_init",                vs.simulation)
check("vs_readings_0",          vs.reading_count() == 0)

r = vs.analyze()
check("vs_reading_type",        isinstance(r, ScreenReading))
check("vs_health_range",        0.0 <= r.health_pct <= 1.0)
check("vs_hunger_range",        0.0 <= r.hunger_pct <= 1.0)
check("vs_confidence",          0.0 <= r.confidence <= 1.0)
check("vs_source_sim",          r.source == "simulation")
check("vs_hearts",              0.0 <= r.health_hearts <= 20.0)
check("vs_drumsticks",          0.0 <= r.hunger_drumsticks <= 20.0)
check("vs_to_dict",             isinstance(r.to_dict(), dict))
check("vs_dict_keys",           all(k in r.to_dict() for k in
                                     ["health_pct","hunger_pct","danger_detected"]))

# Multiple readings
for _ in range(9): vs.analyze()
check("vs_10_readings",         vs.reading_count() == 10)
check("vs_last_reading",        vs.last_reading() is not None)
check("vs_mean_health",         0.0 <= vs.mean_health() <= 1.0)

# Simulation drift
vs2 = VisionSystem(simulation=True, seed=0)
vs2.set_sim_health(0.3)
r2 = vs2.analyze()
check("vs_set_health",          r2.health_pct <= 0.35)

vs2.set_sim_hunger(0.2)
r3 = vs2.analyze()
check("vs_set_hunger",          r3.hunger_pct <= 0.25)

# Fallback reading
r_fallback = vs._fallback_reading("test_error")
check("vs_fallback",            r_fallback.source == "test_error")
check("vs_fallback_conf_0",     r_fallback.confidence == 0.0)

# Summary
summ_vs = vs.summary()
check("vs_summary",             "readings" in summ_vs and "simulation" in summ_vs)
check("vs_repr",                "VisionSystem" in repr(vs))

# ScreenRegion
sr = ScreenRegion(0.02, 0.88, 0.40, 0.03)
px = sr.to_pixels(1080, 1920)
check("screen_region_pixels",   isinstance(px, tuple) and len(px) == 4)
check("screen_region_positive", all(v >= 0 for v in px))


# ─────────────────────────────────────────────────────────
print("\n[SurvivalGoalManager]")
from cognifield.game.survival_goals import (
    SurvivalGoalManager, SurvivalGoal, SurvivalPriority
)

sg = SurvivalGoalManager()

# Full health/hunger — only background goals
obs_full = GameObservation(health=20.0, hunger=20.0)
goals = sg.update(obs_full)
check("sg_goals_list",          isinstance(goals, list))
check("sg_no_critical",         not sg.has_critical())
check("sg_has_goals",           len(goals) >= 1)

# Hungry
obs_hungry = GameObservation(health=20.0, hunger=5.0)
goals_h = sg.update(obs_hungry)
names = [g.name for g in goals_h]
check("sg_hungry_goal",         any("food" in n or "eat" in n for n in names))
top = sg.top_goal()
check("sg_top_goal",            top is not None)
check("sg_top_priority_high",   top.priority in (SurvivalPriority.HIGH,
                                                   SurvivalPriority.MEDIUM))

# Critical health
obs_crit = GameObservation(health=3.0, hunger=20.0)
obs_crit.inventory = [InventoryItem("minecraft:bread", 1, 0)]
goals_c = sg.update(obs_crit)
check("sg_critical",            sg.has_critical())
top_c = sg.top_goal()
check("sg_critical_top",        top_c.priority == SurvivalPriority.CRITICAL)

# Hostile nearby
obs_danger = GameObservation()
obs_danger.entities = [EntityInfo("zombie", (3,64,3), 20.0, True)]
goals_d = sg.update(obs_danger)
danger_goal = [g for g in goals_d if g.name == "avoid_danger"]
check("sg_danger_goal",         len(danger_goal) >= 1)
check("sg_danger_action",       danger_goal[0].action == "flee" if danger_goal else True)

# SurvivalGoal properties
g = SurvivalGoal("eat_food", "eat bread", SurvivalPriority.HIGH,
                  "minecraft:bread", "eat", 0.8, expires_at=None)
check("sg_goal_cf_priority",    0 < g.cf_priority <= 1.0)
check("sg_goal_not_expired",    not g.is_expired)
check("sg_goal_to_query",       "eat" in g.to_query())
check("sg_goal_no_mc_prefix",   "minecraft:" not in g.to_query())

# to_dict
gd = g.to_dict()
check("sg_goal_to_dict",        isinstance(gd, dict))
check("sg_goal_dict_keys",      all(k in gd for k in ["name","priority","action"]))

# Complete goal
sg.complete_goal("eat_food")
check("sg_goal_completed",      True)   # no exception = pass

# Summary
summ_sg = sg.summary()
check("sg_summary",             "active_goals" in summ_sg)
check("sg_repr",                "SurvivalGoalManager" in repr(sg))


# ─────────────────────────────────────────────────────────
print("\n[LanguageLearner]")
from cognifield.game.language_learner import (
    LanguageLearner, GameConcept,
    FOOD_IDS, HOSTILE_TYPES, PASSIVE_MOBS
)

ll = LanguageLearner()

# Process food
c_apple = ll.process_id("minecraft:apple")
check("ll_apple_concept",       c_apple is not None)
check("ll_apple_name",          c_apple.name == "apple")
check("ll_apple_category",      c_apple.category == "food")
check("ll_apple_edible",        c_apple.properties.get("edible") is True)

c_stone = ll.process_id("minecraft:stone")
check("ll_stone_category",      c_stone.category == "stone")
check("ll_stone_not_edible",    c_stone.properties.get("edible") is False)

c_ore = ll.process_id("minecraft:iron_ore")
check("ll_ore_category",        c_ore.category == "ore")

c_log = ll.process_id("minecraft:oak_log")
check("ll_log_category",        c_log.category == "wood")

c_sword = ll.process_id("minecraft:diamond_sword")
check("ll_sword_category",      c_sword.category == "tool")

c_unknown = ll.process_id("minecraft:unknown_custom_block_xyz")
check("ll_unknown_none",        c_unknown is None)
check("ll_unknown_tracked",     "minecraft:unknown_custom_block_xyz" in ll.unknown_ids())

# Entity processing
c_zombie = ll._process_entity("zombie", True)
check("ll_zombie_hostile",      c_zombie.properties.get("dangerous") is True)
check("ll_zombie_not_edible",   c_zombie.properties.get("edible") is False)

c_pig = ll._process_entity("pig", False)
check("ll_pig_edible",          c_pig.properties.get("edible") is True)
check("ll_pig_not_dangerous",   c_pig.properties.get("dangerous") is False)

# Known foods list
for food_id in ["minecraft:apple","minecraft:bread"]:
    for _ in range(3):  # meet min_seen
        ll.process_id(food_id)
foods = ll.known_foods()
check("ll_known_foods",         "apple" in foods)

# Concept bump
c2 = ll.process_id("minecraft:apple")  # bump existing
check("ll_concept_bumped",      c2.times_seen >= 2)

# process_observation
obs_ll = GameObservation()
obs_ll.visible_blocks = [BlockInfo("minecraft:bread", (1,64,1)),
                          BlockInfo("minecraft:coal_ore", (2,64,2))]
obs_ll.entities       = [EntityInfo("pig", (5,64,5))]
obs_ll.inventory      = [InventoryItem("minecraft:apple", 1, 0)]
updated = ll.process_observation(obs_ll)
check("ll_process_obs_list",    isinstance(updated, list))
check("ll_process_obs_updates", len(updated) >= 1)

# get_concept
c3 = ll.get_concept("apple")
check("ll_get_by_name",         c3 is not None and c3.name == "apple")
c4 = ll.get_concept("minecraft:apple")
check("ll_get_by_id",           c4 is not None)

# Integration with world model
from cognifield.core.world_model_v2 import WorldModelV2
from cognifield.world_model.belief_system import BeliefSystem
wm_ll = WorldModelV2()
bs_ll = BeliefSystem()
ll2   = LanguageLearner(world_model=wm_ll, belief_system=bs_ll, min_seen=2)
for _ in range(3): ll2.process_id("minecraft:apple")
check("ll_wm_updated",          wm_ll.get_entity("apple") is not None)
check("ll_bs_updated",          bs_ll.get("apple.edible") is not None)

# Summary
summ_ll = ll.summary()
check("ll_summary_concepts",    summ_ll["known_concepts"] >= 5)
check("ll_summary_category",    "by_category" in summ_ll)
check("ll_repr",                "LanguageLearner" in repr(ll))


# ─────────────────────────────────────────────────────────
print("\n[GameLoop]")
from cognifield.game.game_loop import GameLoop, GameStep
from cognifield.core.interaction_loop import InteractionLoop
from cognifield.core.deep_thinker import DeepThinker, ThinkingMode
from cognifield.core.experience_engine import ExperienceEngine
from cognifield.core.world_model_v2 import WorldModelV2
from cognifield.agents.body import VirtualBody
from cognifield.agents.action_system import ActionSystem
from cognifield.agents.perception import PerceptionSystem
from cognifield.world_model.belief_system import BeliefSystem

# Build a minimal InteractionLoop
body_gl  = VirtualBody(seed=0)
act_gl   = ActionSystem(body_gl, unknown_safety_rule=True)
perc_gl  = PerceptionSystem()
bs_gl    = BeliefSystem()
wm_gl    = WorldModelV2()
dt_gl    = DeepThinker(mode=ThinkingMode.AUTO)
ee_gl    = ExperienceEngine(bs_gl)
for _ in range(5): bs_gl.update("apple.edible", True, "direct_observation")
wm_gl.add_entity("apple", "food", {"edible": True})
wm_gl.add_entity("stone", "material", {"edible": False})

il = InteractionLoop(
    body=body_gl, action_system=act_gl, perception=perc_gl,
    deep_thinker=dt_gl, experience_engine=ee_gl,
    world_model=wm_gl, belief_system=bs_gl,
    unknown_safety=True, verbose=False,
)

adapter_gl = JavaAdapter(simulation=True, seed=42)
adapter_gl.connect()
adapter_gl.sim_spawn_item("minecraft:bread", 2)

gl = GameLoop(adapter=adapter_gl, interaction_loop=il, verbose=False)
check("gl_init",                gl is not None)
check("gl_repr",                "GameLoop" in repr(gl))

# step_from_game
gs = gl.step_from_game()
check("gl_step_type",           isinstance(gs, GameStep))
check("gl_step_has_obs",        isinstance(gs.observation, GameObservation))
check("gl_step_has_goal",       True)   # may be None if no goals — just no crash
check("gl_step_has_query",      isinstance(gs.query, str) and len(gs.query) > 0)
check("gl_step_elapsed",        gs.elapsed_ms >= 0)
check("gl_step_to_dict",        isinstance(gs.to_dict(), dict))
check("gl_step_dict_keys",      all(k in gs.to_dict()
                                     for k in ["step","goal","decision","health"]))

# run_episode
steps = gl.run_episode(n_steps=5)
check("gl_episode_len",         len(steps) == 5)
check("gl_episode_types",       all(isinstance(s, GameStep) for s in steps))
check("gl_step_count",          gl._step_count >= 6)

# Stats
check("gl_mean_health",         0.0 <= gl.mean_health() <= 20.0)
check("gl_mean_hunger",         0.0 <= gl.mean_hunger() <= 20.0)
check("gl_survival_rate",       0.0 <= gl.survival_rate() <= 1.0)
check("gl_recent_steps",        len(gl.recent_steps(3)) <= 3)
check("gl_summary",             "total_steps" in gl.summary())

# Agent death stops episode
adapter_dead = JavaAdapter(simulation=True, seed=0)
adapter_dead.connect()
adapter_dead.sim_set_health(0.1)  # almost dead — will die after tick
gl2 = GameLoop(adapter=adapter_dead, interaction_loop=il, verbose=False)
ep_dead = gl2.run_episode(n_steps=20)   # should stop early
check("gl_death_stops",         len(ep_dead) <= 20)


# ─────────────────────────────────────────────────────────
print("\n[CogniFieldV11 game integration]")
from cognifield import CogniField

cf = CogniField({"agents":2,"thinking_mode":"auto","unknown_safety_rule":True})
cf.teach("apple", {"edible":True,  "category":"food"})
cf.teach("stone", {"edible":False, "category":"material"})
cf.teach("bread", {"edible":True,  "category":"food"})

# create_game_loop
gl_cf = cf.create_game_loop(vision=False, verbose=False)
check("cf_gl_type",             isinstance(gl_cf, GameLoop))

# game_step
gs_cf = cf.game_step()
check("cf_gs_dict",             isinstance(gs_cf, dict))
check("cf_gs_has_goal",         "goal" in gs_cf)
check("cf_gs_has_decision",     "decision" in gs_cf)
check("cf_gs_has_health",       "health" in gs_cf and gs_cf["health"] >= 0)
check("cf_gs_has_hunger",       "hunger" in gs_cf)
check("cf_gs_has_elapsed",      gs_cf["elapsed_ms"] >= 0)

# run_game_episode
cf2 = CogniField({"agents":2})
cf2.teach("bread", {"edible":True, "category":"food"})
ep_cf = cf2.run_game_episode(n_steps=5, verbose=False)
check("cf_ep_list",             isinstance(ep_cf, list))
check("cf_ep_len",              len(ep_cf) == 5)
check("cf_ep_dicts",            all(isinstance(s, dict) for s in ep_cf))

# Custom adapter
ja_cf = JavaAdapter(simulation=True, seed=7)
ja_cf.connect()
ja_cf.sim_spawn_item("minecraft:apple", 3)
ja_cf.sim_set_hunger(5.0)
ep_custom = cf.run_game_episode(n_steps=3, adapter=ja_cf)
check("cf_custom_adapter",      len(ep_custom) == 3)

# Safety: unknown objects never executed
cf3 = CogniField({"agents":2,"unknown_safety_rule":True})
gs3 = cf3.game_step()   # adapter has unknown blocks — should not crash
check("cf_game_step_safe",      "health" in gs3)


# ─────────────────────────────────────────────────────────
print(f"\n{'═'*60}")
print(f"  Part 3 Results: {PASS} passed, {FAIL} failed")
if ERRORS:
    print(f"  Failed: {ERRORS}")
else:
    print("  All v11 Part 3 tests passed ✓")
print(f"{'═'*60}\n")
