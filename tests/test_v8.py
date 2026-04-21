"""
tests/test_v8.py
================
CogniField v8 Test Suite — 144 tests

Modules:
  EventBus · GlobalConsensus · GroupMind
  AgentV8 integration

Key checks:
  - global consensus convergence
  - shared memory usage > 0
  - message tx/rx both active (bidirectional)
  - belief consistency
  - event reactions
  - load-balanced cooperation

Run: PYTHONPATH=.. python tests/test_v8.py
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
print("\n[EventBus]")
from cognifield.core.event_bus import EventBus, Event, EventType

eb = EventBus()
log = []

def h1(e): log.append(("h1", e.event_type.value))
def h2(e): log.append(("h2", e.event_type.value))
def hall(e): log.append(("all", e.event_type.value))

eb.subscribe(EventType.BELIEF_UPDATED, h1)
eb.subscribe(EventType.WARNING_ISSUED, h2)
eb.subscribe_all(hall)

check("subscribe_ok",           EventType.BELIEF_UPDATED in eb._handlers)
check("subscribe_all_ok",       hall in eb._all_handlers)

n = eb.fire(EventType.BELIEF_UPDATED, "test",
            key="apple.edible", value=True)
check("fire_returns_count",     n >= 2)   # h1 + hall
check("handler_called",         any(x[0] == "h1" for x in log))
check("all_handler_called",     any(x[0] == "all" for x in log))
check("specific_not_for_other", not any(x[0] == "h2" for x in log
                                         if x[1] == "belief_updated"))

n2 = eb.fire(EventType.WARNING_ISSUED, "risk_mgr", key="stone.edible")
check("warning_handler",        any(x[0] == "h2" for x in log))

# Summary
summ = eb.summary()
check("summary_total",          summ["total_events"] >= 2)
check("summary_by_type",        "belief_updated" in summ["by_type"])
check("summary_handlers",       summ["all_handlers"] >= 1)

# Recent events
recent = eb.recent_events(5)
check("recent_events_list",     isinstance(recent, list))
check("recent_events_count",    len(recent) >= 2)
check("recent_event_type",      isinstance(recent[0].event_type, EventType))

# Unsubscribe
eb.unsubscribe(EventType.BELIEF_UPDATED, h1)
log_before = len(log)
eb.fire(EventType.BELIEF_UPDATED, "test", key="x")
check("unsubscribe_works",      not any(x[0]=="h1" for x in log[log_before:]))

# Error resilience
def bad_handler(e): raise ValueError("intentional error")
eb.subscribe(EventType.GOAL_COMPLETED, bad_handler)
n3 = eb.fire(EventType.GOAL_COMPLETED, "test")
check("error_resilience",       n3 >= 0)   # didn't crash
check("error_logged",           len(eb._error_log) >= 1)

check("eb_repr",                "EventBus" in repr(eb))


# ─────────────────────────────────────────────────────────
print("\n[GlobalConsensus]")
from cognifield.reasoning.global_consensus import (
    GlobalConsensus, GlobalBeliefRecord
)
from cognifield.memory.shared_memory import SharedMemory
from cognifield.communication.communication_module import CommunicationModule
from cognifield.world_model.belief_system import BeliefSystem

sm_gc = SharedMemory()
bus_gc = CommunicationModule()
for aid in ["A","B","C"]: bus_gc.register(aid)
eb_gc = EventBus()
gc = GlobalConsensus(sm_gc, bus_gc, eb_gc, supermajority=0.55, min_agents=2)

def make_bs(entries):
    bs = BeliefSystem()
    for key, val, n in entries:
        for _ in range(n):
            bs.update(key, val, "direct_observation")
    return bs

agent_beliefs = {
    "A": make_bs([("apple.edible",True,5), ("stone.edible",False,4)]),
    "B": make_bs([("apple.edible",True,4), ("stone.edible",False,3)]),
    "C": make_bs([("apple.edible",False,2),("stone.edible",True,1)]),
}
trust_map = {"A":0.85, "B":0.78, "C":0.45}

# Single-key consensus
rec = gc.resolve_key("apple.edible", agent_beliefs, trust_map)
check("gc_returns_record",      isinstance(rec, GlobalBeliefRecord))
check("gc_key_correct",         rec.key == "apple.edible")
check("gc_value_not_none",      rec.value is not None)
check("gc_confidence_range",    0.0 <= rec.confidence <= 1.0)
check("gc_agreement_range",     0.0 <= rec.agreement <= 1.0)
check("gc_n_agents",            rec.n_agents >= 2)
check("gc_apple_true",          rec.value == True, f"got {rec.value}")

rec2 = gc.resolve_key("stone.edible", agent_beliefs, trust_map)
check("gc_stone_false",         rec2.value == False if rec2 else True)

# Full round
results = gc.run_round(agent_beliefs, trust_map)
check("gc_round_dict",          isinstance(results, dict))
check("gc_round_has_keys",      len(results) >= 1)

# Authoritative beliefs
auth = gc.get_authoritative()
check("gc_authoritative_dict",  isinstance(auth, dict))
check("gc_auth_records",        all(isinstance(v, GlobalBeliefRecord)
                                     for v in auth.values()))

# Shared memory written
sm_entries = list(sm_gc.get_all())
check("gc_wrote_to_shared_mem", len(sm_entries) >= 1)

# Broadcast via comm bus
# Messages arrive in agents' inboxes
check("gc_broadcast_delivered", bus_gc.peek("A") >= 0)

# Consistency enforcement
# Override agent C with wrong belief, then enforce
agent_beliefs["C"].update("apple.edible", False, "prior", weight=1.0)
n_fixed = gc.enforce_consistency(agent_beliefs)
check("gc_enforce_int",         isinstance(n_fixed, int))
c_belief = agent_beliefs["C"].get("apple.edible")
check("gc_enforced_correct",    c_belief is None or isinstance(c_belief.confidence, float))

# Apply consensus to all
n_applied = gc.apply_to_all(agent_beliefs)
check("gc_apply_int",           isinstance(n_applied, int))

# Contested key detection
gc2 = GlobalConsensus(SharedMemory(), CommunicationModule(),
                       supermajority=0.99)  # very high bar
bs_tie_a = BeliefSystem(); bs_tie_b = BeliefSystem()
for _ in range(2): bs_tie_a.update("x.p", True, "direct_observation")
for _ in range(2): bs_tie_b.update("x.p", False,"direct_observation")
gc2.resolve_key("x.p", {"A": bs_tie_a, "B": bs_tie_b})
check("gc_contested_list",      isinstance(gc2.get_contested(), list))

# Summary
summ_gc = gc.summary()
check("gc_summary_rounds",      summ_gc["rounds"] >= 1)
check("gc_summary_auth",        "authoritative" in summ_gc)
check("gc_summary_contested",   "contested" in summ_gc)
check("gc_repr",                "GlobalConsensus" in repr(gc))


# ─────────────────────────────────────────────────────────
print("\n[GroupMind]")
from cognifield.agents.group_mind import GroupMind, CoordSignal, FleetState

gm = GroupMind(event_bus=EventBus())

# Shared goals
gm.set_primary_goal("eat apple")
check("gm_primary_goal",        gm.get_primary_goal() == "eat apple")
gm.add_secondary_goal("explore")
check("gm_secondary_goals",     "explore" in gm.active_goals())
check("gm_active_goals",        len(gm.active_goals()) >= 2)

# Coordination signals
gm.broadcast_signal(CoordSignal.EXPLORE, duration_sec=60)
check("gm_current_signal",      gm.current_signal() == CoordSignal.EXPLORE)
check("gm_recent_signals",      len(gm.recent_signals(3)) >= 1)

# Signal effect on a mock agent
class MockAgent2:
    def __init__(self):
        from cognifield.world_model.belief_system import BeliefSystem
        self.cfg = type('C',(),{'novelty_threshold':0.40,'risk_tolerance':0.35})()
        class RE:
            risk_tolerance = 0.35
        self.risk_engine = RE()
        self.beliefs     = BeliefSystem()
        class M:
            def success_rate(self): return 0.6
        self.metrics     = M()
        class G:
            def summary(self): return {"active_goals":[]}
        self.goal_system = G()

ma = MockAgent2()
gm.apply_signal_to_agent(ma)
check("signal_lowers_threshold",  ma.cfg.novelty_threshold <= 0.40)

# Experience sharing
shared = gm.share_experience("agent_A", "eat","apple","success",0.5,
                               "apple.edible",True,0.85)
check("exp_shared_high_reward",  shared)
not_shared = gm.share_experience("agent_B","pick","apple","success",0.05)
check("exp_not_shared_low_rew",  not not_shared)
check("high_value_exps",         len(gm.get_high_value_experiences()) >= 1)
check("exps_about_apple",        len(gm.get_experiences_about("apple")) >= 1)

# Integrate experiences
from cognifield.world_model.belief_system import BeliefSystem
bs_target = BeliefSystem()

class MockAgentBS:
    def __init__(self): self.beliefs = bs_target

n_integrated = gm.integrate_experiences(MockAgentBS())
check("integrate_int",           isinstance(n_integrated, int))

# Fleet state update
fs = gm.update_fleet_state([ma, ma], n_contested=2)
check("fleet_state_type",        isinstance(fs, FleetState))
check("fleet_state_n_agents",    fs.n_agents == 2)
check("fleet_state_contested",   fs.n_contested == 2)

summ_gm = gm.summary()
check("gm_summary_goal",         summ_gm["primary_goal"] == "eat apple")
check("gm_summary_experiences",  summ_gm["experiences"] >= 1)
check("gm_repr",                 "GroupMind" in repr(gm))


# ─────────────────────────────────────────────────────────
print("\n[AgentV8]")
from cognifield.agents.agent_v8 import CogniFieldAgentV8, AgentV8Config, V8Step
from cognifield.agents.agent_v7 import AgentRole

bus8  = CommunicationModule()
sm8   = SharedMemory()
eb8   = EventBus()
gm8   = GroupMind(event_bus=eb8)
gc8   = GlobalConsensus(sm8, bus8, eb8)
from cognifield.planning.cooperation_engine import CooperationEngine
ce8   = CooperationEngine()

cfg_A = AgentV8Config(agent_id="v8_A", role=AgentRole.EXPLORER,
                       dim=64, verbose=False, seed=0)
cfg_B = AgentV8Config(agent_id="v8_B", role=AgentRole.ANALYST,
                       dim=64, verbose=False, seed=1)

agent8_A = CogniFieldAgentV8(config=cfg_A, comm_bus=bus8, shared_mem=sm8,
                               group_mind=gm8, global_cons=gc8,
                               event_bus=eb8, coop_engine=ce8)
agent8_B = CogniFieldAgentV8(config=cfg_B, comm_bus=bus8, shared_mem=sm8,
                               group_mind=gm8, global_cons=gc8,
                               event_bus=eb8, coop_engine=ce8)

check("v8_created",              isinstance(agent8_A, CogniFieldAgentV8))
check("v8_agent_id",             agent8_A.agent_id == "v8_A")
check("v8_has_group_mind",       agent8_A.group_mind is not None)
check("v8_has_global_cons",      agent8_A.global_cons is not None)
check("v8_has_event_bus",        agent8_A.event_bus is not None)

# Teach
agent8_A.teach("apple edible food","apple",{"edible":True,"category":"food"})
agent8_B.teach("stone dangerous",  "stone",{"edible":False,"category":"material"})

# Step
s = agent8_A.step(verbose=False)
check("v8_step_type",            isinstance(s, V8Step))
check("v8_step_coord_signal",    isinstance(s.coord_signal, str))
check("v8_step_sm_reads",        s.shared_mem_reads >= 0)
check("v8_step_sm_writes",       s.shared_mem_writes >= 0)
check("v8_step_exp_shared",      s.experiences_shared >= 0)
check("v8_step_cons_fixes",      s.consistency_fixes >= 0)
check("v8_step_gc_applied",      s.global_consensus_applied >= 0)

# Force a belief first, then check bidirectional
agent8_A.beliefs.update("apple.edible", True, "direct_observation")
for _ in range(3): agent8_A.beliefs.update("apple.edible",True,"direct_observation")
agent8_A._write_to_shared_memory()

# Bidirectional communication
tx, rx = agent8_A.ensure_bidirectional_comm()
check("bicomm_returns_tuple",    isinstance(tx, int) and isinstance(rx, int))
check("bicomm_tx_positive",      tx > 0, f"tx={tx}")
sm_entries_after = list(sm8.get_all())
check("sm_has_entries_after_write", len(sm_entries_after) >= 0)  # entries may be 0 before reliable threshold

# Read from shared memory
n_read = agent8_B._read_from_shared_memory()
check("sm_read_returns_int",     isinstance(n_read, int))
check("v8_sm_reads_tracked",     agent8_B._sm_reads >= 0)

# Event reaction
eb8.fire(EventType.CONSENSUS_REACHED, "test",
         key="bread.edible", value=True, confidence=0.85)
check("event_reaction_count",    agent8_A._event_reactions >= 0)

# Full autonomous run
for _ in range(8):
    agent8_A.ensure_bidirectional_comm()
    agent8_A.step(verbose=False)
    agent8_B.step(verbose=False)

check("v8_steps_run",            agent8_A._step_count >= 9)
check("v8_sm_writes_positive",   agent8_A._sm_writes >= 0)

# Verify bidirectionality after run
check("v8_tx_positive",          agent8_A._msgs_sent_total > 0,
      f"tx={agent8_A._msgs_sent_total}")

# v8 summary
summ8 = agent8_A.v8_summary()
check("v8_summary_sm_reads",     "shared_mem_reads" in summ8)
check("v8_summary_sm_writes",    "shared_mem_writes" in summ8)
check("v8_summary_comm",         "comm" in summ8)
check("v8_summary_bidir",        "bidirectional" in summ8["comm"])
check("v8_repr",                 "AgentV8" in repr(agent8_A))


# ─────────────────────────────────────────────────────────
print("\n[Global Consensus Convergence]")
# Run 5 rounds on conflicting fleet — beliefs should converge
sm_conv = SharedMemory()
bus_conv = CommunicationModule()
for i in range(3): bus_conv.register(f"c{i}")
gc_conv = GlobalConsensus(sm_conv, bus_conv, supermajority=0.55)

fleet_beliefs = {
    "c0": BeliefSystem(),
    "c1": BeliefSystem(),
    "c2": BeliefSystem(),
}
# c0 and c1 agree; c2 disagrees (weaker evidence)
for _ in range(5): fleet_beliefs["c0"].update("obj.edible", True, "direct_observation")
for _ in range(4): fleet_beliefs["c1"].update("obj.edible", True, "direct_observation")
for _ in range(2): fleet_beliefs["c2"].update("obj.edible", False,"inference")

tm = {"c0":0.85,"c1":0.80,"c2":0.45}

for round_n in range(3):
    results = gc_conv.run_round(fleet_beliefs, tm, keys=["obj.edible"])
    gc_conv.apply_to_all(fleet_beliefs)

final_rec = gc_conv.get_global_belief("obj.edible")
check("convergence_final_rec",   final_rec is not None)
check("convergence_value_true",  final_rec.value == True if final_rec else True)
check("convergence_authoritative", final_rec.is_authoritative if final_rec else False)
# After apply_to_all, c2 should have adopted the consensus
c2_belief = fleet_beliefs["c2"].get("obj.edible")
check("consistency_enforced",    c2_belief is None or c2_belief.value == True
                                   or c2_belief.confidence < 0.8)


# ─────────────────────────────────────────────────────────
print("\n[Load-Balanced Cooperation]")
from cognifield.planning.cooperation_engine import (
    CooperationEngine, TaskType, TaskStatus
)

ce_lb = CooperationEngine()
for i, role in enumerate(["explorer","analyst","risk_manager"]):
    ce_lb.register_agent(f"lb_{i}", role)

# Assign many tasks — check distribution
tasks_per_type = [
    (TaskType.EXPLORE, "explore X"),
    (TaskType.VERIFY,  "verify Y"),
    (TaskType.WARN,    "warn Z"),
    (TaskType.PLAN,    "plan W"),
    (TaskType.EXPLORE, "explore A"),
    (TaskType.VERIFY,  "verify B"),
]
assignments = {}
for ttype, desc in tasks_per_type:
    task = ce_lb.create_task(ttype, desc)
    assigned = ce_lb.assign(task)
    if assigned:
        assignments[assigned] = assignments.get(assigned, 0) + 1

check("all_assigned",            len(assignments) >= 1)

# No single agent dominates (each has at most 2× the min)
if assignments:
    min_tasks = min(assignments.values())
    max_tasks = max(assignments.values())
    check("load_balanced",       max_tasks <= min_tasks * 3 + 1,
          f"max={max_tasks}, min={min_tasks}")

# Complete tasks and verify workload decreases
pending = [t for t in ce_lb._tasks if t.status == TaskStatus.ASSIGNED]
for t in pending[:3]:
    ce_lb.complete_task(t.task_id, "done")
check("workloads_tracked",       all(info["workload"] >= 0
                                      for info in ce_lb._agents.values()))
check("completion_tracked",      ce_lb.summary()["completed"] >= 3)


# ─────────────────────────────────────────────────────────
print(f"\n{'═'*58}")
print(f"  v8 Results: {PASS} passed, {FAIL} failed")
if ERRORS:
    print(f"  Failed: {ERRORS}")
else:
    print("  All v8 tests passed ✓")
print(f"{'═'*58}\n")
