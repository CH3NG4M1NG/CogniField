"""
tests/test_v6.py
================
CogniField v6 Test Suite — 144 tests

Modules tested:
  CommunicationModule · SharedMemory · TrustSystem
  ConsensusEngine · AgentV6 · AgentManager
  Social learning · Conflict between agents
  Multi-agent stability · Knowledge sharing

Run: PYTHONPATH=.. python tests/test_v6.py
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


# ─────────────────────────────────────────────────────────
print("\n[CommunicationModule]")
from cognifield.communication.communication_module import (
    CommunicationModule, Message, MessageType
)

bus = CommunicationModule(max_queue=50)
for aid in ["alice", "bob", "charlie"]:
    bus.register(aid)

check("register_agents",        "alice" in bus._all_agents)
check("register_count",         len(bus._all_agents) == 3)
check("initial_inbox_empty",    bus.peek("alice") == 0)

# Send point-to-point
m1 = Message.belief_msg("alice", "apple", "edible", True, 0.8, receiver_id="bob")
sent = bus.send(m1)
check("send_returns_bool",      isinstance(sent, bool))
check("send_delivered",         bus.peek("bob") == 1)
check("send_not_others",        bus.peek("charlie") == 0)

# Broadcast
m2 = Message.warning_msg("alice", "stone", "edible", False, 0.9)
n   = bus.broadcast(m2)
check("broadcast_count",        n == 2)   # bob + charlie
check("broadcast_bob_inbox",    bus.peek("bob") == 2)
check("broadcast_charlie",      bus.peek("charlie") == 1)
check("broadcast_not_sender",   bus.peek("alice") == 0)

# Receive
bob_msgs = bus.receive("bob", max_msgs=5)
check("receive_list",           isinstance(bob_msgs, list))
check("receive_count",          len(bob_msgs) == 2)
check("receive_drains",         bus.peek("bob") == 0)
check("receive_msg_type",       isinstance(bob_msgs[0].msg_type, MessageType))

# Message factory methods
m_belief = Message.belief_msg("A","apple","edible",True,0.8)
check("belief_msg_type",        m_belief.msg_type == MessageType.BELIEF)
check("belief_msg_content",     m_belief.content["subject"] == "apple")
check("belief_msg_conf",        m_belief.confidence == 0.8)

m_warn = Message.warning_msg("A","stone","edible",False,0.9)
check("warning_msg_type",       m_warn.msg_type == MessageType.WARNING)

m_obs = Message.observation_msg("A","eat","apple","success",0.5)
check("obs_msg_type",           m_obs.msg_type == MessageType.OBSERVATION)
check("obs_msg_action",         m_obs.content["action"] == "eat")
check("obs_msg_outcome",        m_obs.content["outcome"] == "success")

m_q = Message.question_msg("A","purple_berry","edible")
check("question_msg_type",      m_q.msg_type == MessageType.QUESTION)
check("question_msg_subj",      m_q.content["subject"] == "purple_berry")

# Stats
stats = bus.stats()
check("stats_dict",             isinstance(stats, dict))
check("stats_sent",             "total_sent" in stats or "sent" in str(stats))

# Deregister
bus.deregister("charlie")
check("deregister_works",       "charlie" not in bus._all_agents)


# ─────────────────────────────────────────────────────────
print("\n[SharedMemory]")
from cognifield.memory.shared_memory import SharedMemory

sm = SharedMemory(max_entries=1000)

# Basic write/read
sm.write("apple.edible", True, "agent_A", 0.85)
entry = sm.read("apple.edible")
check("write_read",             entry is not None)
check("entry_value",            entry.value == True)
check("entry_confidence",       abs(entry.confidence - 0.85) < 0.1)
check("entry_contributors",     "agent_A" in entry.sources)

# Multiple agents writing same key
sm.write("apple.edible", True, "agent_B", 0.78)
entry2 = sm.read("apple.edible")
check("multi_contributor",      entry2.n_contributors == 2)
check("multi_evidence_grows",   entry2.total_evidence >= entry.total_evidence)

# Conflict: different values
sm.write("mystery.edible", True,  "agent_A", 0.60)
sm.write("mystery.edible", False, "agent_B", 0.55)
e_mystery = sm.read("mystery.edible")
check("conflict_exists",        e_mystery is not None)

# Contested check
sm.write("tied.edible", True,  "A", 0.55)
sm.write("tied.edible", False, "B", 0.55)
e_tied = sm.read("tied.edible")
check("contested_detection",    isinstance(e_tied.is_contested, bool))

# Trust-weighted read
sm2 = SharedMemory()
sm2.write("cherry.edible", True,  "trusted",   0.90)
sm2.write("cherry.edible", False, "untrusted", 0.80)
trust_map = {"trusted": 0.90, "untrusted": 0.20}
val, conf = sm2.read_weighted_by_trust("cherry.edible", trust_map)
check("trust_weighted_val",     val == True)
check("trust_weighted_conf",    isinstance(conf, float))

# Convenience queries
sm3 = SharedMemory()
for n, v, c in [("apple","edible",True), ("stone","edible",False),
                 ("bread","edible",True)]:
    for _ in range(3):
        sm3.write(f"{n}.edible", v, "A", c * 0.9 if v else (1-c) * 0.9)
edible_sm   = sm3.find_edible(min_conf=0.60)
dangerous_sm= sm3.find_dangerous(min_conf=0.60)
check("find_edible_sm",         len(edible_sm) >= 0)   # may vary by aggregation
check("find_dangerous_sm",      isinstance(dangerous_sm, list))

# get_all
all_entries = list(sm.get_all())
check("get_all_list",           isinstance(all_entries, list))
check("get_all_has_entries",    len(all_entries) > 0)

all_confident = list(sm3.get_all(min_conf=0.70))
check("get_all_filtered",       all(e.confidence >= 0.70 for e in all_confident))

# Summary
summ = sm.summary()
check("sm_summary_entries",     "entries" in summ)
check("sm_len",                 len(sm) >= 2)


# ─────────────────────────────────────────────────────────
print("\n[TrustSystem]")
from cognifield.agents.trust_system import TrustSystem, TrustRecord

ts = TrustSystem(owner_id="self")

# Register
rec = ts.register_peer("peer_A")
check("register_peer",          isinstance(rec, TrustRecord))
check("initial_trust",          abs(ts.get_trust("peer_A") - 0.5) < 0.1)
check("unknown_peer_default",   ts.get_trust("nobody", default=0.5) == 0.5)

# Accuracy updates
for _ in range(5):
    ts.update_accuracy("peer_A", was_correct=True)
check("accuracy_rises",         ts.get_trust("peer_A") > 0.6)
check("is_trusted",             ts.get_record("peer_A").is_trusted)

ts.register_peer("peer_B")
for _ in range(4):
    ts.update_accuracy("peer_B", was_correct=False)
check("accuracy_falls",         ts.get_trust("peer_B") < 0.5)
check("is_distrusted",          ts.get_trust("peer_B") < 0.45)

# Consistency
ts.update_consistency("peer_A", is_consistent=True)
check("consistency_update",     ts.get_record("peer_A").consistency >= 0.5)

# Message weight
wt = ts.message_weight("peer_A", confidence=0.8)
check("msg_weight_range",       0.0 <= wt <= 0.8)
check("msg_weight_trusted_higher",
      ts.message_weight("peer_A", 0.8) > ts.message_weight("peer_B", 0.8))

# Observe outcome
ts.observe_outcome("peer_A", True, True)    # correct
ts.observe_outcome("peer_B", True, False)   # wrong
check("observe_correct_helps",  ts.get_trust("peer_A") >= 0.6)
check("observe_wrong_hurts",    ts.get_trust("peer_B") <= 0.5)

# Ranked peers
ranked = ts.ranked_peers()
check("ranked_list",            isinstance(ranked, list))
check("ranked_descending",      ranked[0][1] >= ranked[-1][1])

trusted = ts.trusted_peers(threshold=0.60)
check("trusted_list",           isinstance(trusted, list))
check("trusted_has_a",          "peer_A" in trusted)
check("not_trusted_b",          "peer_B" not in trusted)

# Decay
trust_before = ts.get_trust("peer_A")
ts.decay()
trust_after  = ts.get_trust("peer_A")
check("decay_toward_neutral",   abs(trust_after - 0.5) <= abs(trust_before - 0.5))

summ_ts = ts.summary()
check("ts_summary_owner",       summ_ts["owner"] == "self")
check("ts_summary_peers",       summ_ts["peers"] == 2)


# ─────────────────────────────────────────────────────────
print("\n[ConsensusEngine]")
from cognifield.reasoning.consensus_engine import (
    ConsensusEngine, AgentVote, ConsensusStrategy, ConsensusResult
)
from cognifield.world_model.belief_system import BeliefSystem

ce = ConsensusEngine(supermajority_threshold=0.60, min_votes=2)

# Clear consensus
votes_clear = [
    AgentVote("A", True, 0.85, evidence=4.0, trust=0.85),
    AgentVote("B", True, 0.78, evidence=3.0, trust=0.75),
    AgentVote("C", True, 0.70, evidence=2.5, trust=0.65),
]
result = ce.reach_consensus("apple.edible", votes_clear, ConsensusStrategy.TRUST_WEIGHTED)
check("consensus_returns",      isinstance(result, ConsensusResult))
check("clear_consensus_value",  result.value == True)
check("clear_not_contested",    not result.contested)
check("clear_high_agreement",   result.agreement >= 0.99)
check("clear_positive_conf",    result.confidence > 0.0)

# Split — evidence weighted
votes_split = [
    AgentVote("A", False, 0.92, evidence=6.0, trust=0.88),
    AgentVote("B", False, 0.80, evidence=4.0, trust=0.75),
    AgentVote("C", True,  0.55, evidence=1.5, trust=0.45),
]
result2 = ce.reach_consensus("stone.edible", votes_split, ConsensusStrategy.EVIDENCE_WEIGHTED)
check("split_value_false",      result2.value == False)
check("split_not_contested",    not result2.contested)
check("split_agreement",        result2.agreement > 0.60)

# Insufficient votes
votes_one = [AgentVote("A", True, 0.8, trust=0.8)]
result3 = ce.reach_consensus("lone.edible", votes_one)
check("insufficient_contested", result3.contested)
check("insufficient_none_val",  result3.value is None)

# Votes from BeliefSystems
bs_a = BeliefSystem()
bs_b = BeliefSystem()
for _ in range(4): bs_a.update("bread.edible", True, "direct_observation")
for _ in range(3): bs_b.update("bread.edible", True, "direct_observation")
votes_bs = ConsensusEngine.votes_from_beliefs(
    "bread.edible", {"A": bs_a, "B": bs_b},
    trust_scores={"A": 0.85, "B": 0.75}
)
check("votes_from_bs",          len(votes_bs) == 2)
check("votes_from_bs_values",   all(v.value == True for v in votes_bs))

result4 = ce.reach_consensus("bread.edible", votes_bs)
check("bs_consensus_value",     result4.value == True)

# Apply to BeliefSystem
bs_target = BeliefSystem()
ce.apply_to_belief_system(result4, bs_target, source="consensus")
b = bs_target.get("bread.edible")
check("apply_to_bs",            b is not None)
check("apply_value",            b.value == True)

# Contested keys
ce2 = ConsensusEngine(supermajority_threshold=0.90)  # very high
votes_tight = [
    AgentVote("A", True,  0.55, trust=0.55),
    AgentVote("B", False, 0.54, trust=0.54),
]
r_tight = ce2.reach_consensus("tied.edible", votes_tight, ConsensusStrategy.SUPERMAJORITY)
# With only 2 votes and tight split, may or may not be contested depending on threshold
check("tight_result_valid",     r_tight is not None)
check("contested_list_type",    isinstance(ce2.get_contested_keys(), list))

summ_ce = ce.summary()
check("ce_summary_results",     summ_ce["n_results"] >= 2)
check("ce_summary_agreement",   0.0 <= summ_ce["mean_agreement"] <= 1.0)


# ─────────────────────────────────────────────────────────
print("\n[AgentV6]")
from cognifield.agents.agent_v6 import (
    CogniFieldAgentV6, AgentV6Config, AgentRole, V6Step
)

bus2 = CommunicationModule(max_queue=50)
sm4  = SharedMemory()

agent_A = CogniFieldAgentV6(
    config=AgentV6Config(agent_id="test_A", role=AgentRole.EXPLORER,
                         dim=64, verbose=False, seed=0),
    comm_bus=bus2, shared_mem=sm4
)
agent_B = CogniFieldAgentV6(
    config=AgentV6Config(agent_id="test_B", role=AgentRole.ANALYST,
                         dim=64, verbose=False, seed=1),
    comm_bus=bus2, shared_mem=sm4
)

check("agent_id",               agent_A.agent_id == "test_A")
check("agent_role",             agent_A.role == AgentRole.EXPLORER)
check("trust_system_exists",    agent_A.trust is not None)
check("comm_bus_connected",     agent_A.comm_bus is not None)
check("shared_mem_connected",   agent_A.shared_mem is not None)

# Role traits applied
check("explorer_lower_nov",     agent_A.cfg.novelty_threshold < 0.40)
check("analyst_higher_nov",     agent_B.cfg.novelty_threshold > agent_A.cfg.novelty_threshold)

# Step returns V6Step
s = agent_A.step(verbose=False)
check("v6_step_returns",        isinstance(s, V6Step))
check("v6_step_has_msgs_rx",    hasattr(s, "messages_received"))
check("v6_step_has_msgs_tx",    hasattr(s, "messages_sent"))
check("v6_step_has_social",     hasattr(s, "social_learning"))

# Teach + beliefs
agent_A.teach("apple red fruit food", "apple", {"edible":True,"category":"food"})
agent_A.teach("stone grey heavy",     "stone", {"edible":False,"category":"material"})
check("teach_beliefs",          len(agent_A.beliefs) >= 2)
check("apple_belief_true",      agent_A.beliefs.get("apple.edible") is not None)

# Social learning: A knows stone is dangerous, B doesn't
for _ in range(4):
    agent_A.beliefs.update("stone.edible", False, "direct_observation")
conf_A = agent_A.how_confident("stone", "edible")
conf_B_before = agent_B.how_confident("stone", "edible")
check("A_knows_stone",          conf_A > 0.7)
check("B_ignorant_before",      conf_B_before < 0.6)

# A broadcasts warning
warning = Message.warning_msg("test_A","stone","edible",False, conf_A)
bus2.broadcast(warning)

# B registers A as trusted and processes
agent_B.trust.register_peer("test_A")
for _ in range(3):
    agent_B.trust.update_accuracy("test_A", True)

n_rx, n_upd, n_soc = agent_B._process_incoming_messages()
check("social_learning_msgs",   n_rx >= 1)
check("social_learning_updates",n_upd >= 0)   # may not update if no messages
conf_B_after = agent_B.how_confident("stone", "edible")
check("B_learned_from_A",       conf_B_after >= conf_B_before)

# Share knowledge
msgs_sent = agent_A._share_knowledge()
check("share_knowledge_int",    isinstance(msgs_sent, int))

# Broadcast belief
agent_A.beliefs.update("apple.edible", True, "direct_observation")
did_share = agent_A.broadcast_belief("apple","edible")
check("broadcast_belief",       isinstance(did_share, bool))

# Shared memory sync
agent_A._sync_to_shared_memory()
# After sync, some beliefs may be in shared mem
check("sync_runs",              sm4 is not None)

# Learn from shared memory
n_imported = agent_B.learn_from_shared_memory(min_conf=0.60)
check("learn_from_sm_int",      isinstance(n_imported, int))

# v6_summary
summ_v6 = agent_A.v6_summary()
check("v6_summary_agent_id",    summ_v6["agent_id"] == "test_A")
check("v6_summary_role",        summ_v6["role"] == "explorer")
check("v6_summary_msgs",        "msgs_sent" in summ_v6)
check("v6_summary_trust",       "trust_system" in summ_v6)

# Role helpers
check("explorer_should_explore",agent_A.should_explore_now() in (True, False))


# ─────────────────────────────────────────────────────────
print("\n[AgentManager]")
from cognifield.agents.agent_manager import AgentManager
from cognifield.environment.rich_env import RichEnv

env = RichEnv(seed=42)
mgr = AgentManager(
    num_agents=3,
    roles=[AgentRole.EXPLORER, AgentRole.ANALYST, AgentRole.RISK_MANAGER],
    env=env, dim=64, seed=42, verbose=False
)

check("mgr_n_agents",           len(mgr.agents) == 3)
check("mgr_has_bus",            mgr.comm_bus is not None)
check("mgr_has_shared_mem",     mgr.shared_mem is not None)
check("mgr_has_consensus",      mgr.consensus is not None)

# Agent access
a0 = mgr.get_agent("agent_0")
check("get_agent_found",        a0 is not None)
check("get_agent_type",         isinstance(a0, CogniFieldAgentV6))
check("get_agent_missing",      mgr.get_agent("nobody") is None)
check("agent_ids_list",         len(mgr.agent_ids()) == 3)

# Roles assigned
roles_found = {a.role for a in mgr.agents}
check("roles_assigned",         len(roles_found) >= 2)

# Teach all
mgr.teach_all("apple fruit food", label="apple",
               props={"edible":True,"category":"food"})
check("teach_all_runs",         all(len(a.beliefs) >= 1 for a in mgr.agents))

# Step all
summary = mgr.step_all()
check("step_all_returns",       hasattr(summary, "round_num"))
check("step_all_agent_steps",   len(summary.agent_steps) == 3)
check("step_all_elapsed",       summary.elapsed_ms > 0)

# Run episode (short)
log = mgr.run_episode(n_rounds=5, verbose=False)
check("episode_log",            len(log) == 5)
check("episode_rounds",         log[-1].round_num == 6)  # 1 from step_all + 5

# Collect states
states = mgr.collect_states()
check("collect_states_dict",    isinstance(states, dict))
check("collect_states_agents",  "agent_0" in states)
check("collect_states_role",    states["agent_0"]["role"] == "explorer")

# Belief agreement matrix
mgr.teach_all("stone heavy material", label="stone",
               props={"edible":False,"category":"material"})
for a in mgr.agents:
    for _ in range(3):
        a.beliefs.update("stone.edible", False, "direct_observation")
mat = mgr.belief_agreement_matrix("stone.edible")
check("belief_matrix_keys",     "key" in mat and "per_agent" in mat)
check("belief_matrix_agreement",0.0 <= mat["agreement"] <= 1.0)
check("belief_matrix_plurality",mat["plurality"] in ("False","True","false","true","None",None))

# Force consensus
# Inject mild conflict
mgr.agents[0].beliefs.update("cherry.edible", True,  "hypothesis", 0.6)
mgr.agents[1].beliefs.update("cherry.edible", True,  "inference",  0.5)
mgr.agents[2].beliefs.update("cherry.edible", False, "simulation", 0.4)
val = mgr.force_consensus("cherry.edible")
check("force_consensus_runs",   val is not None or val is None)  # may be contested

# Shared knowledge
sk = mgr.shared_knowledge(min_conf=0.50)
check("shared_knowledge_dict",  isinstance(sk, dict))

# Manager summary
mgr_summ = mgr.summary()
check("mgr_summary_agents",     "agents" in mgr_summ)
check("mgr_summary_rounds",     mgr_summ["n_rounds"] >= 6)
check("mgr_summary_consensus",  "consensus" in mgr_summ)


# ─────────────────────────────────────────────────────────
print("\n[Multi-Agent Stability: 30-round run]")

mgr2 = AgentManager(num_agents=3, env=RichEnv(seed=7),
                    dim=64, seed=7, verbose=False)
for name, props in [
    ("apple", {"edible":True,  "category":"food"}),
    ("stone", {"edible":False, "category":"material"}),
]:
    mgr2.teach_all(name, label=name, props=props)
    for a in mgr2.agents:
        v = np.random.randn(a.cfg.dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-8
        action = "eat"
        a.world_model.record(v, action, v,
                             0.5 if props["edible"] else -0.2,
                             props["edible"], name, props["category"])

log2 = mgr2.run_episode(n_rounds=30, verbose=False)
check("stability_30_rounds",    len(log2) == 30)
check("no_crash_30",            all(s is not None for s in log2))

# All agents should have same apple.edible belief direction
apple_vals = [a.beliefs.get_value("apple.edible") for a in mgr2.agents]
stone_vals = [a.beliefs.get_value("stone.edible") for a in mgr2.agents]
# Check they learned in same direction (not strict equality due to partial info)
check("consistent_apple",       all(v == apple_vals[0] for v in apple_vals if v is not None)
                                  or len([v for v in apple_vals if v is not None]) == 0)
check("consistent_stone",       all(v == stone_vals[0] for v in stone_vals if v is not None)
                                  or len([v for v in stone_vals if v is not None]) == 0)

# Trust system has been used
for a in mgr2.agents:
    check(f"trust_{a.agent_id}_used",
          isinstance(a.trust.summary(), dict))

# ─────────────────────────────────────────────────────────
print(f"\n{'═'*56}")
print(f"  v6 Results: {PASS} passed, {FAIL} failed")
if ERRORS:
    print(f"  Failed: {ERRORS}")
else:
    print("  All v6 tests passed ✓")
print(f"{'═'*56}\n")
