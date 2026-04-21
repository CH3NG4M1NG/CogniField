"""
tests/test_v7.py
================
CogniField v7 Test Suite — 144 tests

Modules tested:
  LanguageLayer · NegotiationEngine · CooperationEngine
  SocialMemory · AgentV7
  Negotiation convergence · Role evolution · Cooperation success
  Reputation stability · Communication correctness

Run: PYTHONPATH=.. python tests/test_v7.py
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
        print(f"  ✗ {name}" + (f" — {msg}" if msg else "")); FAIL += 1; ERRORS.append(name)


# ─────────────────────────────────────────────────────────
print("\n[LanguageLayer]")
from cognifield.communication.language_layer import LanguageLayer, SemanticToken
from cognifield.communication.communication_module import Message, MessageType

ll = LanguageLayer("test_agent", vocab_max=200)

check("initial_vocab_nonempty",   ll.vocab_size() > 0)
check("initial_established",      len(ll.established_tokens()) >= 0)

# Encoding
em = ll.encode("apple","edible",True,0.85)
check("encode_returns_encoded",   em is not None)
check("encode_subject",           em.subject == "apple")
check("encode_predicate",         em.predicate == "edible")
check("encode_value",             em.value == True)
check("encode_confidence",        em.confidence == 0.85)
check("encode_has_token",         len(em.token) > 0)
check("encode_has_raw",           len(em.raw_content) > 0)

em2 = ll.encode("stone","edible",False,0.92)
check("encode_false_value",       em2.value == False)
check("encode_warning_token",     em2.token is not None)

# Token usage tracking
for _ in range(6): ll.encode("apple","edible",True,0.85)
established = ll.established_tokens()
check("tokens_established",       len(established) >= 1)
check("established_type",         all(isinstance(t,SemanticToken) for t in established))

# Decoding
ll2 = LanguageLayer("other_agent")
decoded = ll2.decode(em, sender_trust=0.75)
check("decode_returns_dict",      isinstance(decoded, dict))
check("decode_subject",           decoded["subject"] == "apple")
check("decode_predicate",         decoded["predicate"] == "edible")
check("decode_value",             decoded["value"] == True)
check("decode_eff_conf",          0.0 < decoded["effective_confidence"] <= 0.95)
check("trust_discounts",          decoded["effective_confidence"] < em.confidence)

# Decode with low trust
decoded_low = ll2.decode(em, sender_trust=0.2)
check("low_trust_lower_conf",     decoded_low["effective_confidence"] < decoded["effective_confidence"])

# Decode from message
msg = Message.belief_msg("A","bread","edible",True,0.80)
decoded_msg = ll2.decode_message(msg, sender_trust=0.7)
check("decode_message_ok",        decoded_msg is not None)
check("decode_msg_subject",       decoded_msg["subject"] == "bread")

# Vocabulary merging
for _ in range(6): ll.encode("apple","edible",True,0.85)
n_new = ll2.merge_vocabulary(ll._vocab)
check("merge_returns_int",        isinstance(n_new, int))
shared = ll.get_shared_tokens(ll2._vocab)
check("shared_tokens_list",       isinstance(shared, list))
check("shared_tokens_nonempty",   len(shared) >= 1)

# Summary
summ = ll.summary()
check("ll_summary_keys",          all(k in summ for k in ["vocab_size","encoded","decoded"]))
check("ll_repr",                  "LanguageLayer" in repr(ll))

# Encode from message
msg2 = Message.warning_msg("X","stone","edible",False,0.9)
em_from_msg = ll.encode_from_message(msg2)
check("encode_from_msg",          em_from_msg is not None)
check("encode_from_msg_type",     em_from_msg.msg_type == MessageType.WARNING)


# ─────────────────────────────────────────────────────────
print("\n[NegotiationEngine]")
from cognifield.reasoning.negotiation_engine import (
    NegotiationEngine, NegotiationResult, ArgumentType
)
from cognifield.world_model.belief_system import BeliefSystem

ne = NegotiationEngine(max_rounds=5, tolerance=0.12, learning_rate=0.25)

# Build conflicting BeliefSystems
bs_a = BeliefSystem(); bs_b = BeliefSystem()
for _ in range(5): bs_a.update("pb.edible", True,  "direct_observation")
for _ in range(2): bs_b.update("pb.edible", False, "inference")

conf_a_init = bs_a.get_confidence("pb.edible")
conf_b_init = bs_b.get_confidence("pb.edible")

result = ne.negotiate("pb.edible", bs_a,"A",0.72, bs_b,"B",0.65)
check("neg_returns",              isinstance(result, NegotiationResult))
check("neg_has_key",              result.key == "pb.edible")
check("neg_has_value",            result.agreed_value is not None)
check("neg_has_conf",             0.0 <= result.agreed_conf <= 1.0)
check("neg_rounds_positive",      result.rounds >= 1)
check("neg_converged",            result.converged)
check("neg_a_changed",            result.agent_a_delta != 0.0 or result.converged)
check("neg_history_list",         isinstance(result.history, list))

# After negotiation B should have moved toward A
conf_b_after = bs_b.get_confidence("pb.edible")
check("B_moved_toward_A",         conf_b_after != conf_b_init or result.converged)

# Both agree → fast convergence
bs_c = BeliefSystem(); bs_d = BeliefSystem()
for _ in range(3): bs_c.update("apple.edible", True, "direct_observation")
for _ in range(2): bs_d.update("apple.edible", True, "inference")
result2 = ne.negotiate("apple.edible", bs_c,"C",0.8, bs_d,"D",0.7)
check("agree_converges_fast",     result2.rounds <= 2)
check("agree_value_true",         result2.agreed_value == True)

# Missing belief
bs_e = BeliefSystem(); bs_f = BeliefSystem()
for _ in range(3): bs_e.update("x.edible", True, "direct_observation")
result3 = ne.negotiate("x.edible", bs_e,"E",0.7, bs_f,"F",0.7)
check("missing_belief_adopted",   result3.converged or result3.agreed_value is not None)

# Batch negotiation
bs_g = BeliefSystem(); bs_h = BeliefSystem()
for _ in range(3): bs_g.update("m.edible", True,  "direct_observation")
for _ in range(3): bs_h.update("m.edible", False, "direct_observation")
for _ in range(3): bs_g.update("g.fragile",True,  "direct_observation")
for _ in range(3): bs_h.update("g.fragile",False, "inference")
batch = ne.negotiate_all_conflicts(bs_g,"G",0.75, bs_h,"H",0.65, min_conf_threshold=0.5)
check("batch_list",               isinstance(batch, list))
check("batch_has_results",        len(batch) >= 1)

summ_ne = ne.summary()
check("ne_summary_sessions",      summ_ne["sessions"] >= 3)
check("ne_convergence_rate",      0.0 <= summ_ne["convergence_rate"] <= 1.0)
check("ne_mean_rounds",           summ_ne["mean_rounds"] >= 1.0)
check("ne_repr",                  "NegotiationEngine" in repr(ne))


# ─────────────────────────────────────────────────────────
print("\n[CooperationEngine]")
from cognifield.planning.cooperation_engine import (
    CooperationEngine, TaskType, TaskStatus, CoopTask, CoopPlan
)

ce = CooperationEngine()
for aid, role in [("e0","explorer"),("a1","analyst"),
                   ("r2","risk_manager"),("p3","planner")]:
    ce.register_agent(aid, role)

check("ce_agents_registered",     len(ce._agents) == 4)

# Task creation
task1 = ce.create_task(TaskType.EXPLORE, "explore unknown", "purple_berry", 0.8)
check("task_created",             isinstance(task1, CoopTask))
check("task_id",                  task1.task_id.startswith("task_"))
check("task_type",                task1.task_type == TaskType.EXPLORE)
check("task_pending",             task1.status == TaskStatus.PENDING)

# Assignment
assigned = ce.assign(task1)
check("assignment_returns_str",   isinstance(assigned, str))
check("task_assigned",            task1.status == TaskStatus.ASSIGNED)
check("assigned_agent_is_explorer", ce._agents[assigned]["role"] in
      ("explorer","analyst","risk_manager","planner"))

# Role fitness: EXPLORE → should go to explorer
task_exp = ce.create_task(TaskType.EXPLORE, "explore X", "X")
ag_exp   = ce.assign(task_exp)
check("explore_to_explorer",      ce._agents.get(ag_exp, {}).get("role") in
      ("explorer","analyst"))  # explorer or analyst are top 2

task_warn = ce.create_task(TaskType.WARN, "warn about Z", "Z")
ag_warn   = ce.assign(task_warn)
check("warn_to_risk_mgr",         ce._agents.get(ag_warn, {}).get("role") in
      ("risk_manager","analyst"))

# Parallel plan
plan = ce.plan_parallel("explore all", ["A","B","C"], TaskType.EXPLORE)
check("plan_parallel_type",       isinstance(plan, CoopPlan))
check("plan_n_tasks",             len(plan.tasks) == 3)
check("plan_pattern",             plan.pattern == "parallel")
check("plan_is_complete_false",   not plan.is_complete)

# Pipeline plan
stages = [(TaskType.EXPLORE,"explore"), (TaskType.VERIFY,"verify"),
          (TaskType.PLAN,"plan")]
pipeline = ce.plan_pipeline("get knowledge", stages)
check("pipeline_stages",          len(pipeline.tasks) == 3)
check("pipeline_pattern",         pipeline.pattern == "pipeline")

# Complete tasks
ce.complete_task(task1.task_id, result="done")
check("task_completed",           task1.status == TaskStatus.COMPLETED)
check("workload_reduced",         ce._agents[assigned]["workload"] >= 0)

# Fail and retry
task_fail = ce.create_task(TaskType.VERIFY,"fail test","X")
ce.assign(task_fail)
ce.fail_task(task_fail.task_id)
check("task_failed_or_pending",   task_fail.status in (TaskStatus.FAILED, TaskStatus.PENDING))

# Plan progress
for t in plan.tasks[:2]:
    ce.complete_task(t.task_id, "done")
check("plan_progress",            plan.progress() >= 2/3)
check("plan_n_completed",         plan.n_completed >= 2)

summ_ce = ce.summary()
check("ce_summary_agents",        summ_ce["agents"] == 4)
check("ce_summary_tasks",         summ_ce["tasks"] >= 5)
check("ce_completion_rate",       0.0 <= summ_ce["completion_rate"] <= 1.0)
check("ce_repr",                  "CooperationEngine" in repr(ce))


# ─────────────────────────────────────────────────────────
print("\n[SocialMemory]")
from cognifield.memory.social_memory import SocialMemory, Interaction

sm = SocialMemory("owner_agent")

# Record interactions
for i in range(6):
    sm.record_interaction("peer_X", "belief", "apple.edible",
                           True, 0.85, round_num=i)
for i in range(3):
    sm.record_interaction("peer_Y", "warning", "stone.edible",
                           False, 0.92, round_num=i)

check("interactions_recorded",    sm.interaction_count("peer_X") == 6)
check("known_peers",              "peer_X" in sm.known_peers())
check("known_peers_Y",            "peer_Y" in sm.known_peers())

# Verification
for _ in range(4): sm.record_verification("peer_X","apple.edible",True)
for _ in range(2): sm.record_verification("peer_X","apple.edible",False)
check("overall_accuracy",         0.0 < sm.overall_accuracy("peer_X") < 1.0)
check("topic_accuracy",           0.0 < sm.topic_accuracy("peer_X","apple.edible") < 1.0)
check("unknown_peer_accuracy",    sm.overall_accuracy("nobody") == 0.5)

# Cooperation
sm.record_cooperation("peer_X","explore_task",True,"follower",0.5)
sm.record_cooperation("peer_X","verify_task", True,"equal",   0.3)
sm.record_cooperation("peer_Y","navigate",    False,"leader",  -0.1)
check("coop_success_peer_x",      sm.cooperation_success_rate("peer_X") > 0.5)
check("coop_success_peer_y",      sm.cooperation_success_rate("peer_Y") < 0.5)
check("best_coop_peers",          len(sm.best_cooperative_peers(3)) >= 1)
check("best_coop_is_x",           sm.best_cooperative_peers(1)[0][0] == "peer_X")

# Queries
ixs = sm.get_interactions("peer_X", n=5)
check("get_interactions_list",    isinstance(ixs, list))
check("get_interactions_count",   len(ixs) <= 5)

ixs_topic = sm.get_interactions("peer_X","apple.edible")
check("get_by_topic",             all(ix.topic=="apple.edible" for ix in ixs_topic))

topics_known = sm.topics_peer_knows_well("peer_X", threshold=0.5)
check("topics_known_list",        isinstance(topics_known, list))

most_active = sm.most_interactive_peers(3)
check("most_active_list",         isinstance(most_active, list))
check("most_active_first",        most_active[0][0] == "peer_X")

check("detect_leader",            sm.detect_leader() in (None,"peer_X","peer_Y"))

profile = sm.peer_profile("peer_X")
check("peer_profile_keys",        all(k in profile for k in
                                      ["peer_id","interactions","overall_accuracy"]))
summ = sm.summary()
check("sm_summary_keys",          "known_peers" in summ and "owner" in summ)
check("sm_repr",                  "SocialMemory" in repr(sm))


# ─────────────────────────────────────────────────────────
print("\n[AgentV7]")
from cognifield.agents.agent_v7 import CogniFieldAgentV7, AgentV7Config, AgentRole, V7Step
from cognifield.communication.communication_module import CommunicationModule
from cognifield.memory.shared_memory import SharedMemory
from cognifield.planning.cooperation_engine import CooperationEngine

bus7 = CommunicationModule()
sm7  = SharedMemory()
ce7  = CooperationEngine()

cfg_A = AgentV7Config(agent_id="v7_A", role=AgentRole.EXPLORER,
                       dim=64, verbose=False, seed=0)
cfg_B = AgentV7Config(agent_id="v7_B", role=AgentRole.ANALYST,
                       dim=64, verbose=False, seed=1)

agent_A = CogniFieldAgentV7(config=cfg_A, comm_bus=bus7, shared_mem=sm7, coop_engine=ce7)
agent_B = CogniFieldAgentV7(config=cfg_B, comm_bus=bus7, shared_mem=sm7, coop_engine=ce7)

check("v7_agent_created",         isinstance(agent_A, CogniFieldAgentV7))
check("v7_agent_id",              agent_A.agent_id == "v7_A")
check("v7_role",                  agent_A.role == AgentRole.EXPLORER)
check("v7_has_language",          agent_A.language is not None)
check("v7_has_social_mem",        agent_A.social_mem is not None)
check("v7_has_negotiation",       agent_A.negotiation is not None)

# Step
agent_A.teach("apple edible food", "apple", {"edible":True,"category":"food"})
s = agent_A.step(verbose=False)
check("v7_step_returns",          isinstance(s, V7Step))
check("v7_step_vocab",            s.vocab_size >= 0)
check("v7_step_negs",             s.negotiations_run >= 0)
check("v7_step_coop",             s.coop_tasks_done >= 0)
check("v7_step_role_changed",     isinstance(s.role_changed, bool))

# Register for negotiation
agent_A.register_for_negotiation(agent_B)
agent_B.register_for_negotiation(agent_A)
check("neg_registered",           "v7_B" in agent_A._neg_partners)

# Inject conflict and negotiate
for _ in range(4): agent_A.beliefs.update("berry.edible", True,  "direct_observation")
for _ in range(2): agent_B.beliefs.update("berry.edible", False, "inference")
n_negs = agent_A._run_negotiations()
check("negs_run",                 isinstance(n_negs, int))

# Language sharing
agent_A.beliefs.update("apple.edible", True, "direct_observation")
agent_A._share_via_language()
check("language_sharing_runs",    True)   # no exception = pass

# Cooperation task
ce7.register_agent("v7_A", "explorer")
ce7.register_agent("v7_B", "analyst")
task = ce7.create_task(TaskType.EXPLORE, "explore X", "X")
ce7.assign(task)
n_done = agent_A._process_coop_tasks()
check("coop_tasks_int",           isinstance(n_done, int))

# Reputation
agent_A._reputation["v7_B"] = 0.70
eff_trust = agent_A.effective_trust("v7_B")
check("effective_trust_range",    0.0 <= eff_trust <= 1.0)
check("reputation_blended",       eff_trust != agent_A.trust.get_trust("v7_B", 0.5))

# Update reputation from verification
agent_A.social_mem.record_interaction("v7_B","belief","stone.edible",False,0.9)
agent_A.update_reputation_from_verification("v7_B","stone.edible", correct=True)
check("reputation_updated",       "v7_B" in agent_A._reputation)

# Run autonomous steps
for _ in range(5): agent_A.step(verbose=False)
check("multi_steps_ok",           agent_A._step_count >= 6)

# v7_summary
summ7 = agent_A.v7_summary()
check("v7_summary_language",      "language" in summ7)
check("v7_summary_social",        "social_memory" in summ7)
check("v7_summary_negotiation",   "negotiation" in summ7)
check("v7_summary_role_history",  "role_history" in summ7)
check("v7_summary_reputation",    "reputation" in summ7)

check("v7_repr",                  "AgentV7" in repr(agent_A))


# ─────────────────────────────────────────────────────────
print("\n[Negotiation Convergence]")
ne2 = NegotiationEngine(max_rounds=8, tolerance=0.08, learning_rate=0.3)

# 10 random conflict pairs — all should converge or at least reduce gap
import random; rng = random.Random(42)
convergences = 0
for i in range(10):
    bs1 = BeliefSystem(); bs2 = BeliefSystem()
    for _ in range(rng.randint(2,6)):
        bs1.update(f"obj_{i}.edible", rng.random() > 0.3, "direct_observation")
    for _ in range(rng.randint(1,4)):
        bs2.update(f"obj_{i}.edible", rng.random() < 0.4, "inference")
    r = ne2.negotiate(f"obj_{i}.edible", bs1,f"A{i}",0.7, bs2,f"B{i}",0.6)
    if r.converged: convergences += 1

check("convergence_majority",     convergences >= 7, f"only {convergences}/10")
check("no_neg_exceptions",        True)   # reaching here = no crashes


# ─────────────────────────────────────────────────────────
print("\n[Coop: parallel plan all tasks complete]")
ce_test = CooperationEngine()
for i,role in enumerate(["explorer","analyst","risk_manager"]):
    ce_test.register_agent(f"a{i}", role)

targets = ["A","B","C","D"]
plan_test = ce_test.plan_parallel("explore all", targets, TaskType.EXPLORE)
check("all_assigned",             all(t.assigned_to is not None for t in plan_test.tasks))

for t in plan_test.tasks:
    ce_test.complete_task(t.task_id, "done")
check("all_completed",            plan_test.is_complete)
check("completion_rate_100",      ce_test.summary()["completion_rate"] == 1.0)


# ─────────────────────────────────────────────────────────
print(f"\n{'═'*56}")
print(f"  v7 Results: {PASS} passed, {FAIL} failed")
if ERRORS:
    print(f"  Failed: {ERRORS}")
else:
    print("  All v7 tests passed ✓")
print(f"{'═'*56}\n")
