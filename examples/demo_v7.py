"""
examples/demo_v7.py
====================
CogniField v7 — Social Intelligence Demo

Demonstrates:
  1. Language Layer       — semantic encoding/decoding + vocabulary evolution
  2. Negotiation Engine   — agents argue beliefs without direct experiments
  3. Cooperation Engine   — task assignment and parallel exploration
  4. Social Memory        — per-peer interaction history + topic accuracy
  5. Dynamic Role Evolution — roles shift based on performance
  6. Full v7 Loop         — all systems running together
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np


def sep(title): print(f"\n{'═'*62}\n  {title}\n{'═'*62}")
def section(title): print(f"\n  {'─'*58}\n  {title}\n  {'─'*58}")


# ═══════════════════════════════════════════════════════════
# 1. Language Layer
# ═══════════════════════════════════════════════════════════
def demo_language_layer():
    sep("1. Language Layer — Semantic Encoding & Vocabulary Evolution")
    from cognifield.communication.language_layer import LanguageLayer, SemanticToken
    from cognifield.communication.communication_module import MessageType

    ll_a = LanguageLayer("agent_A", vocab_max=200)
    ll_b = LanguageLayer("agent_B", vocab_max=200)

    print(f"\n  Initial vocab sizes: A={ll_a.vocab_size()}, B={ll_b.vocab_size()}")

    section("Encoding beliefs into semantic messages")
    beliefs_to_encode = [
        ("apple",       "edible",   True,  0.85, MessageType.BELIEF),
        ("stone",       "edible",   False, 0.92, MessageType.WARNING),
        ("bread",       "edible",   True,  0.80, MessageType.BELIEF),
        ("apple",       "category", "food",0.90, MessageType.BELIEF),
        ("purple_berry","edible",   True,  0.62, MessageType.BELIEF),
    ]
    encoded_msgs = []
    for subject, pred, val, conf, mtype in beliefs_to_encode:
        em = ll_a.encode(subject, pred, val, conf, mtype)
        encoded_msgs.append(em)
        print(f"  {em.raw_content:40s} → token: '{em.token}'")

    print(f"\n  A's vocab size after encoding: {ll_a.vocab_size()}")

    section("Decoding received messages")
    for em in encoded_msgs[:3]:
        decoded = ll_b.decode(em, sender_trust=0.75)
        print(f"  token='{em.token}' → {decoded['subject']}.{decoded['predicate']}="
              f"{decoded['value']} (eff_conf={decoded['effective_confidence']:.3f})")

    section("Vocabulary evolution via vocabulary merging")
    # Simulate many uses to establish tokens
    for _ in range(6):
        ll_a.encode("apple", "edible", True, 0.85)
        ll_a.encode("stone", "edible", False, 0.92)

    established_A = ll_a.established_tokens()
    print(f"  A established tokens ({len(established_A)}): "
          f"{[t.token for t in established_A[:5]]}")

    # B merges A's vocabulary
    new_tokens = ll_b.merge_vocabulary(ll_a._vocab)
    print(f"  B adopted {new_tokens} new tokens from A")
    shared = ll_a.get_shared_tokens(ll_b._vocab)
    print(f"  Shared tokens A∩B: {shared[:5]}")

    print(f"\n  Final vocab: A={ll_a.vocab_size()}, B={ll_b.vocab_size()}")
    print(f"  Language summaries:")
    print(f"    A: {ll_a.summary()}")
    print(f"    B: {ll_b.summary()}")


# ═══════════════════════════════════════════════════════════
# 2. Negotiation Engine
# ═══════════════════════════════════════════════════════════
def demo_negotiation():
    sep("2. Negotiation Engine — Resolving Belief Disagreements")
    from cognifield.reasoning.negotiation_engine import (
        NegotiationEngine, ArgumentType
    )
    from cognifield.world_model.belief_system import BeliefSystem

    ne = NegotiationEngine(max_rounds=5, tolerance=0.10, learning_rate=0.30)

    section("Scenario: Both agents disagree on purple_berry.edible")
    bs_a = BeliefSystem()
    bs_b = BeliefSystem()

    # A has more evidence — thinks it IS edible
    for _ in range(5):
        bs_a.update("purple_berry.edible", True, "direct_observation")
    # B has weaker evidence — thinks it is NOT edible
    for _ in range(2):
        bs_b.update("purple_berry.edible", False, "inference")

    conf_a_init = bs_a.get_confidence("purple_berry.edible")
    conf_b_init = bs_b.get_confidence("purple_berry.edible")
    print(f"\n  Before negotiation:")
    print(f"    A: purple_berry.edible={bs_a.get('purple_berry.edible').value} "
          f"(conf={conf_a_init:.3f}, ev={bs_a.get('purple_berry.edible').total_evidence:.1f})")
    print(f"    B: purple_berry.edible={bs_b.get('purple_berry.edible').value} "
          f"(conf={conf_b_init:.3f}, ev={bs_b.get('purple_berry.edible').total_evidence:.1f})")

    result = ne.negotiate(
        "purple_berry.edible",
        bs_a, "agent_A", 0.72,
        bs_b, "agent_B", 0.65,
    )

    print(f"\n  Negotiation ran {result.rounds} rounds:")
    for rnd in result.history:
        print(f"    Round {rnd.round_num}: "
              f"A {rnd.conf_a_before:.3f}→{rnd.conf_a_after:.3f}  "
              f"B {rnd.conf_b_before:.3f}→{rnd.conf_b_after:.3f}  "
              f"gap={rnd.delta:.3f}")

    print(f"\n  Result: value={result.agreed_value}, conf={result.agreed_conf:.3f}")
    print(f"  Converged: {result.converged}")
    print(f"  A delta: {result.agent_a_delta:+.3f}, B delta: {result.agent_b_delta:+.3f}")
    print(f"  Notes: {result.notes}")

    section("Scenario: Agents agree — just merge confidence")
    bs_c = BeliefSystem()
    bs_d = BeliefSystem()
    for _ in range(3): bs_c.update("apple.edible", True, "direct_observation")
    for _ in range(2): bs_d.update("apple.edible", True, "inference")

    result2 = ne.negotiate("apple.edible",
                            bs_c, "C", 0.80,
                            bs_d, "D", 0.70)
    print(f"  apple.edible (both True): converged={result2.converged}, "
          f"conf={result2.agreed_conf:.3f}")

    section("Batch: negotiate all conflicts")
    bs_e = BeliefSystem()
    bs_f = BeliefSystem()
    for _ in range(3): bs_e.update("mango.edible",  True,  "direct_observation")
    for _ in range(3): bs_f.update("mango.edible",  False, "direct_observation")
    for _ in range(4): bs_e.update("glass.fragile", True,  "direct_observation")
    for _ in range(2): bs_f.update("glass.fragile", False, "inference")

    results_batch = ne.negotiate_all_conflicts(bs_e,"E",0.75, bs_f,"F",0.65)
    print(f"  Batch negotiated {len(results_batch)} conflicts")
    for r in results_batch:
        print(f"    {r.key}: → {r.agreed_value} (converged={r.converged})")

    print(f"\n  Engine summary: {ne.summary()}")


# ═══════════════════════════════════════════════════════════
# 3. Cooperation Engine
# ═══════════════════════════════════════════════════════════
def demo_cooperation():
    sep("3. Cooperation Engine — Task Assignment & Coordination")
    from cognifield.planning.cooperation_engine import (
        CooperationEngine, TaskType, TaskStatus
    )

    ce = CooperationEngine()
    agents = {
        "agent_0": "explorer",
        "agent_1": "analyst",
        "agent_2": "risk_manager",
        "agent_3": "planner",
    }
    for aid, role in agents.items():
        ce.register_agent(aid, role)

    print(f"\n  Registered {len(agents)} agents with roles: "
          f"{list(agents.values())}")

    section("Task assignment by role fitness")
    tasks_to_assign = [
        (TaskType.EXPLORE,  "Explore unknown purple_berry"),
        (TaskType.VERIFY,   "Verify apple.edible=True"),
        (TaskType.WARN,     "Warn about stone.edible=False"),
        (TaskType.PLAN,     "Plan eat-apple sequence"),
        (TaskType.NEGOTIATE,"Negotiate mango.edible conflict"),
        (TaskType.EXPERIMENT,"Experiment on glowing_cube"),
    ]
    print(f"\n  {'Task':35s} | {'Assigned To':12s} | Note")
    print(f"  {'─'*35} | {'─'*12} | {'─'*20}")
    for ttype, desc in tasks_to_assign:
        task     = ce.create_task(ttype, desc, priority=0.7)
        assigned = ce.assign(task)
        role_of  = agents.get(assigned, "?") if assigned else "none"
        print(f"  {desc:35s} | {assigned or 'NONE':12s} | role={role_of}")

    section("Parallel exploration plan")
    targets = ["purple_berry", "glowing_cube", "mystery_powder"]
    plan = ce.plan_parallel(
        goal="explore all unknowns",
        targets=targets,
        task_type=TaskType.EXPLORE,
    )
    print(f"  Plan type: {plan.pattern}, tasks: {len(plan.tasks)}")
    for t in plan.tasks:
        print(f"    {t.task_id}: {t.description[:40]} → {t.assigned_to}")

    section("Pipeline: explore → verify → plan")
    stages = [
        (TaskType.EXPLORE,    "Agent explores new object"),
        (TaskType.VERIFY,     "Analyst verifies the finding"),
        (TaskType.PLAN,       "Planner creates execution plan"),
    ]
    pipeline = ce.plan_pipeline("get apple", stages)
    print(f"  Pipeline: {len(pipeline.tasks)} stages")
    for t in pipeline.tasks:
        print(f"    Stage {pipeline.tasks.index(t)+1}: {t.description[:40]} "
              f"→ {t.assigned_to}")

    section("Complete tasks + check progress")
    for task in plan.tasks[:2]:
        ce.complete_task(task.task_id, result={"explored": task.target})
    print(f"  Plan progress: {plan.progress():.0%} "
          f"({plan.n_completed}/{len(plan.tasks)} tasks done)")
    print(f"\n  Summary: {ce.summary()}")


# ═══════════════════════════════════════════════════════════
# 4. Social Memory
# ═══════════════════════════════════════════════════════════
def demo_social_memory():
    sep("4. Social Memory — Interaction History & Reliability")
    from cognifield.memory.social_memory import SocialMemory

    sm = SocialMemory("self")

    # Record interactions with peer A (mostly accurate)
    for i in range(8):
        sm.record_interaction("peer_A", "belief", "apple.edible",
                               True, 0.85, round_num=i)
    for i in range(5):
        sm.record_verification("peer_A", "apple.edible", correct=True)
    for i in range(2):
        sm.record_verification("peer_A", "stone.edible", correct=False)

    # Peer B (accurate on food, inaccurate on tools)
    for i in range(6):
        sm.record_interaction("peer_B", "belief", "bread.edible",
                               True, 0.78, round_num=i)
    for i in range(4):
        sm.record_verification("peer_B", "bread.edible", correct=True)
    for i in range(3):
        sm.record_verification("peer_B", "hammer.category", correct=False)

    # Cooperative tasks
    sm.record_cooperation("peer_A", "explore_unknowns", True, "follower", 0.5)
    sm.record_cooperation("peer_A", "verify_beliefs",   True, "follower", 0.3)
    sm.record_cooperation("peer_B", "navigate_world",   False,"equal",   -0.1)

    print(f"\n  Interaction counts: "
          f"A={sm.interaction_count('peer_A')}, "
          f"B={sm.interaction_count('peer_B')}")

    print(f"\n  Overall accuracy: A={sm.overall_accuracy('peer_A'):.3f}, "
          f"B={sm.overall_accuracy('peer_B'):.3f}")

    print(f"\n  Topic accuracy (peer_A):")
    print(f"    apple.edible:  {sm.topic_accuracy('peer_A','apple.edible'):.3f}")
    print(f"    stone.edible:  {sm.topic_accuracy('peer_A','stone.edible'):.3f}")

    print(f"\n  Topics peer_B knows well: "
          f"{sm.topics_peer_knows_well('peer_B', threshold=0.70)}")

    print(f"\n  Cooperation success: "
          f"A={sm.cooperation_success_rate('peer_A'):.1%}, "
          f"B={sm.cooperation_success_rate('peer_B'):.1%}")

    print(f"\n  Best cooperative peers: {sm.best_cooperative_peers(3)}")
    print(f"  Detected leader: {sm.detect_leader()}")

    print(f"\n  Peer profiles:")
    for pid in ["peer_A","peer_B"]:
        print(f"    {pid}: {sm.peer_profile(pid)}")

    print(f"\n  Summary: {sm.summary()}")


# ═══════════════════════════════════════════════════════════
# 5. Full v7 Agent Loop
# ═══════════════════════════════════════════════════════════
def demo_full_v7():
    sep("5. Full v7 System — Negotiation, Cooperation & Social Intelligence")
    from cognifield.agents.agent_v7 import CogniFieldAgentV7, AgentV7Config, AgentRole
    from cognifield.agents.agent_manager import AgentManager
    from cognifield.planning.cooperation_engine import CooperationEngine, TaskType
    from cognifield.environment.rich_env import RichEnv
    from cognifield.agents.goals import GoalType

    env      = RichEnv(seed=42)
    coop_eng = CooperationEngine()

    # Build three v7 agents
    agents = []
    roles  = [AgentRole.EXPLORER, AgentRole.ANALYST, AgentRole.RISK_MANAGER]
    from cognifield.communication.communication_module import CommunicationModule
    from cognifield.memory.shared_memory import SharedMemory

    bus = CommunicationModule()
    sm  = SharedMemory()

    for i, role in enumerate(roles):
        cfg = AgentV7Config(
            agent_id=f"v7_agent_{i}",
            role=role, dim=64,
            verbose=False, seed=42+i,
        )
        a = CogniFieldAgentV7(config=cfg, env=env, comm_bus=bus,
                               shared_mem=sm, coop_engine=coop_eng)
        agents.append(a)
        coop_eng.register_agent(a.agent_id, role.value)

    # Register negotiation partners
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            if i != j:
                a.register_for_negotiation(b)

    # ── Phase A: Teach world knowledge ──
    section("Phase A: Teaching world knowledge")
    for name, props in [
        ("apple",  {"edible":True,  "category":"food"}),
        ("stone",  {"edible":False, "category":"material"}),
        ("bread",  {"edible":True,  "category":"food"}),
    ]:
        for a in agents:
            a.teach(f"{name} {' '.join(f'{k} {v}' for k,v in props.items())}",
                    name, props)
            v = np.random.randn(a.cfg.dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-8
            a.world_model.record(v, "eat", v,
                                  0.5 if props["edible"] else -0.2,
                                  props["edible"], name, props["category"])

    print(f"  Agents created: {[a.agent_id for a in agents]}")
    print(f"  Roles: {[a.role.value for a in agents]}")
    print(f"  Initial shared memory: {len(sm)} entries")

    # ── Phase B: Inject disagreement ──
    section("Phase B: Inject belief conflict on purple_berry")
    agents[0].beliefs.update("purple_berry.edible", True,  "direct_observation",
                               weight=0.9)
    agents[0].beliefs.update("purple_berry.edible", True,  "direct_observation",
                               weight=0.9)
    agents[1].beliefs.update("purple_berry.edible", False, "inference", weight=0.5)
    agents[2].beliefs.update("purple_berry.edible", False, "simulation", weight=0.4)

    print(f"\n  purple_berry.edible before negotiation:")
    for a in agents:
        b = a.beliefs.get("purple_berry.edible")
        if b:
            print(f"    {a.agent_id}({a.role.value}): {b.value} "
                  f"(conf={b.confidence:.3f})")

    # Run negotiation round
    for a in agents:
        a._run_negotiations(verbose=True)

    print(f"\n  purple_berry.edible after negotiation:")
    for a in agents:
        b = a.beliefs.get("purple_berry.edible")
        if b:
            print(f"    {a.agent_id}: {b.value} (conf={b.confidence:.3f})")

    # ── Phase C: Cooperative exploration plan ──
    section("Phase C: Cooperative parallel exploration")
    unknowns = ["purple_berry", "glowing_cube", "mystery_dust"]
    plan = coop_eng.plan_parallel(
        goal="explore all unknowns",
        targets=unknowns,
        task_type=TaskType.EXPLORE,
        trust_scores={a.agent_id: 0.70 for a in agents},
    )
    print(f"  Plan created: {len(plan.tasks)} tasks (pattern='{plan.pattern}')")
    for t in plan.tasks:
        print(f"    {t.task_id}: explore({t.target}) → {t.assigned_to}")

    # Execute tasks
    for a in agents:
        a._process_coop_tasks(verbose=True)

    print(f"\n  Plan progress: {plan.progress():.0%}")

    # ── Phase D: Run 15-round autonomous loop ──
    section("Phase D: 15-round autonomous run")
    for a in agents:
        a.add_goal("eat apple", GoalType.EAT_OBJECT, target="apple", priority=0.85)

    print(f"\n  {'Agent':12s} | {'Role':14s} | {'Step':4s} | "
          f"{'Negs':4s} | {'Coop':4s} | {'Vocab':5s} | "
          f"{'Beliefs':7s} | {'Grade'}")
    print(f"  {'─'*12} | {'─'*14} | {'─'*4} | "
          f"{'─'*4} | {'─'*4} | {'─'*5} | {'─'*7} | {'─'*5}")

    for round_n in range(15):
        for a in agents:
            s = a.step(verbose=False)
            if round_n % 5 == 4:   # print every 5 rounds
                print(f"  {a.agent_id:12s} | {a.role.value:14s} | "
                      f"{s.step:4d} | {s.negotiations_run:4d} | "
                      f"{s.coop_tasks_done:4d} | {s.vocab_size:5d} | "
                      f"{len(a.beliefs):7d} | {a.metrics.stability_grade():5s}")

    # ── Phase E: Language vocab comparison ──
    section("Phase E: Language vocabulary evolution")
    for a in agents:
        ll_summ = a.language.summary()
        print(f"  {a.agent_id}({a.role.value}): "
              f"vocab={ll_summ['vocab_size']}, "
              f"established={ll_summ['established']}, "
              f"encoded={ll_summ['encoded']}, "
              f"decoded={ll_summ['decoded']}")

    # ── Phase F: Reputation check ──
    section("Phase F: Reputation vs Trust comparison")
    for a in agents:
        print(f"\n  {a.agent_id} ({a.role.value}) peer reputations:")
        for peer_id in [ag.agent_id for ag in agents if ag.agent_id != a.agent_id]:
            trust = a.trust.get_trust(peer_id, 0.5)
            rep   = a._reputation.get(peer_id, 0.5)
            eff   = a.effective_trust(peer_id)
            print(f"    {peer_id}: trust={trust:.3f}, "
                  f"reputation={rep:.3f}, effective={eff:.3f}")

    # ── Phase G: Final summary ──
    section("Final System Summary")
    for a in agents:
        summ = a.v7_summary()
        print(f"\n  {a.agent_id} ({a.role.value}):")
        print(f"    Beliefs: {summ['beliefs']} ({summ['reliable_beliefs']} reliable)")
        print(f"    Steps: {summ['steps']}, Grade: {summ['stability_grade']}")
        print(f"    Negotiations: {summ['negotiation']['sessions']}")
        print(f"    Msgs tx/rx: {summ['msgs_sent']}/{summ['msgs_received']}")
        print(f"    Social peers: {summ['social_memory']['known_peers']}")
        print(f"    Role history: {summ['role_history']}")
        print(f"    Language vocab: {summ['language']['vocab_size']}")

    print(f"\n  Cooperation engine: {coop_eng.summary()}")
    print(f"  Shared memory: {sm.summary()}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   CogniField v7 — Social Intelligence Demo              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    demo_language_layer()
    demo_negotiation()
    demo_cooperation()
    demo_social_memory()
    demo_full_v7()
    print("\n" + "═"*62 + "\n  v7 Demo complete.\n" + "═"*62)
