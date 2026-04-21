"""
examples/demo_v6.py
====================
CogniField v6 — Multi-Agent Collaborative Learning Demo

Demonstrates:
  1. CommunicationModule  — typed message exchange
  2. SharedMemory         — community knowledge store
  3. TrustSystem          — reliability reputation
  4. ConsensusEngine      — belief aggregation
  5. AgentV6 roles        — Explorer/Analyst/Risk Manager/Planner
  6. Social learning      — Agent C learns from A's observation
  7. Full multi-agent loop — all 17 steps running together
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def sep(title: str) -> None:
    print(f"\n{'═'*62}")
    print(f"  {title}")
    print(f"{'═'*62}")


def section(title: str) -> None:
    print(f"\n  {'─'*58}")
    print(f"  {title}")
    print(f"  {'─'*58}")


# ═══════════════════════════════════════════════════════════
# 1. Communication Bus
# ═══════════════════════════════════════════════════════════
def demo_communication():
    sep("1. Communication System — Typed Message Exchange")
    from cognifield.communication.communication_module import (
        CommunicationModule, Message, MessageType
    )

    bus = CommunicationModule(max_queue=50)
    for aid in ["alice", "bob", "charlie"]:
        bus.register(aid)

    print("\n  Sending messages between agents:")
    msgs_sent = []

    # Alice broadcasts a belief
    m1 = Message.belief_msg("alice", "apple", "edible", True, confidence=0.85)
    bus.broadcast(m1)
    msgs_sent.append(("alice→ALL", "belief", "apple.edible=True(0.85)"))

    # Alice warns everyone about stone
    m2 = Message.warning_msg("alice", "stone", "edible", False, confidence=0.92)
    bus.broadcast(m2)
    msgs_sent.append(("alice→ALL", "warning", "stone.edible=False(0.92)"))

    # Bob sends an observation to Charlie specifically
    m3 = Message.observation_msg("bob","eat","apple","success", 0.5)
    bus.send(m3)
    msgs_sent.append(("bob→charlie", "observation", "eat(apple)→success(0.9)"))

    # Alice asks a question
    m4 = Message.question_msg("alice", "purple_berry", "edible")
    bus.broadcast(m4)
    msgs_sent.append(("alice→ALL", "question", "purple_berry.edible?"))

    print(f"  {'From→To':15s} | {'Type':12s} | {'Content'}")
    print(f"  {'─'*15} | {'─'*12} | {'─'*30}")
    for from_to, mtype, content in msgs_sent:
        print(f"  {from_to:15s} | {mtype:12s} | {content}")

    print(f"\n  Inbox sizes: ", {aid: bus.peek(aid) for aid in ["alice","bob","charlie"]})

    # Bob receives
    bob_msgs = bus.receive("bob", max_msgs=10)
    print(f"  Bob received {len(bob_msgs)} messages:")
    for m in bob_msgs:
        print(f"    [{m.msg_type.value}] from {m.sender_id}: {m.content}")

    # Charlie receives
    charlie_msgs = bus.receive("charlie", max_msgs=10)
    print(f"  Charlie received {len(charlie_msgs)} messages:")
    for m in charlie_msgs:
        print(f"    [{m.msg_type.value}] from {m.sender_id}: {m.content}")

    print(f"\n  Bus stats: {bus.stats()}")


# ═══════════════════════════════════════════════════════════
# 2. Shared Memory
# ═══════════════════════════════════════════════════════════
def demo_shared_memory():
    sep("2. Shared Memory — Community Knowledge Store")
    from cognifield.memory.shared_memory import SharedMemory

    sm = SharedMemory()
    print("\n  Three agents writing to shared memory:\n")

    # Agent A writes from direct observation
    sm.write("apple.edible", True,  "agent_A", confidence=0.85)
    sm.write("stone.edible", False, "agent_A", confidence=0.92)

    # Agent B confirms apple
    sm.write("apple.edible", True,  "agent_B", confidence=0.78)
    sm.write("stone.edible", False, "agent_B", confidence=0.80)

    # Agent C has a different experience with purple_berry
    sm.write("purple_berry.edible", True,  "agent_C", confidence=0.60)
    sm.write("purple_berry.edible", False, "agent_A", confidence=0.55)

    print(f"  {'Key':25s} | {'Value':6s} | {'Conf':5s} | {'Ev':5s} | {'N':2s} | {'Contested'}")
    print(f"  {'─'*25} | {'─'*6} | {'─'*5} | {'─'*5} | {'─'*2} | {'─'*9}")
    for entry in list(sm.get_all()):
        print(f"  {entry.key:25s} | {str(entry.value):6s} | "
              f"{entry.confidence:.3f} | {entry.total_evidence:.1f} | "
              f"{entry.n_contributors:2d} | {entry.is_contested}")

    print(f"\n  Trust-weighted read (apple.edible):")
    trust_map = {"agent_A": 0.85, "agent_B": 0.70, "agent_C": 0.50}
    val, conf = sm.read_weighted_by_trust("apple.edible", trust_map)
    print(f"    result={val}, confidence={conf:.3f}")

    print(f"\n  Contested keys: {sm.contested_keys()}")
    print(f"  Edible (conf>=0.65): {sm.find_edible(min_conf=0.65)}")
    print(f"  Dangerous (conf>=0.65): {sm.find_dangerous(min_conf=0.65)}")
    print(f"  Summary: {sm.summary()}")


# ═══════════════════════════════════════════════════════════
# 3. Trust System
# ═══════════════════════════════════════════════════════════
def demo_trust():
    sep("3. Trust System — Agent Reputation")
    from cognifield.agents.trust_system import TrustSystem

    # Agent A evaluates peers B and C
    trust = TrustSystem(owner_id="agent_A")
    trust.register_peer("agent_B")
    trust.register_peer("agent_C")

    print("\n  Simulating accuracy updates:")
    print(f"  {'Event':40s} | B_trust | C_trust")
    print(f"  {'─'*40} | {'─'*7} | {'─'*7}")

    def show(label):
        bt = trust.get_trust("agent_B")
        ct = trust.get_trust("agent_C")
        print(f"  {label:40s} | {bt:.4f}  | {ct:.4f}")

    show("Initial (neutral 0.5)")

    # B is mostly correct
    for _ in range(5):
        trust.update_accuracy("agent_B", was_correct=True)
    show("B: 5 correct observations")

    # C is mostly wrong
    for _ in range(3):
        trust.update_accuracy("agent_C", was_correct=False)
    show("C: 3 wrong observations")

    # More evidence
    for _ in range(3):
        trust.update_accuracy("agent_B", was_correct=True)
    for _ in range(2):
        trust.update_accuracy("agent_C", was_correct=True)
    show("B: +3 correct, C: +2 correct")

    print(f"\n  Message weights (confidence=0.8):")
    print(f"    B: weight = {trust.message_weight('agent_B', 0.8):.4f}")
    print(f"    C: weight = {trust.message_weight('agent_C', 0.8):.4f}")

    print(f"\n  Ranked peers: {trust.ranked_peers()}")
    print(f"  Trusted peers (threshold=0.6): {trust.trusted_peers(0.6)}")
    print(f"  Distrusted (threshold=0.4):    {trust.distrusted_peers(0.4)}")

    # Decay
    trust.decay()
    show("After decay (toward neutral)")
    print(f"\n  Trust summary: {trust.summary()}")


# ═══════════════════════════════════════════════════════════
# 4. Consensus Engine
# ═══════════════════════════════════════════════════════════
def demo_consensus():
    sep("4. Consensus Engine — Multi-Agent Belief Aggregation")
    from cognifield.reasoning.consensus_engine import (
        ConsensusEngine, AgentVote, ConsensusStrategy
    )
    from cognifield.world_model.belief_system import BeliefSystem

    ce = ConsensusEngine(supermajority_threshold=0.60)

    section("Scenario A: Clear consensus (3/3 agree)")
    votes_agree = [
        AgentVote("A", True,  0.85, evidence=4.0, trust=0.85),
        AgentVote("B", True,  0.78, evidence=3.0, trust=0.70),
        AgentVote("C", True,  0.72, evidence=2.0, trust=0.60),
    ]
    result = ce.reach_consensus("apple.edible", votes_agree,
                                ConsensusStrategy.TRUST_WEIGHTED)
    print(f"  Key: apple.edible")
    print(f"  Votes: A=True(0.85), B=True(0.78), C=True(0.72)")
    print(f"  → value={result.value}, conf={result.confidence:.3f}, "
          f"agreement={result.agreement:.1%}, contested={result.contested}")

    section("Scenario B: Split (2 vs 1, strong evidence)")
    votes_split = [
        AgentVote("A", False, 0.92, evidence=6.0, trust=0.88),
        AgentVote("B", False, 0.80, evidence=4.0, trust=0.75),
        AgentVote("C", True,  0.55, evidence=1.5, trust=0.45),
    ]
    result2 = ce.reach_consensus("stone.edible", votes_split,
                                  ConsensusStrategy.EVIDENCE_WEIGHTED)
    print(f"  Key: stone.edible")
    print(f"  Votes: A=False(0.92/ev=6), B=False(0.80/ev=4), C=True(0.55/ev=1.5)")
    print(f"  → value={result2.value}, conf={result2.confidence:.3f}, "
          f"agreement={result2.agreement:.1%}, contested={result2.contested}")
    print(f"  Notes: {result2.notes}")

    section("Scenario C: True conflict — no supermajority")
    votes_conflict = [
        AgentVote("A", True,  0.70, evidence=3.0, trust=0.70),
        AgentVote("B", False, 0.68, evidence=2.8, trust=0.68),
        AgentVote("C", True,  0.55, evidence=1.5, trust=0.55),
    ]
    result3 = ce.reach_consensus("mystery.edible", votes_conflict,
                                  ConsensusStrategy.SUPERMAJORITY)
    print(f"  Key: mystery.edible")
    print(f"  Votes: A=True(0.70), B=False(0.68), C=True(0.55)")
    print(f"  → value={result3.value}, contested={result3.contested}")
    print(f"  Notes: {result3.notes}")
    print(f"  Contested keys: {ce.get_contested_keys()}")

    section("Building votes from BeliefSystems")
    bs_a = BeliefSystem()
    bs_b = BeliefSystem()
    for _ in range(4): bs_a.update("bread.edible", True, "direct_observation")
    for _ in range(3): bs_b.update("bread.edible", True, "direct_observation")
    votes_bs = ConsensusEngine.votes_from_beliefs(
        "bread.edible",
        {"A": bs_a, "B": bs_b},
        trust_scores={"A": 0.85, "B": 0.72},
    )
    result4 = ce.reach_consensus("bread.edible", votes_bs)
    print(f"  bread.edible consensus → {result4.value} (conf={result4.confidence:.3f})")
    print(f"\n  Engine summary: {ce.summary()}")


# ═══════════════════════════════════════════════════════════
# 5. AgentV6 Roles
# ═══════════════════════════════════════════════════════════
def demo_roles():
    sep("5. Agent Roles — Specialised Behaviour")
    from cognifield.agents.agent_v6 import (
        CogniFieldAgentV6, AgentV6Config, AgentRole
    )
    from cognifield.communication.communication_module import CommunicationModule
    from cognifield.memory.shared_memory import SharedMemory

    bus = CommunicationModule()
    sm  = SharedMemory()

    roles = [AgentRole.EXPLORER, AgentRole.ANALYST,
             AgentRole.RISK_MANAGER, AgentRole.PLANNER]

    print(f"\n  {'Role':15s} | {'nov_thresh':10s} | {'risk_tol':8s} | "
          f"{'share_freq':10s} | {'goal_bias'}")
    print(f"  {'─'*15} | {'─'*10} | {'─'*8} | {'─'*10} | {'─'*15}")

    for role in roles:
        cfg = AgentV6Config(
            agent_id=f"agent_{role.value}",
            role=role, dim=64, verbose=False,
        )
        agent = CogniFieldAgentV6(config=cfg, comm_bus=bus, shared_mem=sm)

        from cognifield.agents.agent_v6 import ROLE_TRAITS
        traits = ROLE_TRAITS[role]
        print(f"  {role.value:15s} | "
              f"{cfg.novelty_threshold:.3f}     | "
              f"{cfg.risk_tolerance:.4f}   | "
              f"{traits['share_freq']:10d} | "
              f"{traits['goal_bias']}")

    print(f"\n  Role-specific behaviours:")
    print(f"    EXPLORER:      lower novelty threshold → explores more")
    print(f"    ANALYST:       higher novelty threshold → more conservative")
    print(f"    RISK_MANAGER:  broadcasts warnings for known dangers")
    print(f"    PLANNER:       focuses on goal execution efficiency")


# ═══════════════════════════════════════════════════════════
# 6. Social Learning Demo
# ═══════════════════════════════════════════════════════════
def demo_social_learning():
    sep("6. Social Learning — Learning from Peers Without Direct Experience")
    from cognifield.agents.agent_v6 import (
        CogniFieldAgentV6, AgentV6Config, AgentRole
    )
    from cognifield.communication.communication_module import CommunicationModule
    from cognifield.memory.shared_memory import SharedMemory

    bus = CommunicationModule()
    sm  = SharedMemory()

    # Create two agents: A (Explorer) and B (Analyst)
    agent_A = CogniFieldAgentV6(
        config=AgentV6Config(agent_id="A", role=AgentRole.EXPLORER,
                             dim=64, verbose=False),
        comm_bus=bus, shared_mem=sm
    )
    agent_B = CogniFieldAgentV6(
        config=AgentV6Config(agent_id="B", role=AgentRole.ANALYST,
                             dim=64, verbose=False),
        comm_bus=bus, shared_mem=sm
    )

    print(f"\n  Initial beliefs about stone.edible:")
    print(f"    A: conf={agent_A.how_confident('stone','edible'):.3f}")
    print(f"    B: conf={agent_B.how_confident('stone','edible'):.3f}")

    # Agent A learns stone is dangerous through direct experience
    section("Agent A directly learns: stone.edible=False")
    for _ in range(4):
        agent_A.beliefs.update("stone.edible", False, "direct_observation", weight=1.0)
    print(f"  A teaches itself: stone.edible=False")
    print(f"  A's confidence: {agent_A.how_confident('stone','edible'):.3f}")

    # A broadcasts its finding
    section("Agent A broadcasts warning to B")
    from cognifield.communication.communication_module import Message
    warning = Message.warning_msg("A", "stone", "edible", False,
                                  confidence=0.90)
    bus.broadcast(warning)
    print(f"  A broadcasts: stone.edible=False (conf=0.90)")
    print(f"  B inbox size: {bus.peek('B')}")

    # B processes the message (social learning)
    section("Agent B processes message (social learning)")
    agent_B.trust.register_peer("A")
    for _ in range(3):
        agent_B.trust.update_accuracy("A", was_correct=True)  # B trusts A

    n_rx, n_updates, n_social = agent_B._process_incoming_messages(verbose=True)

    print(f"\n  B received {n_rx} messages, updated {n_updates} beliefs")
    print(f"  B's stone.edible confidence: "
          f"{agent_B.how_confident('stone','edible'):.3f}")
    print(f"  B learned from A WITHOUT direct experience ✓")

    # A broadcasts an observation after eating apple
    section("Agent A broadcasts: eat(apple) → success")
    agent_A.beliefs.update("apple.edible", True, "direct_observation", weight=1.0)
    obs = Message.observation_msg("A", "eat", "apple", "success", 0.5)
    bus.broadcast(obs)
    n_rx2, n_up2, n_soc2 = agent_B._process_incoming_messages(verbose=True)
    print(f"  B inferred apple.edible from observation: "
          f"conf={agent_B.how_confident('apple','edible'):.3f}")


# ═══════════════════════════════════════════════════════════
# 7. Full Multi-Agent Loop
# ═══════════════════════════════════════════════════════════
def demo_full_multiagent():
    sep("7. Full Multi-Agent System — Collaborative Learning")
    from cognifield.agents.agent_manager import AgentManager
    from cognifield.agents.agent_v6 import AgentRole
    from cognifield.environment.rich_env import RichEnv

    env = RichEnv(seed=42)
    mgr = AgentManager(
        num_agents=3,
        roles=[AgentRole.EXPLORER, AgentRole.ANALYST, AgentRole.RISK_MANAGER],
        env=env,
        dim=64,
        seed=42,
        verbose=False,
    )

    # ── Phase A: Teach all agents ──
    section("Phase A: Teaching world knowledge to all agents")
    for name, props in [
        ("apple",  {"edible":True,  "category":"food",     "color":"red"}),
        ("stone",  {"edible":False, "category":"material", "heavy":True}),
        ("bread",  {"edible":True,  "category":"food",     "color":"yellow"}),
    ]:
        mgr.teach_all(
            f"{name} {' '.join(f'{k} {v}' for k,v in props.items())}",
            label=name, props=props
        )
        for agent in mgr.agents:
            for action, success, reward in [
                ("eat",  name in ("apple","bread"), 0.5 if name in ("apple","bread") else -0.2),
                ("pick", True, 0.1),
            ]:
                v = np.random.randn(agent.cfg.dim).astype(np.float32)
                v /= np.linalg.norm(v) + 1e-8
                agent.world_model.record(v, action, v, reward, success,
                                         name, props["category"])

    print(f"  All {len(mgr.agents)} agents taught {len(mgr.shared_mem)} shared beliefs")
    print(f"  Shared memory: {[e.key for e in mgr.shared_mem.get_all()]}")

    # ── Phase B: Add goals ──
    section("Phase B: Setting goals")
    from cognifield.agent.goals import GoalType
    for agent in mgr.agents:
        agent.add_goal("eat apple", GoalType.EAT_OBJECT, target="apple", priority=0.85)
        agent.add_goal("avoid stone", GoalType.AVOID, target="stone", priority=0.95)

    # ── Phase C: Run 20 rounds ──
    section("Phase C: 20 rounds of multi-agent learning")
    log = mgr.run_episode(n_rounds=20, verbose=True)

    # ── Phase D: Belief agreement matrix ──
    section("Phase D: Belief agreement across agents")
    for key in ["apple.edible", "stone.edible", "bread.edible"]:
        mat = mgr.belief_agreement_matrix(key)
        print(f"\n  Key: {key}")
        print(f"  Agreement: {mat['agreement']:.1%}  Plurality: {mat['plurality']}")
        for aid, info in mat["per_agent"].items():
            b_status = "reliable" if info.get("reliable") else "uncertain"
            print(f"    {aid}: {info['value']} (conf={info['confidence']:.3f}, {b_status})")

    # ── Phase E: Force consensus on contested key ──
    section("Phase E: Forcing consensus on contested beliefs")
    # Inject a conflict
    agents = mgr.agents
    agents[0].beliefs.update("purple_berry.edible", True,  "hypothesis",   weight=0.4)
    agents[1].beliefs.update("purple_berry.edible", False, "simulation",   weight=0.4)
    agents[2].beliefs.update("purple_berry.edible", True,  "direct_observation", weight=0.6)

    print(f"  Injected conflict on purple_berry.edible:")
    for a in agents:
        b = a.beliefs.get("purple_berry.edible")
        if b:
            print(f"    {a.agent_id}({a.role.value}): {b.value} (conf={b.confidence:.3f})")

    consensus_val = mgr.force_consensus("purple_berry.edible")
    print(f"\n  Consensus result: purple_berry.edible = {consensus_val}")
    print(f"  After consensus, all agents believe:")
    for a in agents:
        b = a.beliefs.get("purple_berry.edible")
        if b:
            print(f"    {a.agent_id}: {b.value} (conf={b.confidence:.3f})")

    # ── Phase F: Social learning — C learns from A without testing ──
    section("Phase F: Agent C learns from Agent A via shared memory")
    agent_A = agents[0]
    agent_C = agents[2]

    # A gains more confidence about apple
    for _ in range(3):
        agent_A.beliefs.update("apple.edible", True, "direct_observation")

    # Sync A to shared memory
    agent_A._sync_to_shared_memory()

    # C reads from shared memory
    agent_C.trust.register_peer(agent_A.agent_id)
    for _ in range(4):
        agent_C.trust.update_accuracy(agent_A.agent_id, was_correct=True)

    trust_map = {aid: agent_C.trust.get_trust(aid) for aid in mgr.agent_ids()}
    n_imported = agent_C.learn_from_shared_memory(min_conf=0.70)
    print(f"  C imported {n_imported} beliefs from shared memory")
    print(f"  C's apple.edible confidence: "
          f"{agent_C.how_confident('apple','edible'):.3f}")
    print(f"  C's stone.edible confidence: "
          f"{agent_C.how_confident('stone','edible'):.3f}")

    # ── Phase G: Final system summary ──
    section("Phase G: Final System Summary")
    summary = mgr.summary()
    print(f"  Agents: {summary['n_agents']}  |  "
          f"Rounds: {summary['n_rounds']}  |  "
          f"Shared beliefs: {summary['shared_beliefs']}")
    print(f"\n  Per-agent stats:")
    print(f"  {'Agent':10s} | {'Role':14s} | {'Steps':5s} | "
          f"{'Beliefs':7s} | {'Grade':5s} | {'Msgs Rx':7s} | {'Msgs Tx'}")
    print(f"  {'─'*10} | {'─'*14} | {'─'*5} | "
          f"{'─'*7} | {'─'*5} | {'─'*7} | {'─'*7}")
    for aid, info in summary["agents"].items():
        print(f"  {aid:10s} | {info['role']:14s} | "
              f"{info['steps']:5d} | {info['beliefs']:7d} | "
              f"{info['grade']:5s} | {info['msgs_rx']:7d} | {info['msgs_tx']}")

    print(f"\n  Consensus engine: {summary['consensus']}")
    print(f"  Shared memory: {mgr.shared_mem.summary()}")
    print(f"\n  Shared knowledge (conf >= 0.60):")
    sk = mgr.shared_knowledge(min_conf=0.60)
    for k, v in list(sk.items())[:8]:
        cont = "⚡" if v["contested"] else "✓"
        print(f"    {cont} {k:25s} = {str(v['value']):6s} "
              f"(conf={v['confidence']:.3f}, "
              f"contributors={v['contributors']})")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   CogniField v6 — Multi-Agent Collaborative Demo        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_communication()
    demo_shared_memory()
    demo_trust()
    demo_consensus()
    demo_roles()
    demo_social_learning()
    demo_full_multiagent()

    print("\n" + "═"*62)
    print("  v6 Demo complete.")
    print("═"*62)
