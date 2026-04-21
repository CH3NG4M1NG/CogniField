"""
examples/demo_v8.py
====================
CogniField v8 — Collective Intelligence Demo

Demonstrates all 8 v8 improvements:
  1. EventBus           — event-driven publish/subscribe
  2. GlobalConsensus    — fleet-wide belief aggregation + broadcast
  3. GroupMind          — shared goals + coordination signals
  4. SharedMemory 2.0   — continuous read/write, versioned
  5. Bidirectional comm — tx > 0 AND rx > 0 for every agent
  6. Load-balanced coop — fair task distribution
  7. Consistency        — no contradictions survive
  8. Full v8 scenario   — agents disagree → negotiate → consensus → cooperate
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np


def sep(title): print(f"\n{'═'*64}\n  {title}\n{'═'*64}")
def section(title): print(f"\n  {'─'*60}\n  {title}\n  {'─'*60}")


# ═══════════════════════════════════════════════════════════
# 1. Event Bus
# ═══════════════════════════════════════════════════════════
def demo_event_bus():
    sep("1. Event Bus — Publish / Subscribe")
    from cognifield.core.event_bus import EventBus, Event, EventType

    bus = EventBus()
    reactions = []

    # Subscribe handlers
    def on_belief(event):
        reactions.append(f"belief_handler: {event.payload.get('key')}="
                         f"{event.payload.get('value')}")

    def on_warning(event):
        reactions.append(f"WARNING: {event.payload.get('key')} "
                         f"from {event.source}")

    def on_any(event):
        reactions.append(f"any_handler: {event.event_type.value}")

    bus.subscribe(EventType.BELIEF_UPDATED, on_belief)
    bus.subscribe(EventType.WARNING_ISSUED, on_warning)
    bus.subscribe_all(on_any)

    print(f"\n  Firing events:")
    n1 = bus.fire(EventType.BELIEF_UPDATED, "agent_A",
                  key="apple.edible", value=True, confidence=0.85)
    print(f"    BELIEF_UPDATED  → {n1} handlers called")

    n2 = bus.fire(EventType.WARNING_ISSUED, "risk_manager",
                  key="stone.edible", value=False, confidence=0.92)
    print(f"    WARNING_ISSUED  → {n2} handlers called")

    n3 = bus.fire(EventType.CONSENSUS_REACHED, "global_consensus",
                  key="bread.edible", value=True, agreement=0.95)
    print(f"    CONSENSUS_REACHED → {n3} handlers called")

    n4 = bus.fire(EventType.GOAL_COMPLETED, "agent_B", goal="eat apple")
    print(f"    GOAL_COMPLETED  → {n4} handlers called")

    print(f"\n  Reaction log ({len(reactions)} reactions):")
    for r in reactions[:8]:
        print(f"    • {r}")

    print(f"\n  Summary: {bus.summary()}")
    print(f"  Recent events: {[e.event_type.value for e in bus.recent_events(4)]}")


# ═══════════════════════════════════════════════════════════
# 2. Global Consensus
# ═══════════════════════════════════════════════════════════
def demo_global_consensus():
    sep("2. Global Consensus — Fleet-Wide Belief Aggregation")
    from cognifield.reasoning.global_consensus import GlobalConsensus
    from cognifield.memory.shared_memory import SharedMemory
    from cognifield.communication.communication_module import CommunicationModule
    from cognifield.core.event_bus import EventBus
    from cognifield.world_model.belief_system import BeliefSystem

    sm  = SharedMemory()
    bus = CommunicationModule()
    for aid in ["A","B","C"]: bus.register(aid)
    eb  = EventBus()
    gc  = GlobalConsensus(sm, bus, eb, supermajority=0.55, min_agents=2)

    # Build agent beliefs
    def make_bs(vals):
        bs = BeliefSystem()
        for k, v, conf, n in vals:
            for _ in range(n): bs.update(k, v, "direct_observation")
        return bs

    agent_beliefs = {
        "A": make_bs([("apple.edible",True,0.9,5), ("stone.edible",False,0.9,4)]),
        "B": make_bs([("apple.edible",True,0.8,4), ("stone.edible",False,0.8,3)]),
        "C": make_bs([("apple.edible",False,0.6,2),("stone.edible",True,0.5,1)]),
    }
    trust_map = {"A":0.85,"B":0.78,"C":0.45}

    section("Single-key consensus")
    for key in ["apple.edible", "stone.edible"]:
        rec = gc.resolve_key(key, agent_beliefs, trust_map)
        if rec:
            print(f"  {key}: value={rec.value}, conf={rec.confidence:.3f}, "
                  f"agreement={rec.agreement:.1%}, contested={rec.contested}")

    section("Full fleet round")
    results = gc.run_round(agent_beliefs, trust_map)
    print(f"\n  Resolved {len(results)} beliefs:")
    for k, rec in results.items():
        auth = "✓ auth" if rec.is_authoritative else "~ weak"
        print(f"    {k:25s} = {str(rec.value):6s} ({auth}, "
              f"conf={rec.confidence:.3f}, v{rec.version})")

    print(f"\n  Shared memory after consensus: "
          f"{len(list(sm.get_all()))} entries")
    print(f"\n  EventBus fired {eb.summary()['total_events']} events")

    section("Consistency enforcement")
    # Agent C now has wrong beliefs — enforce
    n_fixes = gc.enforce_consistency(agent_beliefs)
    print(f"  Consistency fixes: {n_fixes}")
    c_apple = agent_beliefs["C"].get("apple.edible")
    if c_apple:
        print(f"  Agent C's apple.edible after enforcement: "
              f"{c_apple.value} (conf={c_apple.confidence:.3f})")

    print(f"\n  GlobalConsensus summary: {gc.summary()}")


# ═══════════════════════════════════════════════════════════
# 3. GroupMind
# ═══════════════════════════════════════════════════════════
def demo_group_mind():
    sep("3. GroupMind — Shared Goals & Coordination Signals")
    from cognifield.agents.group_mind import GroupMind, CoordSignal
    from cognifield.core.event_bus import EventBus

    eb = EventBus()
    gm = GroupMind(event_bus=eb)

    section("Shared goals")
    gm.set_primary_goal("eat apple")
    gm.add_secondary_goal("explore unknowns")
    gm.add_secondary_goal("avoid stone")
    print(f"  Primary goal:    {gm.get_primary_goal()}")
    print(f"  All goals:       {gm.active_goals()}")

    section("Coordination signals")
    signals_fired = []
    for sig, dur in [
        (CoordSignal.EXPLORE,     20),
        (CoordSignal.CAUTIOUS,    15),
        (CoordSignal.ACCELERATE,  10),
        (CoordSignal.SYNC,         5),
    ]:
        gm.broadcast_signal(sig, duration_sec=dur)
        current = gm.current_signal()
        signals_fired.append(f"{sig.value}→current={current.value if current else 'expired'}")

    print(f"  Signal history: {gm.recent_signals(5)}")

    section("Experience sharing")
    experiences = [
        ("explorer", "eat",  "apple",  "success", +0.50, "apple.edible", True),
        ("analyst",  "eat",  "stone",  "failure", -0.20, "stone.edible", False),
        ("explorer", "pick", "bread",  "success", +0.10, "",             None),  # below threshold
        ("analyst",  "eat",  "bread",  "success", +0.50, "bread.edible", True),
    ]
    for src, act, tgt, out, rew, bk, bv in experiences:
        shared = gm.share_experience(src, act, tgt, out, rew, bk, bv, 0.85)
        print(f"  share_experience({src}, {act}({tgt}), r={rew:+.2f}) "
              f"→ {'✓ shared' if shared else '✗ below threshold'}")

    print(f"\n  High-value experiences: {len(gm.get_high_value_experiences())}")
    exps = gm.get_experiences_about("apple")
    print(f"  Experiences about apple: {len(exps)}")

    section("Fleet state update")
    # Mock agents
    class MockAgent:
        def __init__(self, conf, sr):
            class B:
                def summary(self): return {"mean_conf": conf}
            class M:
                def success_rate(self): return sr
            class G:
                def summary(self): return {"active_goals":["eat apple"]}
            from cognifield.world_model.belief_system import BeliefSystem
            self.beliefs      = BeliefSystem()
            self.metrics      = M()
            self.goal_system  = G()

    mock_agents = [MockAgent(0.72, 0.65), MockAgent(0.68, 0.58)]
    fs = gm.update_fleet_state(mock_agents, n_contested=1)
    print(f"  Fleet state: n_agents={fs.n_agents}, "
          f"mean_conf={fs.mean_belief_conf:.3f}, "
          f"mean_sr={fs.mean_success_rate:.3f}")

    print(f"\n  GroupMind summary: {gm.summary()}")


# ═══════════════════════════════════════════════════════════
# 4. Shared Memory 2.0 — continuous read/write
# ═══════════════════════════════════════════════════════════
def demo_shared_memory_v2():
    sep("4. Shared Memory 2.0 — Continuous R/W + Version Tracking")
    from cognifield.memory.shared_memory import SharedMemory

    sm = SharedMemory(max_entries=1000)

    section("Versioned writes")
    print(f"\n  {'Agent':10s} | {'Key':20s} | {'Value':6s} | {'Conf':5s} | "
          f"{'V':2s} | {'Contributors'}")
    print(f"  {'─'*10} | {'─'*20} | {'─'*6} | {'─'*5} | {'─'*2} | {'─'*15}")

    writes = [
        ("agent_A", "apple.edible", True,  0.85),
        ("agent_B", "apple.edible", True,  0.78),
        ("agent_C", "apple.edible", False, 0.55),
        ("agent_A", "stone.edible", False, 0.92),
        ("agent_B", "stone.edible", False, 0.88),
        ("agent_A", "bread.edible", True,  0.81),
        ("agent_B", "bread.edible", True,  0.76),
        ("agent_C", "bread.edible", True,  0.70),
    ]
    for agent, key, val, conf in writes:
        sm.write(key, val, agent, conf)
        e = sm.read(key)
        print(f"  {agent:10s} | {key:20s} | {str(val):6s} | "
              f"{e.confidence:.3f} | {e.write_count:2d} | "
              f"{list(e.sources.keys())}")

    section("Trust-weighted reads")
    trust_map = {"agent_A": 0.90, "agent_B": 0.75, "agent_C": 0.35}
    for key in ["apple.edible", "stone.edible", "bread.edible"]:
        val, conf = sm.read_weighted_by_trust(key, trust_map)
        print(f"  {key:20s}: val={val}, trust-weighted-conf={conf:.3f}")

    section("Contested key detection")
    print(f"\n  Contested keys: {sm.contested_keys()}")
    print(f"  Edible (conf>=0.65): {sm.find_edible(0.65)}")
    print(f"  Dangerous (conf>=0.65): {sm.find_dangerous(0.65)}")
    print(f"\n  Summary: {sm.summary()}")


# ═══════════════════════════════════════════════════════════
# 5. Full v8 System — The Complete Scenario
# ═══════════════════════════════════════════════════════════
def demo_full_v8():
    sep("5. Full v8 System — Collective Intelligence in Action")
    from cognifield.agents.agent_v8 import CogniFieldAgentV8, AgentV8Config
    from cognifield.agents.agent_v7 import AgentRole
    from cognifield.agents.group_mind import GroupMind, CoordSignal
    from cognifield.reasoning.global_consensus import GlobalConsensus
    from cognifield.core.event_bus import EventBus, EventType
    from cognifield.communication.communication_module import CommunicationModule
    from cognifield.memory.shared_memory import SharedMemory
    from cognifield.planning.cooperation_engine import CooperationEngine, TaskType
    from cognifield.environment.rich_env import RichEnv
    from cognifield.agents.goals import GoalType

    # Infrastructure
    env      = RichEnv(seed=42)
    bus      = CommunicationModule()
    sm       = SharedMemory()
    eb       = EventBus()
    gm       = GroupMind(event_bus=eb)
    coop_eng = CooperationEngine()
    gc       = GlobalConsensus(sm, bus, eb, supermajority=0.55)

    event_log = []
    def log_event(e):
        event_log.append(f"{e.event_type.value}:{e.source}")
    eb.subscribe_all(log_event)

    # Create 3 v8 agents
    agents = []
    roles  = [AgentRole.EXPLORER, AgentRole.ANALYST, AgentRole.RISK_MANAGER]
    for i, role in enumerate(roles):
        cfg = AgentV8Config(
            agent_id=f"v8_{i}", role=role, dim=64,
            verbose=False, seed=42+i,
        )
        a = CogniFieldAgentV8(
            config=cfg, env=env, comm_bus=bus, shared_mem=sm,
            group_mind=gm, global_cons=gc, event_bus=eb, coop_engine=coop_eng
        )
        agents.append(a)
        coop_eng.register_agent(a.agent_id, role.value)

    # Register negotiation
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            if i != j:
                a.register_for_negotiation(b)

    # ── Phase A: World knowledge ──
    section("Phase A: Teaching world knowledge")
    for name, props in [
        ("apple", {"edible":True,  "category":"food"}),
        ("stone", {"edible":False, "category":"material"}),
        ("bread", {"edible":True,  "category":"food"}),
    ]:
        for a in agents:
            a.teach(f"{name} {' '.join(f'{k} {v}' for k,v in props.items())}",
                    name, props)
            v = np.random.randn(a.cfg.dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-8
            a.world_model.record(v, "eat", v,
                                  0.5 if props["edible"] else -0.2,
                                  props["edible"], name, props["category"])
    print(f"  Agents taught world knowledge")

    # ── Phase B: Inject disagreement ──
    section("Phase B: Injecting belief conflict (purple_berry)")
    agents[0].beliefs.update("purple_berry.edible", True,  "direct_observation", weight=0.9)
    agents[0].beliefs.update("purple_berry.edible", True,  "direct_observation", weight=0.9)
    agents[1].beliefs.update("purple_berry.edible", False, "inference",          weight=0.5)
    agents[2].beliefs.update("purple_berry.edible", False, "simulation",         weight=0.4)

    print(f"\n  purple_berry.edible — initial state:")
    for a in agents:
        b = a.beliefs.get("purple_berry.edible")
        if b:
            print(f"    {a.agent_id}({a.role.value}): {b.value} "
                  f"(conf={b.confidence:.3f})")

    # ── Phase C: Set group goal + signal ──
    section("Phase C: GroupMind sets primary goal + SYNC signal")
    gm.set_primary_goal("eat apple")
    gm.broadcast_signal(CoordSignal.SYNC, duration_sec=60)
    print(f"  Primary goal: '{gm.get_primary_goal()}'")
    print(f"  Coordination signal: {gm.current_signal().value}")

    # ── Phase D: Autonomous run (20 rounds) ──
    section("Phase D: 20-round autonomous run")
    from cognifield.agents.goals import GoalType
    for a in agents:
        a.add_goal("eat apple", GoalType.EAT_OBJECT, target="apple", priority=0.85)

    print(f"\n  {'Rnd':3s}|{'Agent':8s}|{'Role':14s}|"
          f"{'Action':10s}|{'Rew':5s}|{'TX':3s}|{'RX':3s}|"
          f"{'SM_R':4s}|{'SM_W':4s}|{'Sig':10s}")
    print(f"  {'─'*3}|{'─'*8}|{'─'*14}|"
          f"{'─'*10}|{'─'*5}|{'─'*3}|{'─'*3}|"
          f"{'─'*4}|{'─'*4}|{'─'*10}")

    for round_n in range(20):
        # Update fleet state
        gc_results = {}
        if round_n % 5 == 4:
            trust_map = {}
            for a in agents:
                for peer_id in [ag.agent_id for ag in agents if ag.agent_id != a.agent_id]:
                    trust_map[peer_id] = (trust_map.get(peer_id, 0)
                                          + a.effective_trust(peer_id))
            for k in trust_map: trust_map[k] /= (len(agents) - 1)
            ab = {a.agent_id: a.beliefs for a in agents}
            gc_results = gc.run_round(ab, trust_map)
            gc.apply_to_all(ab)

        gm.update_fleet_state(agents, n_contested=gc.summary()["contested"])

        for a in agents:
            # Ensure bidirectional comm
            a.ensure_bidirectional_comm()
            s = a.step(verbose=False)
            if round_n % 4 == 3:
                rew = f"{s.env_reward:+.2f}" if s.env_reward else "  N/A"
                print(f"  {round_n+1:3d}|{a.agent_id:8s}|{a.role.value:14s}|"
                      f"{(s.action_taken or '–')[:10]:10s}|{rew:5s}|"
                      f"{s.messages_sent:3d}|{s.messages_received:3d}|"
                      f"{s.shared_mem_reads:4d}|{s.shared_mem_writes:4d}|"
                      f"{s.coord_signal[:10]:10s}")

    # ── Phase E: Bidirectional communication check ──
    section("Phase E: Bidirectional Communication Verification")
    print(f"\n  {'Agent':10s} | {'TX':5s} | {'RX':5s} | {'Bidirectional':12s}")
    print(f"  {'─'*10} | {'─'*5} | {'─'*5} | {'─'*12}")
    for a in agents:
        bi = a._msgs_sent_total > 0 and a._msgs_received_total > 0
        icon = "✓" if bi else "✗"
        print(f"  {a.agent_id:10s} | {a._msgs_sent_total:5d} | "
              f"{a._msgs_received_total:5d} | {icon} {bi}")

    # ── Phase F: Global consensus result ──
    section("Phase F: Global Consensus State")
    gc_summ = gc.summary()
    print(f"\n  GlobalConsensus after {gc_summ['rounds']} rounds:")
    print(f"    Authoritative beliefs: {gc_summ['authoritative']}")
    print(f"    Contested:             {gc_summ['contested']}")
    print(f"    Mean confidence:       {gc_summ['mean_confidence']:.3f}")
    print(f"    Mean agreement:        {gc_summ['mean_agreement']:.3f}")
    print(f"\n  Authoritative beliefs:")
    for key, rec in gc.get_authoritative().items():
        print(f"    {key:25s} = {str(rec.value):6s} "
              f"(conf={rec.confidence:.3f}, v{rec.version})")

    # ── Phase G: Shared memory state ──
    section("Phase G: Shared Memory Usage")
    sm_summ = sm.summary()
    print(f"\n  Shared memory: {sm_summ}")
    print(f"\n  Content (conf >= 0.55):")
    for entry in list(sm.get_all(min_conf=0.55)):
        cont = "⚡" if entry.is_contested else "✓"
        print(f"    {cont} {entry.key:25s} = {str(entry.value):6s} "
              f"(conf={entry.confidence:.3f}, "
              f"n_contrib={entry.n_contributors})")

    # ── Phase H: Task load balance ──
    section("Phase H: Cooperative Task Load Balance")
    print(f"\n  Cooperation engine: {coop_eng.summary()}")
    print(f"\n  Agent task completion:")
    for aid, info in coop_eng._agents.items():
        print(f"    {aid}: workload={info['workload']}, "
              f"completed={info['completed']}, failed={info['failed']}")

    # ── Phase I: Event bus summary ──
    section("Phase I: Event Bus Activity")
    eb_summ = eb.summary()
    print(f"\n  Total events fired: {eb_summ['total_events']}")
    print(f"  By type:")
    for t, c in sorted(eb_summ["by_type"].items()):
        print(f"    {t:25s}: {c}")

    # ── Phase J: Consistency check ──
    section("Phase J: Consistency — No Contradictions")
    all_consistent = True
    for a in agents:
        audit = a.consistency_engine.audit()
        ok    = audit["consistent"]
        if not ok:
            all_consistent = False
        print(f"  {a.agent_id}({a.role.value}): consistent={ok}, "
              f"violations={audit['n_violations']}, "
              f"downgraded={audit['downgraded']}")
    print(f"\n  Fleet-wide consistency: {'✓ OK' if all_consistent else '✗ violations'}")

    # ── Final Summary ──
    section("Final System Summary")
    print(f"\n  Agents: {len(agents)}, Rounds: 20, "
          f"Shared beliefs: {len(list(sm.get_all()))}")
    print(f"\n  {'Agent':10s} | {'Role':14s} | {'Beliefs':7s} | "
          f"{'Reliable':8s} | {'Grade':5s} | {'SM_R':5s} | {'SM_W':5s} | "
          f"{'Events':6s}")
    print(f"  {'─'*10} | {'─'*14} | {'─'*7} | "
          f"{'─'*8} | {'─'*5} | {'─'*5} | {'─'*5} | {'─'*6}")
    for a in agents:
        s8 = a.v8_summary()
        print(f"  {a.agent_id:10s} | {a.role.value:14s} | "
              f"{s8['beliefs']:7d} | {s8['reliable_beliefs']:8d} | "
              f"{s8['stability_grade']:5s} | {s8['shared_mem_reads']:5d} | "
              f"{s8['shared_mem_writes']:5d} | {s8['event_reactions']:6d}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   CogniField v8 — Collective Intelligence Demo          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    demo_event_bus()
    demo_global_consensus()
    demo_group_mind()
    demo_shared_memory_v2()
    demo_full_v8()
    print("\n" + "═"*64 + "\n  v8 Demo complete.\n" + "═"*64)
