"""
agents/agent_manager.py
========================
Multi-Agent Manager

Creates, manages, and coordinates a fleet of CogniFieldAgentV6 instances.
Provides synchronised stepping, broadcast messaging, consensus rounds,
and per-agent state collection.

Architecture
------------
  AgentManager
    ├── N × CogniFieldAgentV6   (private beliefs, goals, memory)
    ├── 1 × CommunicationModule  (shared bus)
    ├── 1 × SharedMemory         (shared knowledge store)
    └── 1 × ConsensusEngine      (belief aggregation)

Step modes
----------
SYNCHRONISED  – all agents act in lockstep; env is stepped once per round
ROUND_ROBIN   – agents act one at a time in rotation
PARALLEL      – agents act independently (simulated parallelism)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .agent_v6 import AgentV6Config, CogniFieldAgentV6, AgentRole, V6Step
from .trust_system import TrustSystem
from ..communication.communication_module import CommunicationModule, Message, MessageType
from ..memory.shared_memory import SharedMemory
from ..reasoning.consensus_engine import ConsensusEngine, AgentVote, ConsensusStrategy
from ..world_model.belief_system import BeliefSystem
from ..environment.rich_env import RichEnv


# ---------------------------------------------------------------------------
# Episode summary
# ---------------------------------------------------------------------------

@dataclass
class RoundSummary:
    """Summary statistics for one multi-agent round."""
    round_num:       int
    agent_steps:     Dict[str, V6Step]   # agent_id → step result
    total_reward:    float
    msgs_exchanged:  int
    consensus_keys:  List[str]
    new_shared_beliefs: int
    elapsed_ms:      float
    timestamp:       float = field(default_factory=time.time)


class AgentManager:
    """
    Manages a fleet of CogniFieldAgentV6 agents on a shared environment.

    Parameters
    ----------
    num_agents   : Number of agents to create.
    roles        : Optional list of AgentRole values (one per agent).
                   If shorter than num_agents, cycles through the list.
    env          : Shared RichEnv instance.
    dim          : Latent space dimension.
    seed         : Base random seed.
    verbose      : Print round summaries.
    """

    DEFAULT_ROLE_CYCLE = [
        AgentRole.EXPLORER,
        AgentRole.ANALYST,
        AgentRole.RISK_MANAGER,
        AgentRole.PLANNER,
    ]

    def __init__(
        self,
        num_agents: int = 3,
        roles:      Optional[List[AgentRole]] = None,
        env:        Optional[RichEnv]         = None,
        dim:        int   = 64,
        seed:       int   = 42,
        verbose:    bool  = False,
    ) -> None:
        self.verbose    = verbose
        self._round     = 0
        self._round_log: List[RoundSummary] = []

        # Shared infrastructure
        self.comm_bus   = CommunicationModule(max_queue=200)
        self.shared_mem = SharedMemory(max_entries=5_000)
        self.consensus  = ConsensusEngine(supermajority_threshold=0.60)
        self.env        = env

        # Build agents
        role_cycle = roles or self.DEFAULT_ROLE_CYCLE
        self.agents: List[CogniFieldAgentV6] = []
        for i in range(num_agents):
            role   = role_cycle[i % len(role_cycle)]
            cfg    = AgentV6Config(
                dim=dim,
                agent_id=f"agent_{i}",
                role=role,
                seed=seed + i,
                verbose=False,
                risk_tolerance=0.35,
            )
            agent  = CogniFieldAgentV6(
                config=cfg,
                env=env,
                comm_bus=self.comm_bus,
                shared_mem=self.shared_mem,
            )
            self.agents.append(agent)

        if verbose:
            print(f"  [AgentManager] {num_agents} agents created:")
            for a in self.agents:
                print(f"    {a.agent_id}: role={a.role.value}")

    # ------------------------------------------------------------------
    # Agent access
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> Optional[CogniFieldAgentV6]:
        for a in self.agents:
            if a.agent_id == agent_id:
                return a
        return None

    def agent_ids(self) -> List[str]:
        return [a.agent_id for a in self.agents]

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step_all(
        self,
        text_inputs:  Optional[Dict[str, str]] = None,
        force_actions: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> RoundSummary:
        """
        Step all agents one cycle synchronously.
        Each agent observes the shared environment and exchanges messages.

        Parameters
        ----------
        text_inputs   : {agent_id: text_input} — optional per-agent inputs.
        force_actions : {agent_id: (action, obj)} — optional forced actions.

        Returns
        -------
        RoundSummary
        """
        t0        = time.time()
        self._round += 1
        agent_steps: Dict[str, V6Step] = {}
        total_reward = 0.0

        text_inputs   = text_inputs   or {}
        force_actions = force_actions or {}

        for agent in self.agents:
            txt   = text_inputs.get(agent.agent_id, "")
            force = force_actions.get(agent.agent_id)
            s     = agent.step(text_input=txt, force_action=force, verbose=False)
            agent_steps[agent.agent_id] = s
            if s.env_reward is not None:
                total_reward += s.env_reward

            # After successful eat, share the observation
            if (s.env_success and s.action_taken == "eat"
                    and s.action_obj and agent.comm_bus):
                outcome = "success" if s.env_success else "failure"
                agent.share_observation("eat", s.action_obj, outcome,
                                        s.env_reward or 0.0)

        # Tally messages exchanged this round
        stats = self.comm_bus.stats()
        msgs_exchanged = stats.get("sent_total", 0)

        # Periodic consensus round (every 5 rounds)
        consensus_keys = []
        if self._round % 5 == 0:
            consensus_keys = self._run_consensus_round()

        # Count new shared beliefs
        new_shared = len(list(self.shared_mem.get_all(min_conf=0.60)))

        elapsed = (time.time() - t0) * 1000
        summary = RoundSummary(
            round_num=self._round,
            agent_steps=agent_steps,
            total_reward=total_reward,
            msgs_exchanged=msgs_exchanged,
            consensus_keys=consensus_keys,
            new_shared_beliefs=new_shared,
            elapsed_ms=elapsed,
        )
        self._round_log.append(summary)

        if self.verbose:
            sr = sum(1 for s in agent_steps.values() if s.env_success) / len(agent_steps)
            print(f"  Round {self._round:3d} | reward={total_reward:+.2f} | "
                  f"sr={sr:.0%} | msgs={msgs_exchanged} | "
                  f"shared={new_shared} | {elapsed:.0f}ms")

        return summary

    def run_episode(
        self,
        n_rounds:     int = 20,
        verbose:      bool = True,
        stop_on_goal: Optional[str] = None,
    ) -> List[RoundSummary]:
        """Run n_rounds of synchronised multi-agent steps."""
        log = []
        if verbose:
            print(f"\n  {'Rnd':4s}|{'Rwd':6s}|{'SR':4s}|"
                  f"{'Msgs':5s}|{'Shared':6s}|"
                  f"{'Consensus keys'}")
            print(f"  {'─'*4}|{'─'*6}|{'─'*4}|"
                  f"{'─'*5}|{'─'*6}|{'─'*20}")

        for _ in range(n_rounds):
            s = self.step_all()
            log.append(s)

            if verbose:
                sr = (sum(1 for st in s.agent_steps.values() if st.env_success)
                      / max(len(s.agent_steps), 1))
                ckeys = ",".join(s.consensus_keys[:2]) or "—"
                print(f"  {s.round_num:4d}|{s.total_reward:+.2f}|{sr:.0%}|"
                      f"{s.msgs_exchanged:5d}|{s.new_shared_beliefs:6d}|{ckeys}")

            if stop_on_goal:
                for agent in self.agents:
                    for g in agent.goal_system._completed:
                        if stop_on_goal.lower() in g.label.lower():
                            if verbose:
                                print(f"\n  ✅ Goal '{stop_on_goal}' achieved "
                                      f"by {agent.agent_id} at round {s.round_num}!")
                            return log
        return log

    # ------------------------------------------------------------------
    # Communication utilities
    # ------------------------------------------------------------------

    def broadcast(self, message: Message) -> int:
        """Manager broadcasts a message to all agents."""
        return self.comm_bus.broadcast(message)

    def broadcast_from(self, agent_id: str, msg_type: str, **content) -> None:
        """Convenience: broadcast a typed message from a specific agent."""
        agent = self.get_agent(agent_id)
        if agent is None:
            return
        if msg_type == "belief":
            subject, predicate, value = (
                content.get("subject",""), content.get("predicate",""),
                content.get("value")
            )
            conf = content.get("confidence", 0.7)
            msg  = Message.belief_msg(agent_id, subject, predicate, value, conf)
            self.comm_bus.broadcast(msg)

    # ------------------------------------------------------------------
    # Consensus
    # ------------------------------------------------------------------

    def _run_consensus_round(self) -> List[str]:
        """
        Run consensus on all shared memory keys with multiple contributors.
        Returns list of keys where consensus was reached.
        """
        resolved_keys = []
        trust_map = self._build_cross_trust_map()

        # Find contested keys
        contested = self.shared_mem.contested_keys()
        # Also run consensus on any key held by 2+ agents
        multi_keys = [
            k for k in self.shared_mem._store
            if self.shared_mem._store[k].n_contributors >= 2
        ]
        keys_to_resolve = list(set(contested + multi_keys))[:10]  # limit

        for key in keys_to_resolve:
            agent_beliefs = {a.agent_id: a.beliefs for a in self.agents}
            votes = ConsensusEngine.votes_from_beliefs(
                key, agent_beliefs, trust_scores=trust_map
            )
            if len(votes) < 2:
                continue
            result = self.consensus.reach_consensus(
                key, votes, strategy=ConsensusStrategy.TRUST_WEIGHTED
            )
            if not result.contested and result.value is not None:
                # Apply consensus to all agents' belief systems
                for agent in self.agents:
                    self.consensus.apply_to_belief_system(
                        result, agent.beliefs, source="consensus"
                    )
                # Write to shared memory
                self.shared_mem.write(key, result.value, "consensus", result.confidence)
                resolved_keys.append(key)

        return resolved_keys

    def force_consensus(self, key: str) -> Optional[Any]:
        """
        Force a consensus round on a specific key.
        Returns the consensus value, or None if unresolved.
        """
        trust_map = self._build_cross_trust_map()
        agent_beliefs = {a.agent_id: a.beliefs for a in self.agents}
        votes = ConsensusEngine.votes_from_beliefs(
            key, agent_beliefs, trust_scores=trust_map
        )
        if len(votes) < 1:
            return None
        result = self.consensus.reach_consensus(
            key, votes, strategy=ConsensusStrategy.TRUST_WEIGHTED
        )
        if not result.contested:
            for agent in self.agents:
                self.consensus.apply_to_belief_system(
                    result, agent.beliefs, source="consensus"
                )
            return result.value
        return None

    def _build_cross_trust_map(self) -> Dict[str, float]:
        """Build {agent_id: average_trust} from all agents' perspectives."""
        trust_totals: Dict[str, float] = {}
        trust_counts: Dict[str, int]   = {}

        for agent in self.agents:
            for peer_id, rec in agent.trust._records.items():
                trust_totals[peer_id] = (trust_totals.get(peer_id, 0)
                                         + rec.trust_score)
                trust_counts[peer_id] = trust_counts.get(peer_id, 0) + 1

        return {
            aid: trust_totals[aid] / trust_counts[aid]
            for aid in trust_totals
        }

    # ------------------------------------------------------------------
    # State collection
    # ------------------------------------------------------------------

    def collect_states(self) -> Dict[str, Dict]:
        """Collect summary state from all agents."""
        return {a.agent_id: a.v6_summary() for a in self.agents}

    def shared_knowledge(self, min_conf: float = 0.60) -> Dict[str, Any]:
        """Return shared memory entries above confidence threshold."""
        return {
            entry.key: {
                "value":      entry.value,
                "confidence": round(entry.confidence, 3),
                "contributors": list(entry.sources.keys()),
                "contested":  entry.is_contested,
            }
            for entry in list(self.shared_mem.get_all(min_conf=min_conf))
        }

    def belief_agreement_matrix(self, key: str) -> Dict[str, Any]:
        """
        For a given belief key, show each agent's belief + agreement fraction.
        """
        values = {}
        for a in self.agents:
            b = a.beliefs.get(key)
            if b:
                values[a.agent_id] = {
                    "value":      b.value,
                    "confidence": round(b.confidence, 3),
                    "reliable":   b.is_reliable,
                }
        # Agreement: fraction that share the plurality value
        if values:
            val_counts: Dict[str, int] = {}
            for info in values.values():
                v = str(info["value"])
                val_counts[v] = val_counts.get(v, 0) + 1
            plurality = max(val_counts, key=val_counts.get)
            agreement = val_counts[plurality] / len(values)
        else:
            plurality = None
            agreement = 0.0

        return {
            "key":       key,
            "per_agent": values,
            "plurality": plurality,
            "agreement": round(agreement, 3),
        }

    # ------------------------------------------------------------------
    # Teach all agents
    # ------------------------------------------------------------------

    def teach_all(
        self,
        text:  str,
        label: str = "",
        props: Optional[Dict] = None,
    ) -> None:
        """Teach the same fact to all agents."""
        for agent in self.agents:
            agent.teach(text, label=label, props=props)
        # Also write to shared memory
        if props and self.shared_mem:
            for prop, val in props.items():
                key = f"{label or text[:20]}.{prop}"
                self.shared_mem.write(key, val, "manager", 0.80)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        return {
            "n_agents":       len(self.agents),
            "n_rounds":       self._round,
            "shared_beliefs": len(self.shared_mem),
            "consensus":      self.consensus.summary(),
            "comm_stats":     self.comm_bus.stats(),
            "agents":         {
                a.agent_id: {
                    "role":    a.role.value,
                    "steps":   a._step_count,
                    "beliefs": len(a.beliefs),
                    "grade":   a.metrics.stability_grade(),
                    "msgs_rx": a._msgs_received_total,
                    "msgs_tx": a._msgs_sent_total,
                }
                for a in self.agents
            },
        }

    def __repr__(self) -> str:
        return (f"AgentManager(agents={len(self.agents)}, "
                f"rounds={self._round}, "
                f"shared={len(self.shared_mem)})")
