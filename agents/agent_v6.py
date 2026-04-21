"""
agents/agent_v6.py
===================
CogniField v6 — Multi-Agent Collaborative Cognitive Agent

Extends CogniFieldAgentV5 with:
  - Agent identity (id + role)
  - Communication module access
  - Trust system (per-peer reputation)
  - Shared memory read/write
  - Social learning (observe peers' outcomes)
  - Message integration (trust-weighted belief updates)
  - Collaborative experimentation support
  - Consensus participation

The 17-step v6 loop:
  1.  Observe environment
  2.  Observe peers (incoming messages)
  3.  Update private memory
  4.  Validate knowledge
  5.  Detect novelty
  6.  Receive + evaluate messages (trust-weighted)
  7.  Update beliefs from peer messages
  8.  Share relevant knowledge with peers
  9.  Generate goals (including social goals)
  10. Select goal
  11. Plan hierarchically
  12. Simulate outcomes
  13. Risk check
  14. Act
  15. Receive feedback
  16. Update beliefs + trust scores
  17. Participate in consensus (periodic)
  → Repeat

Roles
-----
EXPLORER     – seeks novelty, experiments on unknown objects
ANALYST      – validates beliefs, runs experiments carefully
PLANNER      – focuses on executing goals efficiently
RISK_MANAGER – monitors danger, issues warnings
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..agents.agent_v5 import CogniFieldAgentV5, AgentV5Config, V5Step
from ..world_model.belief_system import Belief, BeliefSystem
from .trust_system import TrustSystem
from ..communication.communication_module import (
    CommunicationModule, Message, MessageType
)
from ..memory.shared_memory import SharedMemory


# ---------------------------------------------------------------------------
# Role system
# ---------------------------------------------------------------------------

class AgentRole(str, Enum):
    EXPLORER      = "explorer"
    ANALYST       = "analyst"
    PLANNER       = "planner"
    RISK_MANAGER  = "risk_manager"
    GENERALIST    = "generalist"


ROLE_TRAITS: Dict[AgentRole, Dict] = {
    AgentRole.EXPLORER: {
        "novelty_threshold_adj": -0.10,   # lower → more curious
        "risk_tolerance_adj":    +0.05,   # slightly bolder
        "share_freq":             2,       # shares every N steps
        "goal_bias":              "explore",
    },
    AgentRole.ANALYST: {
        "novelty_threshold_adj": +0.05,   # higher → more conservative
        "risk_tolerance_adj":    -0.05,
        "share_freq":             3,
        "goal_bias":              "verify",
    },
    AgentRole.PLANNER: {
        "novelty_threshold_adj":  0.00,
        "risk_tolerance_adj":     0.00,
        "share_freq":             4,
        "goal_bias":              "goal_execution",
    },
    AgentRole.RISK_MANAGER: {
        "novelty_threshold_adj": +0.10,   # most conservative
        "risk_tolerance_adj":    -0.10,
        "share_freq":             1,       # shares warnings most frequently
        "goal_bias":              "safety",
    },
    AgentRole.GENERALIST: {
        "novelty_threshold_adj":  0.00,
        "risk_tolerance_adj":     0.00,
        "share_freq":             3,
        "goal_bias":              "balanced",
    },
}


# ---------------------------------------------------------------------------
# V6 step record
# ---------------------------------------------------------------------------

@dataclass
class V6Step(V5Step):
    messages_received:  int = 0
    messages_sent:      int = 0
    beliefs_from_peers: int = 0
    social_learning:    int = 0   # beliefs adopted without direct experience


# ---------------------------------------------------------------------------
# AgentV6Config
# ---------------------------------------------------------------------------

@dataclass
class AgentV6Config(AgentV5Config):
    role:               AgentRole = AgentRole.GENERALIST
    agent_id:           str       = "agent_0"
    share_beliefs_freq: int       = 3       # share own beliefs every N steps
    max_msgs_per_step:  int       = 10
    trust_decay_freq:   int       = 20      # decay trust every N steps
    consensus_freq:     int       = 15      # participate in consensus every N steps
    min_trust_to_adopt: float     = 0.45   # min trust to accept peer belief
    min_conf_to_share:  float     = 0.65   # min confidence to broadcast a belief


# ---------------------------------------------------------------------------
# CogniFieldAgentV6
# ---------------------------------------------------------------------------

class CogniFieldAgentV6(CogniFieldAgentV5):
    """
    Multi-agent-aware cognitive agent (v6).

    Parameters
    ----------
    config       : AgentV6Config (extends AgentV5Config)
    env          : Shared RichEnv (may be None for offline agents)
    comm_bus     : Shared CommunicationModule — all agents share one bus
    shared_mem   : Shared SharedMemory — all agents read/write here
    """

    def __init__(
        self,
        config:     Optional[AgentV6Config] = None,
        env        = None,
        comm_bus:   Optional[CommunicationModule] = None,
        shared_mem: Optional[SharedMemory]        = None,
    ) -> None:
        cfg = config or AgentV6Config()

        # Apply role-based parameter adjustments before initialising v5
        traits = ROLE_TRAITS.get(cfg.role, ROLE_TRAITS[AgentRole.GENERALIST])
        cfg.novelty_threshold = float(
            cfg.novelty_threshold + traits["novelty_threshold_adj"]
        )
        cfg.risk_tolerance    = float(
            cfg.risk_tolerance + traits["risk_tolerance_adj"]
        )

        # Init v5 base
        super().__init__(config=cfg, env=env)

        # v6 identity
        self.agent_id   = cfg.agent_id
        self.role        = cfg.role
        self.cfg_v6: AgentV6Config = cfg   # typed alias

        # v6 components
        self.trust       = TrustSystem(owner_id=self.agent_id)
        self.comm_bus    = comm_bus
        self.shared_mem  = shared_mem

        # Register with communication bus
        if self.comm_bus:
            self.comm_bus.register(self.agent_id)

        # Social learning counters
        self._msgs_received_total   = 0
        self._msgs_sent_total       = 0
        self._beliefs_from_peers    = 0

    # ══════════════════════════════════════════════════════════════════
    # Override step() with v6 17-step loop
    # ══════════════════════════════════════════════════════════════════

    def step(
        self,
        text_input:   str = "",
        force_action: Optional[Tuple[str, str]] = None,
        verbose:      Optional[bool] = None,
    ) -> V6Step:
        """17-step multi-agent cognitive cycle."""
        verbose = verbose if verbose is not None else self.cfg.verbose

        # Run v5 step (handles steps 1, 3-5, 7-18)
        v5_result = super().step(text_input=text_input,
                                  force_action=force_action,
                                  verbose=False)

        # ── v6 extensions ──

        # Step 2 + 4-8: Process peer messages
        msgs_received, beliefs_from_peers, social = self._process_incoming_messages(
            verbose=verbose
        )

        # Share own knowledge
        msgs_sent = self._share_knowledge(verbose=verbose)

        # Periodically update trust
        if self._step_count % self.cfg_v6.trust_decay_freq == 0:
            self.trust.decay()

        # Write reliable beliefs to shared memory
        if self.shared_mem and self._step_count % 2 == 0:
            self._sync_to_shared_memory()

        if verbose and (msgs_received > 0 or msgs_sent > 0):
            print(f"      [{self.agent_id}] msgs: rx={msgs_received}, "
                  f"tx={msgs_sent}, beliefs_from_peers={beliefs_from_peers}")

        # Build V6Step from V5Step fields + new fields
        s = V6Step(
            step=v5_result.step,
            input_text=v5_result.input_text,
            active_goal=v5_result.active_goal,
            action_taken=v5_result.action_taken,
            action_obj=v5_result.action_obj,
            env_success=v5_result.env_success,
            env_reward=v5_result.env_reward,
            novelty=v5_result.novelty,
            risk_decision=v5_result.risk_decision,
            risk_score=v5_result.risk_score,
            belief_updates=v5_result.belief_updates,
            conflicts_found=v5_result.conflicts_found,
            goals_generated=v5_result.goals_generated,
            experiment_run=v5_result.experiment_run,
            consistency_ok=v5_result.consistency_ok,
            elapsed_ms=v5_result.elapsed_ms,
            # v6
            messages_received=msgs_received,
            messages_sent=msgs_sent,
            beliefs_from_peers=beliefs_from_peers,
            social_learning=social,
        )
        return s

    # ══════════════════════════════════════════════════════════════════
    # Message processing
    # ══════════════════════════════════════════════════════════════════

    def _process_incoming_messages(
        self,
        verbose: bool = False,
    ) -> Tuple[int, int, int]:
        """
        Drain inbox, evaluate each message, update beliefs if trusted.
        Returns (n_received, n_belief_updates, n_social_learning).
        """
        if not self.comm_bus:
            return 0, 0, 0

        msgs = self.comm_bus.receive(self.agent_id,
                                     max_msgs=self.cfg_v6.max_msgs_per_step)
        n_received  = len(msgs)
        n_updates   = 0
        n_social    = 0

        for msg in msgs:
            self._msgs_received_total += 1

            # Register sender in trust system
            self.trust.register_peer(msg.sender_id)

            # Compute effective weight: trust × message confidence
            weight = self.trust.message_weight(msg.sender_id, msg.confidence)

            if weight < self.cfg_v6.min_trust_to_adopt * msg.confidence:
                continue  # sender not trusted enough

            if msg.msg_type in (MessageType.BELIEF, MessageType.WARNING):
                subject   = msg.content.get("subject", "")
                predicate = msg.content.get("predicate", "")
                value     = msg.content.get("value")
                if subject and predicate:
                    key = f"{subject}.{predicate}"
                    # Check consistency before adopting
                    allowed, adj_wt, _ = self.consistency_engine.check_before_update(
                        key, value, source="inference", weight=weight
                    )
                    if allowed and adj_wt >= 0.1:
                        self.beliefs.update(key, value,
                                            source="inference",
                                            weight=adj_wt * 0.8,
                                            notes=f"from {msg.sender_id}")
                        n_updates += 1
                        n_social  += 1

                        # Update trust for responsiveness
                        self.trust.update_responsiveness(msg.sender_id, useful=True)

                        if verbose:
                            print(f"      [{self.agent_id}] adopted "
                                  f"{key}={value} from {msg.sender_id} "
                                  f"(weight={adj_wt:.2f})")

            elif msg.msg_type == MessageType.OBSERVATION:
                action  = msg.content.get("action", "")
                target  = msg.content.get("target", "")
                outcome = msg.content.get("outcome", "")
                reward  = msg.content.get("reward", 0.0)

                if action == "eat" and target:
                    inferred_edible = (outcome == "success" and reward > 0)
                    key = f"{target}.edible"
                    allowed, adj_wt, _ = self.consistency_engine.check_before_update(
                        key, inferred_edible, source="inference", weight=weight
                    )
                    if allowed and adj_wt >= 0.1:
                        self.beliefs.update(key, inferred_edible,
                                            source="inference",
                                            weight=adj_wt * 0.7)
                        n_updates += 1
                        n_social  += 1

                # Also update episodic memory as social observation
                self.episodic_memory.record(
                    step=self._step_count,
                    action=action, target=target,
                    outcome=outcome, reward=reward,
                    context={"source": msg.sender_id, "social": True},
                )

        self._beliefs_from_peers += n_updates
        return n_received, n_updates, n_social

    # ══════════════════════════════════════════════════════════════════
    # Knowledge sharing
    # ══════════════════════════════════════════════════════════════════

    def _share_knowledge(self, verbose: bool = False) -> int:
        """Broadcast reliable beliefs to peers. Returns count sent."""
        if not self.comm_bus:
            return 0
        if self._step_count % self.cfg_v6.share_beliefs_freq != 0:
            return 0

        n_sent = 0

        # Share high-confidence beliefs
        for belief in self.beliefs.reliable_beliefs():
            if belief.confidence < self.cfg_v6.min_conf_to_share:
                continue
            parts = belief.key.split(".")
            if len(parts) != 2:
                continue
            subject, predicate = parts

            # Risk managers send warnings for dangerous knowledge
            if (self.role == AgentRole.RISK_MANAGER
                    and predicate == "edible"
                    and belief.value is False
                    and belief.confidence >= 0.80):
                msg = Message.warning_msg(
                    self.agent_id, subject, predicate,
                    belief.value, belief.confidence
                )
            else:
                msg = Message.belief_msg(
                    self.agent_id, subject, predicate,
                    belief.value, belief.confidence
                )

            self.comm_bus.broadcast(msg)
            n_sent += 1

            # Write to shared memory
            if self.shared_mem:
                self.shared_mem.write(belief.key, belief.value, self.agent_id, belief.confidence)

        self._msgs_sent_total += n_sent
        return n_sent

    def share_observation(
        self,
        action:  str,
        target:  str,
        outcome: str,
        reward:  float,
    ) -> None:
        """Broadcast an observation (e.g. after eating something)."""
        if not self.comm_bus:
            return
        conf = 0.9 if outcome == "success" else 0.85
        msg = Message.observation_msg(
            self.agent_id, action, target, outcome, reward
        )
        self.comm_bus.broadcast(msg)
        self._msgs_sent_total += 1

    def broadcast_belief(
        self,
        subject:   str,
        predicate: str,
        value:     Any = None,
    ) -> bool:
        """Broadcast a specific belief to all peers."""
        if not self.comm_bus:
            return False
        key = f"{subject}.{predicate}"
        if value is None:
            value = self.beliefs.get_value(key)
        conf  = self.beliefs.get_confidence(key, default=0.5)
        if conf < self.cfg_v6.min_conf_to_share:
            return False
        msg = Message.belief_msg(
            self.agent_id, subject, predicate, value, conf
        )
        self.comm_bus.broadcast(msg)
        self._msgs_sent_total += 1
        return True

    def ask_peers(self, subject: str, predicate: str) -> None:
        """Broadcast a question about a belief to all peers."""
        if not self.comm_bus:
            return
        msg = Message.question_msg(self.agent_id, subject, predicate)
        self.comm_bus.broadcast(msg)

    # ══════════════════════════════════════════════════════════════════
    # Trust updates from feedback
    # ══════════════════════════════════════════════════════════════════

    def update_trust_from_outcome(
        self,
        peer_id:       str,
        believed_value: Any,
        actual_value:   Any,
    ) -> None:
        """
        After verifying a belief from a peer via experiment, update trust.
        Called by AgentManager after an experiment resolves a belief.
        """
        self.trust.observe_outcome(peer_id, believed_value, actual_value)

    # ══════════════════════════════════════════════════════════════════
    # Shared memory sync
    # ══════════════════════════════════════════════════════════════════

    def _sync_to_shared_memory(self) -> None:
        """Write all reliable beliefs to shared memory."""
        if not self.shared_mem:
            return
        for belief in self.beliefs.reliable_beliefs():
            self.shared_mem.write(belief.key, belief.value, self.agent_id, belief.confidence)

    def read_shared_memory(
        self,
        key:       str,
        trust_map: Optional[Dict[str, float]] = None,
    ) -> Optional[Any]:
        """
        Read a value from shared memory, optionally weighted by trust.
        trust_map: {agent_id: trust_score}
        """
        if not self.shared_mem:
            return None
        if trust_map:
            val, conf = self.shared_mem.read_weighted_by_trust(key, trust_map)
            return val
        entry = self.shared_mem.read(key)
        return entry.value if entry else None

    def learn_from_shared_memory(self, min_conf: float = 0.70) -> int:
        """
        Bulk-import reliable shared beliefs into private belief system.
        Returns number of beliefs imported.
        """
        if not self.shared_mem:
            return 0
        count = 0
        for entry in list(self.shared_mem.get_all(min_conf=min_conf)):
            # Build trust map from all known contributors
            trust_map = {
                aid: self.trust.get_trust(aid, default=0.5)
                for aid in entry.sources
            }
            # Weight = entry confidence × mean contributor trust
            mean_trust = (sum(trust_map.values()) / len(trust_map)
                          if trust_map else 0.5)
            weight = entry.confidence * mean_trust

            allowed, adj_wt, _ = self.consistency_engine.check_before_update(
                entry.key, entry.value, source="inference", weight=weight
            )
            if allowed and adj_wt >= 0.15:
                self.beliefs.update(entry.key, entry.value,
                                    source="inference", weight=adj_wt,
                                    notes=f"shared_memory (n={entry.n_contributors})")
                count += 1
        return count

    # ══════════════════════════════════════════════════════════════════
    # Collaborative experiment coordination
    # ══════════════════════════════════════════════════════════════════

    def request_collaborative_experiment(
        self,
        target:   str,
        property: str,
    ) -> None:
        """Ask peers to help investigate an unknown property."""
        if not self.comm_bus:
            return
        msg = Message(
            sender_id=self.agent_id,
            receiver_id=None,
            msg_type=MessageType.QUESTION,
            content={
                "subject":   target,
                "predicate": property,
                "request":   "collaborative_experiment",
            },
            confidence=0.5,
        )
        self.comm_bus.broadcast(msg)

    # ══════════════════════════════════════════════════════════════════
    # Role-specific helpers
    # ══════════════════════════════════════════════════════════════════

    def should_explore_now(self) -> bool:
        """Explorer role is more inclined to explore unknowns."""
        if self.role == AgentRole.EXPLORER:
            return self.internal_state.curiosity > 0.4
        return self.internal_state.should_explore_boldly()

    def should_warn_peers(self, subject: str, predicate: str) -> bool:
        """Risk manager role proactively warns about dangers."""
        if self.role != AgentRole.RISK_MANAGER:
            return False
        b = self.beliefs.get(f"{subject}.{predicate}")
        return (b is not None
                and b.value is False
                and b.confidence >= 0.80
                and predicate == "edible")

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════

    def v6_summary(self) -> Dict:
        base = self.summary()
        base.update({
            "agent_id":         self.agent_id,
            "role":             self.role.value,
            "msgs_sent":        self._msgs_sent_total,
            "msgs_received":    self._msgs_received_total,
            "beliefs_from_peers": self._beliefs_from_peers,
            "trust_system":     self.trust.summary(),
        })
        return base

    def __repr__(self) -> str:
        return (f"AgentV6(id={self.agent_id}, role={self.role.value}, "
                f"steps={self._step_count}, "
                f"beliefs={len(self.beliefs)}, "
                f"grade={self.metrics.stability_grade()})")
