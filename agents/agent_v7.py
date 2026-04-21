"""
agents/agent_v7.py
===================
CogniField v7 — Social Intelligence Agent

Extends AgentV6 with:
  - Language layer (semantic encoding/decoding)
  - Negotiation (belief argumentation with peers)
  - Cooperation (task coordination via CooperationEngine)
  - Dynamic role evolution (roles shift based on performance)
  - Social memory (per-peer interaction tracking)
  - Reputation (long-term per-peer accuracy weighting)

The v7 loop (18 steps):
  1.  Observe environment
  2.  Observe peers (incoming messages via language layer)
  3.  Update memory + social memory
  4.  Validate knowledge
  5.  Detect novelty
  6.  Check for negotiation requests
  7.  Generate + select goals
  8.  Check cooperation engine for assigned tasks
  9.  Plan
  10. Simulate outcomes
  11. Risk check
  12. Act
  13. Receive feedback
  14. Update beliefs (Bayesian)
  15. Share knowledge via language layer
  16. Update trust + social memory
  17. Dynamic role adjustment (periodic)
  18. Participate in consensus / negotiation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .agent_v6 import (
    CogniFieldAgentV6, AgentV6Config, AgentRole, V6Step, ROLE_TRAITS
)
from .trust_system import TrustSystem
from ..communication.communication_module import (
    CommunicationModule, Message, MessageType
)
from ..communication.language_layer import LanguageLayer
from ..memory.shared_memory import SharedMemory
from ..memory.social_memory import SocialMemory
from ..reasoning.negotiation_engine import NegotiationEngine
from ..planning.cooperation_engine import CooperationEngine, TaskType


# ---------------------------------------------------------------------------
# V7 Config
# ---------------------------------------------------------------------------

@dataclass
class AgentV7Config(AgentV6Config):
    role_evolution_freq:    int   = 25   # steps between role evaluations
    negotiation_freq:       int   = 10   # steps between negotiation attempts
    min_conf_to_negotiate:  float = 0.55 # min confidence for a disputed belief
    language_vocab_max:     int   = 300
    reputation_weight:      float = 0.30 # weight of long-term reputation vs trust


# ---------------------------------------------------------------------------
# V7 Step record
# ---------------------------------------------------------------------------

@dataclass
class V7Step(V6Step):
    negotiations_run:  int   = 0
    coop_tasks_done:   int   = 0
    role_changed:      bool  = False
    vocab_size:        int   = 0
    reputation_updates: int  = 0


# ---------------------------------------------------------------------------
# CogniFieldAgentV7
# ---------------------------------------------------------------------------

class CogniFieldAgentV7(CogniFieldAgentV6):
    """
    v7 multi-agent social intelligence agent.
    """

    def __init__(
        self,
        config:      Optional[AgentV7Config] = None,
        env          = None,
        comm_bus:    Optional[CommunicationModule] = None,
        shared_mem:  Optional[SharedMemory]        = None,
        coop_engine: Optional[CooperationEngine]   = None,
    ) -> None:
        cfg = config or AgentV7Config()
        super().__init__(config=cfg, env=env, comm_bus=comm_bus, shared_mem=shared_mem)

        self.cfg_v7: AgentV7Config = cfg

        # v7 components
        self.language    = LanguageLayer(self.agent_id,
                                          vocab_max=cfg.language_vocab_max)
        self.social_mem  = SocialMemory(self.agent_id)
        self.negotiation = NegotiationEngine(max_rounds=4, tolerance=0.12)
        self.coop_engine = coop_engine

        # Performance tracking for role evolution
        self._role_perf:    Dict[str, float] = {role.value: 0.5 for role in AgentRole}
        self._role_history: List[AgentRole]  = [cfg.role]
        self._neg_partners: List[str]         = []   # who to negotiate with

        # Reputation: long-term per-peer accuracy (slower to change than trust)
        self._reputation: Dict[str, float] = {}

    # ══════════════════════════════════════════════════════════════════
    # Override step() with v7 18-step loop
    # ══════════════════════════════════════════════════════════════════

    def step(
        self,
        text_input:   str = "",
        force_action: Optional[Tuple[str, str]] = None,
        verbose:      Optional[bool] = None,
    ) -> V7Step:
        verbose = verbose if verbose is not None else self.cfg.verbose

        # Run v6 step (covers steps 1-18 of v6 loop)
        v6_result = super().step(text_input=text_input,
                                  force_action=force_action,
                                  verbose=verbose)

        # ── v7 extensions ──
        negs_run       = 0
        coop_done      = 0
        role_changed   = False
        rep_updates    = 0

        # Step 6: Negotiation (periodic)
        if (self._step_count % self.cfg_v7.negotiation_freq == 0
                and self._neg_partners and self.comm_bus):
            negs_run = self._run_negotiations(verbose=verbose)

        # Step 8: Cooperation task dispatch
        if self.coop_engine:
            coop_done = self._process_coop_tasks(verbose=verbose)

        # Step 15: Encode outgoing messages through language layer
        if self.comm_bus and self._step_count % self.cfg_v6.share_beliefs_freq == 0:
            self._share_via_language(verbose=verbose)

        # Step 16: Update social memory + reputation
        rep_updates = self._update_reputation()

        # Step 17: Dynamic role evolution (periodic)
        if self._step_count % self.cfg_v7.role_evolution_freq == 0:
            role_changed = self._evolve_role(verbose=verbose)

        s = V7Step(
            step=v6_result.step,
            input_text=v6_result.input_text,
            active_goal=v6_result.active_goal,
            action_taken=v6_result.action_taken,
            action_obj=v6_result.action_obj,
            env_success=v6_result.env_success,
            env_reward=v6_result.env_reward,
            novelty=v6_result.novelty,
            risk_decision=v6_result.risk_decision,
            risk_score=v6_result.risk_score,
            belief_updates=v6_result.belief_updates,
            conflicts_found=v6_result.conflicts_found,
            goals_generated=v6_result.goals_generated,
            experiment_run=v6_result.experiment_run,
            consistency_ok=v6_result.consistency_ok,
            elapsed_ms=v6_result.elapsed_ms,
            messages_received=v6_result.messages_received,
            messages_sent=v6_result.messages_sent,
            beliefs_from_peers=v6_result.beliefs_from_peers,
            social_learning=v6_result.social_learning,
            # v7 fields
            negotiations_run=negs_run,
            coop_tasks_done=coop_done,
            role_changed=role_changed,
            vocab_size=self.language.vocab_size(),
            reputation_updates=rep_updates,
        )
        return s

    # ══════════════════════════════════════════════════════════════════
    # Negotiation
    # ══════════════════════════════════════════════════════════════════

    def register_for_negotiation(self, peer_agent: "CogniFieldAgentV7") -> None:
        """Register a peer to negotiate with."""
        if peer_agent.agent_id not in self._neg_partners:
            self._neg_partners.append(peer_agent.agent_id)
        # Store reference (weak coupling)
        if not hasattr(self, "_peer_refs"):
            self._peer_refs = {}
        self._peer_refs[peer_agent.agent_id] = peer_agent

    def _run_negotiations(self, verbose: bool = False) -> int:
        """Negotiate conflicting beliefs with all registered peers."""
        if not hasattr(self, "_peer_refs"):
            return 0
        count = 0
        for peer_id in list(self._neg_partners):
            peer = self._peer_refs.get(peer_id)
            if peer is None:
                continue
            trust_we_have = self.trust.get_trust(peer_id, 0.5)
            trust_peer_has = peer.trust.get_trust(self.agent_id, 0.5)

            results = self.negotiation.negotiate_all_conflicts(
                self.beliefs, self.agent_id, trust_we_have,
                peer.beliefs,  peer.agent_id, trust_peer_has,
                min_conf_threshold=self.cfg_v7.min_conf_to_negotiate,
            )
            count += len(results)

            if verbose and results:
                for r in results[:2]:
                    print(f"      [{self.agent_id}] negotiated {r.key}: "
                          f"{'✓' if r.converged else '~'} → {r.agreed_value} "
                          f"in {r.rounds} rounds")

            # Update social memory
            for r in results:
                self.social_mem.record_interaction(
                    peer_id=peer_id, msg_type="negotiation",
                    topic=r.key, content=r.agreed_value,
                    confidence=r.agreed_conf,
                )
        return count

    # ══════════════════════════════════════════════════════════════════
    # Cooperation
    # ══════════════════════════════════════════════════════════════════

    def _process_coop_tasks(self, verbose: bool = False) -> int:
        """Execute assigned cooperative tasks."""
        if not self.coop_engine:
            return 0
        tasks = self.coop_engine.pending_tasks_for(self.agent_id)
        done = 0
        for task in tasks[:2]:   # max 2 tasks per step
            success = self._execute_coop_task(task, verbose)
            if success:
                self.coop_engine.complete_task(task.task_id, result=success)
            else:
                self.coop_engine.fail_task(task.task_id)
            done += 1
        return done

    def _execute_coop_task(self, task, verbose: bool = False) -> Optional[Any]:
        """Execute one cooperative task. Returns result or None on failure."""
        if task.task_type == TaskType.EXPLORE and self.env:
            # Schedule experiment on the target
            self.schedule_experiment(task.target)
            if verbose:
                print(f"      [{self.agent_id}] exploring {task.target} "
                      f"(coop task {task.task_id})")
            return {"action": "explore", "target": task.target}

        elif task.task_type == TaskType.VERIFY:
            conf = self.how_confident(task.target, "edible")
            if verbose:
                print(f"      [{self.agent_id}] verifying {task.target}.edible: "
                      f"conf={conf:.3f}")
            return {"target": task.target, "conf": conf}

        elif task.task_type == TaskType.WARN:
            self.broadcast_belief(task.target, "edible")
            return {"warned_about": task.target}

        return None

    # ══════════════════════════════════════════════════════════════════
    # Language-encoded sharing
    # ══════════════════════════════════════════════════════════════════

    def _share_via_language(self, verbose: bool = False) -> None:
        """Share reliable beliefs through the language layer."""
        if not self.comm_bus:
            return
        for belief in self.beliefs.reliable_beliefs():
            if belief.confidence < self.cfg_v6.min_conf_to_share:
                continue
            parts = belief.key.split(".")
            if len(parts) != 2:
                continue
            subject, predicate = parts

            # Encode through language layer
            encoded = self.language.encode(
                subject, predicate, belief.value,
                confidence=belief.confidence,
                msg_type=MessageType.BELIEF,
            )
            msg = encoded.to_message()
            msg.sender_id = self.agent_id
            self.comm_bus.broadcast(msg)

    def _process_incoming_messages(self, verbose=False):
        """Override: decode messages through language layer."""
        if not self.comm_bus:
            return 0, 0, 0

        msgs = self.comm_bus.receive(self.agent_id,
                                     max_msgs=self.cfg_v6.max_msgs_per_step)
        n_received = len(msgs)
        n_updates  = 0
        n_social   = 0

        for msg in msgs:
            self.trust.register_peer(msg.sender_id)
            # Decode through language layer
            decoded = self.language.decode_message(
                msg,
                sender_trust=self.effective_trust(msg.sender_id)
            )
            if decoded is None:
                continue

            self.social_mem.record_interaction(
                peer_id=msg.sender_id,
                msg_type=msg.msg_type.value,
                topic=decoded.get("subject","") + "." + decoded.get("predicate",""),
                content=decoded.get("value"),
                confidence=decoded.get("effective_confidence", 0.5),
            )

            # Update belief if trusted enough
            eff_conf = decoded.get("effective_confidence", 0.0)
            if eff_conf >= self.cfg_v6.min_trust_to_adopt * 0.5:
                key     = f"{decoded['subject']}.{decoded['predicate']}"
                value   = decoded["value"]
                allowed, adj_wt, _ = self.consistency_engine.check_before_update(
                    key, value, source="inference", weight=eff_conf
                )
                if allowed and adj_wt >= 0.1:
                    self.beliefs.update(key, value, source="inference",
                                        weight=adj_wt * 0.8)
                    n_updates += 1
                    n_social  += 1
                    self.trust.update_responsiveness(msg.sender_id, useful=True)

        self._beliefs_from_peers += n_updates
        return n_received, n_updates, n_social

    def effective_trust(self, peer_id: str) -> float:
        """Blend short-term trust with long-term reputation."""
        trust  = self.trust.get_trust(peer_id, 0.5)
        rep    = self._reputation.get(peer_id, 0.5)
        w      = self.cfg_v7.reputation_weight
        return float((1 - w) * trust + w * rep)

    # ══════════════════════════════════════════════════════════════════
    # Reputation
    # ══════════════════════════════════════════════════════════════════

    def _update_reputation(self) -> int:
        """Slowly update long-term reputation from social memory."""
        count = 0
        for peer_id in self.social_mem.known_peers():
            acc = self.social_mem.overall_accuracy(peer_id)
            if acc == 0.5:
                continue   # no verified data yet
            old_rep = self._reputation.get(peer_id, 0.5)
            # Reputation moves very slowly (α=0.05)
            self._reputation[peer_id] = 0.95 * old_rep + 0.05 * acc
            count += 1
        return count

    def update_reputation_from_verification(
        self,
        peer_id: str,
        topic:   str,
        correct: bool,
    ) -> None:
        """After verifying a belief, update both social memory and trust."""
        self.social_mem.record_verification(peer_id, topic, correct)
        self.trust.update_accuracy(peer_id, correct)
        # Update reputation
        acc = self.social_mem.overall_accuracy(peer_id)
        old_rep = self._reputation.get(peer_id, 0.5)
        self._reputation[peer_id] = 0.95 * old_rep + 0.05 * acc

    # ══════════════════════════════════════════════════════════════════
    # Dynamic role evolution
    # ══════════════════════════════════════════════════════════════════

    def _evolve_role(self, verbose: bool = False) -> bool:
        """
        Evaluate whether a role change would improve performance.
        Switches if a different role consistently outperforms current role.
        """
        # Evaluate current performance signals
        sr = self.metrics.success_rate()

        # Track per-role performance based on who shares useful info
        # Use social memory: agents with higher coop success → their role is valuable
        coop_peers = self.social_mem.best_cooperative_peers(3)

        # Simple heuristic: if success rate is low, shift toward more analytical role
        if sr < 0.35 and self.role == AgentRole.EXPLORER:
            new_role = AgentRole.ANALYST
        elif sr > 0.70 and self.role == AgentRole.ANALYST:
            new_role = AgentRole.EXPLORER   # comfortable enough to explore more
        elif (self.internal_state.frustration > 0.65
              and self.role != AgentRole.RISK_MANAGER):
            new_role = AgentRole.RISK_MANAGER   # high frustration → be more cautious
        else:
            return False   # no change

        if new_role == self.role:
            return False

        old_role = self.role
        self.role = new_role
        self._role_history.append(new_role)

        # Apply new role traits
        traits = ROLE_TRAITS.get(new_role, ROLE_TRAITS[AgentRole.GENERALIST])
        self.cfg.novelty_threshold = float(
            0.40 + traits["novelty_threshold_adj"]
        )
        self.cfg.risk_tolerance = float(
            0.35 + traits["risk_tolerance_adj"]
        )
        self.risk_engine.risk_tolerance = self.cfg.risk_tolerance

        if verbose:
            print(f"      [{self.agent_id}] 🔄 Role evolved: "
                  f"{old_role.value} → {new_role.value} "
                  f"(sr={sr:.1%})")
        return True

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════

    def v7_summary(self) -> Dict:
        base = self.v6_summary()
        base.update({
            "language":     self.language.summary(),
            "social_memory":self.social_mem.summary(),
            "negotiation":  self.negotiation.summary(),
            "role_history": [r.value for r in self._role_history],
            "reputation":   {k: round(v, 3) for k, v in self._reputation.items()},
            "coop_engine":  self.coop_engine.summary() if self.coop_engine else None,
        })
        return base

    def __repr__(self) -> str:
        return (f"AgentV7(id={self.agent_id}, role={self.role.value}, "
                f"steps={self._step_count}, "
                f"beliefs={len(self.beliefs)}, "
                f"vocab={self.language.vocab_size()}, "
                f"grade={self.metrics.stability_grade()})")
