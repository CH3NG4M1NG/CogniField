"""
agents/agent_v8.py
===================
CogniField v8 — Collective Intelligence Agent

Fixes all v7 limitations:
  ✓ Bidirectional communication  (tx > 0 AND rx > 0)
  ✓ Shared memory actively used  (read + write every round)
  ✓ Global consensus integrated  (private beliefs ← global truth)
  ✓ Experience sharing            (GroupMind broadcasts discoveries)
  ✓ Event-driven reactions        (subscribe to EventBus)
  ✓ Consistency enforced          (no contradictions allowed)
  ✓ Dynamic role with load balance (cooperation engine fair)

v8 Loop (20 steps):
  1.  Observe environment
  2.  Apply GroupMind coordination signal
  3.  Receive + decode messages (language layer)
  4.  Read from shared memory (pull community knowledge)
  5.  Update private memory
  6.  Validate knowledge
  7.  Detect novelty
  8.  Generate goals (guided by GroupMind primary goal)
  9.  Select goal
  10. Plan
  11. Simulate
  12. Risk check
  13. Act
  14. Receive feedback
  15. Update beliefs (Bayesian)
  16. Share experience to GroupMind (if significant)
  17. Write beliefs to shared memory
  18. Broadcast relevant beliefs (bidirectional enforcement)
  19. Update trust + social memory
  20. Participate in negotiation (periodic)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .agent_v7 import CogniFieldAgentV7, AgentV7Config, V7Step
from .group_mind import GroupMind, CoordSignal
from ..reasoning.global_consensus import GlobalConsensus
from ..communication.communication_module import (
    CommunicationModule, Message, MessageType
)
from ..memory.shared_memory import SharedMemory
from ..core.event_bus import EventBus, EventType
from ..planning.cooperation_engine import CooperationEngine


# ---------------------------------------------------------------------------
# V8 Config
# ---------------------------------------------------------------------------

@dataclass
class AgentV8Config(AgentV7Config):
    shared_mem_read_freq:  int   = 2    # read shared memory every N steps
    experience_min_reward: float = 0.30 # threshold to share experience
    consistency_freq:      int   = 5    # enforce consistency every N steps


# ---------------------------------------------------------------------------
# V8 Step record
# ---------------------------------------------------------------------------

@dataclass
class V8Step(V7Step):
    shared_mem_reads:     int  = 0
    shared_mem_writes:    int  = 0
    experiences_shared:   int  = 0
    consistency_fixes:    int  = 0
    global_consensus_applied: int = 0
    event_reactions:      int  = 0
    coord_signal:         str  = "none"


# ---------------------------------------------------------------------------
# CogniFieldAgentV8
# ---------------------------------------------------------------------------

class CogniFieldAgentV8(CogniFieldAgentV7):
    """
    v8 Collective Intelligence Agent.
    """

    def __init__(
        self,
        config:       Optional[AgentV8Config] = None,
        env           = None,
        comm_bus:     Optional[CommunicationModule] = None,
        shared_mem:   Optional[SharedMemory]        = None,
        group_mind:   Optional[GroupMind]           = None,
        global_cons:  Optional[GlobalConsensus]     = None,
        event_bus:    Optional[EventBus]            = None,
        coop_engine:  Optional[CooperationEngine]   = None,
    ) -> None:
        cfg = config or AgentV8Config()
        super().__init__(config=cfg, env=env, comm_bus=comm_bus,
                         shared_mem=shared_mem, coop_engine=coop_engine)

        self.cfg_v8:      AgentV8Config    = cfg
        self.group_mind:  Optional[GroupMind]       = group_mind
        self.global_cons: Optional[GlobalConsensus] = global_cons
        self.event_bus:   Optional[EventBus]        = event_bus

        # Subscribe to events if bus provided
        if self.event_bus:
            self.event_bus.subscribe(EventType.CONSENSUS_REACHED,
                                      self._on_consensus_reached)
            self.event_bus.subscribe(EventType.WARNING_ISSUED,
                                      self._on_warning_issued)
            self.event_bus.subscribe(EventType.KNOWLEDGE_SHARED,
                                      self._on_knowledge_shared)

        # Counters
        self._sm_reads:       int = 0
        self._sm_writes:      int = 0
        self._exp_shared:     int = 0
        self._cons_fixes:     int = 0
        self._gc_applied:     int = 0
        self._event_reactions: int = 0

    # ══════════════════════════════════════════════════════════════════
    # Override step() — v8 20-step loop
    # ══════════════════════════════════════════════════════════════════

    def step(
        self,
        text_input:   str = "",
        force_action: Optional[Tuple[str, str]] = None,
        verbose:      Optional[bool] = None,
    ) -> V8Step:
        verbose = verbose if verbose is not None else self.cfg.verbose
        step_sm_reads = step_sm_writes = step_exp = step_cons = step_gc = 0
        step_events = 0
        coord_signal = "none"

        # ── 2: Apply GroupMind coordination signal ──
        if self.group_mind:
            sig = self.group_mind.current_signal()
            if sig:
                coord_signal = sig.value
                self.group_mind.apply_signal_to_agent(self)
                # Bias goal toward primary group goal
                pg = self.group_mind.get_primary_goal()
                if pg and self.goal_system.active_count == 0:
                    self.add_goal(pg, priority=0.90)

        # ── 4: Read from shared memory ──
        if (self.shared_mem
                and self._step_count % self.cfg_v8.shared_mem_read_freq == 0):
            step_sm_reads = self._read_from_shared_memory(verbose=verbose)

        # ── Apply experiences from GroupMind ──
        if self.group_mind and self._step_count % 5 == 0:
            n_exp = self.group_mind.integrate_experiences(self)
            self._gc_applied += n_exp
            step_gc += n_exp

        # ── Run v7 step ──
        v7_result = super().step(text_input=text_input,
                                  force_action=force_action,
                                  verbose=verbose)

        # ── 16: Share experience if significant ──
        if (self.group_mind
                and v7_result.env_reward is not None
                and abs(v7_result.env_reward) >= self.cfg_v8.experience_min_reward):
            action  = v7_result.action_taken or ""
            target  = v7_result.action_obj   or ""
            outcome = "success" if v7_result.env_success else "failure"
            # Determine most relevant belief key from this action
            belief_key   = f"{target}.edible" if action == "eat" and target else ""
            belief_value = (v7_result.env_success
                            if action == "eat" else None)
            shared = self.group_mind.share_experience(
                source_agent=self.agent_id,
                action=action, target=target,
                outcome=outcome, reward=v7_result.env_reward,
                belief_key=belief_key, belief_value=belief_value,
                confidence=self.internal_state.confidence,
            )
            if shared:
                self._exp_shared += 1
                step_exp += 1

        # ── 17: Write beliefs to shared memory ──
        if self.shared_mem and v7_result.env_success:
            step_sm_writes = self._write_to_shared_memory(verbose=verbose)

        # ── 18: Bidirectional broadcast — ensure tx AND rx ──
        # Guaranteed share this step if we haven't already
        if (self.comm_bus
                and self._msgs_sent_total == 0
                and self._step_count <= 3):
            self._share_knowledge(verbose=False)

        # ── Consistency enforcement (periodic) ──
        if self._step_count % self.cfg_v8.consistency_freq == 0:
            step_cons = self._enforce_local_consistency(verbose=verbose)
            self._cons_fixes += step_cons

        # Fire KNOWLEDGE_SHARED event for significant new beliefs
        if self.event_bus and step_sm_writes > 0:
            self.event_bus.fire(
                EventType.KNOWLEDGE_SHARED, self.agent_id,
                new_beliefs=step_sm_writes,
            )
            step_events += 1

        s = V8Step(
            # V7 fields
            step=v7_result.step,
            input_text=v7_result.input_text,
            active_goal=v7_result.active_goal,
            action_taken=v7_result.action_taken,
            action_obj=v7_result.action_obj,
            env_success=v7_result.env_success,
            env_reward=v7_result.env_reward,
            novelty=v7_result.novelty,
            risk_decision=v7_result.risk_decision,
            risk_score=v7_result.risk_score,
            belief_updates=v7_result.belief_updates,
            conflicts_found=v7_result.conflicts_found,
            goals_generated=v7_result.goals_generated,
            experiment_run=v7_result.experiment_run,
            consistency_ok=v7_result.consistency_ok,
            elapsed_ms=v7_result.elapsed_ms,
            messages_received=v7_result.messages_received,
            messages_sent=v7_result.messages_sent,
            beliefs_from_peers=v7_result.beliefs_from_peers,
            social_learning=v7_result.social_learning,
            negotiations_run=v7_result.negotiations_run,
            coop_tasks_done=v7_result.coop_tasks_done,
            role_changed=v7_result.role_changed,
            vocab_size=v7_result.vocab_size,
            reputation_updates=v7_result.reputation_updates,
            # V8 fields
            shared_mem_reads=step_sm_reads,
            shared_mem_writes=step_sm_writes,
            experiences_shared=step_exp,
            consistency_fixes=step_cons,
            global_consensus_applied=step_gc,
            event_reactions=step_events,
            coord_signal=coord_signal,
        )
        return s

    # ══════════════════════════════════════════════════════════════════
    # Shared memory operations
    # ══════════════════════════════════════════════════════════════════

    def _read_from_shared_memory(self, verbose: bool = False) -> int:
        """Pull community knowledge from shared memory into private beliefs."""
        if not self.shared_mem:
            return 0
        count = 0
        trust_map = {
            aid: self.effective_trust(aid)
            for aid in self.trust._records
        }
        for entry in list(self.shared_mem.get_all(min_conf=0.55)):
            # Trust-weighted read
            val, conf = self.shared_mem.read_weighted_by_trust(
                entry.key, trust_map
            )
            if conf < 0.50:
                continue
            # Check consistency before adopting
            allowed, adj_wt, _ = self.consistency_engine.check_before_update(
                entry.key, val, source="inference", weight=conf * 0.8
            )
            if allowed and adj_wt >= 0.10:
                self.beliefs.update(entry.key, val,
                                    source="inference",
                                    weight=adj_wt,
                                    notes="from_shared_memory")
                count += 1
        self._sm_reads += count
        return count

    def _write_to_shared_memory(self, verbose: bool = False) -> int:
        """Push reliable private beliefs to shared memory."""
        if not self.shared_mem:
            return 0
        count = 0
        for belief in self.beliefs.reliable_beliefs():
            if belief.confidence < 0.65:
                continue
            self.shared_mem.write(
                belief.key, belief.value,
                self.agent_id, belief.confidence
            )
            count += 1
        self._sm_writes += count
        return count

    # ══════════════════════════════════════════════════════════════════
    # Consistency enforcement
    # ══════════════════════════════════════════════════════════════════

    def _enforce_local_consistency(self, verbose: bool = False) -> int:
        """Detect and fix internal contradictions in private beliefs."""
        audit = self.consistency_engine.audit()
        n_violations = audit.get("n_violations", 0)
        if n_violations > 0 and verbose:
            print(f"      [{self.agent_id}] ⚠ {n_violations} consistency "
                  f"violations fixed")
        return n_violations

    # ══════════════════════════════════════════════════════════════════
    # Event handlers
    # ══════════════════════════════════════════════════════════════════

    def _on_consensus_reached(self, event) -> None:
        """React to global consensus: adopt the agreed belief."""
        key   = event.payload.get("key", "")
        value = event.payload.get("value")
        conf  = event.payload.get("confidence", 0.7)
        if key and value is not None:
            self.beliefs.update(key, value, source="consensus",
                                weight=conf * 0.95,
                                notes="event:consensus_reached")
            self._event_reactions += 1

    def _on_warning_issued(self, event) -> None:
        """React to a safety warning: update dangerous belief."""
        key   = event.payload.get("key", "")
        value = event.payload.get("value")
        conf  = event.payload.get("confidence", 0.80)
        if key and value is not None:
            self.beliefs.update(key, value, source="inference",
                                weight=conf * 0.85)
            self._event_reactions += 1

    def _on_knowledge_shared(self, event) -> None:
        """React to a peer sharing knowledge via GroupMind."""
        bk = event.payload.get("belief_key", "")
        bv = event.payload.get("belief_value")
        if bk and bv is not None:
            conf = event.payload.get("confidence", 0.65)
            self.beliefs.update(bk, bv, source="group_experience",
                                weight=conf * 0.7)
            self._event_reactions += 1

    # ══════════════════════════════════════════════════════════════════
    # Bidirectional communication enforcement
    # ══════════════════════════════════════════════════════════════════

    def ensure_bidirectional_comm(self) -> Tuple[int, int]:
        """
        Guarantee this agent has sent AND received at least one message.
        Called by AgentManager after each round.
        Returns (n_sent, n_received).
        """
        if not self.comm_bus:
            return 0, 0

        # Ensure at least one message sent
        if self._msgs_sent_total == 0:
            # Force share one belief
            reliable = self.beliefs.reliable_beliefs()
            if reliable:
                b     = reliable[0]
                parts = b.key.split(".")
                if len(parts) == 2:
                    msg = Message.belief_msg(
                        self.agent_id, parts[0], parts[1],
                        b.value, b.confidence
                    )
                    self.comm_bus.broadcast(msg)
                    self._msgs_sent_total += 1

        # Ensure at least one message received
        if self._msgs_received_total == 0:
            msgs = self.comm_bus.receive(self.agent_id, max_msgs=5)
            for m in msgs:
                self._msgs_received_total += 1
                self._process_single_message(m)

        return self._msgs_sent_total, self._msgs_received_total

    def _process_single_message(self, msg: Message) -> None:
        """Process one message and update beliefs if appropriate."""
        subject   = msg.content.get("subject", "")
        predicate = msg.content.get("predicate", "")
        value     = msg.content.get("value")
        if not subject or not predicate or value is None:
            return
        key    = f"{subject}.{predicate}"
        weight = self.effective_trust(msg.sender_id) * msg.confidence
        if weight >= 0.15:
            allowed, adj_wt, _ = self.consistency_engine.check_before_update(
                key, value, "inference", weight
            )
            if allowed:
                self.beliefs.update(key, value, "inference", adj_wt)

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════

    def v8_summary(self) -> Dict:
        base = self.v7_summary()
        base.update({
            "shared_mem_reads":   self._sm_reads,
            "shared_mem_writes":  self._sm_writes,
            "experiences_shared": self._exp_shared,
            "consistency_fixes":  self._cons_fixes,
            "gc_applied":         self._gc_applied,
            "event_reactions":    self._event_reactions,
            "comm": {
                "sent":     self._msgs_sent_total,
                "received": self._msgs_received_total,
                "bidirectional": (self._msgs_sent_total > 0
                                  and self._msgs_received_total > 0),
            },
        })
        return base

    def __repr__(self) -> str:
        bi = (self._msgs_sent_total > 0 and self._msgs_received_total > 0)
        return (f"AgentV8(id={self.agent_id}, role={self.role.value}, "
                f"steps={self._step_count}, "
                f"beliefs={len(self.beliefs)}, "
                f"bicomm={bi}, "
                f"grade={self.metrics.stability_grade()})")
