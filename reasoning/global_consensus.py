"""
reasoning/global_consensus.py
================================
Global Consensus Engine

Aggregates beliefs from ALL agents in the fleet into a single,
authoritative, community-level belief — then broadcasts it back
so every agent holds the same ground truth.

Pipeline
--------
  1. COLLECT   – read key belief from every agent
  2. AGGREGATE – weighted vote (confidence × trust × evidence)
  3. VALIDATE  – check the winner clears the supermajority bar
  4. STORE     – write consensus to SharedMemory with version tag
  5. BROADCAST – send CONSENSUS message to all agents via CommBus
  6. SYNC      – each agent integrates the consensus into private beliefs

Conflict Escalation
-------------------
If no clear winner:
  → key added to "contested" list
  → experiment requested from the most capable agent
  → revisited next consensus round

The difference from v6 ConsensusEngine:
  v6 – aggregates a small set of votes, standalone call
  v8 – fleet-wide, integrates with SharedMemory + CommBus + EventBus,
       runs automatically every N steps, and enforces consistency
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..world_model.belief_system import Belief, BeliefSystem
from ..memory.shared_memory import SharedMemory
from ..communication.communication_module import (
    CommunicationModule, Message, MessageType
)
from ..core.event_bus import EventBus, EventType


@dataclass
class GlobalBeliefRecord:
    """One entry in the global consensus registry."""
    key:          str
    value:        Any
    confidence:   float
    n_agents:     int
    agreement:    float       # fraction of weighted votes for winner
    version:      int   = 0
    contested:    bool  = False
    timestamp:    float = field(default_factory=time.time)

    @property
    def is_authoritative(self) -> bool:
        return self.confidence >= 0.65 and self.agreement >= 0.60


class GlobalConsensus:
    """
    Fleet-wide belief aggregation with SharedMemory integration.

    Parameters
    ----------
    shared_mem   : SharedMemory instance (written after each consensus).
    comm_bus     : CommunicationModule (broadcasts results to agents).
    event_bus    : EventBus (fires CONSENSUS_REACHED / CONFLICT_DETECTED).
    supermajority: Minimum weighted-vote fraction to declare a winner.
    min_agents   : Minimum agents that must have a belief to run consensus.
    """

    def __init__(
        self,
        shared_mem:    SharedMemory,
        comm_bus:      CommunicationModule,
        event_bus:     Optional[EventBus]  = None,
        supermajority: float = 0.55,
        min_agents:    int   = 2,
    ) -> None:
        self.sm            = shared_mem
        self.bus           = comm_bus
        self.eb            = event_bus
        self.supermajority = supermajority
        self.min_agents    = min_agents

        self._registry:   Dict[str, GlobalBeliefRecord] = {}
        self._contested:  List[str] = []
        self._round       = 0
        self._total_resolved = 0

    # ------------------------------------------------------------------
    # Single-key consensus
    # ------------------------------------------------------------------

    def resolve_key(
        self,
        key:          str,
        agent_beliefs: Dict[str, BeliefSystem],
        trust_map:    Optional[Dict[str, float]] = None,
    ) -> Optional[GlobalBeliefRecord]:
        """
        Compute global consensus for one belief key.

        Parameters
        ----------
        key           : e.g. "apple.edible"
        agent_beliefs : {agent_id: BeliefSystem}
        trust_map     : {agent_id: trust_score} from cross-peer average

        Returns
        -------
        GlobalBeliefRecord or None if too few agents have this belief.
        """
        votes: List[Tuple[Any, float]] = []   # (value, weight)

        for agent_id, bs in agent_beliefs.items():
            b = bs.get(key)
            if b is None:
                continue
            t     = (trust_map or {}).get(agent_id, 0.5)
            w     = b.confidence * b.total_evidence * t
            votes.append((b.value, w))

        if len(votes) < self.min_agents:
            return None

        # Tally
        tally: Dict[str, float] = {}
        for val, w in votes:
            k = str(val).lower()
            tally[k] = tally.get(k, 0.0) + w

        total = sum(tally.values()) + 1e-8
        winner_str    = max(tally, key=tally.get)
        winner_weight = tally[winner_str]
        agreement     = winner_weight / total

        # Map string back to Python value
        winner_val: Any = winner_str
        if winner_str == "true":   winner_val = True
        elif winner_str == "false": winner_val = False
        elif winner_str.replace(".","").replace("-","").isdigit():
            winner_val = float(winner_str)

        # Confidence = agreement × mean confidence of winning side
        winning_confs = [b.confidence for (v, _), (aid, bs)
                         in zip(votes, agent_beliefs.items())
                         if (bel := bs.get(key)) and
                            str(bel.value).lower() == winner_str
                         for b in [bel]]
        if not winning_confs:
            winning_confs = [v[1] / (max(tally.values()) + 1e-8)
                             for v in votes if str(v[0]).lower() == winner_str]
        mean_conf = float(np.mean(winning_confs)) if winning_confs else agreement

        consensus_conf = float(np.clip(agreement * mean_conf, 0.0, 0.95))
        contested      = agreement < self.supermajority

        # Version
        existing = self._registry.get(key)
        version  = (existing.version + 1) if existing else 1

        record = GlobalBeliefRecord(
            key=key, value=winner_val,
            confidence=consensus_conf,
            n_agents=len(votes),
            agreement=agreement,
            version=version,
            contested=contested,
        )
        self._registry[key] = record

        if contested:
            if key not in self._contested:
                self._contested.append(key)
        else:
            if key in self._contested:
                self._contested.remove(key)
            self._total_resolved += 1

            # Store in shared memory
            self.sm.write(key, winner_val, "global_consensus", consensus_conf)

            # Broadcast consensus message
            parts = key.split(".")
            if len(parts) == 2:
                subj, pred = parts
                msg = Message.belief_msg(
                    "global_consensus", subj, pred,
                    winner_val, consensus_conf
                )
                msg.msg_type = MessageType.CONSENSUS
                self.bus.broadcast(msg)

            # Fire event
            if self.eb:
                self.eb.fire(
                    EventType.CONSENSUS_REACHED, "global_consensus",
                    key=key, value=winner_val,
                    confidence=consensus_conf,
                    agreement=agreement,
                    version=version,
                )

        return record

    # ------------------------------------------------------------------
    # Full fleet round
    # ------------------------------------------------------------------

    def run_round(
        self,
        agent_beliefs: Dict[str, BeliefSystem],
        trust_map:    Optional[Dict[str, float]] = None,
        keys:         Optional[List[str]] = None,
        min_conf:     float = 0.40,
    ) -> Dict[str, GlobalBeliefRecord]:
        """
        Run consensus on all relevant keys across the fleet.

        Parameters
        ----------
        agent_beliefs : {agent_id: BeliefSystem}
        trust_map     : {agent_id: mean trust score}
        keys          : Explicit list of keys; if None, auto-detect
        min_conf      : Minimum confidence for a belief to be considered

        Returns
        -------
        Dict of key → GlobalBeliefRecord for all resolved keys.
        """
        self._round += 1

        # Collect candidate keys
        if keys is None:
            key_set: set = set()
            for bs in agent_beliefs.values():
                for b in bs.all_beliefs(min_conf=min_conf):
                    key_set.add(b.key)
            keys = sorted(key_set)

        results: Dict[str, GlobalBeliefRecord] = {}
        for key in keys:
            rec = self.resolve_key(key, agent_beliefs, trust_map)
            if rec:
                results[key] = rec

        # Detect conflict: contested keys that multiple agents hold
        contested_multi = [k for k in self._contested
                           if sum(1 for bs in agent_beliefs.values()
                                  if bs.get(k) is not None) >= 2]
        if contested_multi and self.eb:
            for k in contested_multi[:3]:
                self.eb.fire(
                    EventType.CONFLICT_DETECTED, "global_consensus",
                    key=k, contested_keys=contested_multi,
                )

        return results

    # ------------------------------------------------------------------
    # Apply consensus to agents
    # ------------------------------------------------------------------

    def apply_to_all(
        self,
        agent_beliefs: Dict[str, BeliefSystem],
        min_confidence: float = 0.60,
    ) -> int:
        """
        Push all authoritative global beliefs into every agent's BeliefSystem.
        Returns total number of belief updates made.
        """
        total = 0
        for key, rec in self._registry.items():
            if not rec.is_authoritative:
                continue
            for bs in agent_beliefs.values():
                bs.update(key, rec.value, source="consensus",
                          weight=rec.confidence * 0.9,
                          notes=f"global_consensus v{rec.version}")
                total += 1
        return total

    # ------------------------------------------------------------------
    # Consistency enforcer
    # ------------------------------------------------------------------

    def enforce_consistency(
        self,
        agent_beliefs: Dict[str, BeliefSystem],
    ) -> int:
        """
        For every authoritative global belief, find and fix any agent
        whose private belief contradicts it.
        Returns count of corrections made.
        """
        corrections = 0
        for key, rec in self._registry.items():
            if not rec.is_authoritative:
                continue
            for agent_id, bs in agent_beliefs.items():
                b = bs.get(key)
                if b is None:
                    continue
                if (not Belief._values_agree(b.value, rec.value)
                        and b.confidence < rec.confidence):
                    # Override with global consensus
                    bs.update(key, rec.value, source="consensus",
                              weight=rec.confidence,
                              notes="consistency_enforcer")
                    corrections += 1
                    if self.eb:
                        self.eb.fire(
                            EventType.BELIEF_UPDATED, "consistency_enforcer",
                            agent=agent_id, key=key,
                            old_value=b.value, new_value=rec.value,
                        )
        return corrections

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_global_belief(self, key: str) -> Optional[GlobalBeliefRecord]:
        return self._registry.get(key)

    def get_authoritative(self) -> Dict[str, GlobalBeliefRecord]:
        return {k: v for k, v in self._registry.items() if v.is_authoritative}

    def get_contested(self) -> List[str]:
        return list(self._contested)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        n_auth = len(self.get_authoritative())
        return {
            "rounds":          self._round,
            "total_resolved":  self._total_resolved,
            "registry_size":   len(self._registry),
            "authoritative":   n_auth,
            "contested":       len(self._contested),
            "mean_confidence": round(float(np.mean(
                [r.confidence for r in self._registry.values()]
            )) if self._registry else 0.0, 3),
            "mean_agreement":  round(float(np.mean(
                [r.agreement for r in self._registry.values()]
            )) if self._registry else 0.0, 3),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"GlobalConsensus(rounds={s['rounds']}, "
                f"authoritative={s['authoritative']}, "
                f"contested={s['contested']})")
