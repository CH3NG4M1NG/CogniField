"""
agents/trust_system.py
=======================
Trust and Reputation System

Each agent maintains a trust score for every other agent it has
interacted with. Trust determines how much weight is given to
incoming messages.

Trust Dimensions
----------------
1. ACCURACY      – does this agent's beliefs match ground truth?
2. CONSISTENCY   – does this agent give stable, non-contradictory info?
3. RESPONSIVENESS – does this agent participate meaningfully?

Combined: trust = 0.5*accuracy + 0.3*consistency + 0.2*responsiveness

Trust Dynamics
--------------
  confirmed belief → trust += 0.05
  refuted belief   → trust -= 0.10
  consistent msg   → trust += 0.02
  contradictory    → trust -= 0.05
  new agent        → trust = 0.50 (neutral start)
  long silence     → trust decays slightly

Trust Use
---------
  incoming_weight = base_weight × trust_score
  (messages from trusted agents influence beliefs more)
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TrustRecord:
    """Trust record for one (evaluator, target) pair."""
    target_id:      str
    accuracy:       float = 0.5     # how often this agent is correct
    consistency:    float = 0.5     # how non-contradictory
    responsiveness: float = 0.5     # how often it communicates usefully
    confirms:       int   = 0       # times their belief was confirmed
    refutes:        int   = 0       # times their belief was wrong
    interactions:   int   = 0
    last_interaction: float = field(default_factory=time.time)

    @property
    def trust_score(self) -> float:
        """Combined trust ∈ [0, 1]."""
        return float(np.clip(
            0.5 * self.accuracy
          + 0.3 * self.consistency
          + 0.2 * self.responsiveness,
            0.05, 0.95
        ))

    @property
    def is_trusted(self) -> bool:
        return self.trust_score >= 0.60

    @property
    def is_distrusted(self) -> bool:
        return self.trust_score < 0.35

    def message_weight(self, base_confidence: float = 1.0) -> float:
        """Weight to apply to a message from this agent."""
        return base_confidence * self.trust_score


class TrustSystem:
    """
    Maintains trust scores for a set of peer agents.

    Parameters
    ----------
    owner_id    : The agent that owns this trust system.
    decay_rate  : Per-step decay of trust toward neutral (0.5).
    """

    def __init__(self, owner_id: str, decay_rate: float = 0.001) -> None:
        self.owner_id   = owner_id
        self.decay_rate = decay_rate
        self._records:  Dict[str, TrustRecord] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_peer(self, peer_id: str) -> TrustRecord:
        """Add a new peer with neutral trust."""
        if peer_id not in self._records:
            self._records[peer_id] = TrustRecord(target_id=peer_id)
        return self._records[peer_id]

    # ------------------------------------------------------------------
    # Updating trust
    # ------------------------------------------------------------------

    def update_accuracy(self, peer_id: str, was_correct: bool) -> None:
        """
        Update trust after verifying one of this agent's beliefs.
        was_correct=True  → belief was confirmed by experiment/reality
        was_correct=False → belief was refuted
        """
        rec = self._get_or_create(peer_id)
        rec.interactions += 1
        rec.last_interaction = time.time()

        if was_correct:
            rec.confirms += 1
            delta = +0.06
        else:
            rec.refutes  += 1
            delta = -0.12   # refutations hurt more than confirmations help

        # EMA update of accuracy
        alpha = 0.20
        rec.accuracy = float(np.clip(
            (1 - alpha) * rec.accuracy + alpha * (1.0 if was_correct else 0.0),
            0.05, 0.95
        ))

    def update_consistency(self, peer_id: str, is_consistent: bool) -> None:
        """Update trust based on whether a message is consistent with known facts."""
        rec = self._get_or_create(peer_id)
        alpha = 0.10
        rec.consistency = float(np.clip(
            (1 - alpha) * rec.consistency + alpha * (1.0 if is_consistent else 0.0),
            0.05, 0.95
        ))

    def update_responsiveness(self, peer_id: str, useful: bool) -> None:
        """Update trust for communicating useful (vs useless) information."""
        rec = self._get_or_create(peer_id)
        alpha = 0.08
        rec.responsiveness = float(np.clip(
            (1 - alpha) * rec.responsiveness + alpha * (1.0 if useful else 0.5),
            0.05, 0.95
        ))
        rec.interactions += 1
        rec.last_interaction = time.time()

    def observe_outcome(
        self,
        peer_id:       str,
        believed_value: object,
        actual_value:   object,
    ) -> None:
        """
        Comprehensive update: peer said X, actual outcome was Y.
        Handles both accuracy and consistency.
        """
        correct = (str(believed_value).lower() == str(actual_value).lower())
        self.update_accuracy(peer_id, correct)
        self.update_consistency(peer_id, correct)

    # ------------------------------------------------------------------
    # Reading trust
    # ------------------------------------------------------------------

    def get_trust(self, peer_id: str, default: float = 0.5) -> float:
        rec = self._records.get(peer_id)
        return rec.trust_score if rec else default

    def get_record(self, peer_id: str) -> Optional[TrustRecord]:
        return self._records.get(peer_id)

    def message_weight(
        self,
        peer_id:    str,
        confidence: float = 1.0,
    ) -> float:
        """
        Return the effective weight for a message from peer_id.
        = sender confidence × trust score
        """
        trust = self.get_trust(peer_id, default=0.5)
        return float(confidence * trust)

    def ranked_peers(self) -> List[Tuple[str, float]]:
        """Return [(peer_id, trust_score)] sorted by trust descending."""
        return sorted(
            [(pid, rec.trust_score) for pid, rec in self._records.items()],
            key=lambda x: -x[1],
        )

    def trusted_peers(self, threshold: float = 0.60) -> List[str]:
        return [pid for pid, score in self.ranked_peers() if score >= threshold]

    def distrusted_peers(self, threshold: float = 0.35) -> List[str]:
        return [pid for pid, score in self.ranked_peers() if score < threshold]

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    def decay(self) -> None:
        """Pull all trust scores slowly toward neutral (0.5)."""
        for rec in self._records.values():
            for attr in ("accuracy", "consistency", "responsiveness"):
                val = getattr(rec, attr)
                setattr(rec, attr, val + self.decay_rate * (0.5 - val))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, peer_id: str) -> TrustRecord:
        if peer_id not in self._records:
            self._records[peer_id] = TrustRecord(target_id=peer_id)
        return self._records[peer_id]

    def summary(self) -> Dict:
        if not self._records:
            return {"owner": self.owner_id, "peers": 0}
        scores = [r.trust_score for r in self._records.values()]
        return {
            "owner":       self.owner_id,
            "peers":       len(self._records),
            "mean_trust":  round(float(np.mean(scores)), 3),
            "trusted":     self.trusted_peers(),
            "distrusted":  self.distrusted_peers(),
            "records":     {
                pid: round(rec.trust_score, 3)
                for pid, rec in self._records.items()
            },
        }

    def __repr__(self) -> str:
        return (f"TrustSystem(owner={self.owner_id}, "
                f"peers={len(self._records)})")
