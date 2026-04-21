"""
memory/social_memory.py
========================
Social Memory

Tracks the agent's interaction history with peers:
  - who told me what
  - was it accurate
  - how often do we interact
  - what topics do we discuss

This provides richer context than raw trust scores:
  "Agent B is 70% trustworthy overall, but 90% accurate on food-related
   beliefs and only 40% accurate on tool-related beliefs."

Also tracks cooperative history:
  - joint tasks attempted and their outcomes
  - which agents complement our weaknesses
  - emergent social patterns (who leads, who follows)
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Interaction record
# ---------------------------------------------------------------------------

@dataclass
class Interaction:
    """One interaction with a peer agent."""
    peer_id:    str
    msg_type:   str          # "belief", "observation", "warning", etc.
    topic:      str          # subject.predicate e.g. "apple.edible"
    content:    Any          # what they said
    confidence: float
    correct:    Optional[bool] = None   # was it later verified?
    timestamp:  float = field(default_factory=time.time)
    round_num:  int   = 0


@dataclass
class CoopRecord:
    """Record of a cooperative task with a peer."""
    peer_id:    str
    task:       str
    success:    bool
    my_role:    str          # "leader" | "follower" | "equal"
    reward:     float
    timestamp:  float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Social Memory
# ---------------------------------------------------------------------------

class SocialMemory:
    """
    Maintains an agent's social interaction history.

    Parameters
    ----------
    owner_id  : The agent that owns this social memory.
    max_per_peer : Max interactions stored per peer.
    """

    def __init__(self, owner_id: str, max_per_peer: int = 200) -> None:
        self.owner_id     = owner_id
        self.max_per_peer = max_per_peer

        # Per-peer interaction history
        self._interactions: Dict[str, List[Interaction]] = defaultdict(list)

        # Per-peer, per-topic accuracy
        # {peer_id → {topic → [correct_bool, ...]}}
        self._topic_accuracy: Dict[str, Dict[str, List[bool]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Cooperation history
        self._coop_records: List[CoopRecord] = []

        # Interaction count shortcut
        self._interaction_counts: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Recording interactions
    # ------------------------------------------------------------------

    def record_interaction(
        self,
        peer_id:    str,
        msg_type:   str,
        topic:      str,
        content:    Any,
        confidence: float,
        round_num:  int = 0,
    ) -> Interaction:
        """Record that a peer sent us a message."""
        ix = Interaction(
            peer_id=peer_id, msg_type=msg_type, topic=topic,
            content=content, confidence=confidence, round_num=round_num,
        )
        history = self._interactions[peer_id]
        history.append(ix)
        if len(history) > self.max_per_peer:
            history.pop(0)
        self._interaction_counts[peer_id] += 1
        return ix

    def record_verification(
        self,
        peer_id:  str,
        topic:    str,
        correct:  bool,
    ) -> None:
        """
        After an experiment verifies a belief that came from peer_id,
        record whether they were correct.
        """
        # Mark the most recent interaction with this topic
        for ix in reversed(self._interactions[peer_id]):
            if ix.topic == topic and ix.correct is None:
                ix.correct = correct
                break

        self._topic_accuracy[peer_id][topic].append(correct)

    def record_cooperation(
        self,
        peer_id: str,
        task:    str,
        success: bool,
        my_role: str   = "equal",
        reward:  float = 0.0,
    ) -> None:
        """Record the outcome of a cooperative task with a peer."""
        self._coop_records.append(CoopRecord(
            peer_id=peer_id, task=task, success=success,
            my_role=my_role, reward=reward,
        ))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_interactions(
        self,
        peer_id: str,
        topic:   Optional[str] = None,
        n:       int = 20,
    ) -> List[Interaction]:
        """Return recent interactions with peer_id, optionally filtered by topic."""
        history = self._interactions.get(peer_id, [])
        if topic:
            history = [ix for ix in history if ix.topic == topic]
        return history[-n:]

    def topic_accuracy(self, peer_id: str, topic: str) -> float:
        """
        How accurate has peer_id been on this specific topic?
        Returns 0.5 if no data.
        """
        verdicts = self._topic_accuracy[peer_id].get(topic, [])
        if not verdicts:
            return 0.5
        return float(np.mean(verdicts))

    def overall_accuracy(self, peer_id: str) -> float:
        """Mean accuracy of peer_id across all topics with verified data."""
        all_verdicts = []
        for verdicts in self._topic_accuracy[peer_id].values():
            all_verdicts.extend(verdicts)
        if not all_verdicts:
            return 0.5
        return float(np.mean(all_verdicts))

    def interaction_count(self, peer_id: str) -> int:
        return self._interaction_counts[peer_id]

    def most_interactive_peers(self, n: int = 5) -> List[Tuple[str, int]]:
        """Return peers sorted by interaction count."""
        return sorted(
            self._interaction_counts.items(),
            key=lambda x: -x[1]
        )[:n]

    def cooperation_success_rate(self, peer_id: Optional[str] = None) -> float:
        """Overall or per-peer cooperation success rate."""
        records = self._coop_records
        if peer_id:
            records = [r for r in records if r.peer_id == peer_id]
        if not records:
            return 0.5
        return float(np.mean([r.success for r in records]))

    def best_cooperative_peers(self, n: int = 3) -> List[Tuple[str, float]]:
        """Return peers ranked by cooperation success rate."""
        peers = {r.peer_id for r in self._coop_records}
        if not peers:
            return []
        rated = [(pid, self.cooperation_success_rate(pid)) for pid in peers]
        return sorted(rated, key=lambda x: -x[1])[:n]

    def topics_peer_knows_well(
        self,
        peer_id:   str,
        threshold: float = 0.70,
    ) -> List[str]:
        """Return topics where this peer's accuracy exceeds threshold."""
        return [
            topic for topic, verdicts in self._topic_accuracy[peer_id].items()
            if len(verdicts) >= 2 and float(np.mean(verdicts)) >= threshold
        ]

    def peer_profile(self, peer_id: str) -> Dict:
        """Rich profile of our relationship with a peer."""
        return {
            "peer_id":           peer_id,
            "interactions":      self.interaction_count(peer_id),
            "overall_accuracy":  round(self.overall_accuracy(peer_id), 3),
            "coop_success_rate": round(self.cooperation_success_rate(peer_id), 3),
            "strong_topics":     self.topics_peer_knows_well(peer_id),
            "recent_msgs":       len(self.get_interactions(peer_id, n=5)),
        }

    # ------------------------------------------------------------------
    # Social pattern detection
    # ------------------------------------------------------------------

    def detect_leader(self) -> Optional[str]:
        """
        Identify which peer most frequently sends first (leads discussions).
        Simple heuristic: peer with highest interaction count who is also
        consistently accurate.
        """
        candidates = []
        for peer_id, count in self.most_interactive_peers(5):
            acc = self.overall_accuracy(peer_id)
            if acc >= 0.60 and count >= 3:
                candidates.append((peer_id, count * acc))
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1])[0]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def known_peers(self) -> List[str]:
        return list(self._interaction_counts.keys())

    def summary(self) -> Dict:
        return {
            "owner":          self.owner_id,
            "known_peers":    len(self.known_peers()),
            "total_interactions": sum(self._interaction_counts.values()),
            "cooperative_tasks":  len(self._coop_records),
            "coop_success_rate":  round(self.cooperation_success_rate(), 3),
            "leader":         self.detect_leader(),
        }

    def __repr__(self) -> str:
        return (f"SocialMemory(owner={self.owner_id}, "
                f"peers={len(self.known_peers())}, "
                f"interactions={sum(self._interaction_counts.values())})")
