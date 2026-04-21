"""
memory/episodic_memory.py
==========================
Tri-Memory Architecture

Splits memory into three functionally distinct stores, mirroring
the cognitive neuroscience distinction:

1. EPISODIC MEMORY  — specific experiences with context
   "At step 47, I ate apple, felt satisfied, reward=+0.5"
   - Time-tagged, contextual
   - Decays faster (short-to-medium term)
   - Feeds into semantic memory via consolidation

2. SEMANTIC MEMORY  — general facts and rules
   "apple is edible"  "food category → edible"
   - Context-free, general
   - More stable, slower decay
   - Built by consolidating episodic patterns

3. PROCEDURAL MEMORY — how to do things (action patterns)
   "pick → eat is a good sequence for food objects"
   "inspect before eating unknown objects"
   - Skill/strategy store
   - Updated from successful action sequences
   - Queried during planning

Together they provide different retrieval speeds:
  Semantic: fast, general facts
  Episodic: slow, context-rich details
  Procedural: very fast, habit-like recall
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .memory_store import MemoryStore, MemoryEntry
from ..latent_space.frequency_space import FrequencySpace


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """One episodic memory: a time-stamped experience with context."""
    id:           str
    step:         int
    action:       str
    target:       str
    outcome:      str       # "success" | "failure" | "unknown"
    reward:       float
    state_vec:    Optional[np.ndarray]
    context:      Dict[str, Any]
    importance:   float     = 0.5   # [0,1] — higher = more important
    access_count: int       = 0
    timestamp:    float     = field(default_factory=time.time)

    def reinforce(self, amount: float = 0.1) -> None:
        self.importance = min(1.0, self.importance + amount)
        self.access_count += 1

    def decay(self, rate: float = 0.005) -> None:
        self.importance = max(0.0, self.importance - rate)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp


# ---------------------------------------------------------------------------
# Procedure
# ---------------------------------------------------------------------------

@dataclass
class Procedure:
    """A learned action sequence / skill."""
    name:         str
    sequence:     List[Tuple[str, str]]   # [(action, target), ...]
    trigger:      str                      # context that activates this
    success_rate: float    = 0.5
    use_count:    int      = 0
    last_used:    float    = field(default_factory=time.time)

    def record_outcome(self, success: bool) -> None:
        alpha = 0.15
        self.success_rate = (1 - alpha) * self.success_rate + alpha * float(success)
        self.use_count += 1
        self.last_used  = time.time()


# ---------------------------------------------------------------------------
# EpisodicMemoryStore
# ---------------------------------------------------------------------------

class EpisodicMemoryStore:
    """
    Stores time-tagged experiences with importance scoring.

    Parameters
    ----------
    max_episodes : Maximum episodes before pruning.
    decay_rate   : Per-step decay of episode importance.
    """

    def __init__(self, max_episodes: int = 2000, decay_rate: float = 0.003) -> None:
        self.max_episodes = max_episodes
        self.decay_rate   = decay_rate
        self._episodes:   deque = deque(maxlen=max_episodes)
        self._episode_count = 0

    def record(
        self,
        step:      int,
        action:    str,
        target:    str,
        outcome:   str,
        reward:    float,
        state_vec: Optional[np.ndarray] = None,
        context:   Optional[Dict]       = None,
    ) -> Episode:
        """Record a new episode."""
        importance = self._compute_importance(outcome, reward)
        ep = Episode(
            id=str(uuid.uuid4())[:8],
            step=step,
            action=action,
            target=target,
            outcome=outcome,
            reward=reward,
            state_vec=state_vec,
            context=context or {},
            importance=importance,
        )
        self._episodes.append(ep)
        self._episode_count += 1
        return ep

    def _compute_importance(self, outcome: str, reward: float) -> float:
        """High reward, unexpected outcomes, and rare events are more important."""
        base = abs(reward)           # large rewards/penalties = important
        if outcome == "failure":
            base += 0.2              # failures are more memorable
        if abs(reward) > 0.4:
            base += 0.2              # extreme outcomes
        return float(min(1.0, 0.3 + base))

    def recall_recent(self, k: int = 10) -> List[Episode]:
        """Return k most recent episodes."""
        eps = list(self._episodes)
        return eps[-k:]

    def recall_by_importance(self, k: int = 5) -> List[Episode]:
        """Return top-k most important episodes."""
        eps = sorted(self._episodes, key=lambda e: -e.importance)
        return eps[:k]

    def recall_action_outcomes(
        self,
        action: str,
        target: Optional[str] = None,
    ) -> List[Episode]:
        """Return all episodes with a specific action (optionally filtered by target)."""
        result = [e for e in self._episodes if e.action == action]
        if target:
            result = [e for e in result if e.target == target]
        return result

    def success_rate_for(self, action: str, target: Optional[str] = None) -> float:
        """Compute success rate for an action from episodic memory."""
        eps = self.recall_action_outcomes(action, target)
        if not eps:
            return 0.5
        return sum(1 for e in eps if e.outcome == "success") / len(eps)

    def decay_all(self) -> None:
        for ep in self._episodes:
            ep.decay(self.decay_rate)

    def to_semantic_candidates(self) -> List[Tuple[str, str, Any, float]]:
        """
        Extract (subject, predicate, value, confidence) tuples
        suitable for semantic memory from high-importance episodes.
        """
        results = []
        # Group by (action, target)
        groups: Dict[Tuple[str, str], List[Episode]] = {}
        for ep in self._episodes:
            key = (ep.action, ep.target)
            groups.setdefault(key, []).append(ep)

        for (action, target), eps in groups.items():
            if len(eps) < 2:
                continue
            sr = sum(1 for e in eps if e.outcome == "success") / len(eps)
            conf = min(0.95, 0.5 + 0.3 * len(eps) * abs(sr - 0.5))
            if action == "eat":
                results.append((target, "edible", sr > 0.5, conf))
            elif action == "pick":
                results.append((target, "movable", sr > 0.5, conf))

        return results

    @property
    def size(self) -> int:
        return len(self._episodes)

    def summary(self) -> Dict:
        eps = list(self._episodes)
        if not eps:
            return {"size": 0}
        imports = [e.importance for e in eps]
        outcomes = {}
        for e in eps:
            outcomes[e.outcome] = outcomes.get(e.outcome, 0) + 1
        return {
            "size":           len(eps),
            "total_recorded": self._episode_count,
            "mean_importance": round(float(np.mean(imports)), 3),
            "outcomes":       outcomes,
        }


# ---------------------------------------------------------------------------
# ProceduralMemoryStore
# ---------------------------------------------------------------------------

class ProceduralMemoryStore:
    """
    Stores learned action sequences / skills.

    Parameters
    ----------
    max_procedures : Maximum stored procedures.
    """

    def __init__(self, max_procedures: int = 200) -> None:
        self.max_procedures = max_procedures
        self._procedures:   Dict[str, Procedure] = {}

    def store_procedure(
        self,
        name:     str,
        sequence: List[Tuple[str, str]],
        trigger:  str,
        success_rate: float = 0.5,
    ) -> Procedure:
        if name in self._procedures:
            p = self._procedures[name]
            # Update with new information
            p.sequence = sequence
            p.success_rate = 0.7 * p.success_rate + 0.3 * success_rate
            return p

        p = Procedure(name=name, sequence=sequence,
                      trigger=trigger, success_rate=success_rate)
        self._procedures[name] = p
        return p

    def recall_for_goal(self, goal_label: str) -> Optional[Procedure]:
        """Find the best procedure for a goal."""
        best = None
        best_sr = -1.0
        for p in self._procedures.values():
            if (p.trigger.lower() in goal_label.lower()
                    or goal_label.lower() in p.trigger.lower()):
                if p.success_rate > best_sr:
                    best_sr = p.success_rate
                    best    = p
        return best

    def update_outcome(self, procedure_name: str, success: bool) -> None:
        if procedure_name in self._procedures:
            self._procedures[procedure_name].record_outcome(success)

    def best_procedures(self, k: int = 5) -> List[Procedure]:
        return sorted(
            self._procedures.values(),
            key=lambda p: -p.success_rate
        )[:k]

    @property
    def size(self) -> int:
        return len(self._procedures)

    def summary(self) -> Dict:
        if not self._procedures:
            return {"size": 0}
        srs = [p.success_rate for p in self._procedures.values()]
        return {
            "size":         len(self._procedures),
            "mean_sr":      round(float(np.mean(srs)), 3),
            "best":         [(p.name, round(p.success_rate, 3))
                             for p in self.best_procedures(3)],
        }
