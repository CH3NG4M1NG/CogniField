"""
memory/shared_memory.py
========================
Shared Memory System

A global knowledge store that all agents can read and contribute to.
Differs from private BeliefSystem: knowledge here is community property.

Design Principles
-----------------
1. SOURCE ATTRIBUTION
   Every write records which agent wrote it and when.
   Readers can weight contributions by agent trust.

2. VERSIONING
   Each key has a version counter. When multiple agents write
   the same key, the most evidence-backed version wins
   (not last-write-wins).

3. CONFLICT LOGGING
   When agents disagree about a key, the conflict is logged
   for the consensus engine to resolve.

4. CONFIDENCE AGGREGATION
   Multiple agents writing to the same key aggregate evidence
   (like a distributed BeliefSystem).

Use cases
---------
  - Agent A discovers apple.edible=True → writes to shared memory
  - Agent B reads shared memory → learns without testing
  - Agent C sees conflicting beliefs → consensus engine resolves
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class SharedEntry:
    """One entry in the shared memory store."""
    key:          str
    value:        Any
    confidence:   float
    pos_evidence: float          # aggregated positive evidence
    neg_evidence: float          # aggregated negative evidence
    sources:      Dict[str, float]  # agent_id → their contribution weight
    version:      int  = 0
    last_updated: float = field(default_factory=time.time)
    created_at:   float = field(default_factory=time.time)
    write_count:  int  = 0

    @property
    def total_evidence(self) -> float:
        return self.pos_evidence + self.neg_evidence

    @property
    def n_contributors(self) -> int:
        return len(self.sources)

    @property
    def is_contested(self) -> bool:
        """True if evidence for and against are both substantial."""
        if self.total_evidence < 2:
            return False
        ratio = self.pos_evidence / (self.neg_evidence + 1e-8)
        return 0.25 <= ratio <= 4.0   # within 4:1 — not clearly decided

    def __repr__(self) -> str:
        return (f"SharedEntry({self.key}={self.value}, "
                f"conf={self.confidence:.3f}, "
                f"ev={self.total_evidence:.1f}, "
                f"contributors={self.n_contributors})")


class SharedMemory:
    """
    Global knowledge store shared across all agents.

    Parameters
    ----------
    max_entries      : Maximum stored keys.
    conflict_threshold : If pos/neg evidence ratio is in this range → contested.
    """

    def __init__(
        self,
        max_entries:         int   = 10_000,
        conflict_threshold:  float = 4.0,
    ) -> None:
        self.max_entries        = max_entries
        self.conflict_threshold = conflict_threshold
        self._store:     Dict[str, SharedEntry] = {}
        self._conflicts: List[Dict]             = []
        self._write_log: List[Dict]             = []
        self._read_count  = 0
        self._write_count = 0

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write(
        self,
        key:       str,
        value:     Any,
        agent_id:  str,
        confidence: float,
        source_weight: float = 1.0,
    ) -> SharedEntry:
        """
        Write a belief to shared memory.
        If the key already exists, aggregate evidence rather than overwrite.

        Parameters
        ----------
        key          : Belief key (e.g. "apple.edible").
        value        : Belief value (e.g. True).
        agent_id     : Which agent is writing.
        confidence   : Agent's confidence in this value [0, 1].
        source_weight: How much weight to give this agent's contribution.

        Returns
        -------
        Updated SharedEntry.
        """
        self._write_count += 1
        w = confidence * source_weight

        if key not in self._store:
            # New entry
            entry = SharedEntry(
                key=key,
                value=value,
                confidence=confidence,
                pos_evidence=w,
                neg_evidence=0.5,  # Laplace smoothing
                sources={agent_id: w},
                version=1,
                write_count=1,
            )
            self._store[key] = entry
        else:
            entry = self._store[key]
            entry.write_count += 1
            entry.version     += 1
            entry.last_updated = time.time()

            # Check if this is consistent or contradicting
            values_agree = self._values_agree(value, entry.value)
            if values_agree:
                entry.pos_evidence += w
            else:
                entry.neg_evidence += w
                # If evidence against now dominates, flip value
                if entry.neg_evidence > entry.pos_evidence * 1.5:
                    entry.value = value

            # Update confidence
            entry.confidence = entry.pos_evidence / entry.total_evidence

            # Track source
            entry.sources[agent_id] = entry.sources.get(agent_id, 0) + w

            # Log conflict if contested
            if entry.is_contested and not values_agree:
                self._conflicts.append({
                    "key":       key,
                    "agent_id":  agent_id,
                    "new_value": value,
                    "new_conf":  confidence,
                    "stored_value": entry.value,
                    "stored_conf":  entry.confidence,
                    "timestamp": time.time(),
                })

        self._write_log.append({
            "key": key, "agent_id": agent_id,
            "value": value, "confidence": confidence,
            "timestamp": time.time(),
        })

        if len(self._store) > self.max_entries:
            self._prune()

        return entry

    def write_many(
        self,
        beliefs:   List[Tuple[str, Any, float]],
        agent_id:  str,
        source_weight: float = 1.0,
    ) -> int:
        """Bulk write. beliefs = [(key, value, confidence), ...]"""
        count = 0
        for key, value, conf in beliefs:
            self.write(key, value, agent_id, conf, source_weight)
            count += 1
        return count

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(
        self,
        key:      str,
        min_conf: float = 0.0,
    ) -> Optional[SharedEntry]:
        """Read one entry. Returns None if not found or below min_conf."""
        self._read_count += 1
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.confidence < min_conf:
            return None
        return entry

    def read_value(
        self,
        key:     str,
        default: Any   = None,
        min_conf: float = 0.0,
    ) -> Any:
        entry = self.read(key, min_conf=min_conf)
        return entry.value if entry else default

    def get_confidence(self, key: str, default: float = 0.5) -> float:
        entry = self._store.get(key)
        return entry.confidence if entry else default

    def get_all(
        self,
        min_conf:  float = 0.0,
        predicate: Optional[str] = None,
    ) -> Iterator[SharedEntry]:
        """Iterate over all entries matching criteria."""
        for entry in self._store.values():
            if entry.confidence < min_conf:
                continue
            if predicate and not entry.key.endswith(f".{predicate}"):
                continue
            yield entry

    def find_edible(self, min_conf: float = 0.60) -> List[str]:
        result = []
        for entry in self.get_all(min_conf=min_conf, predicate="edible"):
            if self._values_agree(entry.value, True):
                subject = entry.key.rsplit(".", 1)[0]
                result.append(subject)
        return result

    def find_dangerous(self, min_conf: float = 0.60) -> List[str]:
        result = []
        for entry in self.get_all(min_conf=min_conf, predicate="edible"):
            if self._values_agree(entry.value, False):
                subject = entry.key.rsplit(".", 1)[0]
                result.append(subject)
        return result

    def contested_keys(self) -> List[str]:
        return [k for k, e in self._store.items() if e.is_contested]

    # ------------------------------------------------------------------
    # Agent trust-weighted read
    # ------------------------------------------------------------------

    def read_weighted_by_trust(
        self,
        key:          str,
        trust_scores: Dict[str, float],
    ) -> Optional[Tuple[Any, float]]:
        """
        Compute trust-weighted value and confidence from all agent contributions.

        Returns
        -------
        (value, effective_confidence) or None if no data.
        """
        entry = self._store.get(key)
        if entry is None:
            return None

        if not entry.sources:
            return entry.value, entry.confidence

        # Weighted sum of contributions
        total_weight = 0.0
        weighted_pos = 0.0
        for agent_id, contrib in entry.sources.items():
            trust = trust_scores.get(agent_id, 0.5)
            weighted_pos   += contrib * trust
            total_weight   += contrib * trust

        if total_weight < 1e-8:
            return entry.value, entry.confidence

        eff_conf = min(0.99, weighted_pos / (total_weight + entry.neg_evidence))
        return entry.value, float(eff_conf)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def _prune(self) -> int:
        """Remove oldest low-confidence entries."""
        to_remove = sorted(
            [(k, e) for k, e in self._store.items() if e.confidence < 0.30],
            key=lambda x: x[1].last_updated,
        )
        removed = 0
        for k, _ in to_remove[:len(self._store) - self.max_entries + 100]:
            del self._store[k]
            removed += 1
        return removed

    @staticmethod
    def _values_agree(v1: Any, v2: Any) -> bool:
        return str(v1).lower() == str(v2).lower()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def recent_conflicts(self, n: int = 5) -> List[Dict]:
        return self._conflicts[-n:]

    def summary(self) -> Dict:
        if not self._store:
            return {"entries": 0}
        confs = [e.confidence for e in self._store.values()]
        contributors = [e.n_contributors for e in self._store.values()]
        return {
            "entries":         len(self._store),
            "mean_confidence": round(float(np.mean(confs)), 3),
            "contested_keys":  len(self.contested_keys()),
            "total_writes":    self._write_count,
            "total_reads":     self._read_count,
            "n_conflicts":     len(self._conflicts),
            "mean_contributors": round(float(np.mean(contributors)), 2),
            "edible_known":    self.find_edible(),
            "dangerous_known": self.find_dangerous(),
        }

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        s = self.summary()
        return (f"SharedMemory(entries={s['entries']}, "
                f"mean_conf={s.get('mean_confidence', 0):.3f}, "
                f"contested={s['contested_keys']})")
