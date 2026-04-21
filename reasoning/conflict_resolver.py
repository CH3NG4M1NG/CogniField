"""
reasoning/conflict_resolver.py
================================
Conflict Detection and Resolution System

Detects when the agent holds contradictory beliefs and resolves them
in a principled, interpretable way.

What counts as a conflict?
--------------------------
  - apple.edible=True  (conf=0.85)  vs  apple.edible=False (conf=0.70)
  - eat(food).outcome=success (conf=0.80)  vs  eat(food).outcome=failure (conf=0.30)
  - material.edible=True  CONFLICTS WITH  stone.edible=False if stone is_a material

Resolution Strategies (in priority order)
------------------------------------------
1. EVIDENCE_WINS
   The belief with more total evidence wins.
   If evidence is close, merge into "uncertain".

2. RECENCY_WINS
   If evidence is equal but one is much more recent, prefer the newer.

3. DOWNGRADE_BOTH
   If evidence is truly conflicting and neither is dominant,
   lower both to "uncertain" and flag for re-testing.

4. SOURCE_PRIORITY
   Direct observation > inference > simulation > prior.
   Higher-priority source wins when evidence is tied.

5. REQUEST_MORE_EVIDENCE
   If no clear winner, mark the belief key as "needs_experiment"
   so the agent will test it directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..world_model.belief_system import Belief, BeliefSystem, SOURCE_WEIGHTS


class ResolutionStrategy(str, Enum):
    EVIDENCE_WINS       = "evidence_wins"
    RECENCY_WINS        = "recency_wins"
    DOWNGRADE_BOTH      = "downgrade_both"
    SOURCE_PRIORITY     = "source_priority"
    REQUEST_EXPERIMENT  = "request_experiment"


@dataclass
class ConflictRecord:
    """A detected and resolved conflict."""
    key:          str
    value_a:      Any
    conf_a:       float
    value_b:      Any
    conf_b:       float
    strategy:     ResolutionStrategy
    resolved_to:  Any
    resolved_conf: float
    notes:        str = ""
    timestamp:    float = field(default_factory=time.time)


class ConflictResolver:
    """
    Detects and resolves contradictory beliefs.

    Parameters
    ----------
    min_conflict_gap  : Minimum confidence difference to trigger a conflict.
                        If both beliefs are near 0.5, they're just uncertain
                        rather than conflicting.
    evidence_ratio    : Evidence ratio above which one side clearly wins.
    recency_threshold : Seconds — if one belief is this much more recent,
                        recency can break the tie.
    """

    def __init__(
        self,
        min_conflict_gap:  float = 0.20,
        evidence_ratio:    float = 2.0,
        recency_threshold: float = 300.0,
    ) -> None:
        self.min_conflict_gap  = min_conflict_gap
        self.evidence_ratio    = evidence_ratio
        self.recency_threshold = recency_threshold
        self._resolved: List[ConflictRecord] = []
        self._pending_experiments: List[str] = []

    # ------------------------------------------------------------------
    # Scan a BeliefSystem for conflicts
    # ------------------------------------------------------------------

    def scan(self, belief_system: BeliefSystem) -> List[ConflictRecord]:
        """
        Scan all beliefs for contradictions and resolve them.
        Returns list of ConflictRecords for any resolutions made.
        """
        new_records: List[ConflictRecord] = []

        # 1. Check recent conflict log in the BeliefSystem
        conflicts = belief_system.get_conflicts(recency_seconds=600)
        for c in conflicts:
            key = c["key"]
            b   = belief_system.get(key)
            if b is None:
                continue

            old_val  = c["old_value"]
            old_conf = c["old_conf"]
            new_val  = b.value
            new_conf = b.confidence

            # Is it still a real conflict?
            if Belief._values_agree(old_val, new_val):
                continue   # resolved itself

            record = self._resolve(
                belief_system, key,
                old_val, old_conf,
                new_val, new_conf,
                c.get("source", "unknown"),
            )
            new_records.append(record)

        # 2. Check for logical contradictions (category vs instance)
        logical = self._check_logical_consistency(belief_system)
        new_records.extend(logical)

        self._resolved.extend(new_records)
        return new_records

    # ------------------------------------------------------------------
    # Resolve a single conflict
    # ------------------------------------------------------------------

    def resolve_direct(
        self,
        belief_system: BeliefSystem,
        key:    str,
        val_a:  Any,
        conf_a: float,
        val_b:  Any,
        conf_b: float,
        source: str = "unknown",
    ) -> ConflictRecord:
        """Resolve a direct contradiction between two values for the same key."""
        record = self._resolve(belief_system, key, val_a, conf_a, val_b, conf_b, source)
        self._resolved.append(record)
        return record

    def _resolve(
        self,
        belief_system: BeliefSystem,
        key:    str,
        val_a:  Any,
        conf_a: float,
        val_b:  Any,
        conf_b: float,
        source: str,
    ) -> ConflictRecord:
        """Apply the highest-priority applicable resolution strategy."""
        b = belief_system.get(key)
        ev_a = b.pos_evidence  if b and Belief._values_agree(b.value, val_b) else conf_a
        ev_b = b.neg_evidence  if b else conf_b
        age  = b.age_seconds   if b else 0

        # Strategy 1: Evidence wins
        if ev_a > 0 and ev_b > 0 and max(ev_a, ev_b) / (min(ev_a, ev_b) + 1e-8) >= self.evidence_ratio:
            if ev_a >= ev_b:
                winner, winner_conf = val_a, conf_a
                strategy = ResolutionStrategy.EVIDENCE_WINS
            else:
                winner, winner_conf = val_b, conf_b
                strategy = ResolutionStrategy.EVIDENCE_WINS
            notes = f"Evidence ratio {ev_a:.1f}/{ev_b:.1f} favours '{winner}'"

        # Strategy 2: Source priority
        elif SOURCE_WEIGHTS.get(source, 0.5) >= 0.8 and conf_b >= 0.7:
            winner, winner_conf = val_b, conf_b
            strategy = ResolutionStrategy.SOURCE_PRIORITY
            notes = f"High-trust source '{source}' wins"

        # Strategy 3: Recency (very fresh observation overrides old uncertain belief)
        elif age > self.recency_threshold and conf_b > conf_a + 0.2:
            winner, winner_conf = val_b, conf_b
            strategy = ResolutionStrategy.RECENCY_WINS
            notes = f"Belief is {age:.0f}s old; fresh evidence wins"

        # Strategy 4: Downgrade both to uncertain
        elif abs(conf_a - conf_b) < self.min_conflict_gap:
            winner, winner_conf = None, 0.5
            strategy = ResolutionStrategy.DOWNGRADE_BOTH
            notes = f"Tied evidence ({conf_a:.2f} vs {conf_b:.2f}); marking uncertain"
            # Force belief to uncertain state
            if b:
                b.confidence  = 0.5
                b.pos_evidence = 0.5
                b.neg_evidence = 0.5
                b.notes = f"[CONFLICTED: {val_a} vs {val_b}]"

        # Strategy 5: Request experiment
        else:
            winner, winner_conf = None, 0.5
            strategy = ResolutionStrategy.REQUEST_EXPERIMENT
            notes = f"Cannot resolve {val_a}({conf_a:.2f}) vs {val_b}({conf_b:.2f}); experiment needed"
            if key not in self._pending_experiments:
                self._pending_experiments.append(key)

        return ConflictRecord(
            key=key,
            value_a=val_a, conf_a=conf_a,
            value_b=val_b, conf_b=conf_b,
            strategy=strategy,
            resolved_to=winner,
            resolved_conf=winner_conf,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Logical consistency checks
    # ------------------------------------------------------------------

    def _check_logical_consistency(
        self,
        belief_system: BeliefSystem,
    ) -> List[ConflictRecord]:
        """
        Check for logical contradictions between category rules and instance beliefs.

        Example violation:
          food.edible=True (abstract rule)  but  apple.edible=False  with  apple.is_a=food
        """
        records = []

        for b in belief_system.reliable_beliefs():
            parts = b.key.split(".")
            if len(parts) != 2:
                continue
            subject, predicate = parts
            if predicate not in ("edible", "fragile", "heavy"):
                continue

            # Check if subject has a category
            cat_belief = belief_system.get(f"{subject}.is_a")
            if cat_belief is None or not cat_belief.is_reliable:
                continue
            category = cat_belief.value

            # Check category-level rule
            cat_key = f"{category}.{predicate}"
            cat_b   = belief_system.get(cat_key)
            if cat_b is None or not cat_b.is_reliable:
                continue

            # Conflict if instance and category disagree
            if (not Belief._values_agree(b.value, cat_b.value)
                    and b.confidence >= 0.65
                    and cat_b.confidence >= 0.65):
                record = self._resolve(
                    belief_system,
                    b.key,
                    val_a=b.value,      conf_a=b.confidence,
                    val_b=cat_b.value,  conf_b=cat_b.confidence * 0.9,
                    source="inference",
                )
                record.notes = (f"Instance {b.key}={b.value} contradicts "
                                f"category rule {cat_key}={cat_b.value}")
                records.append(record)

        return records

    # ------------------------------------------------------------------
    # Pending experiments
    # ------------------------------------------------------------------

    def pop_experiment_needed(self) -> Optional[str]:
        """Return the next belief key that needs experimental resolution."""
        if self._pending_experiments:
            return self._pending_experiments.pop(0)
        return None

    def has_pending_experiments(self) -> bool:
        return bool(self._pending_experiments)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        if not self._resolved:
            return {"n_resolved": 0, "pending_experiments": len(self._pending_experiments)}
        strategies = {}
        for r in self._resolved:
            s = r.strategy.value
            strategies[s] = strategies.get(s, 0) + 1
        return {
            "n_resolved":           len(self._resolved),
            "by_strategy":          strategies,
            "pending_experiments":  len(self._pending_experiments),
            "recent":               [
                {"key": r.key, "strategy": r.strategy.value, "notes": r.notes}
                for r in self._resolved[-5:]
            ],
        }

    def __repr__(self) -> str:
        return (f"ConflictResolver("
                f"resolved={len(self._resolved)}, "
                f"pending_exp={len(self._pending_experiments)})")
