"""
reasoning/consistency_engine.py
================================
Consistency Engine

Actively checks that the agent's belief system is internally consistent
before any new belief is committed. Blocks or modifies updates that
would create contradictions.

Consistency Rules
-----------------
1. EDIBILITY CONSISTENCY
   If X is_a food  AND  food.edible=True (confident)
   → X.edible MUST be True (or uncertain)
   → Cannot assert X.edible=False without overriding the category rule

2. TRANSITIVITY
   If A is_a B  AND  B is_a C
   → A is_a C (infer implicitly)

3. TEMPORAL MONOTONICITY
   Confidence should not jump by more than MAX_JUMP in one update
   without strong supporting evidence.

4. ACTION OUTCOME CONSISTENCY
   If eat(X)→success was observed with conf>0.8,
   then X.edible SHOULD be True with reasonable confidence.

5. CONTRADICTION BLOCK
   Proposed update that directly contradicts a certain belief
   (conf >= BLOCK_THRESHOLD) is flagged, logged, and downgraded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..world_model.belief_system import Belief, BeliefSystem, CONFIDENCE_THRESHOLDS


BLOCK_THRESHOLD  = 0.85   # beliefs above this cannot be silently overridden
MAX_CONF_JUMP    = 0.40   # max single-step confidence change


@dataclass
class ConsistencyViolation:
    """A detected consistency violation."""
    kind:        str            # "edibility", "temporal_jump", "action_outcome", etc.
    key:         str            # belief key involved
    proposed:    Any
    existing:    Any
    action:      str            # "blocked" | "downgraded" | "logged"
    notes:       str
    timestamp:   float = field(default_factory=time.time)


class ConsistencyEngine:
    """
    Gate-keeps belief updates to maintain internal consistency.

    Parameters
    ----------
    belief_system : The agent's BeliefSystem to monitor.
    strict_mode   : If True, block all inconsistencies; else only log+downgrade.
    """

    def __init__(
        self,
        belief_system: BeliefSystem,
        strict_mode:   bool = False,
    ) -> None:
        self.bs            = belief_system
        self.strict_mode   = strict_mode
        self._violations:  List[ConsistencyViolation] = []
        self._blocked:     int = 0
        self._downgraded:  int = 0

    # ------------------------------------------------------------------
    # Check before update
    # ------------------------------------------------------------------

    def check_before_update(
        self,
        key:    str,
        value:  Any,
        source: str   = "unknown",
        weight: float = 1.0,
    ) -> Tuple[bool, float, str]:
        """
        Check whether a proposed belief update is consistent.

        Returns
        -------
        (allowed, adjusted_weight, reason)
          allowed          : Whether to proceed with the update.
          adjusted_weight  : Possibly reduced weight for the update.
          reason           : Human-readable explanation.
        """
        allowed  = True
        adj_wt   = weight
        reason   = "ok"

        # Rule 1: Block direct contradiction of certain belief
        existing = self.bs.get(key)
        if existing and existing.confidence >= BLOCK_THRESHOLD:
            if not Belief._values_agree(existing.value, value):
                v = self._record(
                    kind="direct_contradiction",
                    key=key, proposed=value,
                    existing=existing.value,
                    action="blocked" if self.strict_mode else "downgraded",
                    notes=f"Proposed {key}={value} contradicts certain "
                          f"belief {existing.value} (conf={existing.confidence:.3f})",
                )
                if self.strict_mode:
                    self._blocked += 1
                    return False, 0.0, v.notes
                else:
                    adj_wt   = weight * 0.3   # strongly downgrade
                    reason   = f"downgraded: conflicts with certain belief (conf={existing.confidence:.2f})"
                    self._downgraded += 1

        # Rule 2: Temporal monotonicity
        if existing:
            src_weight  = Belief.__class__.__init__  # dummy
            import cognifield.world_model.belief_system as _bs
            w = _bs.SOURCE_WEIGHTS.get(source, 0.5) * weight
            prospective_conf = (existing.pos_evidence + w) / (existing.total_evidence + w)
            jump = abs(prospective_conf - existing.confidence)
            if jump > MAX_CONF_JUMP and w < 0.9:
                adj_wt = weight * 0.5
                reason = f"temporal_jump: conf would jump {jump:.2f}; damped"
                self._record("temporal_jump", key, value, existing.value,
                             "downgraded", reason)
                self._downgraded += 1

        # Rule 3: Edibility-category consistency
        if ".edible" in key:
            subject = key.split(".")[0]
            cat_b   = self.bs.get(f"{subject}.is_a") or self.bs.get(f"{subject}.category")
            if cat_b and cat_b.is_reliable:
                cat_edible = self.bs.get(f"{cat_b.value}.edible")
                if cat_edible and cat_edible.confidence >= 0.80:
                    if not Belief._values_agree(cat_edible.value, value) and weight > 0.5:
                        adj_wt = weight * 0.4
                        reason = (f"category_conflict: {cat_b.value}.edible="
                                  f"{cat_edible.value} (conf={cat_edible.confidence:.2f})")
                        self._record("category_conflict", key, value,
                                     cat_edible.value, "downgraded", reason)
                        self._downgraded += 1

        return allowed, adj_wt, reason

    # ------------------------------------------------------------------
    # Post-update propagation
    # ------------------------------------------------------------------

    def propagate(self, key: str) -> List[str]:
        """
        After updating a belief, propagate logical implications.
        Returns list of keys that were also updated.

        Examples:
          food.edible=True  →  apple.edible reinforced  (if apple is_a food)
          X.is_a=food       →  X.edible reinforced  (if food.edible=True known)
        """
        updated: List[str] = []
        b = self.bs.get(key)
        if b is None:
            return updated

        parts = key.split(".")
        if len(parts) != 2:
            return updated
        subject, predicate = parts

        # Propagate category edibility to all instances
        if predicate == "edible" and b.is_reliable:
            for ib in self.bs.all_beliefs():
                ib_parts = ib.key.split(".")
                if len(ib_parts) != 2:
                    continue
                inst_subj, inst_pred = ib_parts
                if inst_pred not in ("is_a", "category"):
                    continue
                if not Belief._values_agree(ib.value, subject):
                    continue
                # Instance inst_subj is_a subject
                inst_edible_key = f"{inst_subj}.edible"
                if not self.bs.is_known(inst_edible_key, threshold=0.70):
                    self.bs.update(
                        inst_edible_key, b.value,
                        source="inference",
                        weight=b.confidence * 0.7,
                        notes=f"inferred from {key}",
                    )
                    updated.append(inst_edible_key)

        # Propagate is_a → look up category rules
        if predicate in ("is_a", "category") and b.is_reliable:
            category = b.value
            for pred in ("edible", "fragile", "heavy"):
                cat_key = f"{category}.{pred}"
                cat_b   = self.bs.get(cat_key)
                if cat_b and cat_b.is_reliable:
                    inst_key = f"{subject}.{pred}"
                    if not self.bs.is_known(inst_key, threshold=0.70):
                        self.bs.update(
                            inst_key, cat_b.value,
                            source="inference",
                            weight=cat_b.confidence * 0.7,
                            notes=f"inferred: {subject} is_a {category}, {cat_key}={cat_b.value}",
                        )
                        updated.append(inst_key)

        return updated

    # ------------------------------------------------------------------
    # Full consistency audit
    # ------------------------------------------------------------------

    def audit(self) -> Dict:
        """
        Run a full consistency audit of the belief system.
        Returns a report with any violations found.
        """
        violations = []

        # Check all .edible beliefs against category rules
        for b in list(self.bs.all_beliefs(min_conf=0.50)):
            parts = b.key.split(".")
            if len(parts) != 2:
                continue
            subj, pred = parts
            if pred != "edible":
                continue

            cat_b = self.bs.get(f"{subj}.is_a")
            if not cat_b:
                continue
            cat_edible = self.bs.get(f"{cat_b.value}.edible")
            if not cat_edible or not cat_edible.is_reliable:
                continue

            if (not Belief._values_agree(b.value, cat_edible.value)
                    and b.confidence >= 0.50):
                violations.append({
                    "type":    "edibility_category_mismatch",
                    "key":     b.key,
                    "value":   b.value,
                    "cat_key": f"{cat_b.value}.edible",
                    "cat_val": cat_edible.value,
                })

        return {
            "violations": violations,
            "n_violations": len(violations),
            "blocked":     self._blocked,
            "downgraded":  self._downgraded,
            "consistent":  len(violations) == 0,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record(self, kind, key, proposed, existing, action, notes) -> ConsistencyViolation:
        v = ConsistencyViolation(kind=kind, key=key, proposed=proposed,
                                 existing=existing, action=action, notes=notes)
        self._violations.append(v)
        return v

    def summary(self) -> Dict:
        return {
            "total_violations": len(self._violations),
            "blocked":          self._blocked,
            "downgraded":       self._downgraded,
            "recent":           [
                {"kind": v.kind, "key": v.key, "action": v.action}
                for v in self._violations[-5:]
            ],
        }

    def __repr__(self) -> str:
        return (f"ConsistencyEngine(violations={len(self._violations)}, "
                f"blocked={self._blocked}, downgraded={self._downgraded})")
