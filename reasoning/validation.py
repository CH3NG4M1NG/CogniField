"""
reasoning/validation.py
========================
Knowledge Validation System

Periodically re-tests old knowledge against:
  1. Memory  — do remembered experiences still support this belief?
  2. Abstraction rules — is this belief consistent with abstract knowledge?
  3. Environment probes — if possible, verify with a safe real-world test

Validation Strategy
-------------------
For each belief scheduled for review:
  1. Gather all supporting evidence from memory
  2. Re-compute expected confidence from scratch
  3. Compare with current confidence
  4. If they diverge significantly → downgrade or flag

This prevents "knowledge rot": beliefs that were once right but are
no longer supported by the agent's actual experience.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..world_model.belief_system import Belief, BeliefSystem, SOURCE_WEIGHTS
from ..memory.relational_memory import RelationalMemory
from ..world_model.transition_model import TransitionModel


@dataclass
class ValidationResult:
    """Result of validating one belief."""
    key:            str
    current_conf:   float
    expected_conf:  float
    delta:          float           # current - expected
    action:         str             # "confirmed" | "downgraded" | "flagged"
    evidence_count: float
    notes:          str
    timestamp:      float = field(default_factory=time.time)


class KnowledgeValidator:
    """
    Validates existing beliefs against available evidence.

    Parameters
    ----------
    belief_system    : Agent's BeliefSystem.
    rel_memory       : RelationalMemory for cross-reference.
    world_model      : TransitionModel for action-outcome evidence.
    validation_interval : Seconds between re-validations of a belief.
    max_drift        : Maximum allowed divergence before downgrading.
    """

    def __init__(
        self,
        belief_system:       BeliefSystem,
        rel_memory:          RelationalMemory,
        world_model:         TransitionModel,
        validation_interval: float = 60.0,
        max_drift:           float = 0.25,
    ) -> None:
        self.bs        = belief_system
        self.rm        = rel_memory
        self.wm        = world_model
        self.interval  = validation_interval
        self.max_drift = max_drift

        self._last_validated: Dict[str, float] = {}
        self._results: List[ValidationResult] = []
        self._cycle   = 0

    # ------------------------------------------------------------------
    # Main validation cycle
    # ------------------------------------------------------------------

    def validate_all(self, verbose: bool = False) -> List[ValidationResult]:
        """
        Validate all beliefs that are due for re-check.
        Returns list of ValidationResults for beliefs that were re-evaluated.
        """
        self._cycle += 1
        results = []
        now = time.time()

        # Validate high-confidence beliefs most frequently
        for b in list(self.bs.all_beliefs(min_conf=0.40)):
            last = self._last_validated.get(b.key, 0)
            # More confident beliefs validated more often (they matter more)
            check_interval = self.interval * (2 - b.confidence)
            if now - last < check_interval:
                continue

            result = self._validate_one(b)
            if result:
                results.append(result)
                self._last_validated[b.key] = now
                if verbose and result.action != "confirmed":
                    print(f"    [Validation] {b.key}: {result.action} "
                          f"(drift={result.delta:+.3f}) — {result.notes}")

        self._results.extend(results)
        return results

    def validate_key(self, key: str) -> Optional[ValidationResult]:
        """Validate a specific belief key on-demand."""
        b = self.bs.get(key)
        if b is None:
            return None
        result = self._validate_one(b)
        if result:
            self._results.append(result)
            self._last_validated[key] = time.time()
        return result

    # ------------------------------------------------------------------
    # Single belief validation
    # ------------------------------------------------------------------

    def _validate_one(self, belief: Belief) -> Optional[ValidationResult]:
        """
        Validate one belief against available evidence.
        Returns None if no validation was possible.
        """
        parts = belief.key.split(".")
        if len(parts) < 2:
            return None

        subject, predicate = parts[0], ".".join(parts[1:])
        expected_conf, evidence_count = self._compute_expected_confidence(
            subject, predicate, belief.value
        )

        if expected_conf < 0:
            return None   # insufficient evidence to validate

        delta = belief.confidence - expected_conf

        # Determine action
        if abs(delta) <= self.max_drift:
            action = "confirmed"
            notes  = f"Consistent with {evidence_count:.1f} evidence units"
            # Mild reinforcement
            if belief.confidence < 0.95:
                belief.reinforce(0.02)

        elif delta > self.max_drift:
            # Belief is MORE confident than evidence warrants → downgrade
            action = "downgraded"
            notes  = (f"Over-confident: current={belief.confidence:.3f} "
                      f"expected={expected_conf:.3f}; "
                      f"drift={delta:+.3f}")
            belief.confidence  = expected_conf + self.max_drift * 0.5
            belief.pos_evidence = belief.confidence * belief.total_evidence
            belief.neg_evidence = (1 - belief.confidence) * belief.total_evidence

        else:
            # Belief is LESS confident than evidence warrants → upgrade
            action = "upgraded"
            notes  = (f"Under-confident: current={belief.confidence:.3f} "
                      f"expected={expected_conf:.3f}")
            belief.confidence  = min(0.95, expected_conf - self.max_drift * 0.5)
            belief.pos_evidence = belief.confidence * belief.total_evidence
            belief.neg_evidence = (1 - belief.confidence) * belief.total_evidence

        return ValidationResult(
            key=belief.key,
            current_conf=belief.confidence,
            expected_conf=expected_conf,
            delta=delta,
            action=action,
            evidence_count=evidence_count,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Evidence gathering
    # ------------------------------------------------------------------

    def _compute_expected_confidence(
        self,
        subject:   str,
        predicate: str,
        value:     Any,
    ) -> Tuple[float, float]:
        """
        Re-derive expected confidence from available evidence sources.

        Returns
        -------
        (expected_confidence, evidence_count)  or (-1, 0) if no evidence.
        """
        pos = 0.5  # Laplace smoothing
        neg = 0.5
        evidence_units = 0.0

        # Source 1: Relational memory facts
        rm_val = self.rm.get_value(subject, predicate)
        if rm_val is not None:
            if Belief._values_agree(rm_val, value):
                pos += 1.0
            else:
                neg += 0.8
            evidence_units += 1.0

        # Source 2: World model rules
        for rule in self.wm.get_rules():
            if predicate == "edible":
                if rule.action == "eat" and rule.object_category == self.rm.get_category(subject):
                    obs_edible = rule.outcome == "success"
                    if Belief._values_agree(obs_edible, value):
                        pos += rule.confidence * (rule.hit_count + rule.miss_count)
                    else:
                        neg += rule.confidence * (rule.hit_count + rule.miss_count)
                    evidence_units += rule.hit_count + rule.miss_count

        # Source 3: Category-level beliefs
        cat = self.rm.get_category(subject)
        if cat:
            cat_b = self.bs.get(f"{cat}.{predicate}")
            if cat_b and cat_b.is_reliable:
                if Belief._values_agree(cat_b.value, value):
                    pos += cat_b.confidence * 2.0  # category rules carry more weight
                else:
                    neg += cat_b.confidence * 1.5
                evidence_units += 2.0

        if evidence_units < 0.5:
            return -1, 0.0   # not enough evidence

        expected_conf = pos / (pos + neg)
        return float(expected_conf), float(evidence_units)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        if not self._results:
            return {"cycles": self._cycle, "n_validated": 0}
        actions = {}
        for r in self._results:
            actions[r.action] = actions.get(r.action, 0) + 1
        return {
            "cycles":      self._cycle,
            "n_validated": len(self._results),
            "by_action":   actions,
            "recent":      [
                {"key": r.key, "action": r.action, "delta": round(r.delta, 3)}
                for r in self._results[-5:]
            ],
        }

    def __repr__(self) -> str:
        return (f"KnowledgeValidator(cycles={self._cycle}, "
                f"validated={len(self._results)})")
