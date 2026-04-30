"""
core/experience_engine.py
===========================
Experience Engine

Closes the feedback loop: after the system makes a decision and
an outcome is observed, the experience engine:

  1. Records what happened
  2. Compares prediction to reality
  3. Updates belief weights accordingly
  4. Detects systematic errors (self-correction)
  5. Extracts generalised rules from patterns

Learning Rule
-------------
If predicted=proceed, outcome=failure:
  → belief was too optimistic → decay confidence by error_weight
  → mark belief as "needs_verification"

If predicted=avoid, outcome=success (false negative):
  → belief was too pessimistic → slightly increase confidence
  → but apply only small boost (asymmetric: false negatives are cheaper)

If predicted matches outcome:
  → reinforce the belief (small boost)
  → update evidence count

Pattern generalisation:
  After N consistent outcomes for (action, category):
  → extract category-level rule
  → all objects in that category inherit the rule
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Outcome:
    """One observed outcome record."""
    input_text:   str
    subject:      str
    predicate:    str
    prediction:   Any          # what the system predicted
    actual:       Any          # what actually happened
    was_correct:  bool
    confidence_at_time: float
    action_taken: str
    reward:       float
    step:         int
    timestamp:    float = field(default_factory=time.time)


@dataclass
class CorrectionRecord:
    """A belief that was corrected based on outcome feedback."""
    key:           str
    old_value:     Any
    old_confidence: float
    new_confidence: float
    reason:        str
    timestamp:     float = field(default_factory=time.time)


class ExperienceEngine:
    """
    Learns from outcomes and updates the belief system automatically.

    Parameters
    ----------
    belief_system       : The agent's BeliefSystem to update.
    error_penalty       : How much to penalise a wrong confident prediction.
    correct_boost       : How much to boost a correct prediction.
    generalise_after    : Number of consistent outcomes to trigger rule generalisation.
    correction_threshold: Confidence above which a wrong prediction triggers correction.
    """

    def __init__(
        self,
        belief_system,
        error_penalty:          float = 0.15,
        correct_boost:          float = 0.04,
        generalise_after:       int   = 4,
        correction_threshold:   float = 0.65,
    ) -> None:
        self.bs                   = belief_system
        self.error_penalty        = error_penalty
        self.correct_boost        = correct_boost
        self.generalise_after     = generalise_after
        self.correction_threshold = correction_threshold

        self._outcomes:      List[Outcome]         = []
        self._corrections:   List[CorrectionRecord]= []
        # Outcome tracking by (action, subject) for generalisation
        self._action_stats:  Dict[str, deque]      = defaultdict(lambda: deque(maxlen=20))
        # Category-level outcome memory
        self._cat_stats:     Dict[str, deque]      = defaultdict(lambda: deque(maxlen=30))
        self._rules_derived: List[str]             = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def learn_from_outcome(
        self,
        input_text:   str,
        subject:      str,
        predicate:    str,
        prediction:   Any,
        actual:       Any,
        action:       str   = "",
        reward:       float = 0.0,
        step:         int   = 0,
    ) -> List[CorrectionRecord]:
        """
        Record an outcome and update beliefs accordingly.

        Parameters
        ----------
        input_text  : Original input question/text.
        subject     : Object/concept that was acted on.
        predicate   : Property that was predicted.
        prediction  : What the system predicted (True/False/value).
        actual      : What actually happened.
        action      : Action taken (eat, pick, etc.).
        reward      : Observed reward signal.
        step        : Current agent step number.

        Returns
        -------
        List of CorrectionRecord for any beliefs that were updated.
        """
        key = f"{subject}.{predicate}"
        belief = self.bs.get(key)
        conf   = belief.confidence if belief else 0.5

        was_correct = self._outcomes_agree(prediction, actual)

        outcome = Outcome(
            input_text=input_text, subject=subject, predicate=predicate,
            prediction=prediction, actual=actual,
            was_correct=was_correct, confidence_at_time=conf,
            action_taken=action, reward=reward, step=step,
        )
        self._outcomes.append(outcome)
        self._action_stats[f"{action}_{subject}"].append(was_correct)

        # Category-level tracking
        cat_b = self.bs.get(f"{subject}.category") or self.bs.get(f"{subject}.is_a")
        if cat_b:
            self._cat_stats[f"{cat_b.value}_{predicate}"].append(
                (was_correct, actual)
            )

        corrections = []

        if not was_correct:
            c = self._correct_wrong_prediction(key, prediction, actual, conf, reward)
            if c:
                corrections.append(c)
        else:
            self._reinforce_correct_prediction(key, actual, conf)

        # Try to generalise to category level
        gen_corrections = self._try_generalise(subject, predicate, actual, was_correct)
        corrections.extend(gen_corrections)

        self._corrections.extend(corrections)
        return corrections

    # ------------------------------------------------------------------
    # Core learning mechanics
    # ------------------------------------------------------------------

    def _correct_wrong_prediction(
        self,
        key:        str,
        prediction: Any,
        actual:     Any,
        conf:       float,
        reward:     float,
    ) -> Optional[CorrectionRecord]:
        """Penalise an overconfident wrong prediction."""
        belief = self.bs.get(key)
        if belief is None:
            # Create a belief with the correct actual value
            self.bs.update(key, actual, source="direct_observation",
                           weight=0.7, notes="from_experience_engine")
            return CorrectionRecord(
                key=key, old_value=prediction, old_confidence=conf,
                new_confidence=0.7 * 0.5 + 0.3,
                reason=f"No prior belief; created from actual outcome={actual}",
            )

        if conf < self.correction_threshold:
            # Belief was uncertain anyway — just update it gently
            self.bs.update(key, actual, source="direct_observation", weight=0.6)
            return None

        # Confident but wrong → significant penalty
        penalty = self.error_penalty * (conf - 0.5) * 2   # scale by overconfidence
        old_conf = belief.confidence
        belief.confidence   = max(0.10, belief.confidence - penalty)
        belief.neg_evidence += penalty * 3
        belief.pos_evidence  = belief.confidence * belief.total_evidence
        belief.neg_evidence  = (1 - belief.confidence) * belief.total_evidence

        # If reward was very negative, also update the value
        if reward < -0.3:
            self.bs.update(key, actual, source="direct_observation", weight=0.8)

        return CorrectionRecord(
            key=key, old_value=prediction, old_confidence=old_conf,
            new_confidence=belief.confidence,
            reason=(f"Wrong prediction (predicted={prediction}, actual={actual}). "
                    f"Penalty={penalty:.3f}. conf {old_conf:.3f}→{belief.confidence:.3f}"),
        )

    def _reinforce_correct_prediction(
        self,
        key:    str,
        actual: Any,
        conf:   float,
    ) -> None:
        """Gently reinforce a correct prediction."""
        belief = self.bs.get(key)
        if belief and belief.confidence < 0.95:
            belief.reinforce(self.correct_boost)
        elif belief is None:
            self.bs.update(key, actual, source="direct_observation", weight=0.65)

    def _try_generalise(
        self,
        subject:    str,
        predicate:  str,
        actual:     Any,
        correct:    bool,
    ) -> List[CorrectionRecord]:
        """
        After N consistent outcomes for a category, extract a general rule.
        E.g. if all food objects are edible → food.edible = True
        """
        corrections = []
        cat_b = self.bs.get(f"{subject}.category") or self.bs.get(f"{subject}.is_a")
        if not cat_b:
            return corrections

        key   = f"{cat_b.value}_{predicate}"
        stats = list(self._cat_stats[key])
        if len(stats) < self.generalise_after:
            return corrections

        # Check consistency of recent outcomes
        recent_actuals = [s[1] for s in stats[-self.generalise_after:]]
        if len(set(str(a).lower() for a in recent_actuals)) == 1:
            # All consistent → derive category-level rule
            cat_key  = f"{cat_b.value}.{predicate}"
            old_b    = self.bs.get(cat_key)
            old_conf = old_b.confidence if old_b else 0.5
            new_conf = min(0.88, 0.5 + len(stats) * 0.04)

            self.bs.update(cat_key, actual, source="abstraction",
                           weight=new_conf, notes="generalised_from_experience")
            rule_str = f"{cat_b.value}.{predicate}={actual} (from {len(stats)} outcomes)"

            if rule_str not in self._rules_derived:
                self._rules_derived.append(rule_str)
                corrections.append(CorrectionRecord(
                    key=cat_key, old_value=actual, old_confidence=old_conf,
                    new_confidence=new_conf,
                    reason=f"Generalised: {rule_str}",
                ))
        return corrections

    # ------------------------------------------------------------------
    # Batch self-correction
    # ------------------------------------------------------------------

    def audit_and_correct(self) -> List[CorrectionRecord]:
        """
        Scan recent outcomes for systematic errors and correct beliefs.
        Call periodically (e.g. every 10 steps).
        """
        corrections = []

        # Find (subject, predicate) pairs with ≥3 wrong predictions
        error_counts: Dict[str, int] = defaultdict(int)
        for o in self._outcomes[-30:]:
            if not o.was_correct and o.confidence_at_time >= self.correction_threshold:
                error_counts[f"{o.subject}.{o.predicate}"] += 1

        for key, n_errors in error_counts.items():
            if n_errors < 2:
                continue
            belief = self.bs.get(key)
            if belief is None:
                continue
            old_conf = belief.confidence
            # Systematic error: aggressively reduce confidence
            penalty  = min(0.30, n_errors * 0.08)
            belief.confidence = max(0.10, belief.confidence - penalty)
            corrections.append(CorrectionRecord(
                key=key, old_value=belief.value, old_confidence=old_conf,
                new_confidence=belief.confidence,
                reason=f"Systematic error: {n_errors} wrong predictions. Corrected.",
            ))

        self._corrections.extend(corrections)
        return corrections

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def success_rate(self, action: Optional[str] = None) -> float:
        outcomes = self._outcomes
        if action:
            outcomes = [o for o in outcomes if o.action_taken == action]
        if not outcomes:
            return 0.5
        return float(np.mean([o.was_correct for o in outcomes]))

    def derived_rules(self) -> List[str]:
        return list(self._rules_derived)

    @staticmethod
    def _outcomes_agree(pred: Any, actual: Any) -> bool:
        return str(pred).lower() == str(actual).lower()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        n = len(self._outcomes)
        if n == 0:
            return {"outcomes": 0}
        sr = float(np.mean([o.was_correct for o in self._outcomes]))
        return {
            "outcomes":       n,
            "success_rate":   round(sr, 3),
            "corrections":    len(self._corrections),
            "rules_derived":  len(self._rules_derived),
            "rules":          self._rules_derived[-3:],
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"ExperienceEngine(outcomes={s['outcomes']}, "
                f"sr={s.get('success_rate',0):.0%}, "
                f"corrections={s.get('corrections',0)})")
