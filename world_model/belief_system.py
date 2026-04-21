"""
world_model/belief_system.py
=============================
Structured Belief System

Replaces raw rules with principled probabilistic beliefs.

Philosophy
----------
Every piece of knowledge the agent holds is a *belief* — not a fact.
Beliefs have:
  - a value (what we think is true)
  - a confidence (how certain we are)
  - evidence (how many observations support it)
  - recency (when was it last updated)
  - source (where did it come from)

This lets the agent:
  ✓ hold partial knowledge ("apple MIGHT be edible, conf=0.6")
  ✓ update gracefully with new evidence
  ✓ distinguish old uncertain beliefs from fresh certain ones
  ✓ detect contradictions before acting on them

Update Rule — Beta-Bayesian Updating
--------------------------------------
We model belief confidence as a Beta distribution:
  α = positive evidence count (confirms the belief)
  β = negative evidence count (disconfirms)
  confidence = α / (α + β)

Each new observation is weighted by its own reliability (source_weight).
Old beliefs decay slightly without reinforcement.

This gives:
  stone.edible after 5 failures → confidence=0.05 (very sure it's NOT edible)
  purple_berry.edible after 0 obs → confidence=0.50 (pure uncertainty)
  apple.edible after 5 successes → confidence=0.95 (very sure it IS edible)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Belief value enum + constants
# ---------------------------------------------------------------------------

UNKNOWN = "unknown"

SOURCE_WEIGHTS = {
    "direct_observation": 1.0,   # agent directly saw the outcome
    "inference":          0.7,   # derived from other beliefs
    "abstraction":        0.6,   # inferred via categorical rule
    "simulation":         0.4,   # predicted by world model
    "hypothesis":         0.2,   # guessed before testing
    "prior":              0.1,   # default assumption
}

CONFIDENCE_THRESHOLDS = {
    "certain":    0.90,   # act on this without hesitation
    "confident":  0.70,   # act on this with normal caution
    "uncertain":  0.45,   # probe further before acting
    "unknown":    0.00,   # completely unknown — must experiment first
}


# ---------------------------------------------------------------------------
# Single belief
# ---------------------------------------------------------------------------

@dataclass
class Belief:
    """
    A single structured belief with Bayesian confidence tracking.

    Fields
    ------
    key          : Unique identifier, e.g. "apple.edible" or "eat(food).outcome"
    value        : Current best estimate of the truth (any Python value).
    confidence   : Probability estimate in [0, 1].
    pos_evidence : Count of confirming observations (α in Beta dist).
    neg_evidence : Count of disconfirming observations (β in Beta dist).
    source       : Origin of the most recent update.
    last_updated : Unix timestamp of most recent update.
    created_at   : Unix timestamp of initial creation.
    notes        : Human-readable context.
    """
    key:          str
    value:        Any
    confidence:   float        = 0.5
    pos_evidence: float        = 0.5   # Laplace smoothing: start at 0.5
    neg_evidence: float        = 0.5
    source:       str          = "prior"
    last_updated: float        = field(default_factory=time.time)
    created_at:   float        = field(default_factory=time.time)
    notes:        str          = ""

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def total_evidence(self) -> float:
        return self.pos_evidence + self.neg_evidence

    @property
    def certainty_label(self) -> str:
        if self.confidence >= CONFIDENCE_THRESHOLDS["certain"]:
            return "certain"
        if self.confidence >= CONFIDENCE_THRESHOLDS["confident"]:
            return "confident"
        if self.confidence >= CONFIDENCE_THRESHOLDS["uncertain"]:
            return "uncertain"
        return "unknown"

    @property
    def age_seconds(self) -> float:
        return time.time() - self.last_updated

    @property
    def is_reliable(self) -> bool:
        """True if confidence is high enough AND backed by real evidence."""
        return (self.confidence >= CONFIDENCE_THRESHOLDS["confident"]
                and self.total_evidence >= 2.0)

    @property
    def needs_verification(self) -> bool:
        """True if the belief is uncertain enough to warrant testing."""
        return self.confidence < CONFIDENCE_THRESHOLDS["uncertain"]

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        observed_value: Any,
        source:         str   = "direct_observation",
        weight:         float = 1.0,
    ) -> "Belief":
        """
        Bayesian update given a new observation.

        Parameters
        ----------
        observed_value : The newly observed value.
        source         : Where this observation came from.
        weight         : Reliability weight [0,1] of this source.

        Returns self (fluent interface).
        """
        src_weight = SOURCE_WEIGHTS.get(source, 0.5) * weight

        # Check if observation agrees with current value
        agrees = self._values_agree(observed_value, self.value)

        if agrees:
            self.pos_evidence += src_weight
        else:
            self.neg_evidence += src_weight
            # Flip value if evidence against is now dominant
            if self.neg_evidence > self.pos_evidence * 1.5:
                self.value = observed_value

        # Recompute confidence from Beta distribution
        self.confidence = self.pos_evidence / (self.pos_evidence + self.neg_evidence)
        self.source      = source
        self.last_updated = time.time()
        return self

    def decay(self, rate: float = 0.002) -> "Belief":
        """
        Decay confidence toward 0.5 (maximum uncertainty) when unused.
        Knowledge we haven't tested recently becomes less certain.
        """
        self.confidence += rate * (0.5 - self.confidence)
        # Also update pos/neg to match
        total = self.total_evidence
        self.pos_evidence = self.confidence * total
        self.neg_evidence = (1.0 - self.confidence) * total
        return self

    def reinforce(self, amount: float = 0.1) -> "Belief":
        """Strengthen an existing belief (e.g. when confirmed by abstraction)."""
        self.pos_evidence += amount
        self.confidence = self.pos_evidence / (self.pos_evidence + self.neg_evidence)
        self.last_updated = time.time()
        return self

    @staticmethod
    def _values_agree(v1: Any, v2: Any) -> bool:
        """Check if two values represent the same belief."""
        if v1 is None or v2 is None:
            return False
        return str(v1).lower() == str(v2).lower()

    def __repr__(self) -> str:
        return (f"Belief({self.key}={self.value}, "
                f"conf={self.confidence:.3f}, "
                f"ev={self.total_evidence:.1f}, "
                f"src={self.source})")


# ---------------------------------------------------------------------------
# Belief System
# ---------------------------------------------------------------------------

class BeliefSystem:
    """
    Central repository for all agent beliefs with Bayesian updates.

    Supports:
      - Structured evidence aggregation
      - Conflict detection
      - Confidence-based filtering
      - Periodic decay of stale beliefs
      - Serialisation for inspection

    Parameters
    ----------
    decay_rate       : How fast unused beliefs decay toward uncertainty.
    min_confidence   : Minimum confidence to retain a belief.
    max_beliefs      : Maximum stored beliefs (prune oldest when full).
    """

    def __init__(
        self,
        decay_rate:     float = 0.001,
        min_confidence: float = 0.05,
        max_beliefs:    int   = 5_000,
    ) -> None:
        self.decay_rate     = decay_rate
        self.min_confidence = min_confidence
        self.max_beliefs    = max_beliefs
        self._beliefs: Dict[str, Belief] = {}
        self._update_count  = 0
        self._conflict_log: List[Dict] = []

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        key:            str,
        value:          Any,
        source:         str   = "direct_observation",
        weight:         float = 1.0,
        notes:          str   = "",
    ) -> Belief:
        """
        Update or create a belief given new evidence.

        Parameters
        ----------
        key    : Belief identifier (e.g. "apple.edible", "eat(food).outcome").
        value  : Observed value.
        source : Evidence source (see SOURCE_WEIGHTS).
        weight : Additional reliability multiplier [0, 1].

        Returns
        -------
        Updated Belief object.
        """
        self._update_count += 1

        if key in self._beliefs:
            old_belief = self._beliefs[key]
            # Log potential conflict before updating
            if (old_belief.confidence >= 0.7
                    and not Belief._values_agree(old_belief.value, value)):
                self._conflict_log.append({
                    "key":       key,
                    "old_value": old_belief.value,
                    "old_conf":  old_belief.confidence,
                    "new_value": value,
                    "source":    source,
                    "ts":        time.time(),
                })
            old_belief.update(value, source, weight)
            if notes:
                old_belief.notes = notes
            return old_belief
        else:
            # New belief
            src_weight = SOURCE_WEIGHTS.get(source, 0.5) * weight
            initial_conf = 0.5 + 0.5 * src_weight   # e.g. direct_obs → 1.0, prior → 0.55
            initial_conf = min(0.95, initial_conf)

            b = Belief(
                key=key,
                value=value,
                confidence=initial_conf,
                pos_evidence=initial_conf / (1 - initial_conf + 1e-8),
                neg_evidence=1.0,
                source=source,
                notes=notes,
            )
            self._beliefs[key] = b

            if len(self._beliefs) > self.max_beliefs:
                self._prune()
            return b

    def observe(
        self,
        subject:   str,
        predicate: str,
        value:     Any,
        source:    str   = "direct_observation",
        weight:    float = 1.0,
    ) -> Belief:
        """Convenience: update(f'{subject}.{predicate}', value, ...)"""
        return self.update(f"{subject}.{predicate}", value, source, weight)

    # ------------------------------------------------------------------
    # Reading beliefs
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Belief]:
        """Return a belief by key, or None."""
        return self._beliefs.get(key)

    def get_value(
        self,
        key:     str,
        default: Any   = None,
        min_conf: float = 0.0,
    ) -> Any:
        """
        Return the value of a belief if its confidence meets the threshold.
        Returns *default* if belief is missing or below threshold.
        """
        b = self._beliefs.get(key)
        if b is None or b.confidence < min_conf:
            return default
        return b.value

    def get_confidence(self, key: str, default: float = 0.5) -> float:
        b = self._beliefs.get(key)
        return b.confidence if b else default

    def is_known(self, key: str, threshold: float = 0.65) -> bool:
        """Return True if belief exists and is sufficiently confident."""
        b = self._beliefs.get(key)
        return b is not None and b.confidence >= threshold

    def is_uncertain(self, key: str) -> bool:
        b = self._beliefs.get(key)
        if b is None:
            return True
        return b.confidence < CONFIDENCE_THRESHOLDS["uncertain"]

    def needs_verification(self, key: str) -> bool:
        b = self._beliefs.get(key)
        if b is None:
            return True
        return b.needs_verification

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def all_beliefs(self, min_conf: float = 0.0) -> Iterator[Belief]:
        """Yield all beliefs meeting a minimum confidence."""
        for b in self._beliefs.values():
            if b.confidence >= min_conf:
                yield b

    def reliable_beliefs(self) -> List[Belief]:
        return [b for b in self._beliefs.values() if b.is_reliable]

    def uncertain_beliefs(self) -> List[Belief]:
        return [b for b in self._beliefs.values() if b.needs_verification]

    def beliefs_about(self, subject: str) -> List[Belief]:
        """Return all beliefs whose key starts with subject."""
        prefix = subject.lower() + "."
        return [b for k, b in self._beliefs.items()
                if k.lower().startswith(prefix)]

    def find_edible(self, min_conf: float = 0.60) -> List[str]:
        """Return subjects believed to be edible with >= min_conf."""
        result = []
        for key, b in self._beliefs.items():
            if ".edible" in key and b.confidence >= min_conf:
                if Belief._values_agree(b.value, True):
                    subject = key.split(".")[0]
                    result.append(subject)
        return result

    def find_dangerous(self, min_conf: float = 0.60) -> List[str]:
        """Return subjects believed to be NOT edible."""
        result = []
        for key, b in self._beliefs.items():
            if ".edible" in key and b.confidence >= min_conf:
                if Belief._values_agree(b.value, False):
                    subject = key.split(".")[0]
                    result.append(subject)
        return result

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def decay_all(self, steps: int = 1) -> int:
        """Decay all beliefs. Returns count of decayed beliefs."""
        count = 0
        for b in self._beliefs.values():
            if b.age_seconds > 60:   # only decay beliefs older than 60s
                b.decay(self.decay_rate * steps)
                count += 1
        return count

    def _prune(self) -> int:
        """Remove beliefs with confidence below threshold."""
        before = len(self._beliefs)
        to_remove = [
            k for k, b in self._beliefs.items()
            if b.confidence < self.min_confidence
               and b.total_evidence < 2.0
        ]
        for k in to_remove:
            del self._beliefs[k]
        return before - len(self._beliefs)

    def prune(self) -> int:
        return self._prune()

    # ------------------------------------------------------------------
    # Conflict tracking
    # ------------------------------------------------------------------

    def get_conflicts(self, recency_seconds: float = 300) -> List[Dict]:
        now = time.time()
        return [c for c in self._conflict_log
                if now - c["ts"] <= recency_seconds]

    @property
    def n_conflicts(self) -> int:
        return len(self._conflict_log)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        if not self._beliefs:
            return {"size": 0}
        confs = [b.confidence for b in self._beliefs.values()]
        return {
            "size":          len(self._beliefs),
            "mean_conf":     round(float(np.mean(confs)), 3),
            "reliable":      sum(1 for b in self._beliefs.values() if b.is_reliable),
            "uncertain":     sum(1 for b in self._beliefs.values() if b.needs_verification),
            "total_updates": self._update_count,
            "conflicts":     self.n_conflicts,
            "edible_known":  self.find_edible(),
            "dangerous":     self.find_dangerous(),
        }

    def __len__(self) -> int:
        return len(self._beliefs)

    def __repr__(self) -> str:
        s = self.summary()
        return (f"BeliefSystem(size={s['size']}, "
                f"mean_conf={s.get('mean_conf', 0):.3f}, "
                f"conflicts={s['conflicts']})")
