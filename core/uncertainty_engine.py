"""
core/uncertainty_engine.py
============================
Uncertainty Engine

Manages three sources of epistemic and aleatoric uncertainty:

1. OBSERVATION NOISE
   Real environments are noisy. A sensor reading "apple=edible"
   may be wrong 15% of the time. The engine injects realistic noise
   into incoming observations before they reach the belief system.

2. CONFIDENCE DECAY UNDER UNCERTAINTY
   When the environment is volatile (outcomes are inconsistent),
   beliefs should decay faster. When uncertainty is low (stable world),
   beliefs can be held more firmly.

3. PARTIAL OBSERVABILITY
   Not all object properties are observable at once. The engine
   models which properties are currently visible and which are hidden,
   and adjusts belief confidence accordingly.

Uncertainty Modes
-----------------
NONE     – clean observations (testing baseline)
LOW      – 5% noise, slow decay
MEDIUM   – 15% noise, moderate decay
HIGH     – 30% noise, fast decay
CHAOTIC  – 50% noise, very fast decay

The uncertainty level is detected dynamically from recent variance
in outcomes rather than being set manually.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class UncertaintyLevel(str, Enum):
    NONE    = "none"
    LOW     = "low"
    MEDIUM  = "medium"
    HIGH    = "high"
    CHAOTIC = "chaotic"


NOISE_RATES: Dict[UncertaintyLevel, float] = {
    UncertaintyLevel.NONE:    0.00,
    UncertaintyLevel.LOW:     0.05,
    UncertaintyLevel.MEDIUM:  0.15,
    UncertaintyLevel.HIGH:    0.30,
    UncertaintyLevel.CHAOTIC: 0.50,
}

DECAY_RATES: Dict[UncertaintyLevel, float] = {
    UncertaintyLevel.NONE:    0.000,
    UncertaintyLevel.LOW:     0.002,
    UncertaintyLevel.MEDIUM:  0.006,
    UncertaintyLevel.HIGH:    0.015,
    UncertaintyLevel.CHAOTIC: 0.035,
}

# Minimum confidence floor under each uncertainty level
CONFIDENCE_FLOORS: Dict[UncertaintyLevel, float] = {
    UncertaintyLevel.NONE:    0.05,
    UncertaintyLevel.LOW:     0.10,
    UncertaintyLevel.MEDIUM:  0.20,
    UncertaintyLevel.HIGH:    0.30,
    UncertaintyLevel.CHAOTIC: 0.40,
}


@dataclass
class NoisyObservation:
    """An observation that may have been corrupted by noise."""
    original_value:  Any
    observed_value:  Any
    was_corrupted:   bool
    noise_applied:   float    # magnitude of noise
    confidence_weight: float  # how much to trust this observation


class UncertaintyEngine:
    """
    Adds realistic uncertainty to observations and belief updates.

    Parameters
    ----------
    level   : Initial uncertainty level (can be auto-detected).
    seed    : RNG seed for reproducibility.
    auto_detect_window : Steps of outcome history to use for auto-detection.
    """

    def __init__(
        self,
        level:                UncertaintyLevel = UncertaintyLevel.MEDIUM,
        seed:                 int  = 42,
        auto_detect_window:   int  = 20,
    ) -> None:
        self.level  = level
        self._rng   = np.random.default_rng(seed)
        self._window = auto_detect_window

        # Outcome history for auto-detection
        self._outcome_variance: deque = deque(maxlen=auto_detect_window)
        self._reward_history:   deque = deque(maxlen=auto_detect_window)

        # Partially observable properties
        self._visible_props: set = {"edible", "fragile", "heavy", "color"}
        self._hidden_props:  set = set()

        # Stats
        self._n_corrupted  = 0
        self._n_total      = 0

    # ------------------------------------------------------------------
    # Noise injection
    # ------------------------------------------------------------------

    def corrupt(
        self,
        value:      Any,
        confidence: float,
        predicate:  str = "",
    ) -> NoisyObservation:
        """
        Potentially corrupt an observation with noise.

        Returns a NoisyObservation with observed_value (may differ from
        true value) and a confidence weight.
        """
        self._n_total += 1
        noise_rate = NOISE_RATES[self.level]

        # Hidden properties: observation is impossible
        if predicate and predicate in self._hidden_props:
            return NoisyObservation(
                original_value=value,
                observed_value=None,
                was_corrupted=True,
                noise_applied=1.0,
                confidence_weight=0.0,
            )

        # Coin flip for corruption
        if noise_rate > 0 and self._rng.random() < noise_rate:
            # Flip boolean observations; perturb numeric
            if isinstance(value, bool):
                corrupted = not value
            elif isinstance(value, (int, float)):
                corrupted = float(value) + self._rng.normal(0, abs(value) * 0.3 + 0.1)
            else:
                corrupted = value   # strings stay (can't easily corrupt)

            self._n_corrupted += 1
            conf_weight = max(0.1, confidence * (1.0 - noise_rate))
            return NoisyObservation(
                original_value=value,
                observed_value=corrupted,
                was_corrupted=True,
                noise_applied=noise_rate,
                confidence_weight=conf_weight,
            )

        # Clean observation
        conf_weight = confidence * (1.0 - noise_rate * 0.5)
        return NoisyObservation(
            original_value=value,
            observed_value=value,
            was_corrupted=False,
            noise_applied=0.0,
            confidence_weight=float(conf_weight),
        )

    def add_vector_noise(self, vec: np.ndarray) -> np.ndarray:
        """Add noise to a latent vector observation."""
        noise_scale = NOISE_RATES[self.level] * 0.5
        if noise_scale <= 0:
            return vec
        noise = self._rng.normal(0, noise_scale, vec.shape).astype(vec.dtype)
        noisy = vec + noise
        norm  = np.linalg.norm(noisy) + 1e-8
        return (noisy / norm).astype(vec.dtype)

    # ------------------------------------------------------------------
    # Confidence decay
    # ------------------------------------------------------------------

    def apply_decay(
        self,
        confidence: float,
        steps:      int = 1,
    ) -> float:
        """
        Decay confidence toward the floor for the current uncertainty level.
        More uncertain environment → faster decay toward higher floor.
        """
        rate  = DECAY_RATES[self.level] * steps
        floor = CONFIDENCE_FLOORS[self.level]
        decayed = confidence + rate * (floor - confidence)
        return float(np.clip(decayed, floor, 1.0))

    def decay_all_beliefs(self, belief_system, steps: int = 1) -> int:
        """Apply uncertainty decay to every belief in a BeliefSystem."""
        count = 0
        for belief in list(belief_system.all_beliefs()):
            new_conf = self.apply_decay(belief.confidence, steps)
            if abs(new_conf - belief.confidence) > 0.001:
                belief.confidence   = new_conf
                belief.pos_evidence = new_conf * belief.total_evidence
                belief.neg_evidence = (1 - new_conf) * belief.total_evidence
                count += 1
        return count

    # ------------------------------------------------------------------
    # Partial observability
    # ------------------------------------------------------------------

    def hide_property(self, predicate: str) -> None:
        """Make a property unobservable."""
        self._hidden_props.add(predicate)
        self._visible_props.discard(predicate)

    def reveal_property(self, predicate: str) -> None:
        """Make a previously hidden property observable."""
        self._hidden_props.discard(predicate)
        self._visible_props.add(predicate)

    def is_observable(self, predicate: str) -> bool:
        return predicate not in self._hidden_props

    def observability_weight(self, predicate: str) -> float:
        """Weight [0,1] for how reliably this predicate can be observed."""
        if predicate in self._hidden_props:
            return 0.0
        noise = NOISE_RATES[self.level]
        return float(1.0 - noise * 0.5)

    # ------------------------------------------------------------------
    # Auto-detection
    # ------------------------------------------------------------------

    def record_outcome_variance(self, reward: float) -> None:
        """Feed one reward observation into the variance estimator."""
        self._reward_history.append(reward)
        if len(self._reward_history) >= 5:
            variance = float(np.var(list(self._reward_history)))
            self._outcome_variance.append(variance)

    def auto_detect_level(self) -> UncertaintyLevel:
        """Infer uncertainty level from recent reward variance."""
        if len(self._outcome_variance) < 3:
            return self.level
        mean_var = float(np.mean(list(self._outcome_variance)))

        if mean_var < 0.02:
            detected = UncertaintyLevel.LOW
        elif mean_var < 0.06:
            detected = UncertaintyLevel.MEDIUM
        elif mean_var < 0.15:
            detected = UncertaintyLevel.HIGH
        else:
            detected = UncertaintyLevel.CHAOTIC

        self.level = detected
        return detected

    # ------------------------------------------------------------------
    # Consensus slowdown
    # ------------------------------------------------------------------

    def consensus_supermajority(self, base: float = 0.55) -> float:
        """
        Return a higher supermajority threshold under high uncertainty.
        Uncertain environments require more agreement before committing.
        """
        boosts = {
            UncertaintyLevel.NONE:    0.00,
            UncertaintyLevel.LOW:     0.02,
            UncertaintyLevel.MEDIUM:  0.05,
            UncertaintyLevel.HIGH:    0.10,
            UncertaintyLevel.CHAOTIC: 0.18,
        }
        return float(min(0.90, base + boosts.get(self.level, 0.0)))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        return {
            "level":         self.level.value,
            "noise_rate":    NOISE_RATES[self.level],
            "decay_rate":    DECAY_RATES[self.level],
            "n_total":       self._n_total,
            "n_corrupted":   self._n_corrupted,
            "corruption_rate": round(self._n_corrupted / max(self._n_total, 1), 3),
            "hidden_props":  list(self._hidden_props),
            "visible_props": list(self._visible_props),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"UncertaintyEngine(level={s['level']}, "
                f"noise={s['noise_rate']:.0%}, "
                f"corrupted={s['n_corrupted']}/{s['n_total']})")
