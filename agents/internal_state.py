"""
agent/internal_state.py
========================
Internal State System

Tracks the agent's moment-to-moment cognitive state and uses it to
modulate decision making. This is the "emotional tone" of the agent —
not in a sentimental sense, but as a control signal that shifts the
balance between exploration vs exploitation, caution vs boldness.

State Variables
---------------
  confidence   [0,1] – how certain the agent is about its current knowledge
  uncertainty  [0,1] – how unsure it is about the current situation
  curiosity    [0,1] – drive to seek novel information
  fatigue      [0,1] – accumulated effort; reduces exploration
  alertness    [0,1] – sensitivity to environmental changes
  frustration  [0,1] – consequence of repeated failures

How state influences behaviour
-------------------------------
  High curiosity   → pick explore goals, lower novelty threshold
  High uncertainty → prefer safer actions, inspect before eating
  High fatigue     → consolidate memory, reduce goal generation rate
  High frustration → meta-learning trigger, revise strategies
  High confidence  → prefer exploitation over exploration
  High alertness   → increase sensitivity to new percepts
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StateSnapshot:
    """A point-in-time record of the agent's internal state."""
    step:        int
    confidence:  float
    uncertainty: float
    curiosity:   float
    fatigue:     float
    alertness:   float
    frustration: float
    timestamp:   float = field(default_factory=time.time)

    def to_vec(self) -> np.ndarray:
        return np.array([
            self.confidence, self.uncertainty, self.curiosity,
            self.fatigue, self.alertness, self.frustration,
        ], dtype=np.float32)


class InternalState:
    """
    Tracks and updates the agent's internal cognitive state.

    All variables live in [0, 1]. They evolve via:
      - exponential moving averages over recent experience
      - event-driven spikes (e.g. failure → frustration spike)
      - natural decay toward baseline values

    Parameters
    ----------
    baseline : Dict of variable → baseline value (system returns here at rest)
    ema_alpha: Smoothing factor for exponential moving average
    """

    _DEFAULTS: Dict[str, float] = {
        "confidence":  0.50,
        "uncertainty": 0.40,
        "curiosity":   0.60,
        "fatigue":     0.10,
        "alertness":   0.70,
        "frustration": 0.10,
    }
    _DECAY_RATE: Dict[str, float] = {
        "confidence":  0.02,
        "uncertainty": 0.03,
        "curiosity":   0.01,
        "fatigue":     0.05,   # fatigue decays fastest (rest)
        "alertness":   0.02,
        "frustration": 0.04,   # frustration also decays fairly fast
    }

    def __init__(
        self,
        baseline:  Optional[Dict[str, float]] = None,
        ema_alpha: float = 0.15,
    ) -> None:
        self._state   = dict(baseline or self._DEFAULTS)
        self._baseline = dict(baseline or self._DEFAULTS)
        self._alpha   = ema_alpha
        self._step    = 0
        self._history: List[StateSnapshot] = []

    # ------------------------------------------------------------------
    # Reading state
    # ------------------------------------------------------------------

    @property
    def confidence(self)  -> float: return self._state["confidence"]
    @property
    def uncertainty(self) -> float: return self._state["uncertainty"]
    @property
    def curiosity(self)   -> float: return self._state["curiosity"]
    @property
    def fatigue(self)     -> float: return self._state["fatigue"]
    @property
    def alertness(self)   -> float: return self._state["alertness"]
    @property
    def frustration(self) -> float: return self._state["frustration"]

    def get(self, key: str, default: float = 0.5) -> float:
        return self._state.get(key, default)

    def as_dict(self) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in self._state.items()}

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(step=self._step, **{
            k: self._state[k] for k in self._DEFAULTS
        })

    # ------------------------------------------------------------------
    # Updating state from experience
    # ------------------------------------------------------------------

    def on_success(self, reward: float = 0.5) -> None:
        """Update state after a successful action."""
        self._update("confidence",  +0.08 * reward)
        self._update("uncertainty", -0.06)
        self._update("frustration", -0.08)
        self._update("fatigue",     +0.02)
        self._update("curiosity",   -0.03)

    def on_failure(self, penalty: float = 0.2) -> None:
        """Update state after a failed action."""
        self._update("confidence",  -0.10 * penalty)
        self._update("uncertainty", +0.10)
        self._update("frustration", +0.12 * penalty)
        self._update("alertness",   +0.08)
        self._update("fatigue",     +0.03)

    def on_novel_input(self, novelty: float) -> None:
        """Update state when something novel is detected."""
        self._update("curiosity",  +0.15 * novelty)
        self._update("alertness",  +0.10 * novelty)
        self._update("uncertainty",+0.05 * novelty)

    def on_goal_completed(self) -> None:
        """Update state when a goal is achieved."""
        self._update("confidence",  +0.15)
        self._update("frustration", -0.20)
        self._update("curiosity",   +0.05)

    def on_exploration(self) -> None:
        """Update state when the agent actively explores."""
        self._update("curiosity",   -0.05)
        self._update("fatigue",     +0.04)
        self._update("alertness",   +0.05)

    def on_consolidation(self) -> None:
        """Update state when memory is consolidated (rest-like)."""
        self._update("fatigue",     -0.15)
        self._update("confidence",  +0.05)
        self._update("alertness",   -0.05)

    def tick(self) -> None:
        """
        One time step — apply natural decay toward baseline for all variables.
        Called once per agent step.
        """
        self._step += 1
        for key, baseline in self._baseline.items():
            current = self._state[key]
            rate    = self._DECAY_RATE.get(key, 0.02)
            # Exponential decay toward baseline
            self._state[key] = current + rate * (baseline - current)
        # Record snapshot every 10 steps
        if self._step % 10 == 0:
            self._history.append(self.snapshot())

    # ------------------------------------------------------------------
    # Decision modulation
    # ------------------------------------------------------------------

    def exploration_weight(self) -> float:
        """
        How much to weight exploration vs exploitation.
        [0,1] where 1 = fully explore.
        Driven by curiosity, anti-correlated with confidence and fatigue.
        """
        raw = (0.5 * self.curiosity
               + 0.3 * self.uncertainty
               - 0.2 * self.fatigue
               - 0.2 * self.confidence)
        return float(np.clip(raw + 0.3, 0.0, 1.0))

    def effective_novelty_threshold(self, base: float = 0.4) -> float:
        """
        Adjust novelty threshold by curiosity and fatigue.
        High curiosity → lower threshold (more things seem interesting).
        High fatigue   → higher threshold (harder to excite).
        """
        adjustment = -0.1 * self.curiosity + 0.1 * self.fatigue
        return float(np.clip(base + adjustment, 0.1, 0.9))

    def should_consolidate(self) -> bool:
        """Return True if the agent should rest and consolidate memory."""
        return self.fatigue > 0.7

    def should_meta_learn(self) -> bool:
        """Return True if frustration warrants strategy revision."""
        return self.frustration > 0.6

    def should_explore_boldly(self) -> bool:
        """Return True if curiosity is high and fatigue is low."""
        return self.curiosity > 0.7 and self.fatigue < 0.4

    def risk_tolerance(self) -> float:
        """
        How willing the agent is to take risky actions.
        [0,1] — high confidence + low frustration = willing to try new things.
        """
        return float(np.clip(
            0.5 * self.confidence - 0.3 * self.frustration + 0.3,
            0.1, 0.9
        ))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update(self, key: str, delta: float) -> None:
        """Clamp and update a state variable."""
        self._state[key] = float(np.clip(self._state[key] + delta, 0.0, 1.0))

    # ------------------------------------------------------------------
    # History & reporting
    # ------------------------------------------------------------------

    def trend(self, key: str, window: int = 5) -> str:
        """
        Return trend direction over recent history.
        '↑' rising, '↓' falling, '→' stable.
        """
        if len(self._history) < 2:
            return "→"
        recent = [getattr(s, key, 0.5) for s in self._history[-window:]]
        delta  = recent[-1] - recent[0]
        if delta > 0.05:
            return "↑"
        if delta < -0.05:
            return "↓"
        return "→"

    def summary(self) -> Dict:
        return {
            k: f"{v:.3f} {self.trend(k)}"
            for k, v in self._state.items()
        }

    def __repr__(self) -> str:
        return (f"InternalState(conf={self.confidence:.2f}, "
                f"unc={self.uncertainty:.2f}, cur={self.curiosity:.2f}, "
                f"fat={self.fatigue:.2f}, frust={self.frustration:.2f})")
