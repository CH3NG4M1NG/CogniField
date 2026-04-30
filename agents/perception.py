"""
agents/perception.py
=====================
Perception System

Converts raw environment feedback and body action results into
structured, normalised observations that the learning and reasoning
systems can directly consume.

Perception pipeline:

  raw_env_output                   (dict or ActionFeedback)
       │
       ▼
  PerceptionSystem.process()
       │
       ├── extract_signal()        → success / failure / risk
       ├── extract_properties()    → object properties seen
       ├── normalise_reward()      → reward signal [-1, +1]
       └── build_belief_updates()  → list of (key, value, confidence)
       │
       ▼
  Observation(standardised)

The Observation is the canonical hand-off between the environment and
the cognitive layers (BeliefSystem, ExperienceEngine, DeepThinker).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .body import BodyActionResult, ActionStatus


# ---------------------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------------------

class PerceptionSignal(str, Enum):
    SUCCESS  = "success"
    FAILURE  = "failure"
    RISK     = "risk"
    NEUTRAL  = "neutral"
    NOVEL    = "novel"
    DANGER   = "danger"


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """
    Normalised perception of one agent-environment interaction step.

    This is the canonical data structure passed to the learning system
    after every action.
    """
    action:        str                       # e.g. "eat"
    target:        str                       # e.g. "apple"
    signal:        PerceptionSignal          # primary signal classification
    reward:        float                     # normalised reward [-1, +1]
    confidence:    float                     # perception confidence [0, 1]

    # Belief updates derived from this observation
    # Each entry: (belief_key, value, confidence)
    belief_updates: List[Tuple[str, Any, float]] = field(default_factory=list)

    # Raw properties detected about the target
    properties:    Dict[str, Any]  = field(default_factory=dict)

    # Risk flags
    danger_detected: bool  = False
    novelty_detected:bool  = False

    # The effect label (e.g. "satisfied", "damage", "observed")
    effect:        str   = ""

    # Source context
    source:        str   = "environment"
    timestamp:     float = field(default_factory=time.time)

    @property
    def is_success(self) -> bool:
        return self.signal == PerceptionSignal.SUCCESS

    @property
    def is_failure(self) -> bool:
        return self.signal in (PerceptionSignal.FAILURE, PerceptionSignal.DANGER)

    @property
    def primary_belief_update(self) -> Optional[Tuple[str, Any, float]]:
        """The most important belief update from this observation."""
        if not self.belief_updates:
            return None
        return max(self.belief_updates, key=lambda x: x[2])  # highest confidence

    def to_dict(self) -> Dict:
        return {
            "action":          self.action,
            "target":          self.target,
            "signal":          self.signal.value,
            "reward":          round(self.reward, 3),
            "confidence":      round(self.confidence, 3),
            "effect":          self.effect,
            "danger_detected": self.danger_detected,
            "novelty_detected":self.novelty_detected,
            "belief_updates":  [(k, v, round(c, 3))
                                 for k, v, c in self.belief_updates],
            "properties":      self.properties,
        }


# ---------------------------------------------------------------------------
# Perception System
# ---------------------------------------------------------------------------

class PerceptionSystem:
    """
    Converts raw environment output into structured Observations.

    Handles four input types:
      1. BodyActionResult  — direct from body.act()
      2. Dict              — raw environment response
      3. ActionFeedback    — from SimpleEnv / RichEnv
      4. None              — no-op observation

    Parameters
    ----------
    novelty_threshold  : Properties below this confidence are "novel"
    risk_threshold     : Damage above this proportion is "danger"
    """

    def __init__(
        self,
        novelty_threshold: float = 0.40,
        risk_threshold:    float = 0.25,
    ) -> None:
        self.novelty_threshold = novelty_threshold
        self.risk_threshold    = risk_threshold
        self._observations:    List[Observation] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(
        self,
        action:       str,
        target:       str,
        body_result:  Optional[BodyActionResult] = None,
        env_response: Optional[Dict]             = None,
    ) -> Observation:
        """
        Convert raw action results into a structured Observation.

        Parameters
        ----------
        action      : Action string (e.g. "eat")
        target      : Target object name (e.g. "apple")
        body_result : Result from VirtualBody.act() (preferred source)
        env_response: Raw environment dict (fallback)

        Returns
        -------
        Observation with normalised signal, reward, and belief updates.
        """
        # Extract raw data from available sources
        raw = self._merge_sources(body_result, env_response)

        # Classify signal
        signal   = self._classify_signal(action, raw)
        reward   = self._normalise_reward(raw.get("reward", 0.0))
        conf     = float(raw.get("confidence", 0.5))
        effect   = raw.get("effect", "")
        props    = raw.get("properties", {})

        # Infer additional properties from signal
        if action == "eat" and "edible" not in props:
            if signal == PerceptionSignal.SUCCESS:
                props["edible"]  = True
                props["safe"]    = True
            elif signal == PerceptionSignal.FAILURE:
                props["edible"]  = False

        # Build belief updates
        belief_updates = self._build_belief_updates(target, action, signal,
                                                     props, reward, conf)

        # Risk and novelty flags
        danger_detected  = signal in (PerceptionSignal.DANGER,
                                       PerceptionSignal.RISK)
        novelty_detected = conf < self.novelty_threshold

        obs = Observation(
            action=action, target=target, signal=signal,
            reward=reward, confidence=conf,
            belief_updates=belief_updates, properties=props,
            danger_detected=danger_detected,
            novelty_detected=novelty_detected,
            effect=effect,
        )
        self._observations.append(obs)
        return obs

    def process_body_result(self, result: BodyActionResult) -> Observation:
        """Convenience wrapper for a BodyActionResult."""
        env_dict = {
            "reward":     result.reward,
            "effect":     result.effect,
            "confidence": result.confidence,
            "properties": result.observations,
            "status":     result.status.value,
        }
        return self.process(
            action=result.action.value,
            target=result.object_name,
            body_result=result,
            env_response=env_dict,
        )

    # ------------------------------------------------------------------
    # Signal classification
    # ------------------------------------------------------------------

    def _classify_signal(self, action: str, raw: Dict) -> PerceptionSignal:
        """Classify the primary perception signal from raw data."""
        status = raw.get("status", "")
        reward = raw.get("reward", 0.0)
        effect = raw.get("effect", "")

        # Explicit damage / dangerous effects
        if effect in ("damage", "damaged", "poisoned", "hurt", "dead"):
            return PerceptionSignal.DANGER
        if effect in ("uncertain_damage", "injury"):
            return PerceptionSignal.RISK

        # Status-based
        if status in ("success", "satisfied"):
            return PerceptionSignal.SUCCESS
        if status == "failure":
            if abs(reward) > self.risk_threshold:
                return PerceptionSignal.DANGER
            return PerceptionSignal.FAILURE

        # Reward-based fallback
        if reward > 0.10:
            return PerceptionSignal.SUCCESS
        if reward < -0.15:
            return PerceptionSignal.DANGER
        if reward < -0.05:
            return PerceptionSignal.FAILURE

        # Observations and inspections
        if action in ("inspect", "look"):
            return PerceptionSignal.NEUTRAL

        return PerceptionSignal.NEUTRAL

    # ------------------------------------------------------------------
    # Reward normalisation
    # ------------------------------------------------------------------

    def _normalise_reward(self, raw_reward: float) -> float:
        """Clip and scale reward to [-1, +1]."""
        return float(np.clip(raw_reward, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Belief updates
    # ------------------------------------------------------------------

    def _build_belief_updates(
        self,
        target:   str,
        action:   str,
        signal:   PerceptionSignal,
        props:    Dict,
        reward:   float,
        conf:     float,
    ) -> List[Tuple[str, Any, float]]:
        """
        Derive belief updates from an observed outcome.
        Returns list of (belief_key, value, confidence) tuples.
        """
        updates: List[Tuple[str, Any, float]] = []

        # Direct property observations (from inspect)
        for prop, val in props.items():
            if prop in ("inspected", "picked", "visible", "known"):
                continue
            belief_key = f"{target}.{prop}"
            updates.append((belief_key, val, min(conf, 0.85)))

        # Infer edibility from eat outcome
        if action == "eat":
            if signal == PerceptionSignal.SUCCESS:
                updates.append((f"{target}.edible", True,
                                 min(conf + 0.10, 0.92)))
                updates.append((f"{target}.safe", True,
                                 min(conf + 0.05, 0.88)))
            elif signal in (PerceptionSignal.FAILURE, PerceptionSignal.DANGER):
                updates.append((f"{target}.edible", False,
                                 min(conf + 0.10, 0.92)))
                if signal == PerceptionSignal.DANGER:
                    updates.append((f"{target}.toxic", True,
                                     min(conf + 0.05, 0.88)))
                    updates.append((f"{target}.safe", False,
                                     min(conf + 0.05, 0.88)))

        # Infer from pick outcome
        if action == "pick" and signal == PerceptionSignal.SUCCESS:
            heavy = props.get("heavy", False)
            updates.append((f"{target}.heavy", heavy, 0.80))
            fragile = props.get("fragile", False)
            if fragile:
                updates.append((f"{target}.fragile", True, 0.75))

        return updates

    # ------------------------------------------------------------------
    # Source merging
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_sources(
        body_result:  Optional[BodyActionResult],
        env_response: Optional[Dict],
    ) -> Dict:
        """Merge body result and env response into a single raw dict."""
        raw: Dict = {}

        if env_response:
            raw.update(env_response)

        if body_result:
            raw["reward"]     = body_result.reward
            raw["effect"]     = body_result.effect
            raw["confidence"] = body_result.confidence
            raw["status"]     = body_result.status.value
            # Body observations override env response for these keys
            raw.update(body_result.observations)

        return raw

    # ------------------------------------------------------------------
    # History queries
    # ------------------------------------------------------------------

    def recent(self, n: int = 10) -> List[Observation]:
        return self._observations[-n:]

    def successes(self) -> List[Observation]:
        return [o for o in self._observations if o.is_success]

    def failures(self) -> List[Observation]:
        return [o for o in self._observations if o.is_failure]

    def success_rate(self) -> float:
        if not self._observations:
            return 0.5
        n = sum(1 for o in self._observations if o.is_success)
        return n / len(self._observations)

    def summary(self) -> Dict:
        return {
            "total_observations": len(self._observations),
            "success_rate":       round(self.success_rate(), 3),
            "danger_events":      sum(1 for o in self._observations
                                      if o.danger_detected),
            "novel_events":       sum(1 for o in self._observations
                                      if o.novelty_detected),
        }

    def __repr__(self) -> str:
        return (f"PerceptionSystem(observations={len(self._observations)}, "
                f"sr={self.success_rate():.0%})")
