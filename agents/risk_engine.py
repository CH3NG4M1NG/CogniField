"""
agent/risk_engine.py
=====================
Risk Awareness System

Prevents the agent from taking dangerous actions when confidence is low.
Every action gets a risk score before execution. The agent will not
proceed with high-risk actions unless it has sufficient confidence.

Risk Model
----------
risk_score(action, target) = base_risk(action)
                             × uncertainty_factor(target)
                             × (1 - known_safety(target))

Where:
  base_risk          : Inherent danger of the action type
  uncertainty_factor : How uncertain we are about the target
  known_safety       : Whether we know this is safe (from beliefs)

The agent's risk tolerance scales with:
  - Internal state confidence (high conf → more willing to act)
  - Belief confidence about the target
  - Whether simulation endorsed the action

Decision: PROCEED / CAUTION / BLOCK
  PROCEED: risk <= tolerance
  CAUTION: explore safer alternatives first
  BLOCK:   do not execute under any circumstances
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..world_model.belief_system import Belief, BeliefSystem, CONFIDENCE_THRESHOLDS


# Base risk of each action type
ACTION_BASE_RISK: Dict[str, float] = {
    "observe":   0.00,
    "inspect":   0.00,
    "move":      0.05,
    "pick":      0.10,
    "drop":      0.15,
    "combine":   0.20,
    "use":       0.25,
    "eat":       0.70,
}

DECISION_THRESHOLDS = {
    "proceed": 0.35,   # risk below this → proceed
    "caution": 0.60,   # risk below this → caution
    # above 0.60 → block
}


@dataclass
class RiskAssessment:
    """Complete risk assessment for one action."""
    action:          str
    target:          str
    risk_score:      float        # [0, 1] — 0=safe, 1=certain harm
    decision:        str          # "proceed" | "caution" | "block"
    reason:          str
    safer_alternative: Optional[str]  # suggest this action instead
    confidence_needed: float      # what confidence would allow this action
    timestamp:       float = field(default_factory=time.time)


class RiskEngine:
    """
    Evaluates and controls action risk.

    Parameters
    ----------
    belief_system    : For looking up confidence about targets.
    risk_tolerance   : Default maximum acceptable risk score [0, 1].
    """

    def __init__(
        self,
        belief_system: BeliefSystem,
        risk_tolerance: float = 0.35,
    ) -> None:
        self.bs              = belief_system
        self.risk_tolerance  = risk_tolerance
        self._assessments:   List[RiskAssessment] = []
        self._blocked_count: int = 0
        self._caution_count: int = 0

    # ------------------------------------------------------------------
    # Main assessment
    # ------------------------------------------------------------------

    def assess(
        self,
        action:          str,
        target:          str,
        sim_prediction:  Optional[str]  = None,
        sim_confidence:  float          = 0.5,
        agent_confidence: float         = 0.5,   # internal state confidence
    ) -> RiskAssessment:
        """
        Compute a full risk assessment for a proposed action.

        Parameters
        ----------
        action           : Action name (eat, pick, inspect, ...).
        target           : Target object name.
        sim_prediction   : World model simulation result ("success"/"failure").
        sim_confidence   : Confidence of simulation prediction.
        agent_confidence : Agent's current internal confidence level.

        Returns
        -------
        RiskAssessment
        """
        base = ACTION_BASE_RISK.get(action.lower(), 0.20)

        # Uncertainty factor: how much we don't know about the target
        unc   = self._uncertainty_factor(action, target)

        # Safety factor: do we have evidence the target is safe?
        safety = self._known_safety(action, target)

        # Simulation penalty
        sim_penalty = 0.0
        if sim_prediction == "failure" and sim_confidence >= 0.6:
            sim_penalty = sim_confidence * 0.3
        elif sim_prediction == "success" and sim_confidence >= 0.7:
            sim_penalty = -0.1   # simulation says it's safe

        # Agent confidence modifier (more confident agent = slightly lower risk)
        agent_modifier = -0.05 * (agent_confidence - 0.5)

        risk = float(np.clip(
            base * unc * (1 - safety) + sim_penalty + agent_modifier,
            0.0, 1.0
        ))

        # Dynamic tolerance
        effective_tolerance = self.risk_tolerance + 0.1 * (agent_confidence - 0.5)

        # Decision
        if risk <= effective_tolerance:
            decision = "proceed"
            reason   = f"risk={risk:.3f} below tolerance={effective_tolerance:.3f}"
            safer    = None
        elif risk <= DECISION_THRESHOLDS["caution"]:
            decision = "caution"
            reason   = f"risk={risk:.3f} — consider safer alternatives first"
            safer    = self._suggest_safer(action, target)
            self._caution_count += 1
        else:
            decision = "block"
            reason   = self._block_reason(action, target, risk, unc, safety)
            safer    = self._suggest_safer(action, target)
            self._blocked_count += 1

        conf_needed = self._confidence_needed_for(action, target)

        ra = RiskAssessment(
            action=action,
            target=target,
            risk_score=risk,
            decision=decision,
            reason=reason,
            safer_alternative=safer,
            confidence_needed=conf_needed,
        )
        self._assessments.append(ra)
        return ra

    # ------------------------------------------------------------------
    # Component factors
    # ------------------------------------------------------------------

    def _uncertainty_factor(self, action: str, target: str) -> float:
        """
        How uncertain are we about the safety of this action on this target?
        1.0 = completely unknown, 0.0 = perfectly known safe.
        """
        # For eating: uncertainty about edibility
        if action == "eat":
            b = self.bs.get(f"{target}.edible")
            if b is None:
                return 1.0
            # High uncertainty when confidence is near 0.5
            return 1.0 - abs(b.confidence - 0.5) * 2.0

        # For dropping: uncertainty about fragility
        elif action == "drop":
            b = self.bs.get(f"{target}.fragile")
            if b is None:
                return 0.5   # unknown fragility is moderate risk
            return 1.0 - abs(b.confidence - 0.5) * 2.0

        # For picking: uncertainty about weight/movability
        elif action == "pick":
            b = self.bs.get(f"{target}.heavy")
            if b is None:
                return 0.3
            return 0.1 if b.confidence > 0.7 else 0.4

        return 0.3   # default moderate uncertainty

    def _known_safety(self, action: str, target: str) -> float:
        """
        How confident are we that this action is safe?
        Returns a safety credit [0, 1] — 1 = known safe.
        """
        if action == "eat":
            b = self.bs.get(f"{target}.edible")
            if b and Belief._values_agree(b.value, True) and b.confidence >= 0.70:
                return b.confidence   # we know it's edible
            if b and Belief._values_agree(b.value, False):
                return 0.0            # we know it's NOT edible — zero safety
            return 0.3               # unknown — moderate

        elif action in ("observe", "inspect"):
            return 1.0               # always safe

        elif action == "pick":
            b = self.bs.get(f"{target}.movable")
            if b and Belief._values_agree(b.value, True) and b.confidence >= 0.6:
                return 0.85
            return 0.6               # usually safe to pick

        return 0.5

    def _suggest_safer(self, action: str, target: str) -> Optional[str]:
        """Suggest a safer action that provides some information."""
        ladder = ["observe", "inspect", "pick", "combine", "eat"]
        idx = ladder.index(action) if action in ladder else len(ladder)
        if idx > 0:
            return ladder[idx - 1]
        return "inspect"

    def _block_reason(self, action, target, risk, unc, safety) -> str:
        b = self.bs.get(f"{target}.edible")
        if b and Belief._values_agree(b.value, False) and b.confidence >= 0.80:
            return (f"BLOCKED: {target} is known NOT edible "
                    f"(conf={b.confidence:.2f})")
        return (f"BLOCKED: risk={risk:.3f} too high "
                f"(uncertainty={unc:.2f}, safety={safety:.2f})")

    def _confidence_needed_for(self, action: str, target: str) -> float:
        """What belief confidence would allow this action?"""
        if action == "eat":
            return 0.75
        elif action == "drop":
            return 0.55
        elif action == "pick":
            return 0.30
        return 0.20

    # ------------------------------------------------------------------
    # Batch assessment
    # ------------------------------------------------------------------

    def safest_action(
        self,
        candidates: List[Tuple[str, str]],   # [(action, target), ...]
        agent_confidence: float = 0.5,
    ) -> Tuple[Optional[Tuple[str, str]], RiskAssessment]:
        """
        From a list of candidate actions, return the one with the lowest risk.
        """
        best_ra  = None
        best_act = None
        for action, target in candidates:
            ra = self.assess(action, target, agent_confidence=agent_confidence)
            if best_ra is None or ra.risk_score < best_ra.risk_score:
                best_ra  = ra
                best_act = (action, target)
        return best_act, best_ra

    def filter_safe(
        self,
        candidates: List[Tuple[str, str]],
        agent_confidence: float = 0.5,
    ) -> List[Tuple[Tuple[str, str], RiskAssessment]]:
        """Return only candidates with decision=proceed."""
        result = []
        for action, target in candidates:
            ra = self.assess(action, target, agent_confidence=agent_confidence)
            if ra.decision == "proceed":
                result.append(((action, target), ra))
        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def risk_profile(self) -> Dict:
        """Summary statistics of risk assessments."""
        if not self._assessments:
            return {"n_assessments": 0}
        decisions = {}
        for ra in self._assessments:
            decisions[ra.decision] = decisions.get(ra.decision, 0) + 1
        risk_scores = [ra.risk_score for ra in self._assessments]
        return {
            "n_assessments": len(self._assessments),
            "by_decision":   decisions,
            "mean_risk":     round(float(np.mean(risk_scores)), 3),
            "blocked":       self._blocked_count,
            "caution":       self._caution_count,
        }

    def __repr__(self) -> str:
        return (f"RiskEngine(tolerance={self.risk_tolerance:.2f}, "
                f"blocked={self._blocked_count}, caution={self._caution_count})")
