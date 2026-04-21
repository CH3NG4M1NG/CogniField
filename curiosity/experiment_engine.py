"""
curiosity/experiment_engine.py
================================
Structured Experimentation Engine

Instead of acting randomly on unknowns, the agent designs safe,
structured experiments to systematically gather evidence.

Experiment Protocol
-------------------
1. IDENTIFY: What property do we want to learn?
   e.g. "Is purple_berry edible?"

2. HYPOTHESIZE: What do we expect?
   e.g. "Hypothesis: purple_berry.edible = True (conf=0.35)"

3. DESIGN: What is the safest test?
   Inspect → Observe → Smell/Examine → Taste-tiny → Eat-full
   (Safety ladder: cheapest, safest first)

4. SIMULATE: What does the world model predict?
   e.g. "Simulation: eat(purple_berry) → 60% success"

5. EXECUTE: Carry out the safest informative test.
   e.g. "inspect(purple_berry)"

6. COLLECT: Record the outcome.
   e.g. "Revealed: edible=True, fragile=False"

7. UPDATE: Update belief system with new evidence.
   e.g. "purple_berry.edible → conf 0.35 → 0.78"

Safety Ladder
-------------
For unknown edibility:
  Level 0: observe (free, safe)
  Level 1: inspect (safe, reveals properties)
  Level 2: combine with known-safe object (cheap test)
  Level 3: eat tiny portion (risky if wrong — only if conf > 0.6)
  Level 4: eat full  (only if conf > 0.8)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..world_model.belief_system import Belief, BeliefSystem
from ..world_model.simulator import WorldSimulator
from ..curiosity.advanced_curiosity import AdvancedCuriosityEngine, Hypothesis


class SafetyLevel(int, Enum):
    """How risky is this experimental action?"""
    SAFE        = 0   # no risk
    LOW_RISK    = 1   # minimal risk
    MODERATE    = 2   # some risk
    HIGH_RISK   = 3   # dangerous if wrong


SAFETY_LADDER: Dict[str, SafetyLevel] = {
    "observe":  SafetyLevel.SAFE,
    "inspect":  SafetyLevel.SAFE,
    "move":     SafetyLevel.SAFE,
    "pick":     SafetyLevel.LOW_RISK,
    "drop":     SafetyLevel.LOW_RISK,
    "combine":  SafetyLevel.MODERATE,
    "eat":      SafetyLevel.HIGH_RISK,
    "use":      SafetyLevel.MODERATE,
}

# Minimum belief confidence required to attempt action at each risk level
MIN_CONF_FOR_RISK: Dict[SafetyLevel, float] = {
    SafetyLevel.SAFE:       0.00,
    SafetyLevel.LOW_RISK:   0.30,
    SafetyLevel.MODERATE:   0.55,
    SafetyLevel.HIGH_RISK:  0.75,
}


@dataclass
class Experiment:
    """A structured experiment design."""
    target:          str           # the object being tested
    property:        str           # the property to learn (e.g. "edible")
    hypothesis_value: Any          # what we expect to find
    prior_confidence: float        # confidence before the experiment
    action:          str           # action to execute
    safety_level:    SafetyLevel
    sim_prediction:  Optional[str] = None    # world model prediction
    sim_confidence:  float         = 0.0
    status:          str           = "designed"  # designed/executed/completed
    result:          Optional[Any] = None
    post_confidence: float         = 0.0
    executed_at:     Optional[float] = None
    notes:           str           = ""


@dataclass
class ExperimentResult:
    """Outcome of one executed experiment."""
    experiment:     Experiment
    observed_value: Any
    success:        bool
    reward:         float
    belief_update:  Dict[str, float]   # key → new confidence
    insight:        str


class ExperimentEngine:
    """
    Designs and tracks structured experiments to test hypotheses safely.

    Parameters
    ----------
    belief_system  : BeliefSystem to update with results.
    simulator      : WorldSimulator for pre-experiment prediction.
    curiosity      : AdvancedCuriosityEngine for hypothesis management.
    min_conf_to_act: Minimum confidence required before risky actions.
    """

    def __init__(
        self,
        belief_system:   BeliefSystem,
        simulator:       WorldSimulator,
        curiosity:       AdvancedCuriosityEngine,
        min_conf_to_act: float = 0.70,
        dim:             int   = 64,
    ) -> None:
        self.bs              = belief_system
        self.sim             = simulator
        self.curiosity       = curiosity
        self.min_conf_to_act = min_conf_to_act
        self.dim             = dim

        self._experiments: List[Experiment] = []
        self._results:     List[ExperimentResult] = []
        self._rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Design an experiment
    # ------------------------------------------------------------------

    def design(
        self,
        target:   str,
        property: str   = "edible",
        state_vec: Optional[np.ndarray] = None,
    ) -> Experiment:
        """
        Design the safest informative experiment for a target property.

        Parameters
        ----------
        target    : Object name (e.g. "purple_berry").
        property  : Property to learn (e.g. "edible", "fragile").
        state_vec : Current world state vector for simulation.

        Returns
        -------
        Experiment  — ready to execute.
        """
        # Get current belief confidence
        key = f"{target}.{property}"
        prior_conf = self.bs.get_confidence(key, default=0.5)

        # Get hypothesis from curiosity engine
        hyps = [h for h in self.curiosity._hypotheses
                if h.subject == target and h.predicate == property and h.status == "open"]
        hyp_value = hyps[0].predicted if hyps else True

        # Choose action from safety ladder
        action = self._choose_safe_action(target, property, prior_conf)
        safety = SAFETY_LADDER.get(action, SafetyLevel.MODERATE)

        # Pre-simulate
        sim_pred = "unknown"
        sim_conf = 0.0
        if state_vec is not None:
            try:
                result = self.sim.test_hypothesis(action, target, state_vec)
                sim_pred = result.get("predicted_outcome", "unknown")
                sim_conf = result.get("success_rate", 0.5)
            except Exception:
                pass

        exp = Experiment(
            target=target,
            property=property,
            hypothesis_value=hyp_value,
            prior_confidence=prior_conf,
            action=action,
            safety_level=safety,
            sim_prediction=sim_pred,
            sim_confidence=sim_conf,
            notes=f"Testing {target}.{property}; prior_conf={prior_conf:.3f}",
        )
        self._experiments.append(exp)
        return exp

    def _choose_safe_action(
        self,
        target:   str,
        property: str,
        conf:     float,
    ) -> str:
        """Select the safest action that will give information about this property."""
        # Specific mappings
        if property == "edible":
            if conf < 0.30:
                return "inspect"           # cheapest: just look
            elif conf < 0.60:
                return "pick"              # handle it, check properties
            elif conf < 0.75:
                return "combine"           # mix with water — safe test
            else:
                return "eat"               # confident enough to eat
        elif property == "fragile":
            if conf < 0.50:
                return "inspect"
            else:
                return "drop"              # will break if fragile
        elif property == "heavy":
            return "pick"                  # directly tests weight
        else:
            return "inspect"              # safe default

    # ------------------------------------------------------------------
    # Evaluate safety before execution
    # ------------------------------------------------------------------

    def is_safe_to_execute(self, exp: Experiment) -> Tuple[bool, str]:
        """
        Check whether it is safe to execute the experiment right now.

        Returns
        -------
        (safe, reason)
        """
        min_conf = MIN_CONF_FOR_RISK[exp.safety_level]

        if exp.prior_confidence < min_conf:
            return False, (f"Action '{exp.action}' requires conf>={min_conf:.2f}, "
                           f"but prior is {exp.prior_confidence:.3f}")

        # Check simulation prediction
        if (exp.safety_level >= SafetyLevel.HIGH_RISK
                and exp.sim_prediction == "failure"
                and exp.sim_confidence >= 0.70):
            return False, (f"Simulation predicts failure with confidence "
                           f"{exp.sim_confidence:.2f} — too risky")

        # Check if target is known dangerous
        edible_belief = self.bs.get(f"{exp.target}.edible")
        if (edible_belief and edible_belief.confidence >= 0.80
                and Belief._values_agree(edible_belief.value, False)
                and exp.action == "eat"):
            return False, (f"{exp.target} is known not edible "
                           f"(conf={edible_belief.confidence:.2f})")

        return True, "safe to proceed"

    # ------------------------------------------------------------------
    # Process results
    # ------------------------------------------------------------------

    def process_result(
        self,
        exp:            Experiment,
        env_feedback:   Dict,
    ) -> ExperimentResult:
        """
        Process environment feedback from an experiment and update beliefs.

        Parameters
        ----------
        exp          : The experiment that was executed.
        env_feedback : Environment feedback dict from RichEnv.step().

        Returns
        -------
        ExperimentResult
        """
        exp.status      = "completed"
        exp.executed_at = time.time()

        success      = env_feedback.get("success", False)
        reward       = env_feedback.get("reward", 0.0)
        obj_props    = env_feedback.get("object_props", {})
        learned      = env_feedback.get("learned", "")

        # Determine what we observed
        observed = None
        if exp.property in obj_props:
            observed = obj_props[exp.property]
        elif "edible" in exp.property:
            if "not edible" in learned.lower():
                observed = False
            elif "edible" in learned.lower():
                observed = True
            elif success and reward >= 0.3:
                observed = True
            elif not success and reward <= -0.15:
                observed = False

        if observed is None:
            observed = success   # fallback

        exp.result = observed

        # Update belief system
        key    = f"{exp.target}.{exp.property}"
        source = "direct_observation" if exp.action in ("eat","inspect") else "inference"
        weight = 0.9 if success else 0.7   # failures are slightly less informative

        self.bs.update(key, observed, source=source, weight=weight,
                       notes=f"experiment: {exp.action}({exp.target})")
        exp.post_confidence = self.bs.get_confidence(key)

        # Update hypothesis tracking
        self.curiosity.update_hypotheses(exp.target, exp.property, observed)

        # Also update object properties in belief system
        for prop, val in obj_props.items():
            if prop != "name":
                self.bs.observe(exp.target, prop, val, source="direct_observation")

        belief_updates = {key: exp.post_confidence}

        # Generate insight string
        insight = self._generate_insight(exp, observed)

        result = ExperimentResult(
            experiment=exp,
            observed_value=observed,
            success=success,
            reward=reward,
            belief_update=belief_updates,
            insight=insight,
        )
        self._results.append(result)
        return result

    def _generate_insight(self, exp: Experiment, observed: Any) -> str:
        agreed = Belief._values_agree(observed, exp.hypothesis_value)
        conf_before = exp.prior_confidence
        conf_after  = exp.post_confidence
        direction   = "↑" if conf_after > conf_before else "↓"
        return (
            f"{'✓ Confirmed' if agreed else '✗ Refuted'}: "
            f"{exp.target}.{exp.property}={observed} "
            f"(conf: {conf_before:.3f} → {conf_after:.3f} {direction})"
        )

    # ------------------------------------------------------------------
    # Pending experiments queue
    # ------------------------------------------------------------------

    def next_experiment_target(self) -> Optional[Tuple[str, str]]:
        """
        Return the (target, property) most in need of experimentation.
        Priority: uncertain edibility of visible objects.
        """
        uncertain = self.bs.uncertain_beliefs()
        for b in sorted(uncertain, key=lambda b: abs(b.confidence - 0.5)):
            parts = b.key.split(".")
            if len(parts) == 2:
                return parts[0], parts[1]
        return None

    def get_pending_experiments(self) -> List[Experiment]:
        return [e for e in self._experiments if e.status == "designed"]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        completed = [e for e in self._experiments if e.status == "completed"]
        conf_gains = [e.post_confidence - e.prior_confidence
                      for e in completed if e.post_confidence > 0]
        return {
            "total_designed":  len(self._experiments),
            "total_completed": len(completed),
            "pending":         len(self.get_pending_experiments()),
            "mean_conf_gain":  round(float(np.mean(conf_gains)), 3) if conf_gains else 0.0,
            "recent_insights": [r.insight for r in self._results[-3:]],
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"ExperimentEngine(designed={s['total_designed']}, "
                f"completed={s['total_completed']}, "
                f"mean_gain={s['mean_conf_gain']:.3f})")
