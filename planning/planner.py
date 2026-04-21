"""
planning/planner.py
====================
Multi-Step Forward Planner

Generates action sequences to achieve a goal by simulating future steps
through the world model before acting in the real environment.

Architecture: Depth-Limited Greedy Tree Search
----------------------------------------------

          current_state
               │
    ┌──────────┼──────────┐
  eat(a)    pick(a)    move(x,y)
    │          │
  sim_state  sim_state
    │          │
  score      score     ← from world model + goal proximity
    │
  best → recurse to depth D

At each node the planner asks the world model:
  "If I do action X, what happens?"
  "How close does that bring me to the goal?"

The plan with the best total score is returned and executed step-by-step.

Scoring a plan step
-------------------
  step_score = (
      0.5 * goal_proximity(predicted_state, goal_vec)
    + 0.3 * expected_reward
    + 0.2 * rule_confidence
  )
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..world_model.transition_model import TransitionModel
from ..world_model.causal_graph import CausalGraph
from ..latent_space.frequency_space import FrequencySpace


# ---------------------------------------------------------------------------
# Plan structures
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """One step in a plan."""
    action:         str
    object_name:    str             # target object (may be empty)
    predicted_vec:  np.ndarray      # predicted state after this step
    expected_reward: float
    confidence:     float
    score:          float


@dataclass
class Plan:
    """A complete multi-step plan."""
    goal_label:  str
    steps:       List[PlanStep]
    total_score: float
    depth:       int

    @property
    def action_sequence(self) -> List[Tuple[str, str]]:
        """Return list of (action, object_name) pairs."""
        return [(s.action, s.object_name) for s in self.steps]

    @property
    def is_empty(self) -> bool:
        return len(self.steps) == 0

    def describe(self) -> str:
        if self.is_empty:
            return f"[empty plan for goal '{self.goal_label}']"
        steps_str = " → ".join(
            f"{s.action}({s.object_name})" if s.object_name else s.action
            for s in self.steps
        )
        return (f"Plan(goal='{self.goal_label}', "
                f"steps=[{steps_str}], score={self.total_score:.3f})")


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """
    Depth-limited forward planner using the world model.

    Parameters
    ----------
    transition_model : TransitionModel for state prediction.
    causal_graph     : CausalGraph for symbolic reasoning.
    space            : FrequencySpace for goal proximity.
    max_depth        : Maximum plan depth.
    beam_width       : Number of branches to explore per level.
    """

    # All actions the planner knows about
    _PRIMITIVE_ACTIONS = ["eat", "pick", "move", "drop", "inspect", "combine", "observe"]

    def __init__(
        self,
        transition_model: Optional[TransitionModel] = None,
        causal_graph:     Optional[CausalGraph]     = None,
        space:            Optional[FrequencySpace]   = None,
        max_depth:        int   = 4,
        beam_width:       int   = 3,
        dim:              int   = 64,
    ) -> None:
        self.tm         = transition_model if transition_model is not None else TransitionModel(dim=dim)
        self.cg         = causal_graph     if causal_graph     is not None else CausalGraph()
        self.space      = space            if space            is not None else FrequencySpace(dim=dim)
        self.max_depth  = max_depth
        self.beam_width = beam_width
        self.dim        = dim
        self._plan_count = 0

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def plan(
        self,
        goal_label:       str,
        goal_vec:         np.ndarray,
        current_state_vec: np.ndarray,
        available_objects: List[Tuple[str, str]],   # (name, category)
        inventory:        Optional[List[str]] = None,
        context:          Optional[Dict] = None,
    ) -> Plan:
        """
        Generate a plan to reach the goal from the current state.

        Parameters
        ----------
        goal_label        : Human-readable goal description.
        goal_vec          : Goal state as latent vector.
        current_state_vec : Current world state vector.
        available_objects : (name, category) pairs visible to agent.
        inventory         : What agent is currently holding.
        context           : Additional context dict.

        Returns
        -------
        Plan  — best plan found within max_depth steps.
        """
        self._plan_count += 1
        inventory = inventory or []

        # Fast path: symbolic plan if causal graph has enough knowledge
        symbolic = self._symbolic_plan(
            goal_label, goal_vec, current_state_vec,
            available_objects, inventory
        )
        if symbolic and symbolic.total_score >= 0.7:
            return symbolic

        # Fallback: simulation-based beam search
        return self._beam_search(
            goal_label, goal_vec, current_state_vec,
            available_objects, inventory
        )

    # ------------------------------------------------------------------
    # Symbolic planning (rule-based, fast)
    # ------------------------------------------------------------------

    def _symbolic_plan(
        self,
        goal_label:       str,
        goal_vec:         np.ndarray,
        state_vec:        np.ndarray,
        available:        List[Tuple[str, str]],
        inventory:        List[str],
    ) -> Optional[Plan]:
        """
        Build a plan using causal graph knowledge.
        Works well when enough symbolic rules have been learned.
        """
        goal_lower = goal_label.lower()
        steps = []

        # Goal: eat something
        if any(word in goal_lower for word in ["eat", "food", "hungry", "consume"]):
            plan = self._plan_eat(goal_vec, state_vec, available, inventory)
            if plan:
                return plan

        # Goal: pick up something
        if any(word in goal_lower for word in ["pick", "grab", "hold", "carry"]):
            for obj_name, obj_cat in available:
                if obj_name not in inventory:
                    step = self._make_step("pick", obj_name, state_vec, goal_vec)
                    if step.score > 0.3:
                        return Plan(goal_label, [step], step.score, 1)

        # Goal: explore / investigate
        if any(word in goal_lower for word in ["explore", "unknown", "investigate", "learn"]):
            step = self._make_step("inspect", "", state_vec, goal_vec)
            return Plan(goal_label, [step], 0.5, 1)

        return None

    def _plan_eat(
        self,
        goal_vec:  np.ndarray,
        state_vec: np.ndarray,
        available: List[Tuple[str, str]],
        inventory: List[str],
    ) -> Optional[Plan]:
        """Plan steps to eat something edible."""
        # Find edible candidates
        edible_candidates = []
        for obj_name, obj_cat in available:
            is_edible = self.cg.is_edible(obj_name)
            if is_edible is True:
                edible_candidates.append((obj_name, obj_cat, 1.0))
            elif is_edible is None and obj_cat == "food":
                edible_candidates.append((obj_name, obj_cat, 0.6))

        if not edible_candidates:
            # No known edible objects — explore
            step = self._make_step("observe", "", state_vec, goal_vec)
            return Plan("find_food", [step], 0.3, 1)

        # Sort by confidence
        edible_candidates.sort(key=lambda x: -x[2])
        obj_name, obj_cat, confidence = edible_candidates[0]

        steps = []
        current = state_vec.copy()

        # Step 1: pick if not in inventory
        if obj_name not in inventory:
            pick_step = self._make_step("pick", obj_name, current, goal_vec, obj_cat)
            steps.append(pick_step)
            current, _ = self.tm.predict_next_state(current, "pick")
            inventory = list(inventory) + [obj_name]  # simulate picking up

        # Step 2: eat
        eat_step = self._make_step("eat", obj_name, current, goal_vec, obj_cat)
        steps.append(eat_step)

        total_score = confidence * (sum(s.score for s in steps) / len(steps))
        return Plan("eat_" + obj_name, steps, total_score, len(steps))

    # ------------------------------------------------------------------
    # Simulation-based beam search
    # ------------------------------------------------------------------

    def _beam_search(
        self,
        goal_label:  str,
        goal_vec:    np.ndarray,
        state_vec:   np.ndarray,
        available:   List[Tuple[str, str]],
        inventory:   List[str],
    ) -> Plan:
        """
        Depth-limited beam search over action space.
        Explores the most promising branches first.
        """
        # Beam: list of (partial_plan_steps, current_state_vec, cumulative_score)
        beam: List[Tuple[List[PlanStep], np.ndarray, float]] = [
            ([], state_vec.copy(), 0.0)
        ]

        best_plan: Plan = Plan(goal_label, [], 0.0, 0)

        for depth in range(self.max_depth):
            candidates = []

            for partial_steps, current_vec, cum_score in beam:
                # Generate candidate next steps
                next_steps = self._generate_candidates(
                    current_vec, goal_vec, available, inventory
                )

                for step in next_steps:
                    new_steps = partial_steps + [step]
                    new_score = (cum_score * depth + step.score) / (depth + 1)
                    candidates.append((new_steps, step.predicted_vec, new_score))

                    # Track best plan
                    if new_score > best_plan.total_score:
                        best_plan = Plan(goal_label, new_steps, new_score, depth + 1)

            # Keep top beam_width candidates
            candidates.sort(key=lambda x: -x[2])
            beam = candidates[:self.beam_width]

            if not beam:
                break

        return best_plan

    def _generate_candidates(
        self,
        state_vec:  np.ndarray,
        goal_vec:   np.ndarray,
        available:  List[Tuple[str, str]],
        inventory:  List[str],
    ) -> List[PlanStep]:
        """Generate all candidate next steps and score them."""
        steps = []

        # Object-specific actions
        for obj_name, obj_cat in available[:5]:  # limit for tractability
            for action in ["eat", "pick", "inspect"]:
                # Filter invalid actions
                if action == "eat" and obj_name not in inventory:
                    continue   # can't eat what you don't hold
                if action == "pick" and obj_name in inventory:
                    continue   # already held

                step = self._make_step(action, obj_name, state_vec, goal_vec, obj_cat)
                steps.append(step)

        # Navigation
        step = self._make_step("move", "", state_vec, goal_vec)
        steps.append(step)

        # Observe
        step = self._make_step("observe", "", state_vec, goal_vec)
        steps.append(step)

        return sorted(steps, key=lambda s: -s.score)[:self.beam_width * 2]

    # ------------------------------------------------------------------
    # Step evaluation
    # ------------------------------------------------------------------

    def _make_step(
        self,
        action:    str,
        obj_name:  str,
        state_vec: np.ndarray,
        goal_vec:  np.ndarray,
        obj_cat:   str = "unknown",
    ) -> PlanStep:
        """Create and score a single plan step."""
        # Predict next state from world model
        predicted_vec, exp_reward = self.tm.predict_next_state(state_vec, action)

        # Symbolic confidence from causal graph / rules
        outcome, rule_reward, conf = self.tm.predict_outcome(action, obj_cat, obj_name)

        # Score: how close does predicted state bring us to the goal?
        goal_proximity = (self.space.similarity(predicted_vec, goal_vec) + 1.0) / 2.0

        # Avoid bad outcomes
        if outcome == "failure" and conf >= 0.7:
            goal_proximity *= 0.2   # penalise known-bad actions heavily

        # Combine signals
        effective_reward = (exp_reward + rule_reward) / 2.0
        score = (
            0.50 * goal_proximity
          + 0.30 * min(1.0, max(0.0, effective_reward + 0.5))  # map reward to [0,1]
          + 0.20 * conf
        )

        return PlanStep(
            action=action,
            object_name=obj_name,
            predicted_vec=predicted_vec,
            expected_reward=effective_reward,
            confidence=conf,
            score=float(np.clip(score, 0.0, 1.0)),
        )

    # ------------------------------------------------------------------
    # Plan safety check
    # ------------------------------------------------------------------

    def is_safe(self, plan: Plan, danger_threshold: float = 0.3) -> bool:
        """
        Returns True if the plan contains no steps with high danger.
        A step is dangerous if its expected reward is strongly negative
        with high confidence.
        """
        for step in plan.steps:
            if step.expected_reward < -danger_threshold and step.confidence >= 0.6:
                return False
        return True

    def filter_dangerous_steps(self, steps: List[PlanStep]) -> List[PlanStep]:
        """Remove steps that are known to be harmful."""
        return [s for s in steps if not (
            s.expected_reward < -0.2 and s.confidence >= 0.6
        )]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"Planner(max_depth={self.max_depth}, "
                f"beam_width={self.beam_width}, "
                f"plans_generated={self._plan_count})")
