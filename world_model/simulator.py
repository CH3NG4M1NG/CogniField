"""
world_model/simulator.py
=========================
World Model Simulator — "Imagination Engine"

Enables the agent to simulate future trajectories before committing
to actions in the real environment. This is the core of model-based
reasoning: "what would happen if I did X, then Y, then Z?"

Architecture
------------
The simulator runs entirely in latent space using the TransitionModel
to roll out action sequences:

  current_state
       │
  simulate(action_1) → predicted_state_1 (+ reward_1)
       │
  simulate(action_2) → predicted_state_2 (+ reward_2)
       │
  ...
       │
  simulate(action_N) → predicted_state_N (+ total_reward)

Multiple rollout strategies:
  GREEDY  – always take the locally best action
  RANDOM  – sample actions randomly (for diversity)
  GUIDED  – guided by goal proximity + world model rules

Uses
----
  - Plan evaluation before execution
  - Hypothesis testing without real-world risk
  - "What if" reasoning about counterfactuals
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .transition_model import TransitionModel
from .causal_graph import CausalGraph
from ..latent_space.frequency_space import FrequencySpace


# ---------------------------------------------------------------------------
# Simulation record
# ---------------------------------------------------------------------------

@dataclass
class SimStep:
    """One step of a simulated rollout."""
    action:       str
    object_name:  str
    state_before: np.ndarray
    state_after:  np.ndarray
    reward:       float
    confidence:   float
    outcome:      str           # "success" | "failure" | "unknown"


@dataclass
class SimulationResult:
    """Result of a complete simulated trajectory."""
    steps:         List[SimStep]
    total_reward:  float
    final_state:   np.ndarray
    goal_reached:  bool
    confidence:    float        # mean confidence across steps

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def action_sequence(self) -> List[Tuple[str, str]]:
        return [(s.action, s.object_name) for s in self.steps]

    def describe(self) -> str:
        seq = " → ".join(
            f"{s.action}({s.object_name})" if s.object_name else s.action
            for s in self.steps
        )
        return (f"Simulation[{seq}] "
                f"reward={self.total_reward:+.3f} "
                f"reached={self.goal_reached}")


class WorldSimulator:
    """
    Simulates sequences of actions in latent space.

    Parameters
    ----------
    transition_model : Trained TransitionModel.
    causal_graph     : CausalGraph for symbolic outcome lookup.
    space            : FrequencySpace.
    """

    def __init__(
        self,
        transition_model: TransitionModel,
        causal_graph:     CausalGraph,
        space:            FrequencySpace,
        dim:              int = 64,
    ) -> None:
        self.tm    = transition_model
        self.cg    = causal_graph
        self.space = space
        self.dim   = dim
        self._sim_count = 0

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        state:           np.ndarray,
        action_sequence: List[Tuple[str, str]],   # [(action, object_name), ...]
        goal_vec:        Optional[np.ndarray] = None,
        goal_threshold:  float = 0.80,
    ) -> SimulationResult:
        """
        Simulate a fixed action sequence from the given state.

        Parameters
        ----------
        state            : Starting latent state vector.
        action_sequence  : List of (action, object_name) to execute.
        goal_vec         : Goal state vector (for goal-reached detection).
        goal_threshold   : Cosine similarity threshold for "goal reached".

        Returns
        -------
        SimulationResult
        """
        self._sim_count += 1
        current  = state.copy()
        steps    = []
        total_r  = 0.0
        goal_reached = False

        for action, obj_name in action_sequence:
            obj_category = self.cg.get_category(obj_name) or "unknown"

            # Predict next state from vector model
            next_state, vec_reward = self.tm.predict_next_state(current, action)

            # Predict symbolic outcome
            outcome, sym_reward, conf = self.tm.predict_outcome(
                action, obj_category, obj_name
            )

            # Blend vector and symbolic rewards
            reward = 0.6 * vec_reward + 0.4 * sym_reward

            step = SimStep(
                action=action,
                object_name=obj_name,
                state_before=current.copy(),
                state_after=next_state.copy(),
                reward=reward,
                confidence=conf,
                outcome=outcome,
            )
            steps.append(step)
            total_r += reward
            current  = next_state

            # Check if goal reached
            if goal_vec is not None:
                goal_sim = self.space.similarity(current, goal_vec)
                if (goal_sim + 1) / 2 >= goal_threshold:
                    goal_reached = True
                    break

        mean_conf = float(np.mean([s.confidence for s in steps])) if steps else 0.0

        return SimulationResult(
            steps=steps,
            total_reward=total_r,
            final_state=current,
            goal_reached=goal_reached,
            confidence=mean_conf,
        )

    # ------------------------------------------------------------------
    # Multi-sequence evaluation (compare alternatives)
    # ------------------------------------------------------------------

    def evaluate_plans(
        self,
        state:    np.ndarray,
        plans:    List[List[Tuple[str, str]]],
        goal_vec: Optional[np.ndarray] = None,
    ) -> List[Tuple[float, SimulationResult]]:
        """
        Simulate multiple alternative action sequences and rank them.

        Returns
        -------
        List of (score, SimulationResult) sorted by score descending.
        """
        results = []
        for plan in plans:
            sim = self.simulate(state, plan, goal_vec)
            score = self._score_simulation(sim, goal_vec)
            results.append((score, sim))
        return sorted(results, key=lambda x: -x[0])

    def _score_simulation(
        self,
        sim:      SimulationResult,
        goal_vec: Optional[np.ndarray],
    ) -> float:
        """Score a simulation result for plan selection."""
        # Base: total reward
        reward_score = float(np.tanh(sim.total_reward))  # bound to (-1, 1)

        # Goal proximity of final state
        goal_score = 0.0
        if goal_vec is not None:
            goal_score = (self.space.similarity(sim.final_state, goal_vec) + 1) / 2

        # Confidence (prefer plans the world model knows about)
        conf_score = sim.confidence

        # Goal reached bonus
        reached_bonus = 0.2 if sim.goal_reached else 0.0

        # Penalty for any step with known-bad outcome
        failure_penalty = sum(
            0.15 for s in sim.steps
            if s.outcome == "failure" and s.confidence >= 0.7
        )

        score = (
            0.35 * (reward_score + 1) / 2  # map to [0,1]
          + 0.35 * goal_score
          + 0.15 * conf_score
          + reached_bonus
          - failure_penalty
        )
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Hypothesis testing via simulation
    # ------------------------------------------------------------------

    def test_hypothesis(
        self,
        hypothesis_action: str,
        hypothesis_object: str,
        current_state:     np.ndarray,
        n_rollouts:        int = 3,
    ) -> Dict:
        """
        Test a hypothesis by simulating the proposed action multiple times.
        Returns a dict with predicted outcome and confidence.
        """
        obj_cat = self.cg.get_category(hypothesis_object) or "unknown"
        outcomes = []
        rewards  = []

        for _ in range(n_rollouts):
            sim = self.simulate(
                current_state,
                [(hypothesis_action, hypothesis_object)]
            )
            if sim.steps:
                outcomes.append(sim.steps[0].outcome)
                rewards.append(sim.steps[0].reward)

        # Aggregate
        success_count = outcomes.count("success")
        success_rate  = success_count / max(len(outcomes), 1)
        mean_reward   = float(np.mean(rewards)) if rewards else 0.0
        predicted_outcome = "success" if success_rate > 0.5 else "failure"

        return {
            "action":            hypothesis_action,
            "object":            hypothesis_object,
            "n_rollouts":        n_rollouts,
            "predicted_outcome": predicted_outcome,
            "success_rate":      round(success_rate, 3),
            "mean_reward":       round(mean_reward, 3),
            "recommendation":    "proceed" if mean_reward > 0 else "avoid",
        }

    # ------------------------------------------------------------------
    # Counterfactual reasoning
    # ------------------------------------------------------------------

    def counterfactual(
        self,
        state:           np.ndarray,
        taken_action:    Tuple[str, str],
        alternative:     Tuple[str, str],
        goal_vec:        Optional[np.ndarray] = None,
    ) -> Dict:
        """
        "What would have happened if I had done X instead of Y?"
        Returns comparison dict.
        """
        sim_taken = self.simulate(state, [taken_action], goal_vec)
        sim_alt   = self.simulate(state, [alternative],  goal_vec)

        score_taken = self._score_simulation(sim_taken, goal_vec)
        score_alt   = self._score_simulation(sim_alt,   goal_vec)

        return {
            "taken_action":      taken_action,
            "alternative":       alternative,
            "taken_reward":      round(sim_taken.total_reward, 3),
            "alt_reward":        round(sim_alt.total_reward,   3),
            "taken_score":       round(score_taken, 3),
            "alt_score":         round(score_alt,   3),
            "better_choice":     alternative if score_alt > score_taken else taken_action,
            "regret":            round(max(0, score_alt - score_taken), 3),
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def sim_count(self) -> int:
        return self._sim_count

    def __repr__(self) -> str:
        return (f"WorldSimulator(simulations={self._sim_count}, "
                f"transitions={self.tm.n_transitions})")
