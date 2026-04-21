"""
world_model/transition_model.py
================================
Learned World Transition Model

Learns cause-effect relationships from environment experience:
    (state_vector, action) → next_state_vector
    (object, action) → outcome_label + reward

Design
------
Two complementary representations:

1. Vector-based transitions:
   Stores observed (state_vec, action_id, next_state_vec) tuples.
   Predicts next state as a weighted mixture of similar past transitions.
   This generalises: similar states + same action → similar next state.

2. Symbolic rules (WorldRule):
   Extracted from repeated patterns.
   e.g. "eat(edible_object) → success + reward=+0.5"
       "eat(inedible_object) → failure + reward=-0.2"
   Rules are stored as (action, object_category, outcome, reward, confidence).

Both are queried together for plan evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..latent_space.frequency_space import FrequencySpace


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """One observed state transition."""
    state_vec:      np.ndarray      # latent vector of state before action
    action:         str             # action name
    next_state_vec: np.ndarray      # latent vector of resulting state
    reward:         float           # observed reward
    success:        bool            # did action succeed?
    object_name:    str = ""        # which object was acted on
    timestamp:      float = field(default_factory=time.time)


@dataclass
class WorldRule:
    """
    An extracted symbolic rule generalising across transitions.
    e.g. action="eat", category="food" → outcome="success", reward=+0.5
    """
    action:       str
    object_category: str          # "food", "tool", "material", "unknown"
    outcome:      str             # "success" | "failure" | "partial"
    reward:       float
    confidence:   float = 0.5     # updated with Bayesian-style counting
    hit_count:    int   = 0
    miss_count:   int   = 0

    def update(self, actual_success: bool, actual_reward: float) -> None:
        """Update rule confidence from a new observation."""
        self.hit_count  += int(actual_success)
        self.miss_count += int(not actual_success)
        total = self.hit_count + self.miss_count + 1e-8
        self.confidence = self.hit_count / total
        # EMA update of expected reward
        self.reward = 0.9 * self.reward + 0.1 * actual_reward

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= 0.6 and (self.hit_count + self.miss_count) >= 2


# ---------------------------------------------------------------------------
# TransitionModel
# ---------------------------------------------------------------------------

class TransitionModel:
    """
    Learns and predicts world transitions from experience.

    Parameters
    ----------
    space   : Shared FrequencySpace for vector operations.
    dim     : Vector dimensionality.
    max_transitions : Max stored raw transitions.
    sim_threshold   : Cosine similarity threshold for "similar state" lookup.
    """

    # Object category mappings (from environment properties)
    _CATEGORY_MAP: Dict[str, str] = {
        "food": "food",
        "material": "material",
        "tool": "tool",
        "combined": "combined",
    }

    def __init__(
        self,
        space: Optional[FrequencySpace] = None,
        dim: int = 64,
        max_transitions: int = 2000,
        sim_threshold: float = 0.75,
    ) -> None:
        self.space          = space if space is not None else FrequencySpace(dim=dim)
        self.dim            = dim
        self.max_transitions = max_transitions
        self.sim_threshold  = sim_threshold

        self._transitions: List[Transition] = []
        self._rules: Dict[Tuple[str, str], WorldRule] = {}  # (action, category) → rule
        self._state_matrix: Optional[np.ndarray] = None    # (N, D) cache
        self._dirty = True

    # ------------------------------------------------------------------
    # Recording transitions
    # ------------------------------------------------------------------

    def record(
        self,
        state_vec:      np.ndarray,
        action:         str,
        next_state_vec: np.ndarray,
        reward:         float,
        success:        bool,
        object_name:    str = "",
        object_category: str = "unknown",
    ) -> None:
        """
        Record one (state, action) → (next_state, reward) transition.
        Also updates symbolic rules.
        """
        t = Transition(
            state_vec=self.space.l2(state_vec.copy()),
            action=action,
            next_state_vec=self.space.l2(next_state_vec.copy()),
            reward=reward,
            success=success,
            object_name=object_name,
        )
        self._transitions.append(t)
        self._dirty = True

        if len(self._transitions) > self.max_transitions:
            self._transitions.pop(0)

        # Update symbolic rule
        self._update_rule(action, object_category, success, reward)

    def _update_rule(
        self,
        action: str,
        category: str,
        success: bool,
        reward: float,
    ) -> None:
        key = (action, category)
        if key not in self._rules:
            outcome = "success" if success else "failure"
            self._rules[key] = WorldRule(
                action=action,
                object_category=category,
                outcome=outcome,
                reward=reward,
                confidence=0.5,
                hit_count=int(success),
                miss_count=int(not success),
            )
        else:
            self._rules[key].update(success, reward)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_next_state(
        self,
        state_vec: np.ndarray,
        action: str,
    ) -> Tuple[np.ndarray, float]:
        """
        Predict the next state vector and expected reward.

        Uses k-nearest-neighbours over stored transitions with the same action.
        Generalises via vector similarity: similar state + same action → similar outcome.

        Returns
        -------
        (predicted_next_state_vec, expected_reward)
        """
        action_transitions = [t for t in self._transitions if t.action == action]
        if not action_transitions:
            # No data: return unchanged state, reward=0
            return state_vec.copy(), 0.0

        # Find most similar stored states for this action
        state_matrix = np.stack([t.state_vec for t in action_transitions])
        sims = self.space.batch_similarity(state_vec, state_matrix)

        top_k = min(5, len(sims))
        top_idx = np.argpartition(sims, -top_k)[-top_k:]
        top_sims = sims[top_idx]

        # Softmax-weighted combination of next states
        exp_sims = np.exp(top_sims * 3.0)   # temperature=3 for sharpness
        weights  = exp_sims / (exp_sims.sum() + 1e-8)

        next_vecs = np.stack([action_transitions[i].next_state_vec for i in top_idx])
        predicted_next = (next_vecs * weights[:, None]).sum(axis=0)
        predicted_next = self.space.l2(predicted_next.astype(np.float32))

        expected_reward = float(
            sum(weights[j] * action_transitions[top_idx[j]].reward
                for j in range(top_k))
        )
        return predicted_next, expected_reward

    def predict_outcome(
        self,
        action: str,
        object_category: str,
        object_name: str = "",
    ) -> Tuple[str, float, float]:
        """
        Predict symbolic outcome using rules.

        Returns
        -------
        (outcome_label, expected_reward, confidence)
        """
        key = (action, object_category)
        if key in self._rules:
            rule = self._rules[key]
            return rule.outcome, rule.reward, rule.confidence

        # Fallback: check if any rule for this action exists
        action_rules = [r for (a, _), r in self._rules.items() if a == action]
        if action_rules:
            # Average outcome of all rules for this action
            avg_reward = float(np.mean([r.reward for r in action_rules]))
            avg_conf   = float(np.mean([r.confidence for r in action_rules]))
            outcome = "success" if avg_reward > 0 else "failure"
            return outcome, avg_reward, avg_conf * 0.5  # half-confidence for generalisation

        return "unknown", 0.0, 0.0

    # ------------------------------------------------------------------
    # Rule queries
    # ------------------------------------------------------------------

    def get_rules(self, action: Optional[str] = None) -> List[WorldRule]:
        """Return all known rules, optionally filtered by action."""
        rules = list(self._rules.values())
        if action:
            rules = [r for r in rules if r.action == action]
        return sorted(rules, key=lambda r: -r.confidence)

    def can_do(self, action: str, object_category: str) -> bool:
        """Returns True if this action on this category is expected to succeed."""
        outcome, reward, conf = self.predict_outcome(action, object_category)
        return outcome == "success" and conf >= 0.5

    def best_action_for_goal(
        self,
        goal_outcome: str,
        available_objects: List[Tuple[str, str]],
    ) -> Optional[Tuple[str, str]]:
        """
        Find the (action, object) pair most likely to produce goal_outcome.
        available_objects: list of (object_name, category) tuples.
        """
        best_conf = -1.0
        best_pair = None

        for obj_name, obj_cat in available_objects:
            for action in ["eat", "pick", "combine", "inspect", "use"]:
                outcome, reward, conf = self.predict_outcome(action, obj_cat, obj_name)
                if outcome == goal_outcome and conf > best_conf:
                    best_conf = conf
                    best_pair = (action, obj_name)

        return best_pair

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def n_transitions(self) -> int:
        return len(self._transitions)

    @property
    def n_rules(self) -> int:
        return len(self._rules)

    def rule_summary(self) -> List[Dict]:
        return [
            {
                "action":    r.action,
                "category":  r.object_category,
                "outcome":   r.outcome,
                "reward":    round(r.reward, 3),
                "confidence": round(r.confidence, 3),
                "count":     r.hit_count + r.miss_count,
                "reliable":  r.is_reliable,
            }
            for r in sorted(self._rules.values(), key=lambda r: -r.confidence)
        ]

    def __repr__(self) -> str:
        return (f"TransitionModel(transitions={self.n_transitions}, "
                f"rules={self.n_rules})")
