"""
agent/goals.py
==============
Goal System

Allows the agent to maintain persistent objectives beyond immediate input.
Goals drive the planning and action-selection process.

Goal Lifecycle
--------------
    PENDING  → ACTIVE → COMPLETED / FAILED / ABANDONED

Goal Priority
-------------
Goals are scored by urgency × relevance × feasibility.
The highest-scoring active goal drives the current planning cycle.

Goal Types
----------
- EAT_OBJECT:   "eat the apple"
- EXPLORE:      "investigate unknown object"
- LEARN:        "learn the property of X"
- NAVIGATE:     "move to location"
- ACQUIRE:      "pick up X"
- AVOID:        "don't eat stone"
- CUSTOM:       arbitrary text goal
"""

from __future__ import annotations

import time
import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class GoalStatus(str, enum.Enum):
    PENDING   = "pending"
    ACTIVE    = "active"
    COMPLETED = "completed"
    FAILED    = "failed"
    ABANDONED = "abandoned"


class GoalType(str, enum.Enum):
    EAT_OBJECT  = "eat_object"
    EXPLORE     = "explore"
    LEARN       = "learn"
    NAVIGATE    = "navigate"
    ACQUIRE     = "acquire"
    AVOID       = "avoid"
    CUSTOM      = "custom"


@dataclass
class Goal:
    """A single agent goal."""
    label:        str
    goal_type:    GoalType
    target:       str                        # target object/location/concept
    goal_vec:     Optional[np.ndarray]       # latent encoding of goal
    priority:     float = 0.5               # 0-1, higher = more urgent
    status:       GoalStatus = GoalStatus.PENDING
    created_at:   float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    attempts:     int = 0
    max_attempts: int = 10
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def activate(self) -> None:
        self.status = GoalStatus.ACTIVE
        self.attempts += 1

    def complete(self) -> None:
        self.status    = GoalStatus.COMPLETED
        self.completed_at = time.time()

    def fail(self) -> None:
        self.attempts += 1
        if self.attempts >= self.max_attempts:
            self.status = GoalStatus.FAILED
        # else stays ACTIVE for retry

    def abandon(self) -> None:
        self.status = GoalStatus.ABANDONED

    @property
    def is_active(self) -> bool:
        return self.status in (GoalStatus.PENDING, GoalStatus.ACTIVE)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


class GoalSystem:
    """
    Manages the agent's goal stack.

    Parameters
    ----------
    max_active : Maximum number of simultaneously active goals.
    """

    def __init__(self, max_active: int = 5) -> None:
        self.max_active = max_active
        self._goals: List[Goal] = []
        self._completed: List[Goal] = []

    # ------------------------------------------------------------------
    # Adding goals
    # ------------------------------------------------------------------

    def add_goal(
        self,
        label:     str,
        goal_type: GoalType,
        target:    str = "",
        priority:  float = 0.5,
        goal_vec:  Optional[np.ndarray] = None,
        metadata:  Optional[Dict] = None,
    ) -> Goal:
        """Add a new goal to the system."""
        goal = Goal(
            label=label,
            goal_type=goal_type,
            target=target,
            goal_vec=goal_vec,
            priority=priority,
            metadata=metadata or {},
        )
        self._goals.append(goal)
        return goal

    def add_eat_goal(self, object_name: str, priority: float = 0.8,
                     goal_vec: Optional[np.ndarray] = None) -> Goal:
        return self.add_goal(
            f"eat {object_name}", GoalType.EAT_OBJECT,
            target=object_name, priority=priority, goal_vec=goal_vec
        )

    def add_explore_goal(self, label: str = "explore unknown",
                         priority: float = 0.5,
                         goal_vec: Optional[np.ndarray] = None) -> Goal:
        return self.add_goal(
            label, GoalType.EXPLORE,
            target="unknown", priority=priority, goal_vec=goal_vec
        )

    def add_acquire_goal(self, object_name: str, priority: float = 0.6,
                         goal_vec: Optional[np.ndarray] = None) -> Goal:
        return self.add_goal(
            f"pick up {object_name}", GoalType.ACQUIRE,
            target=object_name, priority=priority, goal_vec=goal_vec
        )

    def add_avoid_goal(self, object_name: str, priority: float = 0.9) -> Goal:
        return self.add_goal(
            f"avoid eating {object_name}", GoalType.AVOID,
            target=object_name, priority=priority
        )

    # ------------------------------------------------------------------
    # Selecting the current goal
    # ------------------------------------------------------------------

    def select_active_goal(self) -> Optional[Goal]:
        """
        Return the highest-priority active goal.
        Applies urgency scaling: older unresolved goals become more urgent.
        """
        active = [g for g in self._goals if g.is_active]
        if not active:
            return None

        # Score = priority + age_bonus (caps at 0.2) - attempt_penalty
        def score(g: Goal) -> float:
            age_bonus    = min(0.2, g.age_seconds / 300.0)
            attempt_pen  = g.attempts * 0.05
            return g.priority + age_bonus - attempt_pen

        active.sort(key=score, reverse=True)
        goal = active[0]
        goal.activate()
        return goal

    def get_avoidance_goals(self) -> List[Goal]:
        """Return all active AVOID goals."""
        return [g for g in self._goals
                if g.goal_type == GoalType.AVOID and g.is_active]

    # ------------------------------------------------------------------
    # Updating goals
    # ------------------------------------------------------------------

    def mark_completed(self, goal: Goal) -> None:
        goal.complete()
        self._completed.append(goal)
        if goal in self._goals:
            self._goals.remove(goal)

    def mark_failed(self, goal: Goal) -> None:
        goal.fail()
        if goal.status == GoalStatus.FAILED:
            self._completed.append(goal)
            if goal in self._goals:
                self._goals.remove(goal)

    def check_goal_satisfied(
        self,
        goal: Goal,
        env_feedback: Dict[str, Any],
        current_state_vec: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Determine if a goal has been satisfied from environment feedback.
        """
        fb = env_feedback
        success = fb.get("success", False)
        action  = fb.get("action", "")
        obj     = fb.get("object_name", "")

        if goal.goal_type == GoalType.EAT_OBJECT:
            return success and action == "eat" and goal.target in obj

        if goal.goal_type == GoalType.ACQUIRE:
            return success and action == "pick" and goal.target in obj

        if goal.goal_type == GoalType.NAVIGATE:
            return success and action == "move"

        if goal.goal_type == GoalType.EXPLORE:
            return success  # any successful action satisfies explore

        if goal.goal_type == GoalType.LEARN:
            return success and action == "inspect"

        if goal.goal_type == GoalType.AVOID:
            # AVOID is satisfied by NOT doing the action — it's persistent
            if not success and goal.target in obj:
                return True   # we tried and failed → avoided!
            return False

        return success

    # ------------------------------------------------------------------
    # Auto-generate goals from context
    # ------------------------------------------------------------------

    def infer_goals_from_context(
        self,
        known_edible: List[str],
        unknown_objects: List[str],
        inventory: List[str],
    ) -> List[Goal]:
        """
        Automatically suggest goals based on world state.
        """
        new_goals = []

        # If we know edible objects and don't have them → acquire goal
        for obj in known_edible:
            if obj not in inventory:
                existing = [g for g in self._goals
                            if g.target == obj and g.is_active]
                if not existing:
                    g = self.add_acquire_goal(obj, priority=0.6)
                    new_goals.append(g)

        # If there are unknowns → explore
        for obj in unknown_objects[:2]:  # limit
            existing = [g for g in self._goals
                        if obj in g.label and g.is_active]
            if not existing:
                g = self.add_explore_goal(
                    f"investigate {obj}", priority=0.4
                )
                new_goals.append(g)

        return new_goals

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        return sum(1 for g in self._goals if g.is_active)

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    def summary(self) -> Dict:
        active = [g for g in self._goals if g.is_active]
        return {
            "active_goals":    [g.label for g in active],
            "completed":       len(self._completed),
            "pending":         len([g for g in self._goals if g.status == GoalStatus.PENDING]),
        }

    def __repr__(self) -> str:
        return (f"GoalSystem(active={self.active_count}, "
                f"completed={self.completed_count})")
