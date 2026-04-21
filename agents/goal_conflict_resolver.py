"""
agents/goal_conflict_resolver.py
==================================
Goal Conflict Resolver

When an agent has multiple active goals that are mutually incompatible
or resource-constrained, this module arbitrates between them.

Conflict Types
--------------
1. RESOURCE CONFLICT
   "eat apple" AND "eat bread" — both require the agent to eat,
   but eating takes one action slot.

2. VALUE CONFLICT
   "explore purple_berry" AND "avoid unknown objects"
   → directly contradictory.

3. TEMPORAL CONFLICT
   "eat apple NOW" AND "wait and verify first"
   → depend on time horizon.

4. PRIORITY OVERRIDE
   "survive" always beats "explore" — permanent dominance.

Resolution Strategies
---------------------
PRIORITY_ORDER
  Simply pick the highest-priority non-conflicting set.
  Simple, predictable, but may starve lower-priority goals.

UTILITY_MAXIMISATION
  Score each goal by: priority × time_pressure × feasibility × expected_reward
  Pick the goal with the highest combined utility.

SATISFICING
  Find the minimum set of goals that satisfies a threshold utility.
  Trade-off: good-enough over greedy.

PARETO_OPTIMAL
  Build a Pareto frontier of goal combinations.
  No combo where you can improve one goal without hurting another.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .goals import Goal, GoalSystem, GoalType


class ConflictType(str, Enum):
    RESOURCE   = "resource"
    VALUE      = "value"
    TEMPORAL   = "temporal"
    PRIORITY   = "priority"


class ResolutionStrategy(str, Enum):
    PRIORITY_ORDER        = "priority_order"
    UTILITY_MAXIMISATION  = "utility_maximisation"
    SATISFICING           = "satisficing"


@dataclass
class GoalConflict:
    """A detected conflict between two or more goals."""
    goal_a:      Goal
    goal_b:      Goal
    conflict_type: ConflictType
    description: str
    severity:    float     # 0 = minor, 1 = irreconcilable
    resolved_to: Optional[Goal] = None
    strategy:    Optional[ResolutionStrategy] = None
    timestamp:   float = field(default_factory=time.time)


@dataclass
class ResolutionDecision:
    """The output of one conflict resolution round."""
    chosen_goals:   List[Goal]
    dropped_goals:  List[Goal]
    conflicts:      List[GoalConflict]
    strategy_used:  ResolutionStrategy
    utility_score:  float
    notes:          str


class GoalConflictResolver:
    """
    Detects and resolves conflicts between active agent goals.

    Parameters
    ----------
    strategy  : Default resolution strategy.
    sat_threshold : Utility threshold for SATISFICING mode.
    """

    def __init__(
        self,
        strategy:       ResolutionStrategy = ResolutionStrategy.UTILITY_MAXIMISATION,
        sat_threshold:  float = 0.60,
    ) -> None:
        self.strategy      = strategy
        self.sat_threshold = sat_threshold
        self._history:     List[ResolutionDecision] = []
        self._conflict_log: List[GoalConflict] = []

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------

    def detect_conflicts(self, goals: List[Goal]) -> List[GoalConflict]:
        """Scan a list of goals for conflicts."""
        conflicts = []
        for i, ga in enumerate(goals):
            for j, gb in enumerate(goals):
                if j <= i:
                    continue
                conflict = self._check_pair(ga, gb)
                if conflict:
                    conflicts.append(conflict)
                    self._conflict_log.append(conflict)
        return conflicts

    def _check_pair(self, ga: Goal, gb: Goal) -> Optional[GoalConflict]:
        """Check if two goals conflict."""
        la = ga.label.lower()
        lb = gb.label.lower()

        # Value conflict: avoid X + eat X
        if "avoid" in la or "avoid" in lb:
            target_a = ga.target.lower() if ga.target else ""
            target_b = gb.target.lower() if gb.target else ""
            if target_a == target_b and target_a:
                return GoalConflict(
                    goal_a=ga, goal_b=gb,
                    conflict_type=ConflictType.VALUE,
                    description=f"'{ga.label}' contradicts '{gb.label}' for target '{target_a}'",
                    severity=0.9,
                )

        # Resource conflict: two EAT goals
        if (ga.goal_type == GoalType.EAT_OBJECT
                and gb.goal_type == GoalType.EAT_OBJECT):
            return GoalConflict(
                goal_a=ga, goal_b=gb,
                conflict_type=ConflictType.RESOURCE,
                description=f"Can only eat one thing at a time: '{ga.label}' vs '{gb.label}'",
                severity=0.4,
            )

        # Temporal conflict: EXPLORE vs AVOID_UNKNOWN
        if "explore" in la and ("avoid" in lb or "cautious" in lb):
            return GoalConflict(
                goal_a=ga, goal_b=gb,
                conflict_type=ConflictType.TEMPORAL,
                description=f"'{ga.label}' conflicts with cautious '{gb.label}'",
                severity=0.5,
            )

        # Priority dominance: survival beats exploration
        DOMINANT = {"survive", "safety", "avoid", "escape"}
        if any(d in la for d in DOMINANT) and "explore" in lb:
            return GoalConflict(
                goal_a=ga, goal_b=gb,
                conflict_type=ConflictType.PRIORITY,
                description=f"Safety goal '{ga.label}' dominates '{gb.label}'",
                severity=0.7,
            )

        return None

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        goals:           List[Goal],
        belief_system    = None,
        internal_state   = None,
        strategy:        Optional[ResolutionStrategy] = None,
    ) -> ResolutionDecision:
        """
        Given a list of active goals, return which to pursue and which to drop.

        Parameters
        ----------
        goals          : Active goals list.
        belief_system  : For feasibility checking.
        internal_state : For time-pressure estimation.
        strategy       : Override default strategy.
        """
        strategy = strategy or self.strategy
        conflicts = self.detect_conflicts(goals)

        if not conflicts:
            utility = self._total_utility(goals, belief_system, internal_state)
            return ResolutionDecision(
                chosen_goals=goals, dropped_goals=[],
                conflicts=[], strategy_used=strategy,
                utility_score=utility,
                notes="No conflicts detected",
            )

        if strategy == ResolutionStrategy.PRIORITY_ORDER:
            chosen, dropped = self._by_priority(goals, conflicts)
        elif strategy == ResolutionStrategy.UTILITY_MAXIMISATION:
            chosen, dropped = self._by_utility(goals, conflicts,
                                                belief_system, internal_state)
        else:  # SATISFICING
            chosen, dropped = self._by_satisficing(goals, conflicts,
                                                     belief_system, internal_state)

        utility = self._total_utility(chosen, belief_system, internal_state)
        notes   = (f"Resolved {len(conflicts)} conflicts via {strategy.value}. "
                   f"Dropped: {[g.label for g in dropped]}")

        decision = ResolutionDecision(
            chosen_goals=chosen, dropped_goals=dropped,
            conflicts=conflicts, strategy_used=strategy,
            utility_score=utility, notes=notes,
        )
        self._history.append(decision)
        return decision

    def _by_priority(
        self, goals: List[Goal], conflicts: List[GoalConflict]
    ) -> Tuple[List[Goal], List[Goal]]:
        """Keep highest-priority goal from each conflicting pair."""
        to_drop: Set[int] = set()
        for c in conflicts:
            loser = c.goal_b if c.goal_a.priority >= c.goal_b.priority else c.goal_a
            to_drop.add(id(loser))
        chosen  = [g for g in goals if id(g) not in to_drop]
        dropped = [g for g in goals if id(g) in to_drop]
        return chosen, dropped

    def _by_utility(
        self, goals: List[Goal], conflicts: List[GoalConflict],
        bs=None, ist=None,
    ) -> Tuple[List[Goal], List[Goal]]:
        """Keep the subset of goals with maximum combined utility."""
        to_drop: Set[int] = set()
        for c in conflicts:
            util_a = self._goal_utility(c.goal_a, bs, ist)
            util_b = self._goal_utility(c.goal_b, bs, ist)
            loser  = c.goal_b if util_a >= util_b else c.goal_a
            to_drop.add(id(loser))
        chosen  = [g for g in goals if id(g) not in to_drop]
        dropped = [g for g in goals if id(g) in to_drop]
        return chosen, dropped

    def _by_satisficing(
        self, goals: List[Goal], conflicts: List[GoalConflict],
        bs=None, ist=None,
    ) -> Tuple[List[Goal], List[Goal]]:
        """Keep goals until cumulative utility exceeds sat_threshold."""
        scored  = [(g, self._goal_utility(g, bs, ist)) for g in goals]
        scored.sort(key=lambda x: -x[1])
        chosen  = []
        dropped = []
        cum     = 0.0
        conflict_ids = {id(c.goal_b) for c in conflicts} | {id(c.goal_a) for c in conflicts}
        for g, u in scored:
            if cum >= self.sat_threshold and id(g) in conflict_ids:
                dropped.append(g)
            else:
                chosen.append(g)
                cum = min(1.0, cum + u)
        return chosen, dropped

    def _goal_utility(self, goal: Goal, bs=None, ist=None) -> float:
        """
        Estimate utility of pursuing a goal.
        priority × feasibility × urgency
        """
        feasibility = 1.0
        if bs and goal.target:
            conf = bs.get_confidence(f"{goal.target}.edible", default=0.5)
            if goal.goal_type == GoalType.EAT_OBJECT:
                feasibility = conf   # more likely if we know it's edible

        urgency = 1.0
        if ist:
            if ist.frustration > 0.6:
                urgency = 1.2   # frustrated → more urgent to complete anything
            elif ist.fatigue > 0.7:
                urgency = 0.8   # tired → less urgent

        age_penalty = min(0.3, goal.age_seconds / 300)   # penalise very old goals
        return float(np.clip(goal.priority * feasibility * urgency - age_penalty,
                              0.0, 1.0))

    def _total_utility(self, goals: List[Goal], bs=None, ist=None) -> float:
        if not goals:
            return 0.0
        return float(np.mean([self._goal_utility(g, bs, ist) for g in goals]))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def n_conflicts(self) -> int:
        return len(self._conflict_log)

    def conflict_types(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for c in self._conflict_log:
            k = c.conflict_type.value
            counts[k] = counts.get(k, 0) + 1
        return counts

    def summary(self) -> Dict:
        return {
            "total_conflicts":   self.n_conflicts(),
            "by_type":           self.conflict_types(),
            "decisions_made":    len(self._history),
            "strategy":          self.strategy.value,
        }

    def __repr__(self) -> str:
        return (f"GoalConflictResolver(conflicts={self.n_conflicts()}, "
                f"decisions={len(self._history)})")
