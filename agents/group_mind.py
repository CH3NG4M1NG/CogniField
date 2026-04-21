"""
agents/group_mind.py
=====================
Group Mind — Collective Coordination Layer

Coordinates a fleet of agents through shared goals, shared state,
and coordination signals. The GroupMind is not a separate agent —
it is a lightweight overlay that keeps the fleet coherent without
centralising control.

Responsibilities
----------------
1. SHARED GOALS
   The fleet has one active goal at a time (the "primary mission").
   All agents bias their goal selection toward the primary mission.

2. SHARED STATE
   A snapshot of fleet-level metrics (mean confidence, success rate,
   n_contested beliefs, workload distribution) available to all agents.

3. COORDINATION SIGNALS
   Broadcast signals that trigger specific behaviours:
     EXPLORE     → agents with free capacity explore unknowns
     CONSOLIDATE → all agents run memory consolidation
     CAUTIOUS    → raise risk tolerance; avoid experiments
     ACCELERATE  → lower novelty threshold; take more actions
     SYNC        → all agents sync private beliefs to shared memory

4. EXPERIENCE SHARING
   When an agent completes a significant episode
   (high reward / novel discovery), GroupMind broadcasts the
   experience to all agents who integrate it via social learning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.event_bus import EventBus, EventType


class CoordSignal(str, Enum):
    EXPLORE      = "explore"
    CONSOLIDATE  = "consolidate"
    CAUTIOUS     = "cautious"
    ACCELERATE   = "accelerate"
    SYNC         = "sync"
    NEGOTIATE    = "negotiate"
    VOTE         = "vote"


@dataclass
class FleetState:
    """Snapshot of collective fleet metrics."""
    round:              int
    n_agents:           int
    mean_belief_conf:   float
    mean_success_rate:  float
    n_contested:        int
    total_beliefs:      int
    active_goals:       List[str]
    recent_signals:     List[str]
    timestamp:          float = field(default_factory=time.time)


@dataclass
class SharedExperience:
    """An episode worth sharing with the whole fleet."""
    source_agent:    str
    action:          str
    target:          str
    outcome:         str
    reward:          float
    belief_key:      str
    belief_value:    Any
    confidence:      float
    timestamp:       float = field(default_factory=time.time)


class GroupMind:
    """
    Collective coordination overlay for a fleet of agents.

    Parameters
    ----------
    event_bus : Optional EventBus for firing coordination events.
    """

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self.eb              = event_bus
        self._primary_goal:  Optional[str] = None
        self._secondary_goals: List[str]  = []
        self._fleet_state:   Optional[FleetState] = None
        self._signal_history: List[Tuple[float, CoordSignal]] = []
        self._experiences:   List[SharedExperience] = []
        self._round          = 0

        # Experience importance threshold: reward must exceed this to share
        self._share_threshold = 0.30

        # Coordination state
        self._current_signal: Optional[CoordSignal] = None
        self._signal_expiry:  float = 0.0

    # ------------------------------------------------------------------
    # Shared goals
    # ------------------------------------------------------------------

    def set_primary_goal(self, goal: str) -> None:
        """Set the fleet's primary collective objective."""
        self._primary_goal = goal
        if self.eb:
            self.eb.fire(EventType.ROUND_COMPLETE, "group_mind",
                         primary_goal=goal)

    def add_secondary_goal(self, goal: str) -> None:
        if goal not in self._secondary_goals:
            self._secondary_goals.append(goal)

    def get_primary_goal(self) -> Optional[str]:
        return self._primary_goal

    def active_goals(self) -> List[str]:
        goals = []
        if self._primary_goal:
            goals.append(self._primary_goal)
        goals.extend(self._secondary_goals)
        return goals

    # ------------------------------------------------------------------
    # Coordination signals
    # ------------------------------------------------------------------

    def broadcast_signal(
        self,
        signal:       CoordSignal,
        duration_sec: float = 30.0,
        source:       str   = "group_mind",
    ) -> None:
        """Send a coordination signal to all agents."""
        self._current_signal  = signal
        self._signal_expiry   = time.time() + duration_sec
        self._signal_history.append((time.time(), signal))

        if self.eb:
            self.eb.fire(
                EventType.ROUND_COMPLETE, source,
                signal=signal.value,
                duration=duration_sec,
            )

    def current_signal(self) -> Optional[CoordSignal]:
        """Return the active coordination signal, or None if expired."""
        if (self._current_signal is not None
                and time.time() < self._signal_expiry):
            return self._current_signal
        self._current_signal = None
        return None

    def recent_signals(self, n: int = 5) -> List[str]:
        return [s.value for _, s in self._signal_history[-n:]]

    def apply_signal_to_agent(self, agent) -> None:
        """
        Apply the current coordination signal to one agent,
        adjusting its parameters accordingly.
        """
        sig = self.current_signal()
        if sig is None:
            return

        if sig == CoordSignal.EXPLORE:
            agent.cfg.novelty_threshold = max(0.20,
                agent.cfg.novelty_threshold - 0.08)

        elif sig == CoordSignal.CAUTIOUS:
            agent.cfg.novelty_threshold = min(0.70,
                agent.cfg.novelty_threshold + 0.08)
            if hasattr(agent, "risk_engine"):
                agent.risk_engine.risk_tolerance = max(0.20,
                    agent.risk_engine.risk_tolerance - 0.05)

        elif sig == CoordSignal.ACCELERATE:
            agent.cfg.novelty_threshold = max(0.15,
                agent.cfg.novelty_threshold - 0.10)
            if hasattr(agent, "risk_engine"):
                agent.risk_engine.risk_tolerance = min(0.50,
                    agent.risk_engine.risk_tolerance + 0.05)

        elif sig == CoordSignal.CONSOLIDATE:
            if hasattr(agent, "consolidator"):
                agent.consolidator.consolidate()

        elif sig == CoordSignal.SYNC:
            if hasattr(agent, "_sync_to_shared_memory"):
                agent._sync_to_shared_memory()

    # ------------------------------------------------------------------
    # Fleet state
    # ------------------------------------------------------------------

    def update_fleet_state(
        self,
        agents: List,
        n_contested: int = 0,
    ) -> FleetState:
        """Compute and cache the current fleet state."""
        self._round += 1

        if not agents:
            state = FleetState(
                round=self._round, n_agents=0,
                mean_belief_conf=0.5, mean_success_rate=0.0,
                n_contested=n_contested, total_beliefs=0,
                active_goals=self.active_goals(),
                recent_signals=self.recent_signals(),
            )
            self._fleet_state = state
            return state

        mean_conf = float(np.mean([
            a.beliefs.summary().get("mean_conf", 0.5) for a in agents
            if hasattr(a, "beliefs")
        ]))
        mean_sr = float(np.mean([
            a.metrics.success_rate() for a in agents
            if hasattr(a, "metrics")
        ]))
        total_beliefs = sum(
            len(a.beliefs) for a in agents if hasattr(a, "beliefs")
        )
        all_goals = []
        for a in agents:
            if hasattr(a, "goal_system"):
                all_goals += a.goal_system.summary().get("active_goals", [])

        state = FleetState(
            round=self._round,
            n_agents=len(agents),
            mean_belief_conf=round(mean_conf, 3),
            mean_success_rate=round(mean_sr, 3),
            n_contested=n_contested,
            total_beliefs=total_beliefs,
            active_goals=list(set(all_goals))[:5],
            recent_signals=self.recent_signals(3),
        )
        self._fleet_state = state

        # Auto-signal based on fleet state
        self._auto_signal(state)
        return state

    def _auto_signal(self, state: FleetState) -> None:
        """Automatically broadcast coordination signals based on fleet health."""
        if state.mean_success_rate < 0.30 and state.n_contested > 3:
            self.broadcast_signal(CoordSignal.CAUTIOUS, duration_sec=20)
        elif state.mean_success_rate > 0.70 and state.n_contested == 0:
            self.broadcast_signal(CoordSignal.ACCELERATE, duration_sec=15)
        elif self._round % 10 == 0:
            self.broadcast_signal(CoordSignal.SYNC, duration_sec=5)

    def get_fleet_state(self) -> Optional[FleetState]:
        return self._fleet_state

    # ------------------------------------------------------------------
    # Experience sharing
    # ------------------------------------------------------------------

    def share_experience(
        self,
        source_agent: str,
        action:       str,
        target:       str,
        outcome:      str,
        reward:       float,
        belief_key:   str  = "",
        belief_value: Any  = None,
        confidence:   float = 0.7,
    ) -> bool:
        """
        Contribute an experience to the group's collective memory.
        Only high-reward / novel experiences are accepted.
        Returns True if the experience was stored.
        """
        if abs(reward) < self._share_threshold:
            return False

        exp = SharedExperience(
            source_agent=source_agent,
            action=action,
            target=target,
            outcome=outcome,
            reward=reward,
            belief_key=belief_key,
            belief_value=belief_value,
            confidence=confidence,
        )
        self._experiences.append(exp)

        if self.eb:
            self.eb.fire(
                EventType.KNOWLEDGE_SHARED, source_agent,
                action=action, target=target,
                outcome=outcome, reward=reward,
                belief_key=belief_key,
            )
        return True

    def get_experiences_about(
        self,
        target: str,
        n:      int = 10,
    ) -> List[SharedExperience]:
        """Return recent experiences relevant to a target object."""
        relevant = [e for e in self._experiences if e.target == target]
        return relevant[-n:]

    def get_high_value_experiences(
        self,
        min_reward: float = 0.30,
    ) -> List[SharedExperience]:
        """Return experiences with significant positive reward."""
        return [e for e in self._experiences if e.reward >= min_reward]

    def integrate_experiences(self, agent) -> int:
        """
        Push all high-value shared experiences into an agent's beliefs.
        Called when an agent joins or after a sync signal.
        Returns count of beliefs updated.
        """
        count = 0
        for exp in self.get_high_value_experiences():
            if not exp.belief_key or exp.belief_value is None:
                continue
            if not hasattr(agent, "beliefs"):
                continue
            existing = agent.beliefs.get(exp.belief_key)
            if existing and existing.confidence >= exp.confidence:
                continue   # agent already knows better
            agent.beliefs.update(
                exp.belief_key, exp.belief_value,
                source="group_experience",
                weight=exp.confidence * 0.7,
                notes=f"shared by {exp.source_agent}",
            )
            count += 1
        return count

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        fs = self._fleet_state
        return {
            "round":             self._round,
            "primary_goal":      self._primary_goal,
            "secondary_goals":   len(self._secondary_goals),
            "current_signal":    self._current_signal.value if self._current_signal else None,
            "experiences":       len(self._experiences),
            "high_value_exps":   len(self.get_high_value_experiences()),
            "fleet_state":       {
                "n_agents":      fs.n_agents if fs else 0,
                "mean_conf":     fs.mean_belief_conf if fs else 0,
                "mean_sr":       fs.mean_success_rate if fs else 0,
                "n_contested":   fs.n_contested if fs else 0,
            } if fs else {},
        }

    def __repr__(self) -> str:
        sig = self._current_signal.value if self._current_signal else "none"
        return (f"GroupMind(goal={self._primary_goal!r}, "
                f"signal={sig}, "
                f"experiences={len(self._experiences)})")
