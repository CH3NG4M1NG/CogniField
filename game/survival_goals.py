"""
game/survival_goals.py
========================
Survival Goal Manager

Automatically generates, prioritises, and tracks goals based on the
agent's current survival needs. Goals are derived from game observations
so the agent always pursues the most critical objective first.

Goal hierarchy (priority order):
  SURVIVE_HEALTH   – health < 30% → heal immediately (highest priority)
  FIND_FOOD        – hunger < 50% → eat or find food
  AVOID_DANGER     – hostile mob nearby → flee or fight
  GATHER_RESOURCES – default exploration mode
  EXPLORE          – when safe and well-fed

The SurvivalGoalManager feeds directly into CogniField's goal system
through add_goal() calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .base_adapter import GameObservation


class SurvivalPriority(str, Enum):
    CRITICAL  = "critical"     # immediate threat to life
    HIGH      = "high"         # significant risk
    MEDIUM    = "medium"       # important but not urgent
    LOW       = "low"          # background maintenance


@dataclass
class SurvivalGoal:
    """One active survival objective."""
    name:       str
    description:str
    priority:   SurvivalPriority
    target:     str           # target object/entity/direction
    action:     str           # recommended action ("eat", "flee", "explore", ...)
    score:      float         # urgency score 0–1
    expires_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def cf_priority(self) -> float:
        """Convert to CogniField goal priority [0, 1]."""
        mapping = {
            SurvivalPriority.CRITICAL: 0.98,
            SurvivalPriority.HIGH:     0.85,
            SurvivalPriority.MEDIUM:   0.65,
            SurvivalPriority.LOW:      0.40,
        }
        return mapping.get(self.priority, 0.50)

    def to_query(self) -> str:
        """Build a query string for InteractionLoop.step()."""
        # Strip minecraft: prefix so the loop can parse it
        target = self.target
        if ":" in target:
            target = target.split(":")[-1].replace("_", " ")
        return f"{self.action} {target}"

    def to_dict(self) -> Dict:
        return {
            "name":       self.name,
            "description":self.description,
            "priority":   self.priority.value,
            "target":     self.target,
            "action":     self.action,
            "score":      round(self.score, 3),
            "cf_priority":self.cf_priority,
        }


class SurvivalGoalManager:
    """
    Derives survival goals from game observations and manages their lifecycle.

    Parameters
    ----------
    health_critical   : Health threshold for SURVIVE_HEALTH goal (0–1)
    hunger_high       : Hunger threshold for FIND_FOOD goal (0–1)
    danger_distance   : (future) radius for hostile mob detection
    """

    # Minecraft food items the agent knows how to eat
    KNOWN_FOODS = [
        "minecraft:apple", "minecraft:bread", "minecraft:carrot",
        "minecraft:cooked_beef", "minecraft:cooked_porkchop",
        "minecraft:cooked_chicken", "minecraft:sweet_berries",
        "minecraft:glow_berries", "minecraft:mushroom_stew",
        "minecraft:melon_slice", "minecraft:pumpkin_pie",
        "minecraft:cooked_salmon", "minecraft:cooked_cod",
    ]

    SAFE_FOOD_IDS = {f.split(":")[-1].replace("_", " ") for f in KNOWN_FOODS}

    def __init__(
        self,
        health_critical: float = 0.30,
        hunger_high:     float = 0.50,
        danger_distance: float = 16.0,
    ) -> None:
        self.health_critical  = health_critical
        self.hunger_high      = hunger_high
        self.danger_distance  = danger_distance
        self._active:  List[SurvivalGoal] = []
        self._history: List[SurvivalGoal] = []

    # ------------------------------------------------------------------
    # Goal generation
    # ------------------------------------------------------------------

    def update(self, obs: GameObservation) -> List[SurvivalGoal]:
        """
        Re-evaluate survival goals based on a fresh observation.
        Returns the current prioritised goal list (highest first).
        """
        # Prune expired goals
        self._active = [g for g in self._active if not g.is_expired]
        new_goals: List[SurvivalGoal] = []

        # 1 — SURVIVE_HEALTH (critical)
        if obs.health_pct < self.health_critical:
            new_goals.append(self._goal_survive_health(obs))

        # 2 — AVOID_DANGER
        if obs.is_in_danger:
            new_goals.append(self._goal_avoid_danger(obs))

        # 3 — FIND_FOOD
        if obs.is_hungry:
            new_goals.append(self._goal_find_food(obs))

        # 4 — GATHER_RESOURCES
        if not obs.is_in_danger and not obs.is_hungry:
            new_goals.append(self._goal_gather_resources(obs))

        # 5 — EXPLORE (background, lowest priority)
        new_goals.append(self._goal_explore(obs))

        # Merge: if same goal type already active, update score; else add
        merged = self._merge(new_goals)
        self._active = sorted(merged, key=lambda g: -g.score)
        return list(self._active)

    # ------------------------------------------------------------------
    # Individual goal factories
    # ------------------------------------------------------------------

    def _goal_survive_health(self, obs: GameObservation) -> SurvivalGoal:
        urgency = 1.0 - obs.health_pct   # more urgent the lower the health
        # Find a food item if available
        food = self._first_food_in_inventory(obs)
        if food:
            return SurvivalGoal(
                name="survive_health", priority=SurvivalPriority.CRITICAL,
                description=f"Health critical ({obs.health_pct:.0%}). Eat food.",
                target=food, action="eat",
                score=min(1.0, urgency * 1.5),
                expires_at=time.time() + 30,
            )
        return SurvivalGoal(
            name="survive_health", priority=SurvivalPriority.CRITICAL,
            description=f"Health critical. No food. Flee!",
            target="danger", action="flee",
            score=min(1.0, urgency * 1.5),
            expires_at=time.time() + 30,
        )

    def _goal_avoid_danger(self, obs: GameObservation) -> SurvivalGoal:
        hostiles = obs.hostile_entities
        urgency  = min(1.0, len(hostiles) * 0.4 + (1.0 - obs.health_pct) * 0.3)
        nearest  = hostiles[0].entity_type if hostiles else "unknown"
        return SurvivalGoal(
            name="avoid_danger", priority=SurvivalPriority.HIGH,
            description=f"Hostile entity nearby: {nearest}. Flee or fight.",
            target=nearest, action="flee",
            score=urgency,
            expires_at=time.time() + 20,
        )

    def _goal_find_food(self, obs: GameObservation) -> SurvivalGoal:
        urgency = 1.0 - obs.hunger_pct
        # Check inventory first
        food = self._first_food_in_inventory(obs)
        if food:
            return SurvivalGoal(
                name="eat_food", priority=SurvivalPriority.HIGH,
                description=f"Hungry ({obs.hunger_pct:.0%}). Eat {food}.",
                target=food, action="eat",
                score=urgency,
                expires_at=time.time() + 60,
            )
        # Check visible blocks for food
        visible_food = obs.visible_food
        if visible_food:
            blk = visible_food[0].name
            return SurvivalGoal(
                name="pick_food", priority=SurvivalPriority.MEDIUM,
                description=f"Food nearby: {blk}. Pick it up.",
                target=blk, action="pick",
                score=urgency * 0.8,
                expires_at=time.time() + 60,
            )
        # No food available — explore to find some
        return SurvivalGoal(
            name="find_food", priority=SurvivalPriority.MEDIUM,
            description="Hungry. No food in inventory. Explore to find food.",
            target="food_source", action="explore",
            score=urgency * 0.6,
            expires_at=time.time() + 120,
        )

    def _goal_gather_resources(self, obs: GameObservation) -> SurvivalGoal:
        # Identify the most useful resource to gather from visible blocks
        resource_priority = ["minecraft:iron_ore", "minecraft:coal_ore",
                              "minecraft:oak_log", "minecraft:spruce_log"]
        for res_id in resource_priority:
            for blk in obs.visible_blocks:
                if blk.block_id == res_id:
                    return SurvivalGoal(
                        name="gather", priority=SurvivalPriority.LOW,
                        description=f"Gathering {blk.name}.",
                        target=blk.name, action="break",
                        score=0.40,
                        expires_at=time.time() + 60,
                    )
        return SurvivalGoal(
            name="gather", priority=SurvivalPriority.LOW,
            description="Gathering resources. Exploring.",
            target="resources", action="explore",
            score=0.35,
            expires_at=time.time() + 60,
        )

    def _goal_explore(self, obs: GameObservation) -> SurvivalGoal:
        direction = ["north", "south", "east", "west"][
            hash(str(obs.position)) % 4
        ]
        return SurvivalGoal(
            name="explore", priority=SurvivalPriority.LOW,
            description=f"Exploring {direction}. Learning environment.",
            target=direction, action="move",
            score=0.25,
            expires_at=time.time() + 30,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _first_food_in_inventory(self, obs: GameObservation) -> Optional[str]:
        for item in obs.inventory:
            if any(f in item.item_id for f in
                   ["apple","bread","carrot","beef","pork","chicken",
                    "berry","melon","stew","soup","fish","salmon","cod"]):
                return item.item_id
        return None

    def _merge(self, new_goals: List[SurvivalGoal]) -> List[SurvivalGoal]:
        """Keep existing goals that are not superseded by new ones."""
        existing_names = {g.name for g in self._active}
        new_names      = {g.name for g in new_goals}
        # Keep existing goals not covered by new generation
        kept = [g for g in self._active if g.name not in new_names]
        return kept + new_goals

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def top_goal(self) -> Optional[SurvivalGoal]:
        """Return the highest-priority active goal."""
        return self._active[0] if self._active else None

    def goals_by_priority(
        self, priority: SurvivalPriority
    ) -> List[SurvivalGoal]:
        return [g for g in self._active if g.priority == priority]

    def has_critical(self) -> bool:
        return any(g.priority == SurvivalPriority.CRITICAL for g in self._active)

    def complete_goal(self, name: str) -> bool:
        """Mark a goal as completed (remove from active list)."""
        before = len(self._active)
        completed = [g for g in self._active if g.name == name]
        self._active = [g for g in self._active if g.name != name]
        self._history.extend(completed)
        return len(self._active) < before

    def summary(self) -> Dict:
        return {
            "active_goals":    len(self._active),
            "top_goal":        self._active[0].name if self._active else None,
            "has_critical":    self.has_critical(),
            "completed_goals": len(self._history),
        }

    def __repr__(self) -> str:
        top = self._active[0].name if self._active else "none"
        return f"SurvivalGoalManager(active={len(self._active)}, top={top!r})"
