"""
agents/body.py
===============
Virtual Body

Simulates a physical embodiment with three sensory/motor systems:

  EYES   – observe and inspect objects in the environment
  HANDS  – pick up, carry, drop, examine by touch
  MOUTH  – eat, taste, consume

Every action returns a structured BodyActionResult so the perception
layer has a single, consistent format to normalise.

Body state:
  - inventory: objects currently held
  - hunger: 0.0 (full) → 1.0 (starving)
  - health: 1.0 (perfect) → 0.0 (dead)
  - position: (x, y) grid coordinates
  - energy: 0.0 → 1.0 (affects action success probability)

Design principles:
  - All actions are deterministic given body state + environment response
  - Actions can fail for physical reasons (hands full, object too heavy, etc.)
  - Every action is logged for the learning system
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class BodyAction(str, Enum):
    PICK    = "pick"      # grasp an object
    DROP    = "drop"      # release held object
    EAT     = "eat"       # consume an object
    INSPECT = "inspect"   # examine an object carefully
    MOVE    = "move"      # move in a direction
    LOOK    = "look"      # scan surroundings
    WAIT    = "wait"      # do nothing this step


class ActionStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    BLOCKED = "blocked"   # safety rule prevented action
    INVALID = "invalid"   # action not physically possible


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BodyActionResult:
    """
    Structured result of one body action.

    All fields are present so downstream code never needs to key-check.
    """
    action:      BodyAction
    object_name: str
    status:      ActionStatus
    effect:      str                   # "satisfied", "damage", "observed", etc.
    reward:      float                 # signed reward signal
    confidence:  float                 # how certain we are about the effect
    observations: Dict[str, Any]       # what was observed / learned
    body_delta:   Dict[str, float]     # changes to body state (health, hunger, etc.)
    reason:       str                  # human-readable explanation
    timestamp:    float = field(default_factory=time.time)

    @property
    def succeeded(self) -> bool:
        return self.status == ActionStatus.SUCCESS

    @property
    def failed(self) -> bool:
        return self.status == ActionStatus.FAILURE

    def to_dict(self) -> Dict:
        return {
            "action":       self.action.value,
            "object":       self.object_name,
            "status":       self.status.value,
            "effect":       self.effect,
            "reward":       round(self.reward, 3),
            "confidence":   round(self.confidence, 3),
            "observations": self.observations,
            "body_delta":   {k: round(v, 3) for k, v in self.body_delta.items()},
            "reason":       self.reason,
        }


# ---------------------------------------------------------------------------
# Body state
# ---------------------------------------------------------------------------

@dataclass
class BodyState:
    """Complete physical state of the body."""
    health:      float = 1.00   # 0 = dead, 1 = perfect
    hunger:      float = 0.30   # 0 = full, 1 = starving
    energy:      float = 0.90   # 0 = exhausted, 1 = full energy
    position:    Tuple[int, int] = (0, 0)
    inventory:   List[str] = field(default_factory=list)
    max_carry:   int  = 3       # max items in inventory

    @property
    def alive(self) -> bool:
        return self.health > 0.0

    @property
    def can_pick(self) -> bool:
        return len(self.inventory) < self.max_carry and self.energy > 0.05

    @property
    def is_hungry(self) -> bool:
        return self.hunger >= 0.60

    def apply_delta(self, delta: Dict[str, float]) -> None:
        for attr, change in delta.items():
            if hasattr(self, attr):
                val = getattr(self, attr)
                if isinstance(val, float):
                    setattr(self, attr, float(np.clip(val + change, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Virtual Body
# ---------------------------------------------------------------------------

class VirtualBody:
    """
    Virtual body that interfaces between the agent's decisions and the
    physical environment.

    Parameters
    ----------
    max_inventory : Maximum items the body can carry simultaneously.
    seed          : RNG seed for any stochastic effects.
    """

    def __init__(
        self,
        max_inventory:   int = 3,
        seed:            int = 42,
    ) -> None:
        self.state  = BodyState(max_carry=max_inventory)
        self._rng   = np.random.default_rng(seed)
        self._log:  List[BodyActionResult] = []
        self._step  = 0

    # ------------------------------------------------------------------
    # Core action dispatch
    # ------------------------------------------------------------------

    def act(
        self,
        action:      BodyAction,
        object_name: str,
        env_response: Optional[Dict] = None,
    ) -> BodyActionResult:
        """
        Perform a body action.

        Parameters
        ----------
        action      : BodyAction enum value
        object_name : Target object name
        env_response: Optional pre-fetched environment response dict

        Returns
        -------
        BodyActionResult with full outcome details
        """
        if not self.state.alive:
            return self._make_result(action, object_name,
                                     ActionStatus.INVALID, "dead",
                                     0.0, 0.0, {}, {}, "Body is no longer alive.")

        self._step += 1

        if action == BodyAction.EAT:
            result = self._do_eat(object_name, env_response)
        elif action == BodyAction.PICK:
            result = self._do_pick(object_name, env_response)
        elif action == BodyAction.DROP:
            result = self._do_drop(object_name)
        elif action == BodyAction.INSPECT:
            result = self._do_inspect(object_name, env_response)
        elif action == BodyAction.MOVE:
            result = self._do_move(object_name)   # object_name = direction here
        elif action == BodyAction.LOOK:
            result = self._do_look(env_response)
        elif action == BodyAction.WAIT:
            result = self._do_wait()
        else:
            result = self._make_result(
                action, object_name, ActionStatus.INVALID,
                "unknown_action", 0.0, 1.0, {}, {},
                f"Unknown action: {action}"
            )

        # Apply body state changes
        self.state.apply_delta(result.body_delta)
        self._log.append(result)
        return result

    # ------------------------------------------------------------------
    # Individual action implementations
    # ------------------------------------------------------------------

    def _do_eat(
        self,
        obj:          str,
        env_response: Optional[Dict],
    ) -> BodyActionResult:
        """Use mouth to consume an object."""
        # Must have object in inventory or it must be accessible
        if not self._accessible(obj):
            return self._make_result(
                BodyAction.EAT, obj, ActionStatus.INVALID,
                "not_accessible", -0.05, 1.0, {}, {},
                f"Cannot eat {obj}: not in inventory or nearby."
            )

        # Pull result from environment response
        if env_response:
            edible   = env_response.get("edible", None)
            effect   = env_response.get("effect", "unknown")
            reward   = float(env_response.get("reward", 0.0))
            known    = env_response.get("known", False)
            confidence = env_response.get("confidence", 0.5)
        else:
            # No environment info → treat as unknown
            edible, effect, reward, known, confidence = None, "unknown", 0.0, False, 0.3

        if edible is True:
            delta  = {"hunger": -0.35, "energy": +0.20, "health": +0.05}
            status = ActionStatus.SUCCESS
            effect = effect or "satisfied"
            if obj in self.state.inventory:
                self.state.inventory.remove(obj)
        elif edible is False:
            delta  = {"health": -0.20, "hunger": -0.05, "energy": -0.10}
            status = ActionStatus.FAILURE
            effect = effect or "damage"
            if obj in self.state.inventory:
                self.state.inventory.remove(obj)
        else:
            # Unknown edibility → uncertain outcome
            if self._rng.random() < 0.5:
                delta  = {"hunger": -0.15, "energy": +0.05}
                status = ActionStatus.SUCCESS
                effect = "uncertain_success"
                reward = 0.10
            else:
                delta  = {"health": -0.10, "energy": -0.05}
                status = ActionStatus.FAILURE
                effect = "uncertain_damage"
                reward = -0.15

        obs = {"edible": edible, "effect": effect, "known": known}
        reason = (f"Mouth consumed {obj}: edible={edible}, "
                  f"effect={effect}, reward={reward:+.2f}")
        return self._make_result(BodyAction.EAT, obj, status, effect,
                                  reward, confidence, obs, delta, reason)

    def _do_pick(
        self,
        obj:          str,
        env_response: Optional[Dict],
    ) -> BodyActionResult:
        """Use hands to pick up an object."""
        if not self.state.can_pick:
            msg = ("Hands are full."
                   if len(self.state.inventory) >= self.state.max_carry
                   else "Too exhausted to pick.")
            return self._make_result(BodyAction.PICK, obj,
                                     ActionStatus.INVALID, "hands_full",
                                     -0.01, 1.0, {}, {}, msg)

        heavy = env_response.get("heavy", False) if env_response else False
        if heavy and self._rng.random() < 0.3:
            return self._make_result(BodyAction.PICK, obj,
                                     ActionStatus.FAILURE, "too_heavy",
                                     -0.05, 0.7, {"heavy": True},
                                     {"energy": -0.05},
                                     f"{obj} is too heavy to pick up.")

        self.state.inventory.append(obj)
        delta  = {"energy": -0.03}
        obs    = env_response or {"picked": True}
        reason = f"Hands picked up {obj}. Inventory: {self.state.inventory}"
        return self._make_result(BodyAction.PICK, obj, ActionStatus.SUCCESS,
                                  "picked", 0.05, 0.95, obs, delta, reason)

    def _do_drop(self, obj: str) -> BodyActionResult:
        """Release a held object."""
        if obj not in self.state.inventory:
            return self._make_result(BodyAction.DROP, obj,
                                     ActionStatus.INVALID, "not_held",
                                     0.0, 1.0, {}, {},
                                     f"{obj} is not in inventory.")
        self.state.inventory.remove(obj)
        return self._make_result(BodyAction.DROP, obj, ActionStatus.SUCCESS,
                                  "dropped", 0.0, 1.0, {}, {},
                                  f"Dropped {obj}.")

    def _do_inspect(
        self,
        obj:          str,
        env_response: Optional[Dict],
    ) -> BodyActionResult:
        """Use eyes + hands to carefully examine an object."""
        properties = env_response.get("properties", {}) if env_response else {}
        observations = {
            "inspected":  True,
            "properties": properties,
            "visible":    env_response.get("visible", True) if env_response else True,
        }
        delta  = {"energy": -0.02}
        reason = f"Eyes inspected {obj}: found {list(properties.keys())}"
        return self._make_result(BodyAction.INSPECT, obj, ActionStatus.SUCCESS,
                                  "observed", 0.05, 0.90, observations, delta, reason)

    def _do_move(self, direction: str) -> BodyActionResult:
        """Move the body in a direction."""
        dx_dy = {"north": (0,1), "south": (0,-1),
                  "east":  (1,0), "west":  (-1,0),
                  "up":    (0,1), "down":  (0,-1),
                  "left":  (-1,0),"right": (1,0)}
        dx, dy = dx_dy.get(direction.lower(), (0, 0))
        if dx == 0 and dy == 0:
            return self._make_result(BodyAction.MOVE, direction,
                                     ActionStatus.INVALID, "bad_direction",
                                     0.0, 1.0, {}, {},
                                     f"Unknown direction: {direction}")
        x, y = self.state.position
        self.state.position = (x + dx, y + dy)
        delta  = {"energy": -0.04, "hunger": +0.02}
        obs    = {"new_position": self.state.position}
        reason = f"Moved {direction} to {self.state.position}"
        return self._make_result(BodyAction.MOVE, direction, ActionStatus.SUCCESS,
                                  "moved", 0.0, 1.0, obs, delta, reason)

    def _do_look(self, env_response: Optional[Dict]) -> BodyActionResult:
        """Eyes scan the environment."""
        visible = env_response.get("visible_objects", []) if env_response else []
        obs     = {"visible_objects": visible, "position": self.state.position}
        reason  = f"Eyes scanned: saw {visible}"
        return self._make_result(BodyAction.LOOK, "", ActionStatus.SUCCESS,
                                  "observed", 0.02, 1.0, obs, {}, reason)

    def _do_wait(self) -> BodyActionResult:
        """Rest — restore small amount of energy."""
        delta  = {"energy": +0.05, "hunger": +0.03}
        return self._make_result(BodyAction.WAIT, "", ActionStatus.SUCCESS,
                                  "rested", 0.01, 1.0, {}, delta,
                                  "Waited one step. Slight energy recovery.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _accessible(self, obj: str) -> bool:
        """Object is in inventory or assumed accessible (in reach)."""
        # In our virtual world, objects are always reachable unless we
        # specifically require pick-first. For simplicity, allow eating
        # without explicit pick.
        return True

    @staticmethod
    def _make_result(
        action:       BodyAction,
        obj:          str,
        status:       ActionStatus,
        effect:       str,
        reward:       float,
        confidence:   float,
        observations: Dict,
        body_delta:   Dict,
        reason:       str,
    ) -> BodyActionResult:
        return BodyActionResult(
            action=action, object_name=obj, status=status,
            effect=effect, reward=reward, confidence=confidence,
            observations=observations, body_delta=body_delta, reason=reason,
        )

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def health_pct(self) -> float:
        return self.state.health

    def hunger_pct(self) -> float:
        return self.state.hunger

    def inventory_list(self) -> List[str]:
        return list(self.state.inventory)

    def is_motivated(self) -> bool:
        """True if hunger is high enough to motivate eating."""
        return self.state.hunger >= 0.50

    def summary(self) -> Dict:
        return {
            "health":    round(self.state.health, 3),
            "hunger":    round(self.state.hunger, 3),
            "energy":    round(self.state.energy, 3),
            "position":  self.state.position,
            "inventory": self.state.inventory,
            "alive":     self.state.alive,
            "steps":     self._step,
            "actions_taken": len(self._log),
        }

    def action_history(self, n: int = 10) -> List[Dict]:
        return [r.to_dict() for r in self._log[-n:]]

    def __repr__(self) -> str:
        s = self.state
        return (f"VirtualBody(health={s.health:.0%}, hunger={s.hunger:.0%}, "
                f"pos={s.position}, inv={s.inventory})")
