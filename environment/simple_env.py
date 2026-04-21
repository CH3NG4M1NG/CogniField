"""
environment/simple_env.py
=========================
Simple Object World Environment

A minimal simulation with:
  - Objects: named entities with properties and latent vectors
  - Actions: pick, move, observe, drop, combine
  - State: agent inventory + object positions
  - Feedback: reward signal + natural language description

The environment provides grounded training data:
every interaction produces (state_vec, action_vec, feedback) tuples
that the agent can learn from.

This is the "embodiment" component — meaning is grounded in action.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Object definition
# ---------------------------------------------------------------------------

@dataclass
class WorldObject:
    """An object in the simulation."""
    name:       str
    category:   str          # e.g. "food", "tool", "material"
    properties: Dict[str, Any]
    position:   Tuple[int, int]
    is_held:    bool = False
    latent_vec: Optional[np.ndarray] = None   # set when encoder is available

    def describe(self) -> str:
        props = ", ".join(f"{k}={v}" for k, v in self.properties.items())
        return f"{self.name} [{self.category}] at {self.position} ({props})"


# ---------------------------------------------------------------------------
# Default world objects
# ---------------------------------------------------------------------------

DEFAULT_OBJECTS: List[Dict] = [
    {"name": "apple",   "category": "food",     "color": "red",    "weight": "light", "edible": True},
    {"name": "stone",   "category": "material", "color": "grey",   "weight": "heavy", "edible": False},
    {"name": "book",    "category": "tool",     "color": "brown",  "weight": "light", "edible": False},
    {"name": "water",   "category": "food",     "color": "clear",  "weight": "medium","edible": True},
    {"name": "hammer",  "category": "tool",     "color": "grey",   "weight": "heavy", "edible": False},
    {"name": "leaf",    "category": "material", "color": "green",  "weight": "light", "edible": False},
    {"name": "bread",   "category": "food",     "color": "yellow", "weight": "light", "edible": True},
    {"name": "lamp",    "category": "tool",     "color": "white",  "weight": "medium","edible": False},
]

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

ACTIONS = ["pick", "drop", "move", "observe", "eat", "combine", "inspect"]

ActionFeedback = Dict[str, Any]


# ---------------------------------------------------------------------------
# SimpleEnv
# ---------------------------------------------------------------------------

class SimpleEnv:
    """
    A 10×10 grid world with named objects.

    The agent can:
      pick   <name>    : Pick up an object
      drop   <name>    : Drop a held object
      move   <x> <y>  : Move to position (x, y)
      observe          : Describe current surroundings
      eat    <name>    : Eat an edible object
      combine <a> <b>  : Combine two held objects
      inspect <name>   : Get detailed info about an object

    Feedback includes:
      - success   : bool
      - reward    : float ∈ [-1, 1]
      - message   : natural language description
      - state_vec : latent-space encoding of current state
    """

    GRID_SIZE = 10

    def __init__(self, seed: int = 42) -> None:
        self._rng     = random.Random(seed)
        self._np_rng  = np.random.default_rng(seed)
        self._objects: Dict[str, WorldObject] = {}
        self._inventory: List[str] = []
        self._agent_pos: Tuple[int, int] = (5, 5)
        self._step_count = 0
        self._total_reward = 0.0
        self._event_log: List[Dict] = []
        self._encoder = None     # injected by agent

        self._init_objects()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_objects(self) -> None:
        G = self.GRID_SIZE
        for obj_def in DEFAULT_OBJECTS:
            pos  = (self._rng.randint(0, G-1),
                    self._rng.randint(0, G-1))
            props = {k: v for k, v in obj_def.items()
                     if k not in ("name", "category")}
            obj = WorldObject(
                name=obj_def["name"],
                category=obj_def["category"],
                properties=props,
                position=pos,
            )
            self._objects[obj_def["name"]] = obj

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def step(self, action: str, *args) -> ActionFeedback:
        """Execute an action and return feedback."""
        self._step_count += 1
        action = action.lower().strip()

        if action == "pick":
            fb = self._act_pick(args[0] if args else "")
        elif action == "drop":
            fb = self._act_drop(args[0] if args else "")
        elif action == "move":
            x = int(args[0]) if len(args) > 0 else self._agent_pos[0]
            y = int(args[1]) if len(args) > 1 else self._agent_pos[1]
            fb = self._act_move(x, y)
        elif action == "observe":
            fb = self._act_observe()
        elif action == "eat":
            fb = self._act_eat(args[0] if args else "")
        elif action == "combine":
            a = args[0] if len(args) > 0 else ""
            b = args[1] if len(args) > 1 else ""
            fb = self._act_combine(a, b)
        elif action == "inspect":
            fb = self._act_inspect(args[0] if args else "")
        else:
            fb = {"success": False, "reward": -0.1,
                  "message": f"Unknown action: '{action}'",
                  "state_vec": self.state_vector()}

        self._total_reward += fb.get("reward", 0)
        self._event_log.append({
            "step": self._step_count,
            "action": action,
            "args": list(args),
            **fb,
        })
        return fb

    def _act_pick(self, name: str) -> ActionFeedback:
        obj = self._objects.get(name)
        if not obj:
            return {"success": False, "reward": -0.1,
                    "message": f"No object named '{name}' exists.",
                    "state_vec": self.state_vector()}
        dist = self._distance_to(obj.position)
        if dist > 2:
            return {"success": False, "reward": -0.05,
                    "message": f"'{name}' is too far away (distance={dist:.1f}).",
                    "state_vec": self.state_vector()}
        if obj.is_held:
            return {"success": False, "reward": -0.05,
                    "message": f"You are already holding '{name}'.",
                    "state_vec": self.state_vector()}
        if len(self._inventory) >= 3:
            return {"success": False, "reward": -0.05,
                    "message": "Inventory full (max 3 items).",
                    "state_vec": self.state_vector()}
        obj.is_held = True
        self._inventory.append(name)
        return {"success": True, "reward": 0.1,
                "message": f"Picked up '{name}'.",
                "state_vec": self.state_vector()}

    def _act_drop(self, name: str) -> ActionFeedback:
        if name not in self._inventory:
            return {"success": False, "reward": -0.05,
                    "message": f"'{name}' is not in inventory.",
                    "state_vec": self.state_vector()}
        self._objects[name].is_held = False
        self._objects[name].position = self._agent_pos
        self._inventory.remove(name)
        return {"success": True, "reward": 0.0,
                "message": f"Dropped '{name}' at {self._agent_pos}.",
                "state_vec": self.state_vector()}

    def _act_move(self, x: int, y: int) -> ActionFeedback:
        x = int(np.clip(x, 0, self.GRID_SIZE - 1))
        y = int(np.clip(y, 0, self.GRID_SIZE - 1))
        old = self._agent_pos
        self._agent_pos = (x, y)
        dist = abs(x - old[0]) + abs(y - old[1])
        nearby = self._nearby_objects()
        msg = f"Moved to ({x},{y})."
        if nearby:
            msg += f" Nearby: {', '.join(nearby)}."
        return {"success": True, "reward": -0.01 * dist,
                "message": msg,
                "nearby": nearby,
                "state_vec": self.state_vector()}

    def _act_observe(self) -> ActionFeedback:
        nearby  = self._nearby_objects(radius=3)
        held    = list(self._inventory)
        parts   = [f"Position: {self._agent_pos}."]
        if held:
            parts.append(f"Holding: {', '.join(held)}.")
        if nearby:
            parts.append(f"Nearby objects: {', '.join(nearby)}.")
        else:
            parts.append("No objects nearby.")
        return {"success": True, "reward": 0.0,
                "message": " ".join(parts),
                "position": self._agent_pos,
                "inventory": held,
                "nearby": nearby,
                "state_vec": self.state_vector()}

    def _act_eat(self, name: str) -> ActionFeedback:
        if name not in self._inventory:
            return {"success": False, "reward": -0.05,
                    "message": f"'{name}' is not in inventory.",
                    "state_vec": self.state_vector()}
        obj = self._objects[name]
        if not obj.properties.get("edible", False):
            return {"success": False, "reward": -0.2,
                    "message": f"'{name}' is not edible!",
                    "state_vec": self.state_vector()}
        self._inventory.remove(name)
        del self._objects[name]
        return {"success": True, "reward": 0.5,
                "message": f"Ate '{name}'. Delicious!",
                "state_vec": self.state_vector()}

    def _act_combine(self, a: str, b: str) -> ActionFeedback:
        if a not in self._inventory or b not in self._inventory:
            return {"success": False, "reward": -0.05,
                    "message": f"Both '{a}' and '{b}' must be in inventory.",
                    "state_vec": self.state_vector()}
        combo_name = f"{a}_{b}"
        combo_props = {**self._objects[a].properties,
                       **self._objects[b].properties}
        self._inventory.remove(a)
        self._inventory.remove(b)
        combo = WorldObject(
            name=combo_name,
            category="combined",
            properties=combo_props,
            position=self._agent_pos,
            is_held=True,
        )
        self._objects[combo_name] = combo
        self._inventory.append(combo_name)
        return {"success": True, "reward": 0.3,
                "message": f"Combined '{a}' and '{b}' into '{combo_name}'.",
                "state_vec": self.state_vector()}

    def _act_inspect(self, name: str) -> ActionFeedback:
        obj = self._objects.get(name)
        if not obj:
            return {"success": False, "reward": 0.0,
                    "message": f"No object '{name}'.",
                    "state_vec": self.state_vector()}
        return {"success": True, "reward": 0.02,
                "message": obj.describe(),
                "object": obj,
                "state_vec": self.state_vector()}

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def state_vector(self) -> np.ndarray:
        """
        Encode the current world state as a float32 vector.
        Deterministic, no encoder needed.
        """
        dim = 64
        vec = np.zeros(dim, dtype=np.float32)

        # Agent position
        vec[0] = self._agent_pos[0] / self.GRID_SIZE
        vec[1] = self._agent_pos[1] / self.GRID_SIZE

        # Inventory (one-hot style)
        for i, name in enumerate(self._inventory[:3]):
            obj = self._objects.get(name)
            if obj:
                # Use object name hash as feature
                h = hash(name) % 10000
                idx = 2 + i * 8
                for b in range(8):
                    vec[idx + b] = float((h >> b) & 1)

        # Nearby objects
        nearby = self._nearby_objects(radius=3)
        for i, name in enumerate(nearby[:4]):
            obj = self._objects[name]
            idx = 26 + i * 8
            vec[idx]   = obj.position[0] / self.GRID_SIZE
            vec[idx+1] = obj.position[1] / self.GRID_SIZE
            vec[idx+2] = float(obj.properties.get("edible", False))
            vec[idx+3] = 1.0 if obj.category == "food" else 0.0
            vec[idx+4] = 1.0 if obj.category == "tool" else 0.0

        # Global stats
        vec[60] = len(self._inventory) / 3.0
        vec[61] = len(self._objects) / len(DEFAULT_OBJECTS)
        vec[62] = self._step_count / 100.0
        vec[63] = self._total_reward / 10.0

        n = np.linalg.norm(vec)
        return vec / (n + 1e-8)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _distance_to(self, pos: Tuple[int, int]) -> float:
        dx = self._agent_pos[0] - pos[0]
        dy = self._agent_pos[1] - pos[1]
        return (dx**2 + dy**2) ** 0.5

    def _nearby_objects(self, radius: int = 2) -> List[str]:
        result = []
        for name, obj in self._objects.items():
            if not obj.is_held and self._distance_to(obj.position) <= radius:
                result.append(name)
        return sorted(result)

    def reset(self) -> None:
        """Reset to initial state."""
        self._objects    = {}
        self._inventory  = []
        self._agent_pos  = (5, 5)
        self._step_count = 0
        self._total_reward = 0.0
        self._event_log  = []
        self._init_objects()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def inventory(self) -> List[str]:
        return list(self._inventory)

    @property
    def object_names(self) -> List[str]:
        return list(self._objects.keys())

    def stats(self) -> Dict:
        return {
            "steps":        self._step_count,
            "total_reward": round(self._total_reward, 3),
            "inventory":    self.inventory,
            "n_objects":    len(self._objects),
        }

    def __repr__(self) -> str:
        return (f"SimpleEnv(pos={self._agent_pos}, "
                f"inventory={self.inventory}, "
                f"steps={self._step_count})")
