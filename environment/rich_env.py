"""
environment/rich_env.py
========================
Rich Object World Environment (v3)

Upgrades over SimpleEnv:
  - More object properties: edible, fragile, heavy, movable, unknown
  - Partial observability: agent only sees objects within visibility radius
  - Richer feedback: detailed state changes, consequences
  - Unknown objects: agent must investigate before knowing properties
  - Consequence system: fragile objects break, heavy objects slow movement
  - "mystery_object" that triggers curiosity
"""

from __future__ import annotations

import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Object definition
# ---------------------------------------------------------------------------

@dataclass
class RichObject:
    """An object with rich properties, some hidden until inspected."""
    name:          str
    category:      str
    position:      Tuple[int, int]
    # Observable properties (always visible)
    color:         str  = "grey"
    size:          str  = "medium"    # "small", "medium", "large"
    # Hidden properties (revealed by inspect)
    edible:        Optional[bool] = None    # None = unknown until inspected
    fragile:       bool = False
    heavy:         bool = False
    movable:       bool = True
    is_held:       bool = False
    broken:        bool = False
    # State
    _inspected:    bool = False

    def inspect(self) -> Dict[str, Any]:
        """Reveal all properties. Returns full property dict."""
        self._inspected = True
        return {
            "name":     self.name,
            "category": self.category,
            "color":    self.color,
            "size":     self.size,
            "edible":   self.edible,
            "fragile":  self.fragile,
            "heavy":    self.heavy,
            "movable":  self.movable,
        }

    def observable_desc(self) -> str:
        """Description with only observable properties."""
        parts = [f"{self.color} {self.size} {self.name}"]
        if self.broken:
            parts.append("[broken]")
        if self._inspected:
            parts.append(f"[edible={self.edible}]")
        return " ".join(parts)

    def full_desc(self) -> str:
        return (f"{self.name} [{self.category}] at {self.position} | "
                f"edible={self.edible}, fragile={self.fragile}, "
                f"heavy={self.heavy}, broken={self.broken}")


# ---------------------------------------------------------------------------
# Default rich object catalogue
# ---------------------------------------------------------------------------

RICH_CATALOGUE: List[Dict] = [
    # Known food
    {"name": "apple",    "category": "food",     "color": "red",    "size": "small",
     "edible": True,  "fragile": False, "heavy": False},
    {"name": "bread",    "category": "food",     "color": "yellow", "size": "small",
     "edible": True,  "fragile": False, "heavy": False},
    {"name": "water",    "category": "food",     "color": "clear",  "size": "medium",
     "edible": True,  "fragile": True,  "heavy": False},
    # Known non-food
    {"name": "stone",    "category": "material", "color": "grey",   "size": "medium",
     "edible": False, "fragile": False, "heavy": True},
    {"name": "hammer",   "category": "tool",     "color": "grey",   "size": "medium",
     "edible": False, "fragile": False, "heavy": True},
    {"name": "glass_jar","category": "tool",     "color": "clear",  "size": "small",
     "edible": False, "fragile": True,  "heavy": False},
    {"name": "leaf",     "category": "material", "color": "green",  "size": "small",
     "edible": False, "fragile": False, "heavy": False},
    # Mystery objects — properties unknown until inspected
    {"name": "purple_berry", "category": "unknown", "color": "purple", "size": "small",
     "edible": None,  "fragile": False, "heavy": False},
    {"name": "glowing_cube", "category": "unknown", "color": "blue",   "size": "small",
     "edible": None,  "fragile": True,  "heavy": False},
]


ActionFeedback = Dict[str, Any]


class RichEnv:
    """
    10×10 grid world with rich object properties and partial observability.

    Visibility radius: agent only sees objects within VISIBILITY_RADIUS cells.

    Actions:
      pick   <name>         pick up an object (must be nearby + not heavy)
      drop   <name>         drop a held object
      move   <x> <y>        move to position
      eat    <name>         eat a held object
      inspect <name>        fully reveal properties of an object
      observe               see nearby objects
      combine <a> <b>       combine two held objects
      use     <name>        use a tool
    """

    GRID_SIZE          = 10
    VISIBILITY_RADIUS  = 3    # partial observability
    MAX_INVENTORY      = 3

    def __init__(self, seed: int = 42, n_objects: Optional[int] = None) -> None:
        self._rng       = random.Random(seed)
        self._np_rng    = np.random.default_rng(seed)
        self._objects:  Dict[str, RichObject] = {}
        self._inventory: List[str] = []
        self._agent_pos: Tuple[int, int] = (5, 5)
        self._step_count = 0
        self._total_reward = 0.0
        self._event_log: List[Dict] = []
        self._health = 1.0      # agent health [0, 1]
        self._satiation = 0.5   # hunger level [0, 1] — lower = more hungry

        catalogue = RICH_CATALOGUE if n_objects is None else RICH_CATALOGUE[:n_objects]
        self._init_objects(catalogue)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_objects(self, catalogue: List[Dict]) -> None:
        G = self.GRID_SIZE
        for obj_def in catalogue:
            pos = (self._rng.randint(0, G - 1), self._rng.randint(0, G - 1))
            obj = RichObject(
                name=obj_def["name"],
                category=obj_def["category"],
                color=obj_def.get("color", "grey"),
                size=obj_def.get("size", "medium"),
                position=pos,
                edible=obj_def.get("edible", None),
                fragile=obj_def.get("fragile", False),
                heavy=obj_def.get("heavy", False),
                movable=obj_def.get("movable", True),
            )
            self._objects[obj_def["name"]] = obj

    # ------------------------------------------------------------------
    # Partial observability
    # ------------------------------------------------------------------

    def visible_objects(self) -> List[RichObject]:
        """Return only objects within VISIBILITY_RADIUS of agent."""
        result = []
        for obj in self._objects.values():
            if not obj.is_held and self._dist(obj.position) <= self.VISIBILITY_RADIUS:
                result.append(obj)
        return result

    def visible_names(self) -> List[str]:
        return [o.name for o in self.visible_objects()]

    def _dist(self, pos: Tuple[int, int]) -> float:
        dx = self._agent_pos[0] - pos[0]
        dy = self._agent_pos[1] - pos[1]
        return (dx**2 + dy**2) ** 0.5

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def step(self, action: str, *args) -> ActionFeedback:
        self._step_count += 1
        action = action.lower().strip()
        self._satiation = max(0.0, self._satiation - 0.01)  # hunger increases

        if action == "pick":
            fb = self._act_pick(args[0] if args else "")
        elif action == "drop":
            fb = self._act_drop(args[0] if args else "")
        elif action == "move":
            x = int(args[0]) if len(args) > 0 else self._agent_pos[0]
            y = int(args[1]) if len(args) > 1 else self._agent_pos[1]
            fb = self._act_move(x, y)
        elif action == "eat":
            fb = self._act_eat(args[0] if args else "")
        elif action == "inspect":
            fb = self._act_inspect(args[0] if args else "")
        elif action == "observe":
            fb = self._act_observe()
        elif action == "combine":
            a = args[0] if len(args) > 0 else ""
            b = args[1] if len(args) > 1 else ""
            fb = self._act_combine(a, b)
        elif action == "use":
            fb = self._act_use(args[0] if args else "")
        else:
            fb = {"success": False, "reward": -0.1,
                  "message": f"Unknown action '{action}'",
                  "state_vec": self.state_vector()}

        self._total_reward += fb.get("reward", 0)
        fb["action"]     = action
        fb["args"]       = list(args)
        fb["step"]       = self._step_count
        fb["state_vec"]  = self.state_vector()
        fb["health"]     = self._health
        fb["satiation"]  = self._satiation
        fb["object_name"] = args[0] if args else ""

        self._event_log.append(fb)
        return fb

    def _act_pick(self, name: str) -> ActionFeedback:
        obj = self._objects.get(name)
        if not obj:
            return {"success": False, "reward": -0.05,
                    "message": f"No object '{name}' exists."}
        if obj.broken:
            return {"success": False, "reward": -0.02,
                    "message": f"'{name}' is broken, can't pick it up."}
        if self._dist(obj.position) > 2:
            return {"success": False, "reward": -0.05,
                    "message": f"'{name}' is too far away."}
        if obj.is_held:
            return {"success": False, "reward": 0,
                    "message": f"Already holding '{name}'."}
        if len(self._inventory) >= self.MAX_INVENTORY:
            return {"success": False, "reward": -0.02,
                    "message": "Inventory full."}
        if obj.heavy:
            # Heavy objects can be picked but slow movement
            obj.is_held = True
            self._inventory.append(name)
            return {"success": True, "reward": 0.05,
                    "message": f"Picked up '{name}' (heavy — movement slowed).",
                    "object_props": obj.inspect(),
                    "object_category": obj.category}
        obj.is_held = True
        self._inventory.append(name)
        return {"success": True, "reward": 0.1,
                "message": f"Picked up '{name}'.",
                "object_props": {k: v for k, v in obj.inspect().items()
                                 if obj._inspected or k in ("name", "color", "size")},
                "object_category": obj.category}

    def _act_drop(self, name: str) -> ActionFeedback:
        if name not in self._inventory:
            return {"success": False, "reward": -0.02,
                    "message": f"'{name}' not in inventory."}
        obj = self._objects[name]
        obj.is_held = False
        obj.position = self._agent_pos

        # Fragile objects break when dropped
        if obj.fragile and not obj.broken:
            obj.broken = True
            obj.movable = False
            return {"success": True, "reward": -0.3,
                    "message": f"Dropped '{name}' — it shattered! [fragile]",
                    "consequence": "broken",
                    "object_category": obj.category}

        self._inventory.remove(name)
        return {"success": True, "reward": 0.0,
                "message": f"Dropped '{name}'.",
                "object_category": obj.category}

    def _act_move(self, x: int, y: int) -> ActionFeedback:
        x = int(np.clip(x, 0, self.GRID_SIZE - 1))
        y = int(np.clip(y, 0, self.GRID_SIZE - 1))
        # Heavy inventory slows movement
        heavy_items = [n for n in self._inventory if self._objects[n].heavy]
        dist = abs(x - self._agent_pos[0]) + abs(y - self._agent_pos[1])
        reward = -0.01 * dist - 0.01 * len(heavy_items)

        self._agent_pos = (x, y)
        visible = self.visible_names()
        msg = f"Moved to ({x},{y})."
        if visible:
            msg += f" Can see: {', '.join(visible)}."
        return {"success": True, "reward": reward,
                "message": msg, "visible": visible,
                "position": (x, y)}

    def _act_eat(self, name: str) -> ActionFeedback:
        if name not in self._inventory:
            return {"success": False, "reward": -0.05,
                    "message": f"'{name}' not in inventory."}
        obj = self._objects[name]

        # Unknown edibility — risky!
        if obj.edible is None:
            # Eating unknown → random outcome
            outcome = self._rng.random() > 0.5
            if outcome:
                self._satiation = min(1.0, self._satiation + 0.3)
                self._inventory.remove(name)
                del self._objects[name]
                obj.edible = True  # learned
                return {"success": True, "reward": 0.3,
                        "message": f"Ate '{name}' — turned out edible! (+satiation)",
                        "learned": f"{name} is edible",
                        "object_category": obj.category}
            else:
                self._health = max(0.0, self._health - 0.2)
                self._inventory.remove(name)
                del self._objects[name]
                obj.edible = False
                return {"success": False, "reward": -0.4,
                        "message": f"Ate '{name}' — poisonous! (-health)",
                        "learned": f"{name} is NOT edible",
                        "consequence": "poisoned",
                        "object_category": obj.category}

        if not obj.edible:
            self._health = max(0.0, self._health - 0.1)
            return {"success": False, "reward": -0.2,
                    "message": f"Cannot eat '{name}' — it's not edible! (-health)",
                    "object_category": obj.category}

        # Successfully eat edible object
        self._satiation = min(1.0, self._satiation + 0.4)
        self._inventory.remove(name)
        del self._objects[name]
        return {"success": True, "reward": 0.5,
                "message": f"Ate '{name}'. Delicious! (+satiation={self._satiation:.2f})",
                "object_category": obj.category}

    def _act_inspect(self, name: str) -> ActionFeedback:
        obj = self._objects.get(name)
        if not obj:
            # Search visible objects
            visible = self.visible_objects()
            found = [o for o in visible if name.lower() in o.name.lower()]
            if not found:
                return {"success": False, "reward": -0.01,
                        "message": f"Cannot find '{name}' nearby."}
            obj = found[0]

        if self._dist(obj.position) > 3 and not obj.is_held:
            return {"success": False, "reward": -0.01,
                    "message": f"'{name}' too far to inspect."}

        props = obj.inspect()
        return {"success": True, "reward": 0.05,
                "message": obj.full_desc(),
                "object": obj,
                "object_props": props,
                "object_category": obj.category,
                "revealed": True}

    def _act_observe(self) -> ActionFeedback:
        visible = self.visible_objects()
        held    = list(self._inventory)
        msgs    = [f"Position: {self._agent_pos}.",
                   f"Health: {self._health:.2f}, Satiation: {self._satiation:.2f}."]
        if held:
            msgs.append(f"Holding: {', '.join(held)}.")
        if visible:
            vis_desc = [o.observable_desc() for o in visible]
            msgs.append(f"Visible ({len(visible)}): {'; '.join(vis_desc[:4])}.")
        else:
            msgs.append("Nothing visible nearby.")

        unknown = [o for o in visible if o.category == "unknown" or o.edible is None]
        if unknown:
            msgs.append(f"Unknown objects: {', '.join(o.name for o in unknown)}.")

        return {"success": True, "reward": 0.0,
                "message": " ".join(msgs),
                "visible_objects": [o.name for o in visible],
                "unknown_objects": [o.name for o in unknown],
                "inventory": held,
                "health": self._health,
                "satiation": self._satiation}

    def _act_combine(self, a: str, b: str) -> ActionFeedback:
        if a not in self._inventory or b not in self._inventory:
            return {"success": False, "reward": -0.02,
                    "message": f"Both '{a}' and '{b}' must be in inventory."}
        oa, ob = self._objects[a], self._objects[b]
        combo_name = f"{a}_{b}_mix"
        edible_combo = oa.edible and ob.edible
        combo = RichObject(
            name=combo_name, category="combined",
            color="mixed", size="medium",
            position=self._agent_pos, is_held=True,
            edible=edible_combo, fragile=oa.fragile or ob.fragile,
            heavy=oa.heavy or ob.heavy,
        )
        combo._inspected = True
        self._inventory.remove(a); self._inventory.remove(b)
        del self._objects[a]; del self._objects[b]
        self._objects[combo_name] = combo
        self._inventory.append(combo_name)
        return {"success": True, "reward": 0.2,
                "message": f"Combined '{a}' + '{b}' → '{combo_name}' (edible={edible_combo}).",
                "object_category": "combined"}

    def _act_use(self, name: str) -> ActionFeedback:
        if name not in self._inventory:
            return {"success": False, "reward": -0.02,
                    "message": f"'{name}' not in inventory."}
        obj = self._objects[name]
        if obj.category == "tool":
            return {"success": True, "reward": 0.1,
                    "message": f"Used '{name}' — tool applied.",
                    "object_category": obj.category}
        return {"success": False, "reward": -0.02,
                "message": f"'{name}' is not a tool.",
                "object_category": obj.category}

    # ------------------------------------------------------------------
    # State vector (richer than SimpleEnv)
    # ------------------------------------------------------------------

    def state_vector(self) -> np.ndarray:
        dim = 64
        v = np.zeros(dim, dtype=np.float32)
        G = self.GRID_SIZE

        # Agent state
        v[0] = self._agent_pos[0] / G
        v[1] = self._agent_pos[1] / G
        v[2] = self._health
        v[3] = self._satiation

        # Inventory (up to 3 items)
        for i, name in enumerate(self._inventory[:3]):
            obj = self._objects.get(name)
            if obj:
                base = 4 + i * 8
                v[base]   = float(obj.edible or False)
                v[base+1] = float(obj.fragile)
                v[base+2] = float(obj.heavy)
                v[base+3] = float(obj._inspected)
                v[base+4] = hash(obj.category) % 100 / 100.0

        # Visible nearby objects (up to 4)
        vis = self.visible_objects()
        for i, obj in enumerate(vis[:4]):
            base = 28 + i * 8
            v[base]   = obj.position[0] / G
            v[base+1] = obj.position[1] / G
            v[base+2] = float(obj.edible or False)
            v[base+3] = float(obj.fragile)
            v[base+4] = float(obj.category == "unknown")
            v[base+5] = float(obj._inspected)

        # Global stats
        v[60] = len(self._inventory) / self.MAX_INVENTORY
        v[61] = len(self._objects) / len(RICH_CATALOGUE)
        v[62] = min(1.0, self._step_count / 200.0)
        v[63] = min(1.0, max(-1.0, self._total_reward)) / 2.0 + 0.5

        n = np.linalg.norm(v)
        return v / (n + 1e-8)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._objects = {}
        self._inventory = []
        self._agent_pos = (5, 5)
        self._step_count = 0
        self._total_reward = 0.0
        self._event_log = []
        self._health = 1.0
        self._satiation = 0.5
        self._init_objects(RICH_CATALOGUE)

    @property
    def inventory(self) -> List[str]:
        return list(self._inventory)

    @property
    def object_names(self) -> List[str]:
        return list(self._objects.keys())

    def get_object(self, name: str) -> Optional[RichObject]:
        return self._objects.get(name)

    def available_objects(self) -> List[Tuple[str, str]]:
        """Return (name, category) for all visible + held objects."""
        result = [(n, self._objects[n].category) for n in self._inventory
                  if n in self._objects]
        for obj in self.visible_objects():
            if obj.name not in self._inventory:
                result.append((obj.name, obj.category))
        return result

    def stats(self) -> Dict:
        return {
            "steps":        self._step_count,
            "total_reward": round(self._total_reward, 3),
            "health":       round(self._health, 3),
            "satiation":    round(self._satiation, 3),
            "inventory":    self.inventory,
            "n_objects":    len(self._objects),
        }

    def __repr__(self) -> str:
        return (f"RichEnv(pos={self._agent_pos}, "
                f"health={self._health:.2f}, sat={self._satiation:.2f}, "
                f"inv={self.inventory})")
