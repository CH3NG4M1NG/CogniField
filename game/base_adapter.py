"""
game/base_adapter.py
======================
Base Game Adapter Interface

Defines the contract every game adapter must satisfy.
All adapters return the same canonical observation so the
CogniField cognition layer never needs to know which game is running.

Observation schema (always returned, missing fields set to None / []):
  {
    "visible_blocks": [{"id": "minecraft:apple", "pos": (x,y,z)}, ...],
    "entities":       [{"type": "pig", "pos": (x,y,z), "health": 20}],
    "inventory":      [{"id": "minecraft:apple", "count": 3, "slot": 0}],
    "health":         float,      # 0–20 (Minecraft hearts × 2)
    "hunger":         float,      # 0–20
    "position":       (x, y, z),  # float 3-tuple
    "biome":          str,
    "time_of_day":    int,         # 0–24000 ticks
    "dimension":      str,         # "overworld" / "nether" / "end"
    "on_ground":      bool,
    "source":         str,         # adapter identifier
    "timestamp":      float,
    "partial":        bool,        # True if some fields are unavailable
    "missing_fields": [str],       # list of unavailable field names
  }
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    MOVE      = "move"
    LOOK      = "look"
    ATTACK    = "attack"
    USE       = "use"
    BREAK     = "break"
    DROP      = "drop"
    SNEAK     = "sneak"
    SPRINT    = "sprint"
    JUMP      = "jump"
    INVENTORY = "inventory"
    CHAT      = "chat"
    COMMAND   = "command"


@dataclass
class BlockInfo:
    block_id: str
    pos:      Tuple[int, int, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.block_id.split(":")[-1].replace("_", " ")


@dataclass
class EntityInfo:
    entity_type: str
    pos:         Tuple[float, float, float]
    health:      float = 20.0
    hostile:     bool  = False
    metadata:    Dict[str, Any] = field(default_factory=dict)

    @property
    def is_food_source(self) -> bool:
        return self.entity_type in {
            "pig", "cow", "sheep", "chicken", "rabbit",
            "salmon", "cod", "tropical_fish"
        }

    @property
    def is_hostile(self) -> bool:
        return self.hostile or self.entity_type in {
            "zombie", "skeleton", "creeper", "spider", "enderman",
            "witch", "pillager", "phantom"
        }


@dataclass
class InventoryItem:
    item_id:  str
    count:    int
    slot:     int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.item_id.split(":")[-1].replace("_", " ")


@dataclass
class GameObservation:
    """
    Canonical game observation — adapter-agnostic.
    Always use this type when passing observations to the cognition layer.
    """
    visible_blocks: List[BlockInfo]          = field(default_factory=list)
    entities:       List[EntityInfo]         = field(default_factory=list)
    inventory:      List[InventoryItem]      = field(default_factory=list)
    health:         float                    = 20.0
    hunger:         float                    = 20.0
    position:       Tuple[float, float, float] = (0.0, 64.0, 0.0)
    biome:          str                      = "plains"
    time_of_day:    int                      = 6000
    dimension:      str                      = "overworld"
    on_ground:      bool                     = True
    source:         str                      = "unknown"
    timestamp:      float                    = field(default_factory=time.time)
    partial:        bool                     = False
    missing_fields: List[str]                = field(default_factory=list)

    @property
    def health_pct(self) -> float:
        return self.health / 20.0

    @property
    def hunger_pct(self) -> float:
        return self.hunger / 20.0

    @property
    def is_hungry(self) -> bool:
        return self.hunger < 14.0

    @property
    def is_in_danger(self) -> bool:
        return (self.health < 8.0
                or any(e.is_hostile for e in self.entities))

    @property
    def visible_food(self) -> List[BlockInfo]:
        FOOD_IDS = {
            "minecraft:apple", "minecraft:bread", "minecraft:carrot",
            "minecraft:potato", "minecraft:beetroot", "minecraft:melon_slice",
            "minecraft:sweet_berries", "minecraft:glow_berries",
            "minecraft:mushroom_stew", "minecraft:pumpkin_pie",
        }
        return [b for b in self.visible_blocks if b.block_id in FOOD_IDS]

    @property
    def hostile_entities(self) -> List[EntityInfo]:
        return [e for e in self.entities if e.is_hostile]

    def inventory_count(self, item_id: str) -> int:
        return sum(i.count for i in self.inventory if i.item_id == item_id)

    def has_item(self, item_id: str) -> bool:
        return self.inventory_count(item_id) > 0

    def to_dict(self) -> Dict:
        return {
            "visible_blocks": [{"id": b.block_id, "pos": b.pos}
                                for b in self.visible_blocks[:10]],
            "entities":       [{"type": e.entity_type, "health": e.health,
                                 "hostile": e.hostile}
                                for e in self.entities[:10]],
            "inventory":      [{"id": i.item_id, "count": i.count}
                                for i in self.inventory[:20]],
            "health":         self.health,
            "hunger":         self.hunger,
            "position":       self.position,
            "biome":          self.biome,
            "time_of_day":    self.time_of_day,
            "dimension":      self.dimension,
            "on_ground":      self.on_ground,
            "source":         self.source,
            "partial":        self.partial,
            "missing_fields": self.missing_fields,
        }


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class GameAdapter(ABC):
    """Abstract base class for all game adapters."""

    def __init__(self, name: str) -> None:
        self.name         = name
        self.connected    = False
        self._step_count  = 0
        self._last_obs:   Optional[GameObservation] = None
        self._action_log: List[Dict] = []

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection. Returns True on success."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection cleanly."""

    @abstractmethod
    def get_observation(self) -> GameObservation:
        """Read current game state. Never raises."""

    @abstractmethod
    def send_action(self, action: Dict) -> bool:
        """Send action. Returns True if delivered."""

    # Convenience helpers
    def get_inventory(self) -> List[InventoryItem]:
        return self.get_observation().inventory

    def get_position(self) -> Tuple[float, float, float]:
        return self.get_observation().position

    def get_health(self) -> float:
        return self.get_observation().health

    def get_hunger(self) -> float:
        return self.get_observation().hunger

    def is_alive(self) -> bool:
        return self.get_health() > 0.0

    def step(self) -> GameObservation:
        self._step_count += 1
        obs = self.get_observation()
        self._last_obs = obs
        return obs

    def last_observation(self) -> Optional[GameObservation]:
        return self._last_obs

    def log_action(self, action: Dict, success: bool) -> None:
        self._action_log.append({
            "step": self._step_count, "action": action,
            "success": success, "timestamp": time.time(),
        })

    def summary(self) -> Dict:
        return {
            "adapter":      self.name,
            "connected":    self.connected,
            "steps":        self._step_count,
            "actions_sent": len(self._action_log),
        }

    def __repr__(self) -> str:
        status = "connected" if self.connected else "disconnected"
        return f"{self.__class__.__name__}(name={self.name!r}, {status})"


# ---------------------------------------------------------------------------
# Null adapter — safe no-op
# ---------------------------------------------------------------------------

class NullAdapter(GameAdapter):
    """Do-nothing adapter for testing."""

    def __init__(self) -> None:
        super().__init__("null")

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False

    def get_observation(self) -> GameObservation:
        return GameObservation(source="null")

    def send_action(self, action: Dict) -> bool:
        self.log_action(action, True)
        return True
