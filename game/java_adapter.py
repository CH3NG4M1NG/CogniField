"""
game/java_adapter.py
======================
Java Edition Adapter

Simulates integration with a Fabric/Forge mod that exposes a local
JSON-over-socket API. In simulation mode (no real Minecraft running)
the adapter generates deterministic observations from an internal
virtual world state — fully testable offline.

Real integration path (when Minecraft IS running):
  1. Install the CogniField Fabric mod
  2. Mod opens a local TCP socket on 127.0.0.1:25566
  3. This adapter sends JSON requests and reads JSON responses

Simulation mode (default, no Minecraft required):
  - Generates a procedural world with biomes, blocks, entities
  - Actions mutate internal state (eating reduces hunger, etc.)
  - Health/hunger/position all respond realistically
"""

from __future__ import annotations

import json
import math
import random
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

from .base_adapter import (
    GameAdapter, GameObservation, BlockInfo, EntityInfo, InventoryItem,
    ActionType,
)


# ---------------------------------------------------------------------------
# Simulated world
# ---------------------------------------------------------------------------

class _JavaWorld:
    """
    Minimal deterministic world simulation for offline testing.
    Tracks player state and nearby objects procedurally.
    """

    BIOMES = ["plains", "forest", "taiga", "desert", "jungle",
              "swamp", "savanna", "mountains"]

    BLOCKS_BY_BIOME = {
        "plains":    ["minecraft:grass_block", "minecraft:short_grass",
                      "minecraft:dandelion", "minecraft:poppy"],
        "forest":    ["minecraft:oak_log", "minecraft:oak_leaves",
                      "minecraft:apple", "minecraft:mushroom_stew"],
        "taiga":     ["minecraft:spruce_log", "minecraft:spruce_leaves",
                      "minecraft:sweet_berries", "minecraft:snow"],
        "desert":    ["minecraft:sand", "minecraft:sandstone",
                      "minecraft:dead_bush", "minecraft:cactus"],
        "jungle":    ["minecraft:jungle_log", "minecraft:jungle_leaves",
                      "minecraft:cocoa", "minecraft:glow_berries"],
        "swamp":     ["minecraft:mud", "minecraft:lily_pad",
                      "minecraft:brown_mushroom", "minecraft:vine"],
        "savanna":   ["minecraft:acacia_log", "minecraft:tall_grass",
                      "minecraft:grass_block", "minecraft:carrot"],
        "mountains": ["minecraft:stone", "minecraft:gravel",
                      "minecraft:coal_ore", "minecraft:iron_ore"],
    }

    ENTITIES_BY_BIOME = {
        "plains":    [("pig", False), ("cow", False), ("sheep", False)],
        "forest":    [("wolf", False), ("rabbit", False), ("spider", True)],
        "taiga":     [("wolf", False), ("fox", False), ("skeleton", True)],
        "desert":    [("zombie", True), ("husk", True)],
        "jungle":    [("ocelot", False), ("parrot", False), ("creeper", True)],
        "swamp":     [("witch", True), ("slime", True)],
        "savanna":   [("horse", False), ("llama", False)],
        "mountains": [("goat", False), ("skeleton", True)],
    }

    FOOD_ITEMS = [
        "minecraft:apple", "minecraft:bread", "minecraft:carrot",
        "minecraft:potato", "minecraft:cooked_porkchop",
        "minecraft:sweet_berries", "minecraft:glow_berries",
    ]

    def __init__(self, seed: int = 42) -> None:
        self._rng   = random.Random(seed)
        self.health = 20.0
        self.hunger = 20.0
        self.pos    = (0.0, 64.0, 0.0)
        self.biome  = "plains"
        self.time   = 6000
        self.on_ground = True
        self.dim    = "overworld"
        self.inventory: List[InventoryItem] = [
            InventoryItem("minecraft:bread", 3, 0),
            InventoryItem("minecraft:wooden_sword", 1, 1),
        ]
        self._tick  = 0

    def tick(self) -> None:
        """Advance world state by one game tick."""
        self._tick += 1
        self.time   = (self.time + 10) % 24000
        # Passive hunger drain
        self.hunger = max(0.0, self.hunger - 0.02)
        # Health regeneration when hunger >= 18
        if self.hunger >= 18.0 and self.health < 20.0:
            self.health = min(20.0, self.health + 0.1)
        # Starvation damage
        if self.hunger == 0.0:
            self.health = max(0.0, self.health - 0.1)

    def get_visible_blocks(self) -> List[BlockInfo]:
        blocks = []
        pool   = self.BLOCKS_BY_BIOME.get(self.biome, [])
        n      = self._rng.randint(3, 8)
        for i in range(min(n, len(pool))):
            dx = self._rng.randint(-5, 5)
            dz = self._rng.randint(-5, 5)
            x, y, z = self.pos
            blocks.append(BlockInfo(
                block_id=pool[i],
                pos=(int(x)+dx, int(y), int(z)+dz),
            ))
        return blocks

    def get_entities(self) -> List[EntityInfo]:
        entities = []
        pool = self.ENTITIES_BY_BIOME.get(self.biome, [])
        n    = self._rng.randint(0, 3)
        x, y, z = self.pos
        for i in range(min(n, len(pool))):
            etype, hostile = pool[i]
            dx = self._rng.randint(-10, 10)
            dz = self._rng.randint(-10, 10)
            entities.append(EntityInfo(
                entity_type=etype,
                pos=(x+dx, y, z+dz),
                health=float(self._rng.randint(5, 20)),
                hostile=hostile,
            ))
        return entities

    def eat_item(self, item_id: str) -> Tuple[bool, float]:
        """Consume an inventory item. Returns (success, hunger_restored)."""
        hunger_values = {
            "minecraft:apple":           4.0,
            "minecraft:bread":           5.0,
            "minecraft:carrot":          3.0,
            "minecraft:potato":          1.0,
            "minecraft:cooked_porkchop": 8.0,
            "minecraft:sweet_berries":   2.0,
            "minecraft:glow_berries":    2.0,
            "minecraft:mushroom_stew":   6.0,
            "minecraft:cooked_beef":     8.0,
            "minecraft:cooked_chicken":  6.0,
        }
        for item in list(self.inventory):
            if item.item_id == item_id:
                restore = hunger_values.get(item_id, 0.0)
                if restore > 0:
                    item.count -= 1
                    if item.count <= 0:
                        self.inventory.remove(item)
                    self.hunger = min(20.0, self.hunger + restore)
                    return True, restore
                return False, 0.0
        return False, 0.0

    def move(self, dx: float, dz: float) -> None:
        x, y, z = self.pos
        self.pos = (x + dx, y, z + dz)
        self.hunger = max(0.0, self.hunger - 0.05)
        # Biome changes as player moves (simple grid)
        biome_idx = (abs(int(self.pos[0])) // 50 +
                     abs(int(self.pos[2])) // 50) % len(self.BIOMES)
        self.biome = self.BIOMES[biome_idx]

    def take_damage(self, amount: float) -> None:
        self.health = max(0.0, self.health - amount)

    def pick_up_item(self, item_id: str) -> bool:
        """Add item to inventory (simulate picking up from ground)."""
        for item in self.inventory:
            if item.item_id == item_id:
                item.count += 1
                return True
        if len(self.inventory) < 36:
            slot = max((i.slot for i in self.inventory), default=-1) + 1
            self.inventory.append(InventoryItem(item_id, 1, slot))
            return True
        return False


# ---------------------------------------------------------------------------
# Java Adapter
# ---------------------------------------------------------------------------

class JavaAdapter(GameAdapter):
    """
    Minecraft Java Edition adapter.

    Parameters
    ----------
    host          : Mod server host (default 127.0.0.1)
    port          : Mod server port (default 25566)
    simulation    : If True, use built-in world simulation (no Minecraft needed)
    seed          : Simulation RNG seed
    timeout_sec   : Socket timeout in seconds
    """

    def __init__(
        self,
        host:        str  = "127.0.0.1",
        port:        int  = 25566,
        simulation:  bool = True,
        seed:        int  = 42,
        timeout_sec: float = 2.0,
    ) -> None:
        super().__init__("java_edition")
        self.host        = host
        self.port        = port
        self.simulation  = simulation
        self.timeout     = timeout_sec
        self._sim_world  = _JavaWorld(seed=seed) if simulation else None
        self._socket:    Optional[socket.socket] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        if self.simulation:
            self.connected = True
            return True
        # Real socket connection
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.timeout)
            self._socket.connect((self.host, self.port))
            self.connected = True
            return True
        except (socket.error, OSError):
            self.connected = False
            return False

    def disconnect(self) -> None:
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        self.connected = False

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self) -> GameObservation:
        if self.simulation:
            return self._sim_observation()
        return self._real_observation()

    def _sim_observation(self) -> GameObservation:
        """Build observation from simulation world."""
        w = self._sim_world
        w.tick()
        return GameObservation(
            visible_blocks = w.get_visible_blocks(),
            entities       = w.get_entities(),
            inventory      = list(w.inventory),
            health         = w.health,
            hunger         = w.hunger,
            position       = w.pos,
            biome          = w.biome,
            time_of_day    = w.time,
            dimension      = w.dim,
            on_ground      = w.on_ground,
            source         = "java_simulation",
            partial        = False,
            missing_fields = [],
        )

    def _real_observation(self) -> GameObservation:
        """Request observation from real Minecraft mod via socket."""
        if not self._socket:
            return GameObservation(source="java_disconnected", partial=True,
                                   missing_fields=["all"])
        try:
            request = json.dumps({"type": "get_observation"}) + "\n"
            self._socket.sendall(request.encode())
            raw = b""
            while b"\n" not in raw:
                chunk = self._socket.recv(4096)
                if not chunk:
                    break
                raw += chunk
            data = json.loads(raw.decode().strip())
            return self._parse_json_observation(data)
        except Exception:
            return GameObservation(source="java_error", partial=True,
                                   missing_fields=["all"])

    @staticmethod
    def _parse_json_observation(data: Dict) -> GameObservation:
        blocks   = [BlockInfo(b["id"], tuple(b["pos"]))
                    for b in data.get("visible_blocks", [])]
        entities = [EntityInfo(e["type"], tuple(e["pos"]),
                               e.get("health", 20), e.get("hostile", False))
                    for e in data.get("entities", [])]
        inv      = [InventoryItem(i["id"], i["count"], i["slot"])
                    for i in data.get("inventory", [])]
        return GameObservation(
            visible_blocks=blocks, entities=entities, inventory=inv,
            health=float(data.get("health", 20)),
            hunger=float(data.get("hunger", 20)),
            position=tuple(data.get("position", (0, 64, 0))),
            biome=data.get("biome", "plains"),
            time_of_day=int(data.get("time_of_day", 6000)),
            dimension=data.get("dimension", "overworld"),
            on_ground=bool(data.get("on_ground", True)),
            source="java_real",
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def send_action(self, action: Dict) -> bool:
        if self.simulation:
            return self._sim_action(action)
        return self._real_action(action)

    def _sim_action(self, action: Dict) -> bool:
        """Apply action to the simulation world."""
        w    = self._sim_world
        atype = action.get("type", "")
        ok   = True

        if atype == ActionType.MOVE:
            direction = action.get("direction", "forward")
            speed     = float(action.get("speed", 1.0))
            dx_dz     = {
                "forward": (0.0, speed),  "back": (0.0, -speed),
                "left": (-speed, 0.0),    "right": (speed, 0.0),
                "north": (0.0, -speed),   "south": (0.0, speed),
                "east":  (speed, 0.0),    "west":  (-speed, 0.0),
            }
            dx, dz = dx_dz.get(direction, (0.0, 0.0))
            w.move(dx, dz)

        elif atype == ActionType.USE:
            target = action.get("target", "")
            if target:
                ok = w.eat_item(target) if self._is_food(target) else False
            # Using a tool / block
            if not ok:
                ok = w.pick_up_item(target)

        elif atype == ActionType.ATTACK:
            # Simulate combat damage taken
            w.take_damage(float(action.get("damage_taken", 2.0)))
            ok = True

        elif atype == ActionType.DROP:
            item_id = action.get("item", "")
            for item in list(w.inventory):
                if item.item_id == item_id:
                    item.count -= 1
                    if item.count <= 0:
                        w.inventory.remove(item)
                    ok = True
                    break

        elif atype == ActionType.JUMP:
            x, y, z = w.pos
            w.pos = (x, y + 1.0, z)
            w.hunger = max(0.0, w.hunger - 0.02)

        self.log_action(action, ok)
        return ok

    def _real_action(self, action: Dict) -> bool:
        if not self._socket:
            return False
        try:
            request = json.dumps(action) + "\n"
            self._socket.sendall(request.encode())
            raw = self._socket.recv(1024)
            resp = json.loads(raw.decode().strip())
            success = resp.get("success", False)
            self.log_action(action, success)
            return success
        except Exception:
            return False

    @staticmethod
    def _is_food(item_id: str) -> bool:
        return (item_id.startswith("minecraft:") and
                any(f in item_id for f in
                    ["apple", "bread", "carrot", "potato", "beef",
                     "pork", "chicken", "fish", "berry", "melon",
                     "stew", "soup", "cake", "cookie"]))

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def move(self, direction: str, steps: int = 1) -> bool:
        """Move in a compass direction N steps."""
        ok = True
        for _ in range(steps):
            ok = self.send_action({"type": ActionType.MOVE,
                                   "direction": direction, "speed": 1.0})
        return ok

    def eat(self, item_id: str) -> bool:
        """Eat an item from inventory."""
        result = self._sim_action({"type": ActionType.USE, "target": item_id})                  if self.simulation else self._real_action({"type": ActionType.USE, "target": item_id})
        return bool(result)

    def eat_food(self) -> bool:
        """Eat the first available food item from inventory."""
        obs = self.get_observation()
        for item in obs.inventory:
            if self._is_food(item.item_id):
                return self.eat(item.item_id)
        return False

    def sim_spawn_item(self, item_id: str, count: int = 1) -> None:
        """Simulation only: add an item to inventory."""
        if self.simulation and self._sim_world:
            for _ in range(count):
                self._sim_world.pick_up_item(item_id)

    def sim_set_health(self, health: float) -> None:
        if self.simulation and self._sim_world:
            self._sim_world.health = max(0.0, min(20.0, health))

    def sim_set_hunger(self, hunger: float) -> None:
        if self.simulation and self._sim_world:
            self._sim_world.hunger = max(0.0, min(20.0, hunger))

    def summary(self) -> Dict:
        base = super().summary()
        if self.simulation and self._sim_world:
            w = self._sim_world
            base["sim_health"]   = round(w.health, 1)
            base["sim_hunger"]   = round(w.hunger, 1)
            base["sim_position"] = w.pos
            base["sim_biome"]    = w.biome
        base["mode"] = "simulation" if self.simulation else "real"
        return base
