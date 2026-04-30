"""
game/bedrock_adapter.py
========================
Bedrock Edition Adapter

Bedrock provides fewer programmatic hooks than Java.
This adapter exposes what IS available while clearly marking gaps.

Available via WebSocket GameTest API (Bedrock 1.18+):
  - /wsserver ws://127.0.0.1:19131
  - Receives typed event packets from Minecraft

Limited data (vs Java):
  ✓ health
  ✓ hunger (food level)
  ✓ player position
  ✓ dimension
  ✗ visible_blocks   → placeholder list, not real
  ✗ entity list      → partial (only nearby mobs via event stream)
  ✓ inventory        → available via GameTest API
  ✗ biome            → not exposed directly

All missing fields are clearly listed in observation.missing_fields
and observation.partial = True.
"""

from __future__ import annotations

import json
import socket
import time
import random
from typing import Any, Dict, List, Optional, Tuple

from .base_adapter import (
    GameAdapter, GameObservation, BlockInfo, EntityInfo, InventoryItem,
    ActionType,
)


# ---------------------------------------------------------------------------
# Simulation world for Bedrock (simpler than Java)
# ---------------------------------------------------------------------------

class _BedrockWorld:
    """Minimal Bedrock simulation — health/hunger/position only."""

    def __init__(self, seed: int = 42) -> None:
        self._rng    = random.Random(seed)
        self.health  = 20.0
        self.hunger  = 20.0
        self.pos     = (0.0, 64.0, 0.0)
        self.dim     = "overworld"
        self.inventory: List[InventoryItem] = [
            InventoryItem("minecraft:bread", 2, 0),
        ]
        self._tick   = 0

    def tick(self) -> None:
        self._tick += 1
        self.hunger = max(0.0, self.hunger - 0.03)
        if self.hunger >= 18.0 and self.health < 20.0:
            self.health = min(20.0, self.health + 0.05)
        if self.hunger == 0.0:
            self.health = max(0.0, self.health - 0.1)

    def eat(self, item_id: str) -> bool:
        RESTORE = {
            "minecraft:bread": 5.0, "minecraft:apple": 4.0,
            "minecraft:carrot": 3.0, "minecraft:cooked_beef": 8.0,
        }
        for item in list(self.inventory):
            if item.item_id == item_id:
                restore = RESTORE.get(item_id, 2.0)
                self.hunger = min(20.0, self.hunger + restore)
                item.count -= 1
                if item.count <= 0:
                    self.inventory.remove(item)
                return True
        return False

    def move(self, dx: float, dz: float) -> None:
        x, y, z = self.pos
        self.pos = (x+dx, y, z+dz)
        self.hunger = max(0.0, self.hunger - 0.04)


# ---------------------------------------------------------------------------
# Bedrock Adapter
# ---------------------------------------------------------------------------

class BedrockAdapter(GameAdapter):
    """
    Minecraft Bedrock Edition adapter.

    Uses the WebSocket GameTest API when available.
    Falls back to simulation mode when no game is running.

    Parameters
    ----------
    host       : WebSocket host (default 127.0.0.1)
    port       : WebSocket port (default 19131)
    simulation : Use offline simulation (default True)
    seed       : Simulation seed
    """

    # Fields not available from Bedrock API
    MISSING = ["visible_blocks", "biome", "time_of_day"]

    def __init__(
        self,
        host:       str  = "127.0.0.1",
        port:       int  = 19131,
        simulation: bool = True,
        seed:       int  = 42,
    ) -> None:
        super().__init__("bedrock_edition")
        self.host       = host
        self.port       = port
        self.simulation = simulation
        self._world     = _BedrockWorld(seed=seed) if simulation else None
        self._ws_socket: Optional[socket.socket] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        if self.simulation:
            self.connected = True
            return True
        try:
            self._ws_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._ws_socket.settimeout(2.0)
            self._ws_socket.connect((self.host, self.port))
            self.connected = True
            return True
        except (socket.error, OSError):
            self.connected = False
            return False

    def disconnect(self) -> None:
        if self._ws_socket:
            try:
                self._ws_socket.close()
            except Exception:
                pass
        self.connected = False

    # ------------------------------------------------------------------
    # Observation (partial — clearly marked)
    # ------------------------------------------------------------------

    def get_observation(self) -> GameObservation:
        if self.simulation:
            return self._sim_observation()
        return self._real_observation()

    def _sim_observation(self) -> GameObservation:
        w = self._world
        w.tick()
        return GameObservation(
            visible_blocks = [],          # NOT available on Bedrock
            entities       = [],          # partial
            inventory      = list(w.inventory),
            health         = w.health,
            hunger         = w.hunger,
            position       = w.pos,
            biome          = "unknown",   # NOT available on Bedrock
            time_of_day    = 6000,        # NOT available
            dimension      = w.dim,
            on_ground      = True,
            source         = "bedrock_simulation",
            partial        = True,
            missing_fields = list(self.MISSING),
        )

    def _real_observation(self) -> GameObservation:
        """
        In a real implementation this would parse WebSocket event packets.
        Bedrock sends asynchronous events; we'd buffer them here.
        Returns partial observation with what IS available.
        """
        return GameObservation(
            source         = "bedrock_real",
            partial        = True,
            missing_fields = list(self.MISSING),
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def send_action(self, action: Dict) -> bool:
        if self.simulation:
            return self._sim_action(action)
        # Real: send /wsserver command via WebSocket
        return self._ws_action(action)

    def _sim_action(self, action: Dict) -> bool:
        w     = self._world
        atype = action.get("type", "")
        ok    = False

        if atype == ActionType.MOVE:
            direction = action.get("direction", "forward")
            speed     = float(action.get("speed", 1.0))
            dx_dz = {"forward":(0,speed),"back":(0,-speed),
                     "left":(-speed,0),"right":(speed,0),
                     "north":(0,-speed),"south":(0,speed),
                     "east":(speed,0),"west":(-speed,0)}
            dx, dz = dx_dz.get(direction, (0, 0))
            w.move(dx, dz)
            ok = True

        elif atype == ActionType.USE:
            target = action.get("target", "")
            ok = w.eat(target)

        elif atype in (ActionType.JUMP, ActionType.SNEAK):
            ok = True   # no-op in simulation

        elif atype == ActionType.COMMAND:
            # Bedrock supports slash commands via in-game chat
            ok = True

        self.log_action(action, ok)
        return ok

    def _ws_action(self, action: Dict) -> bool:
        """Convert action to Bedrock command string and send via WS."""
        cmd = self._action_to_command(action)
        if not cmd or not self._ws_socket:
            return False
        try:
            self._ws_socket.sendall((cmd + "\n").encode())
            return True
        except Exception:
            return False

    @staticmethod
    def _action_to_command(action: Dict) -> str:
        """Translate action dict to a Bedrock slash command."""
        atype = action.get("type", "")
        if atype == ActionType.COMMAND:
            return action.get("command", "")
        if atype == ActionType.CHAT:
            return f"/say {action.get('message', '')}"
        if atype == ActionType.USE:
            item = action.get("target", "")
            return f"/replaceitem entity @s slot.hotbar 0 {item} 1"
        return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def sim_set_health(self, v: float) -> None:
        if self._world:
            self._world.health = max(0.0, min(20.0, v))

    def sim_set_hunger(self, v: float) -> None:
        if self._world:
            self._world.hunger = max(0.0, min(20.0, v))

    def sim_add_item(self, item_id: str, count: int = 1) -> None:
        if self._world:
            self._world.inventory.append(
                InventoryItem(item_id, count, len(self._world.inventory))
            )

    def summary(self) -> Dict:
        base = super().summary()
        base["partial_observation"] = True
        base["missing_fields"] = self.MISSING
        base["mode"] = "simulation" if self.simulation else "real"
        return base
