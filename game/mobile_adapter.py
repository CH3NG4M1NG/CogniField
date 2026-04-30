"""
game/mobile_adapter.py
=======================
Mobile (Android/iOS) Adapter — ADB-based

Controls Minecraft Pocket Edition / Bedrock Mobile via ADB
(Android Debug Bridge).

ADB must be installed and device connected over USB or WiFi.

Architecture:
  1. send_action() translates game actions to ADB shell commands
     (tap, swipe, keyevent)
  2. get_observation() uses placeholder stub; full vision-based
     observation requires the VisionSystem (see vision/vision_system.py)

ADB command mapping:
  MOVE forward  → swipe (hold up joystick)
  LOOK          → swipe (rotate camera)
  JUMP          → tap jump button
  ATTACK        → tap attack button (long press for break)
  USE           → tap use/interact button
  INVENTORY     → keyevent 82 (menu key)

Screen coordinates are for a 1080×1920 Pixel layout.
Override COORDS for other device resolutions.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .base_adapter import (
    GameAdapter, GameObservation, BlockInfo, EntityInfo, InventoryItem,
    ActionType,
)


# ---------------------------------------------------------------------------
# Screen layout (1080×1920 portrait, Minecraft PE default UI)
# ---------------------------------------------------------------------------

@dataclass
class ScreenLayout:
    """Button coordinates for a specific screen resolution."""
    resolution:   Tuple[int, int]
    jump_btn:     Tuple[int, int]
    attack_btn:   Tuple[int, int]
    use_btn:      Tuple[int, int]
    inventory_btn:Tuple[int, int]
    joystick_center: Tuple[int, int]   # left thumb-stick centre
    camera_center:   Tuple[int, int]   # right camera area centre


LAYOUTS = {
    "1080x1920": ScreenLayout(
        resolution=(1080, 1920),
        jump_btn=(950, 1450),
        attack_btn=(100, 1350),
        use_btn=(950, 1250),
        inventory_btn=(540, 1800),
        joystick_center=(200, 1600),
        camera_center=(720, 960),
    ),
    "720x1280": ScreenLayout(
        resolution=(720, 1280),
        jump_btn=(640, 970),
        attack_btn=(80, 900),
        use_btn=(640, 840),
        inventory_btn=(360, 1200),
        joystick_center=(135, 1080),
        camera_center=(480, 640),
    ),
    "1440x2960": ScreenLayout(
        resolution=(1440, 2960),
        jump_btn=(1280, 2200),
        attack_btn=(160, 2000),
        use_btn=(1280, 1900),
        inventory_btn=(720, 2800),
        joystick_center=(260, 2500),
        camera_center=(960, 1480),
    ),
}


# ---------------------------------------------------------------------------
# ADB wrapper
# ---------------------------------------------------------------------------

class _ADB:
    """Thin wrapper around ADB shell commands."""

    def __init__(self, device_id: Optional[str] = None, dry_run: bool = True) -> None:
        self.device_id = device_id
        self.dry_run   = dry_run   # if True: print commands, don't run
        self._log:     List[str] = []

    def _run(self, cmd: str) -> bool:
        full = f"adb {'-s ' + self.device_id + ' ' if self.device_id else ''}{cmd}"
        self._log.append(full)
        if self.dry_run:
            return True   # pretend it worked
        try:
            result = subprocess.run(
                full.split(), capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def tap(self, x: int, y: int) -> bool:
        return self._run(f"shell input tap {x} {y}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> bool:
        return self._run(f"shell input swipe {x1} {y1} {x2} {y2} {duration_ms}")

    def keyevent(self, keycode: int) -> bool:
        return self._run(f"shell input keyevent {keycode}")

    def screencap(self, path: str = "/sdcard/cf_screen.png") -> bool:
        return self._run(f"shell screencap -p {path}")

    def pull(self, remote: str, local: str) -> bool:
        return self._run(f"pull {remote} {local}")

    def devices(self) -> List[str]:
        if self.dry_run:
            return ["emulator-5554"]
        try:
            r = subprocess.run(["adb", "devices"], capture_output=True, text=True)
            lines = r.stdout.strip().split("\n")[1:]
            return [l.split("\t")[0] for l in lines if "device" in l]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Mobile Adapter
# ---------------------------------------------------------------------------

class MobileAdapter(GameAdapter):
    """
    Minecraft Pocket / Bedrock Mobile adapter.

    Uses ADB to control the game on a connected Android device.
    Observation is stub-based (vision system integration is a future step).

    Parameters
    ----------
    device_id    : ADB device ID (None = use first connected device)
    resolution   : Screen resolution key e.g. "1080x1920"
    dry_run      : If True, print ADB commands without running them
    """

    # ADB keyevent codes
    KEY_BACK   = 4
    KEY_MENU   = 82
    KEY_HOME   = 3
    KEY_ENTER  = 66
    KEY_SPACE  = 62

    def __init__(
        self,
        device_id:  Optional[str] = None,
        resolution: str = "1080x1920",
        dry_run:    bool = True,
    ) -> None:
        super().__init__("mobile_adb")
        self.layout     = LAYOUTS.get(resolution, LAYOUTS["1080x1920"])
        self._adb       = _ADB(device_id=device_id, dry_run=dry_run)
        self.dry_run    = dry_run
        self._obs_stub  = GameObservation(
            source         = "mobile_stub",
            partial        = True,
            missing_fields = ["visible_blocks", "entities", "inventory",
                              "biome", "time_of_day"],
        )
        # Stub health/hunger tracking (updated from vision when available)
        self._stub_health = 20.0
        self._stub_hunger = 20.0

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        devices = self._adb.devices()
        if devices or self.dry_run:
            self.connected = True
            return True
        self.connected = False
        return False

    def disconnect(self) -> None:
        self.connected = False

    # ------------------------------------------------------------------
    # Observation (stub — full CV integration is a future phase)
    # ------------------------------------------------------------------

    def get_observation(self) -> GameObservation:
        """
        Returns a stub observation.
        In a full implementation this would:
          1. Call screencap via ADB
          2. Pull the screenshot
          3. Run VisionSystem.analyze_screenshot()
          4. Return the parsed observation
        """
        self._obs_stub.health    = self._stub_health
        self._obs_stub.hunger    = self._stub_hunger
        self._obs_stub.timestamp = time.time()
        return self._obs_stub

    def update_from_vision(
        self,
        health:  Optional[float] = None,
        hunger:  Optional[float] = None,
    ) -> None:
        """Update stub values from vision system output."""
        if health is not None:
            self._stub_health = float(health)
        if hunger is not None:
            self._stub_hunger = float(hunger)

    # ------------------------------------------------------------------
    # Actions → ADB commands
    # ------------------------------------------------------------------

    def send_action(self, action: Dict) -> bool:
        atype = action.get("type", "")

        if atype == ActionType.MOVE:
            ok = self._adb_move(action.get("direction", "forward"))
        elif atype == ActionType.LOOK:
            ok = self._adb_look(action.get("dx", 0), action.get("dy", 0))
        elif atype == ActionType.JUMP:
            ok = self._adb.tap(*self.layout.jump_btn)
        elif atype == ActionType.ATTACK:
            ok = self._adb.swipe(*self.layout.attack_btn,
                                  *self.layout.attack_btn, 500)
        elif atype == ActionType.USE:
            ok = self._adb.tap(*self.layout.use_btn)
        elif atype == ActionType.INVENTORY:
            ok = self._adb.tap(*self.layout.inventory_btn)
        elif atype == ActionType.SNEAK:
            ok = self._adb.keyevent(self.KEY_SPACE)
        elif atype == ActionType.CHAT:
            ok = self._send_chat(action.get("message", ""))
        elif atype == "tap":
            ok = self._adb.tap(action.get("x", 540), action.get("y", 960))
        elif atype == "swipe":
            ok = self._adb.swipe(
                action.get("x1", 540), action.get("y1", 960),
                action.get("x2", 540), action.get("y2", 800),
                action.get("duration_ms", 200),
            )
        elif atype == "keyevent":
            ok = self._adb.keyevent(action.get("keycode", 0))
        else:
            ok = False

        self.log_action(action, ok)
        return ok

    def _adb_move(self, direction: str) -> bool:
        """Simulate joystick movement via swipe."""
        cx, cy = self.layout.joystick_center
        offsets = {
            "forward": (cx, cy - 150), "back": (cx, cy + 150),
            "left":    (cx - 150, cy), "right": (cx + 150, cy),
            "north":   (cx, cy - 150), "south": (cx, cy + 150),
            "east":    (cx + 150, cy), "west":  (cx - 150, cy),
        }
        tx, ty = offsets.get(direction, (cx, cy - 150))
        return self._adb.swipe(cx, cy, tx, ty, 300)

    def _adb_look(self, dx: int, dy: int) -> bool:
        """Simulate camera rotation via swipe on right side of screen."""
        cx, cy = self.layout.camera_center
        return self._adb.swipe(cx, cy, cx + dx * 5, cy + dy * 5, 100)

    def _send_chat(self, message: str) -> bool:
        """Open chat and type a message."""
        self._adb.keyevent(self.KEY_MENU)
        time.sleep(0.1)
        for char in message:
            # Text entry via ADB text input
            self._adb._run(f"shell input text {char}")
        return self._adb.keyevent(self.KEY_ENTER)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_adb_log(self, n: int = 20) -> List[str]:
        """Return the last N ADB commands sent."""
        return self._adb._log[-n:]

    def summary(self) -> Dict:
        base = super().summary()
        base["dry_run"]       = self.dry_run
        base["resolution"]    = f"{self.layout.resolution[0]}x{self.layout.resolution[1]}"
        base["adb_commands"]  = len(self._adb._log)
        base["partial"]       = True
        base["missing_fields"]= self._obs_stub.missing_fields
        return base
