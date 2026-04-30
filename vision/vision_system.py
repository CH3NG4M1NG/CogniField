"""
vision/vision_system.py
========================
Vision System

Processes visual data from the game screen to extract structured
observations. In simulation mode it generates synthetic readings;
in real mode it analyses screenshot pixel data.

Detection capabilities:
  - Health bar (hearts)
  - Hunger bar (drumsticks)
  - Hotbar items (slot parsing)
  - Basic object/block detection (colour-region heuristic)

Real screenshot analysis pipeline (when image data is available):
  1. Load PIL image from bytes / file path
  2. Sample health bar region → count red pixels → infer hearts
  3. Sample hunger bar region → count orange pixels → infer food level
  4. Run block-colour heuristic → rough block type list
  5. Return ScreenReading

All coordinates are for a 1080×1920 layout; override regions for
other resolutions via VisionSystem.set_layout().
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Screen regions (normalised 0–1 for resolution independence)
# ---------------------------------------------------------------------------

@dataclass
class ScreenRegion:
    """A normalised screen region [0,1]."""
    x:      float   # left
    y:      float   # top
    width:  float
    height: float

    def to_pixels(self, screen_w: int, screen_h: int) -> Tuple[int,int,int,int]:
        """Return (x0, y0, x1, y1) in pixel coordinates."""
        return (
            int(self.x * screen_w),
            int(self.y * screen_h),
            int((self.x + self.width) * screen_w),
            int((self.y + self.height) * screen_h),
        )


DEFAULT_REGIONS = {
    "health_bar": ScreenRegion(0.02, 0.88, 0.40, 0.03),
    "hunger_bar": ScreenRegion(0.52, 0.88, 0.40, 0.03),
    "hotbar":     ScreenRegion(0.10, 0.93, 0.80, 0.06),
    "crosshair":  ScreenRegion(0.45, 0.45, 0.10, 0.10),
    "minimap":    ScreenRegion(0.80, 0.02, 0.18, 0.18),
}

# Approximate BGR colour anchors for common Minecraft blocks
BLOCK_COLOUR_HINTS = {
    "grass":   (34, 139, 34),
    "stone":   (128, 128, 128),
    "sand":    (255, 223, 128),
    "water":   (64, 164, 223),
    "wood":    (139, 90, 43),
    "leaves":  (0, 100, 0),
    "snow":    (240, 240, 255),
    "lava":    (207, 70, 0),
}


# ---------------------------------------------------------------------------
# Screen reading output
# ---------------------------------------------------------------------------

@dataclass
class ScreenReading:
    """
    Parsed visual data extracted from one game screenshot.
    All values normalised to [0.0, 1.0] ranges.
    """
    health_pct:      float              # 0.0 (dead) → 1.0 (full)
    hunger_pct:      float              # 0.0 (starving) → 1.0 (full)
    hotbar_items:    List[str]          # detected item IDs (may be empty)
    detected_blocks: List[str]          # rough block types visible
    detected_objects:List[str]          # entity/object labels
    danger_detected: bool               # hostile mob visible?
    food_visible:    bool               # food item on screen?
    confidence:      float              # overall detection confidence
    source:          str                # "simulation" | "screenshot" | "stub"
    timestamp:       float = field(default_factory=time.time)

    # Convenience conversions to Minecraft scale
    @property
    def health_hearts(self) -> float:
        return self.health_pct * 20.0

    @property
    def hunger_drumsticks(self) -> float:
        return self.hunger_pct * 20.0

    def to_dict(self) -> Dict:
        return {
            "health_pct":       round(self.health_pct, 3),
            "hunger_pct":       round(self.hunger_pct, 3),
            "health_hearts":    round(self.health_hearts, 1),
            "hunger_drumsticks":round(self.hunger_drumsticks, 1),
            "hotbar_items":     self.hotbar_items,
            "detected_blocks":  self.detected_blocks,
            "detected_objects": self.detected_objects,
            "danger_detected":  self.danger_detected,
            "food_visible":     self.food_visible,
            "confidence":       round(self.confidence, 3),
            "source":           self.source,
        }


# ---------------------------------------------------------------------------
# Vision System
# ---------------------------------------------------------------------------

class VisionSystem:
    """
    Processes game screenshots to extract structured observations.

    Parameters
    ----------
    screen_w, screen_h  : Target screen resolution.
    simulation          : If True, generate synthetic readings (no image needed).
    seed                : RNG seed for simulation mode.
    """

    FOOD_COLOURS = {
        "red_mushroom":   (200, 50, 50),
        "apple":          (220, 30, 30),
        "carrot":         (255, 140, 0),
        "wheat":          (240, 210, 100),
        "melon":          (100, 200, 50),
    }

    HOSTILE_PALETTE = {
        "zombie": (50, 150, 50),
        "skeleton": (240, 240, 240),
        "creeper": (80, 180, 80),
    }

    def __init__(
        self,
        screen_w:   int   = 1080,
        screen_h:   int   = 1920,
        simulation: bool  = True,
        seed:       int   = 42,
    ) -> None:
        self.screen_w   = screen_w
        self.screen_h   = screen_h
        self.simulation = simulation
        self._regions   = dict(DEFAULT_REGIONS)
        self._rng       = random.Random(seed)
        self._readings: List[ScreenReading] = []

        # Simulated environment state (drifts over time)
        self._sim_health = 1.0
        self._sim_hunger = 1.0
        self._sim_tick   = 0

    # ------------------------------------------------------------------
    # Main analysis entry points
    # ------------------------------------------------------------------

    def analyze(self, image_data: Optional[bytes] = None) -> ScreenReading:
        """
        Analyse a screenshot or generate a simulated reading.

        Parameters
        ----------
        image_data : Raw PNG/JPEG bytes (None → simulation mode)

        Returns
        -------
        ScreenReading with all detected values.
        """
        if self.simulation or image_data is None:
            return self._simulate_reading()
        return self._analyze_screenshot(image_data)

    def analyze_file(self, path: str) -> ScreenReading:
        """Load a screenshot file and analyse it."""
        try:
            with open(path, "rb") as f:
                return self.analyze(f.read())
        except (IOError, OSError):
            return self._fallback_reading("file_error")

    # ------------------------------------------------------------------
    # Simulation reading
    # ------------------------------------------------------------------

    def _simulate_reading(self) -> ScreenReading:
        """Generate a plausible reading that drifts over time."""
        self._sim_tick += 1

        # Health/hunger drift
        self._sim_health = max(0.05, self._sim_health - self._rng.uniform(0, 0.008))
        self._sim_hunger = max(0.05, self._sim_hunger - self._rng.uniform(0, 0.012))

        # Simulate occasional food/danger detection
        food_visible   = self._rng.random() < 0.30
        danger_nearby  = self._rng.random() < 0.15

        # Block palette depends on "biome simulation"
        biome_idx = (self._sim_tick // 10) % 4
        block_sets = [
            ["grass", "stone", "wood"],
            ["sand", "stone"],
            ["snow", "stone", "wood"],
            ["grass", "leaves", "wood"],
        ]
        detected_blocks = block_sets[biome_idx]

        # Objects
        detected_objects = []
        if danger_nearby:
            mob = self._rng.choice(["zombie", "skeleton", "creeper"])
            detected_objects.append(mob)
        if food_visible:
            item = self._rng.choice(["apple", "carrot", "mushroom"])
            detected_objects.append(item)

        hotbar = self._rng.choice([
            ["minecraft:bread", "minecraft:wooden_sword"],
            ["minecraft:apple", "minecraft:stone_pickaxe"],
            [],
        ])

        reading = ScreenReading(
            health_pct       = round(self._sim_health, 3),
            hunger_pct       = round(self._sim_hunger, 3),
            hotbar_items     = hotbar,
            detected_blocks  = detected_blocks,
            detected_objects = detected_objects,
            danger_detected  = danger_nearby,
            food_visible     = food_visible,
            confidence       = round(self._rng.uniform(0.70, 0.92), 3),
            source           = "simulation",
        )
        self._readings.append(reading)
        return reading

    # ------------------------------------------------------------------
    # Real screenshot analysis
    # ------------------------------------------------------------------

    def _analyze_screenshot(self, image_data: bytes) -> ScreenReading:
        """
        Parse real screenshot bytes.
        Uses a pixel-sampling heuristic — no heavy CV library required.
        """
        try:
            # Try PIL for image loading
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
            w, h = img.size

            health_pct = self._sample_bar(img, w, h, "health_bar", "red")
            hunger_pct = self._sample_bar(img, w, h, "hunger_bar", "orange")
            blocks     = self._detect_blocks(img, w, h)
            objects    = self._detect_objects(img, w, h)
            food_vis   = any(f in objects for f in
                             ["apple","carrot","mushroom","bread","melon"])
            danger     = any(d in objects for d in
                             ["zombie","skeleton","creeper","spider","enderman"])

            reading = ScreenReading(
                health_pct       = health_pct,
                hunger_pct       = hunger_pct,
                hotbar_items     = [],
                detected_blocks  = blocks,
                detected_objects = objects,
                danger_detected  = danger,
                food_visible     = food_vis,
                confidence       = 0.75,
                source           = "screenshot_pil",
            )
        except ImportError:
            # PIL not available — use stub
            reading = self._fallback_reading("no_pil")
        except Exception:
            reading = self._fallback_reading("parse_error")

        self._readings.append(reading)
        return reading

    def _sample_bar(
        self, img, w: int, h: int, region_key: str, colour: str
    ) -> float:
        """
        Sample a UI bar region and estimate fill level from colour density.
        Returns a value in [0.0, 1.0].
        """
        region = self._regions.get(region_key)
        if not region:
            return 0.5

        x0, y0, x1, y1 = region.to_pixels(w, h)
        try:
            bar_region = img.crop((x0, y0, x1, y1))
            pixels     = list(bar_region.getdata())
        except Exception:
            return 0.5

        if not pixels:
            return 0.5

        target_r = {"red": (180, 0, 0), "orange": (200, 100, 0)}.get(colour, (128,128,128))
        total    = len(pixels)
        matching = sum(
            1 for p in pixels
            if abs(p[0] - target_r[0]) < 60
            and abs(p[1] - target_r[1]) < 60
            and abs(p[2] - target_r[2]) < 60
        )
        return float(matching / max(total, 1))

    def _detect_blocks(self, img, w: int, h: int) -> List[str]:
        """Sample central region for dominant block colours."""
        cx, cy = w // 2, h // 2
        sample_size = min(200, w // 5, h // 5)
        try:
            center = img.crop((cx - sample_size, cy - sample_size,
                                cx + sample_size, cy + sample_size))
            pixels = list(center.getdata())[:500]
        except Exception:
            return []

        detected = []
        for block_name, (tr, tg, tb) in BLOCK_COLOUR_HINTS.items():
            match = sum(
                1 for p in pixels
                if abs(p[0]-tr)<40 and abs(p[1]-tg)<40 and abs(p[2]-tb)<40
            )
            if match / max(len(pixels), 1) > 0.05:
                detected.append(block_name)
        return detected[:5]

    def _detect_objects(self, img, w: int, h: int) -> List[str]:
        """
        Placeholder object detector.
        Returns empty list — extend with a real detector.
        """
        return []

    def _fallback_reading(self, source: str) -> ScreenReading:
        return ScreenReading(
            health_pct=0.5, hunger_pct=0.5,
            hotbar_items=[], detected_blocks=[],
            detected_objects=[], danger_detected=False,
            food_visible=False, confidence=0.0, source=source,
        )

    # ------------------------------------------------------------------
    # Simulation controls
    # ------------------------------------------------------------------

    def set_sim_health(self, v: float) -> None:
        self._sim_health = max(0.0, min(1.0, v))

    def set_sim_hunger(self, v: float) -> None:
        self._sim_hunger = max(0.0, min(1.0, v))

    def set_layout(self, regions: Dict[str, ScreenRegion]) -> None:
        """Override screen regions for a custom layout."""
        self._regions.update(regions)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def reading_count(self) -> int:
        return len(self._readings)

    def last_reading(self) -> Optional[ScreenReading]:
        return self._readings[-1] if self._readings else None

    def mean_health(self) -> float:
        if not self._readings:
            return 1.0
        return sum(r.health_pct for r in self._readings) / len(self._readings)

    def summary(self) -> Dict:
        return {
            "readings":      len(self._readings),
            "simulation":    self.simulation,
            "resolution":    f"{self.screen_w}×{self.screen_h}",
            "mean_health":   round(self.mean_health(), 3),
        }

    def __repr__(self) -> str:
        mode = "sim" if self.simulation else "real"
        return f"VisionSystem(mode={mode}, readings={len(self._readings)})"
