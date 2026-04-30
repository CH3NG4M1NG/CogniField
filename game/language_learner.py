"""
game/language_learner.py
=========================
Language Learner

Extracts object names from game observations and maps them to
CogniField world-model concepts without any external NLP library.

Pipeline:
  game observation → extract object names → classify category
    → infer properties (edible? dangerous? fragile?)
    → push to WorldModelV2 + BeliefSystem

Minecraft ID mapping:
  "minecraft:apple"   → name="apple",    category="food", edible=True
  "minecraft:zombie"  → name="zombie",   category="mob",  dangerous=True
  "minecraft:oak_log" → name="oak log",  category="wood", edible=False
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .base_adapter import GameObservation, BlockInfo, EntityInfo


# ---------------------------------------------------------------------------
# Concept registry
# ---------------------------------------------------------------------------

@dataclass
class GameConcept:
    """A learned game concept mapped to CogniField world model."""
    mc_id:       str               # e.g. "minecraft:apple"
    name:        str               # e.g. "apple"
    category:    str               # food / mob / block / tool / plant
    properties:  Dict[str, Any]    # edible, dangerous, fragile, ...
    confidence:  float = 0.80
    times_seen:  int   = 0
    first_seen:  float = field(default_factory=time.time)
    last_seen:   float = field(default_factory=time.time)

    def bump(self) -> None:
        self.times_seen += 1
        self.last_seen   = time.time()
        self.confidence  = min(0.95, self.confidence + 0.01)


# ---------------------------------------------------------------------------
# Static knowledge base (Minecraft-specific)
# ---------------------------------------------------------------------------

FOOD_IDS: Set[str] = {
    "minecraft:apple", "minecraft:bread", "minecraft:carrot",
    "minecraft:potato", "minecraft:beetroot", "minecraft:melon_slice",
    "minecraft:sweet_berries", "minecraft:glow_berries",
    "minecraft:cooked_beef", "minecraft:cooked_porkchop",
    "minecraft:cooked_chicken", "minecraft:cooked_salmon",
    "minecraft:cooked_cod", "minecraft:mushroom_stew",
    "minecraft:pumpkin_pie", "minecraft:cake",
}

HOSTILE_TYPES: Set[str] = {
    "zombie", "skeleton", "creeper", "spider", "witch",
    "enderman", "pillager", "ravager", "phantom", "husk",
    "stray", "drowned", "vindicator", "evoker",
}

PASSIVE_MOBS: Set[str] = {
    "pig", "cow", "sheep", "chicken", "rabbit", "horse",
    "donkey", "llama", "fox", "ocelot", "cat", "wolf",
    "parrot", "bee", "panda", "polar_bear", "salmon", "cod",
}

WOOD_IDS: Set[str] = {
    "minecraft:oak_log", "minecraft:spruce_log",
    "minecraft:birch_log", "minecraft:jungle_log",
    "minecraft:acacia_log", "minecraft:dark_oak_log",
    "minecraft:mangrove_log", "minecraft:cherry_log",
}

STONE_IDS: Set[str] = {
    "minecraft:stone", "minecraft:cobblestone",
    "minecraft:granite", "minecraft:diorite", "minecraft:andesite",
    "minecraft:deepslate", "minecraft:tuff",
}

ORE_IDS: Set[str] = {
    "minecraft:coal_ore", "minecraft:iron_ore",
    "minecraft:gold_ore", "minecraft:diamond_ore",
    "minecraft:redstone_ore", "minecraft:lapis_ore",
    "minecraft:copper_ore", "minecraft:emerald_ore",
}


# ---------------------------------------------------------------------------
# Language Learner
# ---------------------------------------------------------------------------

class LanguageLearner:
    """
    Extracts concepts from Minecraft observations and pushes them to
    the CogniField world model.

    Parameters
    ----------
    world_model   : WorldModelV2 instance to update.
    belief_system : BeliefSystem instance to update.
    min_seen      : Minimum observations before concept is considered reliable.
    """

    def __init__(
        self,
        world_model   = None,
        belief_system = None,
        min_seen:  int = 2,
    ) -> None:
        self.wm         = world_model
        self.bs         = belief_system
        self.min_seen   = min_seen
        self._concepts: Dict[str, GameConcept] = {}
        self._unknown:  List[str]              = []
        self._n_updates = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_observation(self, obs: GameObservation) -> List[GameConcept]:
        """
        Extract all learnable concepts from one game observation.
        Automatically updates world model and belief system.

        Returns list of new/updated concepts.
        """
        updated: List[GameConcept] = []

        # Process visible blocks
        for block in obs.visible_blocks:
            c = self._process_id(block.block_id)
            if c:
                updated.append(c)

        # Process inventory items
        for item in obs.inventory:
            c = self._process_id(item.item_id)
            if c:
                updated.append(c)

        # Process entities
        for entity in obs.entities:
            c = self._process_entity(entity.entity_type,
                                      entity.hostile)
            if c:
                updated.append(c)

        return updated

    def process_id(self, mc_id: str) -> Optional[GameConcept]:
        """Process one Minecraft ID and return the concept (if learned)."""
        return self._process_id(mc_id)

    # ------------------------------------------------------------------
    # ID processing
    # ------------------------------------------------------------------

    def _process_id(self, mc_id: str) -> Optional[GameConcept]:
        """Classify one Minecraft ID and update world model."""
        if not mc_id or not isinstance(mc_id, str):
            return None

        # Check cache
        if mc_id in self._concepts:
            c = self._concepts[mc_id]
            c.bump()
            if c.times_seen >= self.min_seen:
                self._push_to_world_model(c)
            return c

        # Parse and classify
        name       = self._mc_id_to_name(mc_id)
        category   = self._classify_block(mc_id)
        properties = self._infer_properties(mc_id, category)

        if category == "unknown":
            if mc_id not in self._unknown:
                self._unknown.append(mc_id)
            return None

        concept = GameConcept(
            mc_id=mc_id, name=name,
            category=category, properties=properties,
            confidence=0.80, times_seen=1,
        )
        self._concepts[mc_id] = concept

        if concept.times_seen >= self.min_seen:
            self._push_to_world_model(concept)

        return concept

    def _process_entity(
        self, entity_type: str, hostile: bool
    ) -> Optional[GameConcept]:
        """Process an entity type."""
        mc_id = f"entity:{entity_type}"
        if mc_id in self._concepts:
            c = self._concepts[mc_id]
            c.bump()
            return c

        if entity_type in HOSTILE_TYPES or hostile:
            cat   = "mob_hostile"
            props = {"dangerous": True, "edible": False,
                     "hostile": True, "avoid": True}
        elif entity_type in PASSIVE_MOBS:
            cat   = "mob_passive"
            # Passive mobs can be food sources
            props = {"dangerous": False,
                     "edible": entity_type in
                               {"pig","cow","sheep","chicken","rabbit",
                                "salmon","cod"},
                     "hostile": False}
        else:
            cat   = "mob"
            props = {"dangerous": False, "edible": False}

        concept = GameConcept(
            mc_id=mc_id, name=entity_type.replace("_", " "),
            category=cat, properties=props,
            confidence=0.80, times_seen=1,
        )
        self._concepts[mc_id] = concept
        self._push_to_world_model(concept)
        return concept

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _mc_id_to_name(mc_id: str) -> str:
        """Convert "minecraft:oak_log" → "oak log"."""
        name = mc_id.split(":")[-1]
        return name.replace("_", " ")

    def _classify_block(self, mc_id: str) -> str:
        if mc_id in FOOD_IDS:
            return "food"
        if mc_id in WOOD_IDS:
            return "wood"
        if mc_id in STONE_IDS:
            return "stone"
        if mc_id in ORE_IDS:
            return "ore"
        # Heuristic matching
        tail = mc_id.split(":")[-1]
        if any(k in tail for k in
               ["log","plank","wood","slab","stair","door"]):
            return "wood"
        if any(k in tail for k in
               ["stone","cobble","brick","granite","diorite","andesite"]):
            return "stone"
        if any(k in tail for k in
               ["ore","crystal","gem","ingot"]):
            return "ore"
        if any(k in tail for k in
               ["flower","grass","fern","vine","leaf","leaves",
                "sapling","bush"]):
            return "plant"
        if any(k in tail for k in
               ["sword","axe","pickaxe","shovel","hoe","bow","arrow"]):
            return "tool"
        if any(k in tail for k in
               ["helmet","chestplate","leggings","boots","armour","armor"]):
            return "armour"
        if any(k in tail for k in
               ["apple","bread","carrot","potato","stew","soup","cake",
                "berry","melon","fish","beef","pork","chicken","rabbit"]):
            return "food"
        if any(k in tail for k in
               ["sand","gravel","dirt","clay","mud","soul"]):
            return "terrain"
        if any(k in tail for k in
               ["chest","barrel","shulker","furnace","crafting",
                "anvil","enchanting"]):
            return "container"
        if any(k in tail for k in
               ["glass","wool","carpet","concrete","terracotta","banner"]):
            return "decorative"
        return "unknown"

    def _infer_properties(
        self, mc_id: str, category: str
    ) -> Dict[str, Any]:
        """Infer boolean properties for a classified item."""
        props: Dict[str, Any] = {"category": category}

        if category == "food":
            props["edible"]  = True
            props["safe"]    = True
        elif category in ("stone", "wood", "ore", "terrain",
                           "container", "decorative"):
            props["edible"]  = False
            props["safe"]    = True
        elif category == "plant":
            # Some plants are food; check known food IDs
            props["edible"]  = mc_id in FOOD_IDS
            props["safe"]    = True
            props["natural"] = True
        elif category == "tool":
            props["edible"]  = False
            props["safe"]    = True
            props["fragile"] = True
        elif category == "armour":
            props["edible"]  = False
            props["safe"]    = True
            props["protective"] = True

        return props

    # ------------------------------------------------------------------
    # World model + belief sync
    # ------------------------------------------------------------------

    def _push_to_world_model(self, concept: GameConcept) -> None:
        """Push a learned concept to WorldModelV2 and BeliefSystem."""
        self._n_updates += 1

        if self.wm is not None:
            self.wm.add_entity(
                concept.name,
                category=concept.category,
                properties={k: v for k, v in concept.properties.items()
                            if k != "category"},
                confidence=concept.confidence,
            )

        if self.bs is not None:
            for prop, val in concept.properties.items():
                if prop == "category":
                    continue
                key = f"{concept.name}.{prop}"
                existing = self.bs.get(key)
                if existing is None or existing.confidence < concept.confidence:
                    self.bs.update(
                        key, val,
                        source="language_learner",
                        weight=concept.confidence * 0.85,
                        notes=f"learned_from_mc:{concept.mc_id}",
                    )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_concept(self, name_or_id: str) -> Optional[GameConcept]:
        # Try direct mc_id lookup
        if name_or_id in self._concepts:
            return self._concepts[name_or_id]
        # Try name match
        for c in self._concepts.values():
            if c.name == name_or_id:
                return c
        return None

    def known_foods(self) -> List[str]:
        return [c.name for c in self._concepts.values()
                if c.category == "food"]

    def known_dangers(self) -> List[str]:
        return [c.name for c in self._concepts.values()
                if c.properties.get("dangerous") is True]

    def unknown_ids(self) -> List[str]:
        return list(self._unknown)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        cats: Dict[str, int] = {}
        for c in self._concepts.values():
            cats[c.category] = cats.get(c.category, 0) + 1
        return {
            "known_concepts":   len(self._concepts),
            "unknown_ids":      len(self._unknown),
            "by_category":      cats,
            "world_model_updates": self._n_updates,
            "known_foods":      self.known_foods(),
            "known_dangers":    self.known_dangers(),
        }

    def __repr__(self) -> str:
        return (f"LanguageLearner(known={len(self._concepts)}, "
                f"unknown={len(self._unknown)}, "
                f"updates={self._n_updates})")
