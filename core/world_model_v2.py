"""
core/world_model_v2.py
========================
Structured World Model

Maintains a hierarchical world model:
  object → category → property → effect

Example structure:
  apple       is_a  food
  food        has   edible=True
  food        has   digestible=True
  apple       has   color=red
  eat(apple)  causes satisfied, reward=+0.5

This gives the system explicit background knowledge about:
  - What categories things belong to
  - What properties categories have (inheritance)
  - What effects actions have on objects

The model is populated by:
  1. Direct teach() calls
  2. Inference from observations
  3. Generalisation from the experience engine

It enables forward reasoning:
  "I know apple is_a food, food is edible → apple is edible (inferred)"
  "I know eat(food) → satisfied → eating apple will satisfy me (inferred)"
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class WorldEntity:
    """A concept in the world model."""
    name:       str
    category:   Optional[str]      = None
    properties: Dict[str, Any]     = field(default_factory=dict)
    confidence: Dict[str, float]   = field(default_factory=dict)  # per property
    created_at: float              = field(default_factory=time.time)
    updated_at: float              = field(default_factory=time.time)

    def get_property(self, prop: str, default: Any = None) -> Any:
        return self.properties.get(prop, default)

    def get_confidence(self, prop: str, default: float = 0.5) -> float:
        return self.confidence.get(prop, default)

    def set_property(self, prop: str, value: Any, conf: float = 0.8) -> None:
        self.properties[prop] = value
        self.confidence[prop] = float(np.clip(conf, 0.0, 1.0))
        self.updated_at       = time.time()

    def inherit_from(self, parent: "WorldEntity", weight: float = 0.75) -> int:
        """Inherit unset properties from a parent category."""
        inherited = 0
        for prop, val in parent.properties.items():
            if prop not in self.properties:
                self.set_property(prop, val,
                                  parent.get_confidence(prop) * weight)
                inherited += 1
        return inherited


@dataclass
class ActionEffect:
    """Recorded effect of an action on an object."""
    action:    str
    target:    str
    effect:    str          # e.g. "satisfied", "damaged", "moved"
    reward:    float
    confidence:float = 0.7
    n_observed:int   = 1


class WorldModelV2:
    """
    Structured hierarchical world model.

    Provides:
      - Category hierarchy (is_a relationships)
      - Property inheritance (food.edible=True → apple.edible=True)
      - Effect knowledge (eat(food) → satisfied)
      - Forward inference chain
    """

    def __init__(self) -> None:
        self._entities: Dict[str, WorldEntity]   = {}
        self._effects:  Dict[str, ActionEffect]  = {}
        self._rules:    List[Tuple[str, str, Any, float]] = []  # (subj, pred, val, conf)
        self._inferences_made = 0

        # Seed with basic world knowledge
        self._seed_defaults()

    # ------------------------------------------------------------------
    # Building the model
    # ------------------------------------------------------------------

    def add_entity(
        self,
        name:       str,
        category:   Optional[str]  = None,
        properties: Optional[Dict] = None,
        confidence: float          = 0.80,
    ) -> WorldEntity:
        """Add or update an entity in the world model."""
        if name not in self._entities:
            self._entities[name] = WorldEntity(name=name, category=category)
        e = self._entities[name]
        if category:
            e.category = category
        if properties:
            for k, v in properties.items():
                e.set_property(k, v, confidence)
        # Inherit from category if known
        if e.category and e.category in self._entities:
            e.inherit_from(self._entities[e.category])
        return e

    def add_effect(
        self,
        action:   str,
        target:   str,
        effect:   str,
        reward:   float,
        confidence: float = 0.7,
    ) -> None:
        """Record that action(target) causes effect."""
        key = f"{action}({target})"
        if key in self._effects:
            ae = self._effects[key]
            # Update with EMA
            ae.reward     = 0.8 * ae.reward + 0.2 * reward
            ae.confidence = min(0.95, ae.confidence + 0.03)
            ae.n_observed += 1
        else:
            self._effects[key] = ActionEffect(
                action=action, target=target,
                effect=effect, reward=reward, confidence=confidence
            )

        # Also record for category
        e = self._entities.get(target)
        if e and e.category:
            cat_key = f"{action}({e.category})"
            if cat_key not in self._effects:
                self._effects[cat_key] = ActionEffect(
                    action=action, target=e.category,
                    effect=effect, reward=reward, confidence=confidence * 0.8
                )

    def add_rule(
        self,
        subject:    str,
        predicate:  str,
        value:      Any,
        confidence: float = 0.75,
    ) -> None:
        """Add a general rule: subject.predicate = value."""
        self._rules.append((subject, predicate, value, confidence))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer_property(
        self,
        name:     str,
        prop:     str,
    ) -> Tuple[Optional[Any], float]:
        """
        Infer a property value for an entity using the hierarchy.

        Returns (value, confidence), or (None, 0.5) if unknown.
        """
        # Direct entity lookup
        e = self._entities.get(name)
        if e and prop in e.properties:
            return e.get_property(prop), e.get_confidence(prop)

        # Category-level inference
        if e and e.category:
            cat = self._entities.get(e.category)
            if cat and prop in cat.properties:
                self._inferences_made += 1
                return cat.get_property(prop), cat.get_confidence(prop) * 0.80

        # Rule-based inference
        for subj, pred, val, conf in self._rules:
            if subj == name and pred == prop:
                return val, conf
            # Category-level rule
            if e and e.category == subj and pred == prop:
                return val, conf * 0.80

        return None, 0.5

    def infer_effect(
        self,
        action: str,
        target: str,
    ) -> Tuple[str, float, float]:
        """
        Infer the effect of action(target).
        Returns (effect_label, reward, confidence).
        """
        # Direct lookup
        key = f"{action}({target})"
        if key in self._effects:
            ae = self._effects[key]
            return ae.effect, ae.reward, ae.confidence

        # Category inference
        e = self._entities.get(target)
        if e and e.category:
            cat_key = f"{action}({e.category})"
            if cat_key in self._effects:
                ae = self._effects[cat_key]
                return ae.effect, ae.reward, ae.confidence * 0.80

        return "unknown", 0.0, 0.30

    def causal_chains(self, subject: str, predicate: str) -> Dict:
        """Return causal chain data for the deep thinker."""
        val, conf = self.infer_property(subject, predicate)
        if val is None:
            return {}
        e = self._entities.get(subject)
        if e and e.category:
            cat_val, cat_conf = self.infer_property(e.category, predicate)
            if cat_val is not None:
                return {
                    f"{subject}.{predicate}": {
                        "cause":            f"{subject} is_a {e.category}",
                        "effect":           f"{predicate}={val}",
                        "confidence_boost": (cat_conf - 0.5) * 0.1,
                    }
                }
        return {}

    def get_entity(self, name: str) -> Optional[WorldEntity]:
        return self._entities.get(name)

    def known_categories(self) -> List[str]:
        return list({e.category for e in self._entities.values()
                     if e.category is not None})

    def entities_in_category(self, category: str) -> List[str]:
        return [name for name, e in self._entities.items()
                if e.category == category]

    # ------------------------------------------------------------------
    # Sync with BeliefSystem
    # ------------------------------------------------------------------

    def sync_to_beliefs(self, belief_system, min_conf: float = 0.65) -> int:
        """Push high-confidence world model entries into a BeliefSystem."""
        count = 0
        for name, entity in self._entities.items():
            for prop, val in entity.properties.items():
                conf = entity.get_confidence(prop)
                if conf >= min_conf:
                    key = f"{name}.{prop}"
                    existing = belief_system.get(key)
                    if existing is None or existing.confidence < conf:
                        belief_system.update(key, val, source="world_model",
                                             weight=conf * 0.9,
                                             notes="from_world_model_v2")
                        count += 1
        return count

    # ------------------------------------------------------------------
    # Internal seeding
    # ------------------------------------------------------------------

    def _seed_defaults(self) -> None:
        """Load basic background knowledge."""
        # Category defaults
        categories = {
            "food":     {"edible": True,  "digestible": True,  "safe": True},
            "material": {"edible": False, "digestible": False, "safe": True},
            "tool":     {"edible": False, "fragile": True,     "safe": True},
            "plant":    {"edible": None,  "natural": True},
            "animal":   {"edible": None,  "alive": True},
        }
        for cat, props in categories.items():
            self.add_entity(cat, properties=props, confidence=0.85)

        # Common action effects
        effects = [
            ("eat",    "food",     "satisfied", +0.50),
            ("eat",    "material", "damaged",   -0.30),
            ("pick",   "food",     "held",      +0.10),
            ("pick",   "material", "held",      +0.10),
            ("drop",   "fragile",  "broken",    -0.20),
            ("inspect","object",   "observed",  +0.05),
        ]
        for action, target, effect, reward in effects:
            self.add_effect(action, target, effect, reward, confidence=0.80)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        return {
            "entities":         len(self._entities),
            "effects":          len(self._effects),
            "rules":            len(self._rules),
            "inferences_made":  self._inferences_made,
            "categories":       self.known_categories(),
        }

    def __repr__(self) -> str:
        return (f"WorldModelV2(entities={len(self._entities)}, "
                f"effects={len(self._effects)}, "
                f"inferences={self._inferences_made})")
