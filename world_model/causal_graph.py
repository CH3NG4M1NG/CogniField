"""
world_model/causal_graph.py
============================
Causal Knowledge Graph

Stores symbolic cause-effect relationships discovered during interaction:

    eat(apple)     → [satisfied, reward=+0.5]
    eat(stone)     → [damaged,   reward=-0.2]
    pick(heavy)    → [slow,      reward=-0.05]
    drop(fragile)  → [broken,    reward=-0.3]
    combine(a, b)  → [new_item,  reward=+0.3]

Additionally stores conceptual property relationships:
    apple → is_a    → food
    apple → color   → red
    apple → edible  → True
    stone → weight  → heavy

These are used by the planner to reason about what to do and what to avoid.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class CausalEdge:
    """A directed causal relationship."""
    source:     str           # "eat(apple)" or "apple"
    relation:   str           # "causes", "is_a", "has_property", "leads_to"
    target:     str           # "satisfied", "food", "True"
    weight:     float = 1.0   # strength of relationship
    count:      int   = 1     # observed occurrences
    is_causal:  bool  = True  # True = causal, False = attribute

    def reinforce(self, amount: float = 0.1) -> None:
        self.weight  = min(1.0, self.weight + amount)
        self.count  += 1

    def weaken(self, amount: float = 0.05) -> None:
        self.weight = max(0.0, self.weight - amount)


class CausalGraph:
    """
    Directed graph of causal and property relationships.

    Supports:
    - Adding/updating edges
    - Querying what causes what
    - Querying properties of objects
    - Finding objects with desired properties
    """

    def __init__(self) -> None:
        # Adjacency: source → {relation → [CausalEdge]}
        self._graph: Dict[str, Dict[str, List[CausalEdge]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Reverse index: target → [CausalEdge]
        self._reverse: Dict[str, List[CausalEdge]] = defaultdict(list)
        # Object properties cache
        self._properties: Dict[str, Dict[str, Any]] = defaultdict(dict)

    # ------------------------------------------------------------------
    # Adding knowledge
    # ------------------------------------------------------------------

    def add_causal(
        self,
        action_obj: str,
        effect: str,
        weight: float = 0.8,
    ) -> CausalEdge:
        """
        Add or reinforce: action_obj → causes → effect
        e.g. add_causal("eat(apple)", "satisfied")
        """
        edges = self._graph[action_obj]["causes"]
        for e in edges:
            if e.target == effect:
                e.reinforce()
                return e
        edge = CausalEdge(action_obj, "causes", effect, weight, is_causal=True)
        self._graph[action_obj]["causes"].append(edge)
        self._reverse[effect].append(edge)
        return edge

    def add_property(
        self,
        obj: str,
        prop: str,
        value: Any,
        weight: float = 1.0,
    ) -> None:
        """
        Add: obj → has_property → prop=value
        e.g. add_property("apple", "edible", True)
        """
        self._properties[obj][prop] = value
        edges = self._graph[obj]["has_property"]
        prop_str = f"{prop}={value}"
        for e in edges:
            if e.target == prop_str:
                e.reinforce(0.05)
                return
        edge = CausalEdge(obj, "has_property", prop_str, weight, is_causal=False)
        self._graph[obj]["has_property"].append(edge)

    def add_is_a(self, obj: str, category: str) -> None:
        """Add: obj → is_a → category"""
        edges = self._graph[obj]["is_a"]
        for e in edges:
            if e.target == category:
                e.reinforce(0.05)
                return
        edge = CausalEdge(obj, "is_a", category, 1.0, is_causal=False)
        self._graph[obj]["is_a"].append(edge)
        self._reverse[category].append(edge)

    def add_leads_to(
        self,
        action_obj: str,
        next_action_obj: str,
        weight: float = 0.7,
    ) -> None:
        """Add sequential relationship: doing X often leads to doing Y."""
        edges = self._graph[action_obj]["leads_to"]
        for e in edges:
            if e.target == next_action_obj:
                e.reinforce()
                return
        edge = CausalEdge(action_obj, "leads_to", next_action_obj, weight)
        self._graph[action_obj]["leads_to"].append(edge)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_effects(self, action_obj: str) -> List[Tuple[str, float]]:
        """
        What does action_obj cause?
        Returns list of (effect, weight) sorted by weight.
        """
        edges = self._graph.get(action_obj, {}).get("causes", [])
        return sorted([(e.target, e.weight) for e in edges],
                      key=lambda x: -x[1])

    def get_property(self, obj: str, prop: str, default: Any = None) -> Any:
        """Get a cached property value."""
        return self._properties.get(obj, {}).get(prop, default)

    def get_all_properties(self, obj: str) -> Dict[str, Any]:
        """Get all cached properties of an object."""
        return dict(self._properties.get(obj, {}))

    def get_category(self, obj: str) -> Optional[str]:
        """Return the most strongly held is_a category."""
        edges = self._graph.get(obj, {}).get("is_a", [])
        if not edges:
            return None
        return max(edges, key=lambda e: e.weight).target

    def is_edible(self, obj: str) -> Optional[bool]:
        """Query whether obj is edible."""
        return self.get_property(obj, "edible", None)

    def is_fragile(self, obj: str) -> Optional[bool]:
        return self.get_property(obj, "fragile", None)

    def find_edible_objects(self) -> List[str]:
        """Return all objects known to be edible."""
        result = []
        for obj, props in self._properties.items():
            if props.get("edible") is True:
                result.append(obj)
        return result

    def find_objects_by_category(self, category: str) -> List[str]:
        """Return all objects with is_a = category."""
        return [
            e.source for edges in self._reverse.get(category, [])
            for e in [edges] if e.relation == "is_a"
        ]

    def what_causes(self, effect: str) -> List[Tuple[str, float]]:
        """What actions/objects cause this effect? Returns (cause, weight)."""
        edges = self._reverse.get(effect, [])
        causal = [(e.source, e.weight) for e in edges if e.relation == "causes"]
        return sorted(causal, key=lambda x: -x[1])

    def get_next_actions(self, action_obj: str) -> List[Tuple[str, float]]:
        """What actions typically follow this one?"""
        edges = self._graph.get(action_obj, {}).get("leads_to", [])
        return sorted([(e.target, e.weight) for e in edges],
                      key=lambda x: -x[1])

    # ------------------------------------------------------------------
    # Bulk loading from environment feedback
    # ------------------------------------------------------------------

    def ingest_feedback(
        self,
        action: str,
        object_name: str,
        object_props: Dict[str, Any],
        success: bool,
        reward: float,
    ) -> None:
        """
        Automatically extract knowledge from one environment interaction.
        """
        # 1. Object properties
        for prop, val in object_props.items():
            self.add_property(object_name, prop, val)

        # 2. Category from props
        if "category" in object_props:
            self.add_is_a(object_name, object_props["category"])

        # 3. Causal effect of this action
        action_key = f"{action}({object_name})"
        if success:
            if reward >= 0.3:
                self.add_causal(action_key, "satisfied", weight=0.8)
            elif reward > 0:
                self.add_causal(action_key, "partial_success", weight=0.6)
        else:
            if reward <= -0.2:
                self.add_causal(action_key, "damaged", weight=0.8)
            else:
                self.add_causal(action_key, "failed", weight=0.6)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        total_edges = sum(
            len(elist)
            for rel_dict in self._graph.values()
            for elist in rel_dict.values()
        )
        return {
            "n_nodes":      len(self._graph),
            "n_edges":      total_edges,
            "n_objects":    len(self._properties),
            "edible":       self.find_edible_objects(),
        }

    def describe_object(self, obj: str) -> str:
        props = self.get_all_properties(obj)
        cat   = self.get_category(obj) or "unknown"
        effects = self.get_effects(f"eat({obj})") + self.get_effects(f"pick({obj})")
        efx_str = ", ".join(f"{e}({w:.1f})" for e, w in effects[:3]) or "unknown"
        prop_str = ", ".join(f"{k}={v}" for k, v in list(props.items())[:4])
        return f"{obj} [{cat}]: {prop_str} | effects: {efx_str}"

    def __repr__(self) -> str:
        s = self.summary()
        return (f"CausalGraph(nodes={s['n_nodes']}, edges={s['n_edges']}, "
                f"edible={s['edible']})")
