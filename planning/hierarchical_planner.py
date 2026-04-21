r"""
planning/hierarchical_planner.py
==================================
Hierarchical Planning System

Breaks high-level goals into subgoal trees, then recursively plans
primitive actions for each subgoal.

Architecture
------------
                  Goal: "survive"
                  /              \
          "find food"        "avoid danger"
          /         \              |
    "locate apple"  "pick apple"  "don't eat stone"
         |               |
    "move to apple"  "eat apple"

Each level in the hierarchy:
  - Goal node: abstract objective
  - Subgoal nodes: concrete sub-objectives
  - Action nodes: primitive actions from RichEnv

Decomposition Library
---------------------
Pre-defined decomposition rules plus learned ones:

  "eat X"  →  if X not in inventory: pick(X) → eat(X)
  "find X" →  observe → move(toward X) → pick(X)
  "survive" → find food + avoid danger
  "explore" → observe → move → inspect(unknown)

Learned decompositions are discovered from successful plan executions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .planner import Planner, Plan, PlanStep
from ..world_model.transition_model import TransitionModel
from ..world_model.causal_graph import CausalGraph
from ..world_model.simulator import WorldSimulator
from ..latent_space.frequency_space import FrequencySpace


# ---------------------------------------------------------------------------
# Hierarchical structures
# ---------------------------------------------------------------------------

@dataclass
class SubGoal:
    """A node in the goal decomposition tree."""
    label:        str
    depth:        int
    parent:       Optional[str]
    children:     List["SubGoal"] = field(default_factory=list)
    actions:      List[Tuple[str, str]] = field(default_factory=list)  # primitive steps
    score:        float = 0.0
    completed:    bool  = False

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def all_actions(self) -> List[Tuple[str, str]]:
        """Flatten the subtree into a list of primitive actions."""
        if self.is_leaf():
            return list(self.actions)
        result = []
        for child in self.children:
            result.extend(child.all_actions())
        return result


@dataclass
class HierarchicalPlan:
    """A hierarchical plan with subgoal tree + flattened action sequence."""
    root_goal:   str
    tree:        SubGoal           # root of the decomposition tree
    flat_actions: List[Tuple[str, str]]  # primitive action sequence
    total_score: float
    depth:       int

    def describe(self, indent: int = 0) -> str:
        lines = [" " * indent + f"[{self.root_goal}] score={self.total_score:.3f}"]
        def _walk(node: SubGoal, lvl: int) -> None:
            prefix = " " * (2 * lvl)
            lines.append(prefix + f"└─ {node.label}")
            if node.actions:
                for a, o in node.actions:
                    lines.append(prefix + f"   → {a}({o or '–'})")
            for child in node.children:
                _walk(child, lvl + 1)
        _walk(self.tree, 1)
        return "\n".join(lines)

    @property
    def is_empty(self) -> bool:
        return not self.flat_actions


# ---------------------------------------------------------------------------
# HierarchicalPlanner
# ---------------------------------------------------------------------------

class HierarchicalPlanner:
    """
    Plans by decomposing goals into subgoal trees.

    Parameters
    ----------
    base_planner : Flat Planner for leaf-level action planning.
    simulator    : WorldSimulator for plan evaluation.
    space        : FrequencySpace.
    max_depth    : Maximum decomposition depth.
    """

    # Static decomposition library: goal_keyword → subgoal list
    _DECOMPOSITION_LIBRARY: Dict[str, List[str]] = {
        "survive":       ["find food", "avoid danger"],
        "eat":           ["acquire target", "consume target"],
        "acquire":       ["locate target", "pick target"],
        "explore":       ["observe surroundings", "move to unknown", "inspect unknown"],
        "understand":    ["locate target", "inspect target", "record findings"],
        "avoid danger":  ["identify dangerous objects", "maintain distance"],
        "improve":       ["analyse failures", "revise strategy", "practice"],
        "find food":     ["observe", "move to food area", "pick food"],
    }

    # Learned decompositions (updated from successful executions)
    _learned: Dict[str, List[str]] = {}

    def __init__(
        self,
        base_planner: Planner,
        simulator:    WorldSimulator,
        space:        FrequencySpace,
        max_depth:    int = 3,
        dim:          int = 64,
    ) -> None:
        self.base_planner = base_planner
        self.simulator    = simulator
        self.space        = space
        self.max_depth    = max_depth
        self.dim          = dim
        self._plan_count  = 0
        self._successful_decompositions: List[Dict] = []

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def plan_hierarchical(
        self,
        goal_label:       str,
        goal_vec:         np.ndarray,
        current_state:    np.ndarray,
        available_objects: List[Tuple[str, str]],
        inventory:        List[str],
    ) -> HierarchicalPlan:
        """
        Generate a hierarchical plan by decomposing the goal.

        Parameters
        ----------
        goal_label        : Human-readable goal.
        goal_vec          : Goal latent vector.
        current_state     : Current world state vector.
        available_objects : (name, category) pairs.
        inventory         : Currently held items.

        Returns
        -------
        HierarchicalPlan
        """
        self._plan_count += 1

        # Build the decomposition tree
        root = self._decompose(
            goal_label, goal_vec, current_state,
            available_objects, inventory, depth=0
        )

        # Flatten to primitive actions
        flat_actions = root.all_actions()

        # Score via simulation
        score = 0.5
        if flat_actions:
            sim_result = self.simulator.simulate(current_state, flat_actions, goal_vec)
            score = self.simulator._score_simulation(sim_result, goal_vec)
            root.score = score

        depth = self._tree_depth(root)

        return HierarchicalPlan(
            root_goal=goal_label,
            tree=root,
            flat_actions=flat_actions,
            total_score=score,
            depth=depth,
        )

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------

    def _decompose(
        self,
        goal_label:       str,
        goal_vec:         np.ndarray,
        state:            np.ndarray,
        available:        List[Tuple[str, str]],
        inventory:        List[str],
        depth:            int,
    ) -> SubGoal:
        """Recursively decompose a goal into subgoals or primitive actions."""
        node = SubGoal(label=goal_label, depth=depth, parent=None)

        # At max depth or if goal is already primitive → use flat planner
        if depth >= self.max_depth or self._is_primitive(goal_label):
            flat_plan = self.base_planner.plan(
                goal_label, goal_vec, state, available, inventory
            )
            node.actions = flat_plan.action_sequence[:3]  # limit to 3 steps
            return node

        # Find decomposition
        subgoal_labels = self._find_decomposition(goal_label)

        if not subgoal_labels:
            # No decomposition found → use flat planner
            flat_plan = self.base_planner.plan(
                goal_label, goal_vec, state, available, inventory
            )
            node.actions = flat_plan.action_sequence[:3]
            return node

        # Recursively plan each subgoal
        sim_state = state.copy()
        for sg_label in subgoal_labels:
            sg_label_expanded = self._expand_template(sg_label, goal_label, available)
            sg_vec = self._subgoal_vec(sg_label_expanded, goal_vec)

            child = self._decompose(
                sg_label_expanded, sg_vec, sim_state,
                available, inventory, depth + 1
            )
            child.parent = goal_label
            node.children.append(child)

            # Advance simulated state through child's actions
            for action, obj in child.all_actions():
                sim_state, _ = self.simulator.tm.predict_next_state(sim_state, action)

        return node

    def _find_decomposition(self, goal_label: str) -> List[str]:
        """Look up decomposition rules for a goal."""
        label_lower = goal_label.lower()

        # Exact match in library
        if label_lower in self._DECOMPOSITION_LIBRARY:
            return self._DECOMPOSITION_LIBRARY[label_lower]

        # Keyword match
        for keyword, subgoals in self._DECOMPOSITION_LIBRARY.items():
            if keyword in label_lower or label_lower in keyword:
                return subgoals

        # Check learned decompositions
        for learned_goal, subgoals in self._learned.items():
            if learned_goal in label_lower:
                return subgoals

        return []

    def _is_primitive(self, goal_label: str) -> bool:
        """Is this goal a primitive action (no further decomposition needed)?"""
        primitives = {"pick", "eat", "drop", "move", "inspect",
                      "observe", "combine", "use"}
        first_word = goal_label.split()[0].lower()
        return first_word in primitives

    def _expand_template(
        self,
        template:   str,
        parent:     str,
        available:  List[Tuple[str, str]],
    ) -> str:
        """
        Replace "target" in subgoal templates with specific object names.
        e.g. "pick target" from parent "eat apple" → "pick apple"
        """
        if "target" not in template:
            return template

        # Extract target from parent goal
        parent_words = parent.split()
        if len(parent_words) >= 2:
            target = parent_words[-1]
            return template.replace("target", target)

        # Fallback: use first available object
        if available:
            return template.replace("target", available[0][0])

        return template

    def _subgoal_vec(self, label: str, parent_vec: np.ndarray) -> np.ndarray:
        """Generate a vector for a subgoal (blend parent vec with noise)."""
        rng    = np.random.default_rng(hash(label) % 2**32)
        noise  = rng.standard_normal(self.dim).astype(np.float32)
        noise /= np.linalg.norm(noise) + 1e-8
        return self.space.l2(parent_vec + 0.3 * noise)

    def _tree_depth(self, node: SubGoal) -> int:
        if node.is_leaf():
            return node.depth
        return max(self._tree_depth(c) for c in node.children)

    # ------------------------------------------------------------------
    # Learning decompositions from success
    # ------------------------------------------------------------------

    def record_success(
        self,
        goal_label:    str,
        flat_actions:  List[Tuple[str, str]],
    ) -> None:
        """
        When a hierarchical plan succeeds, store the decomposition
        as a learnable template for future use.
        """
        self._successful_decompositions.append({
            "goal": goal_label,
            "actions": flat_actions,
        })
        # Extract subgoal labels from action sequence
        if len(flat_actions) >= 2:
            sg_labels = [f"{a}({o})" if o else a for a, o in flat_actions]
            self._learned[goal_label.lower()] = sg_labels[:3]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"HierarchicalPlanner(plans={self._plan_count}, "
                f"learned={len(self._learned)})")
