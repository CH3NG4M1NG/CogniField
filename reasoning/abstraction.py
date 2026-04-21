"""
reasoning/abstraction.py
=========================
Knowledge Abstraction Engine

Extracts general rules from specific experiences, building a hierarchy
of increasingly abstract knowledge.

Example Inference Chain
-----------------------
Specific experiences:
  eat(apple) → success (+0.5)
  eat(bread) → success (+0.5)
  eat(water) → success (+0.5)
  apple → is_a → food
  bread → is_a → food
  water → is_a → food

Abstract rule extracted:
  food → edible → True (confidence=0.9)
  eat(food) → success (confidence=0.9)

Second-order abstraction:
  multiple food items eaten → satiation+
  → infer: eating restores satiation

Abstraction Methods
-------------------
1. CATEGORICAL INDUCTION
   All X with property P → category(X) has property P

2. ACTION GENERALISATION
   If eat(A)→success AND eat(B)→success AND similar(A,B)
   → eat(category)→success

3. TEMPORAL PATTERN DETECTION
   sequence: pick(X) → eat(X) → success
   → "pick then eat" is a valid strategy for food

4. NEGATIVE INFERENCE
   eat(stone) → failure
   stone → is_a → material
   → material → edible=False
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..memory.relational_memory import RelationalMemory, Relation
from ..world_model.transition_model import TransitionModel, WorldRule
from ..world_model.causal_graph import CausalGraph
from ..latent_space.frequency_space import FrequencySpace


@dataclass
class AbstractRule:
    """A generalised rule extracted from specific experiences."""
    subject:     str            # e.g. "food", "material"
    predicate:   str            # e.g. "edible", "action_outcome"
    value:       Any            # e.g. True, "success"
    confidence:  float          # how confident we are
    support:     int            # number of specific examples supporting this
    method:      str            # which abstraction method produced it
    examples:    List[str] = field(default_factory=list)
    timestamp:   float = field(default_factory=time.time)

    @property
    def is_strong(self) -> bool:
        return self.confidence >= 0.70 and self.support >= 2


class AbstractionEngine:
    """
    Extracts abstract rules from grounded experience.

    Parameters
    ----------
    rel_memory     : RelationalMemory with specific facts.
    world_model    : TransitionModel with action rules.
    causal_graph   : CausalGraph with causal relationships.
    space          : FrequencySpace for vector operations.
    min_support    : Minimum examples needed to form an abstract rule.
    min_confidence : Minimum confidence threshold for new rules.
    """

    def __init__(
        self,
        rel_memory:     RelationalMemory,
        world_model:    TransitionModel,
        causal_graph:   CausalGraph,
        space:          FrequencySpace,
        min_support:    int   = 2,
        min_confidence: float = 0.60,
    ) -> None:
        self.rel_mem  = rel_memory
        self.wm       = world_model
        self.cg       = causal_graph
        self.space    = space
        self.min_support    = min_support
        self.min_confidence = min_confidence

        self._rules:      List[AbstractRule] = []
        self._rule_index: Dict[Tuple[str,str], AbstractRule] = {}
        self._cycle_count = 0

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def run(self, verbose: bool = False) -> List[AbstractRule]:
        """
        Run all abstraction methods and return new rules discovered.
        Also stores strong rules back into relational memory.
        """
        self._cycle_count += 1
        new_rules: List[AbstractRule] = []

        new_rules += self._categorical_induction()
        new_rules += self._action_generalisation()
        new_rules += self._negative_inference()
        new_rules += self._temporal_patterns()

        # Deduplicate
        new_rules = [r for r in new_rules if self._is_novel(r)]

        # Store strong rules
        stored = 0
        for rule in new_rules:
            self._rules.append(rule)
            key = (rule.subject, rule.predicate)
            self._rule_index[key] = rule
            if rule.is_strong:
                self.rel_mem.add_fact(
                    rule.subject, rule.predicate, rule.value,
                    confidence=rule.confidence
                )
                stored += 1

        if verbose and new_rules:
            print(f"  [Abstraction] Found {len(new_rules)} rules "
                  f"({stored} stored to memory)")
            for r in new_rules[:4]:
                print(f"    {r.subject} → {r.predicate}={r.value} "
                      f"(conf={r.confidence:.2f}, support={r.support}, "
                      f"method={r.method})")

        return new_rules

    # ------------------------------------------------------------------
    # Method 1: Categorical induction
    # ------------------------------------------------------------------

    def _categorical_induction(self) -> List[AbstractRule]:
        """
        If multiple objects of the same category share a property,
        infer that property for the category.

        apple→edible=True, bread→edible=True, water→edible=True
        apple→is_a=food, bread→is_a=food, water→is_a=food
        → food→edible=True (if ≥ min_support objects)
        """
        rules = []
        # Group objects by category
        cat_to_objects: Dict[str, List[str]] = {}
        for concept in self.rel_mem._graph:
            cat = self.rel_mem.get_category(concept)
            if cat and cat != concept:  # avoid self-reference
                cat_to_objects.setdefault(cat, []).append(concept)

        for cat, objects in cat_to_objects.items():
            if len(objects) < self.min_support:
                continue

            # For each property, count how many objects in this category have it
            prop_values: Dict[str, Dict[Any, int]] = {}
            for obj in objects:
                for rel in self.rel_mem.get_facts(obj):
                    if rel.predicate in ("is_a", "category"):
                        continue
                    prop_values.setdefault(rel.predicate, {})
                    val_str = str(rel.obj)
                    prop_values[rel.predicate][val_str] = (
                        prop_values[rel.predicate].get(val_str, 0) + 1
                    )

            for pred, val_counts in prop_values.items():
                for val_str, count in val_counts.items():
                    if count >= self.min_support:
                        confidence = min(0.95, 0.5 + 0.15 * count)
                        if confidence >= self.min_confidence:
                            # Parse value back from string
                            val: Any = val_str
                            if val_str == "True":   val = True
                            elif val_str == "False": val = False
                            elif val_str.replace(".", "").replace("-", "").isdigit():
                                val = float(val_str)

                            rules.append(AbstractRule(
                                subject=cat,
                                predicate=pred,
                                value=val,
                                confidence=confidence,
                                support=count,
                                method="categorical_induction",
                                examples=objects[:4],
                            ))

        return rules

    # ------------------------------------------------------------------
    # Method 2: Action generalisation
    # ------------------------------------------------------------------

    def _action_generalisation(self) -> List[AbstractRule]:
        """
        If eat(A)→success, eat(B)→success and A,B are same category,
        generalise: eat(category)→success.
        """
        rules = []
        world_rules = self.wm.get_rules()

        # Group rules by (action, outcome)
        action_outcomes: Dict[Tuple[str,str], List[WorldRule]] = {}
        for wr in world_rules:
            key = (wr.action, wr.outcome)
            action_outcomes.setdefault(key, []).append(wr)

        for (action, outcome), rule_list in action_outcomes.items():
            if len(rule_list) < self.min_support:
                continue

            # Confidence = average across rules, boosted by support
            avg_conf = np.mean([r.confidence for r in rule_list])
            if avg_conf < self.min_confidence:
                continue

            # Most common category
            cat_counts: Dict[str, int] = {}
            for r in rule_list:
                cat_counts[r.object_category] = cat_counts.get(r.object_category, 0) + 1
            best_cat = max(cat_counts, key=lambda c: cat_counts[c])

            rules.append(AbstractRule(
                subject=f"{action}({best_cat})",
                predicate="outcome",
                value=outcome,
                confidence=min(0.95, avg_conf + 0.1),
                support=len(rule_list),
                method="action_generalisation",
                examples=[f"{r.action}({r.object_category})" for r in rule_list[:3]],
            ))

        return rules

    # ------------------------------------------------------------------
    # Method 3: Negative inference
    # ------------------------------------------------------------------

    def _negative_inference(self) -> List[AbstractRule]:
        """
        eat(stone)→failure, stone→material
        → material→edible=False
        """
        rules = []
        dangerous = self.rel_mem.find_dangerous()

        # Group by category
        cat_dangerous: Dict[str, List[str]] = {}
        for obj in dangerous:
            cat = self.rel_mem.get_category(obj)
            if cat:
                cat_dangerous.setdefault(cat, []).append(obj)

        for cat, objects in cat_dangerous.items():
            if len(objects) >= self.min_support:
                confidence = min(0.90, 0.6 + 0.1 * len(objects))
                rules.append(AbstractRule(
                    subject=cat,
                    predicate="edible",
                    value=False,
                    confidence=confidence,
                    support=len(objects),
                    method="negative_inference",
                    examples=objects[:4],
                ))

        return rules

    # ------------------------------------------------------------------
    # Method 4: Temporal pattern detection
    # ------------------------------------------------------------------

    def _temporal_patterns(self) -> List[AbstractRule]:
        """
        Detect patterns in transition sequences.
        pick(X) then eat(X) → success
        → "pick_then_eat" is a strategy for food.
        """
        rules = []
        transitions = self.wm._transitions

        if len(transitions) < 4:
            return rules

        # Look for consecutive pick→eat success sequences
        pick_eat_successes = 0
        pick_eat_attempts  = 0

        for i in range(len(transitions) - 1):
            t1 = transitions[i]
            t2 = transitions[i + 1]
            if t1.action == "pick" and t2.action == "eat":
                pick_eat_attempts += 1
                if t1.success and t2.success:
                    pick_eat_successes += 1

        if pick_eat_attempts >= self.min_support:
            confidence = pick_eat_successes / (pick_eat_attempts + 1e-8)
            if confidence >= self.min_confidence:
                rules.append(AbstractRule(
                    subject="pick_then_eat",
                    predicate="strategy_outcome",
                    value="success",
                    confidence=confidence,
                    support=pick_eat_successes,
                    method="temporal_pattern",
                    examples=["pick→eat sequence"],
                ))

        return rules

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_novel(self, rule: AbstractRule) -> bool:
        """Return True if this rule is not already known."""
        key = (rule.subject, rule.predicate)
        if key in self._rule_index:
            existing = self._rule_index[key]
            # Novel if confidence is significantly higher
            return rule.confidence > existing.confidence + 0.1
        # Check relational memory
        existing_val = self.rel_mem.get_value(rule.subject, rule.predicate)
        if existing_val is not None and str(existing_val) == str(rule.value):
            return False
        return True

    def query(self, subject: str, predicate: str) -> Optional[AbstractRule]:
        """Look up an abstract rule."""
        return self._rule_index.get((subject, predicate))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        strong  = [r for r in self._rules if r.is_strong]
        methods = {}
        for r in self._rules:
            methods[r.method] = methods.get(r.method, 0) + 1
        return {
            "total_rules":  len(self._rules),
            "strong_rules": len(strong),
            "by_method":    methods,
        }

    def __repr__(self) -> str:
        return (f"AbstractionEngine(rules={len(self._rules)}, "
                f"strong={sum(1 for r in self._rules if r.is_strong)})")
