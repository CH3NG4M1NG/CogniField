"""
curiosity/advanced_curiosity.py
================================
Advanced Curiosity System with Hypothesis Testing

Upgrades over basic novelty detection:

  1. Hypothesis Generation
     When unknown input detected → formulate hypotheses about its properties.
     e.g. "purple_berry may be edible (similar to known food)"
          "glowing_cube may be fragile (similar color to glass_jar)"

  2. Hypothesis Testing
     Rank hypotheses by testability + importance.
     Suggest actions that would confirm or refute them.
     e.g. "to test if purple_berry is edible: inspect it"

  3. Learning Gap Tracking
     Track which concepts have the most uncertainty.
     Prioritise exploration of high-uncertainty concepts.

  4. Exploration State
     Explored set: concepts whose properties are known.
     Frontier: concepts partially known, worth investigating.
     Unknown: no information.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..latent_space.frequency_space import FrequencySpace
from ..memory.memory_store import MemoryStore
from ..memory.relational_memory import RelationalMemory


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """A testable hypothesis about an unknown concept."""
    subject:     str           # the unknown object
    predicate:   str           # property to test (e.g. "edible")
    predicted:   Any           # predicted value (e.g. True)
    confidence:  float         # how confident we are before testing [0,1]
    basis:       str           # why we think this (e.g. "similar to apple")
    test_action: str           # action that would confirm/refute it
    status:      str = "open"  # "open" | "confirmed" | "refuted"
    created_at:  float = field(default_factory=time.time)

    def confirm(self) -> None:
        self.status     = "confirmed"
        self.confidence = min(1.0, self.confidence + 0.3)

    def refute(self) -> None:
        self.status     = "refuted"
        self.confidence = max(0.0, self.confidence - 0.5)


# ---------------------------------------------------------------------------
# AdvancedCuriosityEngine
# ---------------------------------------------------------------------------

class AdvancedCuriosityEngine:
    """
    Curiosity engine with hypothesis generation and testing.

    Parameters
    ----------
    space             : Shared FrequencySpace.
    rel_memory        : RelationalMemory for concept lookups.
    vec_memory        : Basic MemoryStore for vector similarity.
    novelty_threshold : Score above which something is "novel".
    """

    def __init__(
        self,
        space:             Optional[FrequencySpace]   = None,
        rel_memory:        Optional[RelationalMemory] = None,
        vec_memory:        Optional[MemoryStore]      = None,
        novelty_threshold: float = 0.4,
        dim:               int   = 64,
        seed:              int   = 42,
    ) -> None:
        self.space             = space      if space      is not None else FrequencySpace(dim=dim)
        self.rel_memory        = rel_memory if rel_memory is not None else RelationalMemory(dim=dim)
        self.vec_memory        = vec_memory if vec_memory is not None else MemoryStore(dim=dim)
        self.novelty_threshold = novelty_threshold
        self._rng              = np.random.default_rng(seed)

        # Tracking
        self._hypotheses:    List[Hypothesis] = []
        self._explored:      set = set()      # concepts fully known
        self._frontier:      set = set()      # concepts partially known
        self._unknown_count  = 0
        self._exploration_log: List[Dict] = []

    # ------------------------------------------------------------------
    # Novelty
    # ------------------------------------------------------------------

    def detect_novelty(
        self,
        vector: np.ndarray,
        concept_label: str = "",
    ) -> float:
        """
        Novelty score ∈ [0,1].
        Combines vector distance + symbolic knowledge gap.
        """
        # Vector-based novelty
        recalls = self.vec_memory.retrieve(vector, k=3)
        if not recalls:
            vec_novelty = 1.0
        else:
            max_sim = max(s for s, _ in recalls)
            vec_novelty = 1.0 - float(max_sim)

        # Symbolic novelty: is this concept known?
        sym_novelty = 0.0
        if concept_label and not self.rel_memory.is_known(concept_label):
            sym_novelty = 0.5

        # Combined
        novelty = 0.7 * vec_novelty + 0.3 * sym_novelty
        return float(np.clip(novelty, 0.0, 1.0))

    def is_novel(self, vector: np.ndarray, label: str = "") -> bool:
        return self.detect_novelty(vector, label) >= self.novelty_threshold

    # ------------------------------------------------------------------
    # Hypothesis generation
    # ------------------------------------------------------------------

    def generate_hypotheses(
        self,
        unknown_concept: str,
        unknown_vec:     np.ndarray,
        known_props:     Optional[Dict[str, Any]] = None,
    ) -> List[Hypothesis]:
        """
        Generate hypotheses about an unknown concept.

        Strategy:
        1. Find nearest known concept by vector similarity.
        2. Inherit its properties as hypotheses.
        3. Adjust confidence by similarity score.
        """
        hypotheses = []
        recalls = self.vec_memory.retrieve(unknown_vec, k=3)

        if recalls:
            best_sim, best_entry = recalls[0]
            similar_concept = best_entry.label

            # Inherit properties from similar concept
            if best_sim >= 0.4:
                sim_facts = self.rel_memory.get_facts(similar_concept)
                for fact in sim_facts[:4]:
                    conf = best_sim * fact.confidence * 0.8  # discounted confidence
                    test_actions = self._test_action_for(fact.predicate)
                    h = Hypothesis(
                        subject=unknown_concept,
                        predicate=fact.predicate,
                        predicted=fact.obj,
                        confidence=conf,
                        basis=f"similar to '{similar_concept}' (sim={best_sim:.2f})",
                        test_action=test_actions,
                    )
                    hypotheses.append(h)

            # Category hypothesis
            category = best_entry.metadata.get("category",
                        self.rel_memory.get_category(similar_concept))
            if category and best_sim >= 0.3:
                h = Hypothesis(
                    subject=unknown_concept,
                    predicate="category",
                    predicted=category,
                    confidence=best_sim * 0.7,
                    basis=f"similar to '{similar_concept}' [{category}]",
                    test_action="inspect",
                )
                hypotheses.append(h)

        # Known props from environment (override with observed truth)
        if known_props:
            for prop, val in known_props.items():
                # If we have a hypothesis that contradicts, refute it
                for h in hypotheses:
                    if h.predicate == prop and str(h.predicted) != str(val):
                        h.refute()
                # Add confirmed fact as high-confidence hypothesis
                h = Hypothesis(
                    subject=unknown_concept,
                    predicate=prop,
                    predicted=val,
                    confidence=1.0,
                    basis="directly observed",
                    test_action="none",
                    status="confirmed",
                )
                hypotheses.append(h)

        self._hypotheses.extend(hypotheses)
        self._frontier.add(unknown_concept)
        return hypotheses

    def _test_action_for(self, predicate: str) -> str:
        """Suggest action to test a predicate."""
        mapping = {
            "edible":   "eat",
            "fragile":  "drop",
            "heavy":    "pick",
            "category": "inspect",
            "color":    "observe",
            "size":     "observe",
            "is_a":     "inspect",
        }
        return mapping.get(predicate, "inspect")

    # ------------------------------------------------------------------
    # Hypothesis testing
    # ------------------------------------------------------------------

    def update_hypotheses(
        self,
        concept:   str,
        predicate: str,
        observed:  Any,
    ) -> List[Hypothesis]:
        """
        Update hypotheses about concept.predicate based on observed value.
        Returns list of affected hypotheses.
        """
        affected = []
        for h in self._hypotheses:
            if h.subject == concept and h.predicate == predicate and h.status == "open":
                if str(h.predicted) == str(observed):
                    h.confirm()
                else:
                    h.refute()
                affected.append(h)

        # Move to explored if enough confirmed facts
        confirmed = [h for h in self._hypotheses
                     if h.subject == concept and h.status == "confirmed"]
        if len(confirmed) >= 2:
            self._explored.add(concept)
            self._frontier.discard(concept)

        return affected

    def best_hypothesis_to_test(self) -> Optional[Hypothesis]:
        """
        Return the most valuable open hypothesis to test.
        Prioritises: high-uncertainty (0.3–0.6 confidence) + important predicates.
        """
        open_hs = [h for h in self._hypotheses if h.status == "open"]
        if not open_hs:
            return None

        important_preds = {"edible", "fragile", "category", "is_a"}

        def priority(h: Hypothesis) -> float:
            importance = 2.0 if h.predicate in important_preds else 1.0
            uncertainty = 1.0 - abs(h.confidence - 0.5) * 2  # max at 0.5
            return importance * uncertainty

        return max(open_hs, key=priority)

    # ------------------------------------------------------------------
    # Exploration priority
    # ------------------------------------------------------------------

    def exploration_priority(self, concept: str, vector: np.ndarray) -> float:
        """
        Score how much to prioritise exploring this concept [0,1].
        High priority = novel + potentially important.
        """
        novelty = self.detect_novelty(vector, concept)
        in_frontier = float(concept in self._frontier)
        is_unknown  = float(concept not in self._explored)

        # Danger uncertainty bonus: if we're not sure it's safe, explore carefully
        edible_val = self.rel_memory.get_value(concept, "edible")
        safety_uncertainty = 0.3 if edible_val is None else 0.0

        return float(np.clip(
            0.5 * novelty + 0.2 * in_frontier + 0.2 * is_unknown + 0.1 * safety_uncertainty,
            0.0, 1.0
        ))

    def curiosity_weight(self, vector: np.ndarray, label: str = "") -> float:
        """Return learning-rate multiplier [1.0, 3.0] for novel inputs."""
        novelty = self.detect_novelty(vector, label)
        return 1.0 + 2.0 * novelty

    # ------------------------------------------------------------------
    # Full exploration trigger
    # ------------------------------------------------------------------

    def explore(
        self,
        concept:     str,
        vector:      np.ndarray,
        known_props: Optional[Dict[str, Any]] = None,
        modality:    str = "unknown",
    ) -> Dict:
        """
        Fully trigger exploration of an unknown concept.
        Returns exploration report with hypotheses and suggested actions.
        """
        self._unknown_count += 1
        novelty      = self.detect_novelty(vector, concept)
        hypotheses   = self.generate_hypotheses(concept, vector, known_props)
        best_h       = self.best_hypothesis_to_test()
        priority_score = self.exploration_priority(concept, vector)

        # Store in vector memory
        self.vec_memory.store(
            vector, label=concept, modality=modality,
            metadata={"novelty": novelty, "n_hypotheses": len(hypotheses)},
        )

        report = {
            "concept":          concept,
            "novelty":          novelty,
            "priority":         priority_score,
            "n_hypotheses":     len(hypotheses),
            "hypotheses":       [(h.predicate, h.predicted, h.confidence, h.basis)
                                 for h in hypotheses[:5]],
            "suggested_action": best_h.test_action if best_h else "inspect",
            "best_test":        best_h.subject + "." + best_h.predicate if best_h else "–",
        }
        self._exploration_log.append(report)
        return report

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def n_explorations(self) -> int:
        return len(self._exploration_log)

    def summary(self) -> Dict:
        open_h  = [h for h in self._hypotheses if h.status == "open"]
        conf_h  = [h for h in self._hypotheses if h.status == "confirmed"]
        ref_h   = [h for h in self._hypotheses if h.status == "refuted"]
        return {
            "explorations":       self.n_explorations,
            "explored_concepts":  len(self._explored),
            "frontier":           len(self._frontier),
            "hypotheses_open":    len(open_h),
            "hypotheses_confirmed": len(conf_h),
            "hypotheses_refuted": len(ref_h),
            "edible_known":       self.rel_memory.find_edible(),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"AdvancedCuriosityEngine("
                f"explorations={s['explorations']}, "
                f"explored={s['explored_concepts']}, "
                f"hypotheses={s['hypotheses_open']} open)")
