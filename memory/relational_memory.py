"""
memory/relational_memory.py
============================
Relational Memory Layer

Extends the vector MemoryStore with a graph of typed relationships:

    apple  → is_a    → food
    apple  → color   → red
    apple  → edible  → True
    eat(apple) → causes → satisfied

Supports natural language-style queries:
    "what is apple?"          → get_facts("apple")
    "what can I eat?"         → query("edible", True)
    "what does eating do?"    → query_cause("eat")

Also maintains a concept graph for graph-traversal reasoning.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .memory_store import MemoryStore, MemoryEntry
from ..latent_space.frequency_space import FrequencySpace


class Relation:
    """A typed directed relation between two concepts."""
    __slots__ = ("subject", "predicate", "obj", "confidence", "count")

    def __init__(self, subject: str, predicate: str, obj: Any,
                 confidence: float = 1.0) -> None:
        self.subject    = subject
        self.predicate  = predicate
        self.obj        = obj
        self.confidence = confidence
        self.count      = 1

    def reinforce(self, amount: float = 0.05) -> None:
        self.count     += 1
        self.confidence = min(1.0, self.confidence + amount)

    def __repr__(self) -> str:
        return f"({self.subject} --{self.predicate}--> {self.obj} [{self.confidence:.2f}])"


class RelationalMemory:
    """
    Graph-augmented memory: combines vector retrieval with symbolic relations.

    Parameters
    ----------
    dim    : Vector dimension (must match encoder output).
    space  : Shared FrequencySpace.
    """

    def __init__(
        self,
        dim:   int = 64,
        space: Optional[FrequencySpace] = None,
    ) -> None:
        self.dim    = dim
        self.space  = space if space is not None else FrequencySpace(dim=dim)
        self.vectors = MemoryStore(dim=dim)

        # Graph storage: subject → predicate → [Relation]
        self._graph: Dict[str, Dict[str, List[Relation]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Reverse index: (predicate, object) → [subject]
        self._reverse: Dict[Tuple[str, Any], List[str]] = defaultdict(list)

    def _get_relation(self, subject: str, predicate: str, obj: Any) -> "Optional[Relation]":
        """Return existing Relation or None."""
        for rel in self._graph.get(subject, {}).get(predicate, []):
            if str(rel.obj) == str(obj):
                return rel
        return None

    # ------------------------------------------------------------------
    # Adding knowledge
    # ------------------------------------------------------------------

    def add_fact(
        self,
        subject:    str,
        predicate:  str,
        obj:        Any,
        confidence: float = 1.0,
        vector:     Optional[np.ndarray] = None,
    ) -> None:
        """
        Add a relational fact.
        e.g. add_fact("apple", "is_a", "food")
             add_fact("apple", "edible", True)
        """
        existing = self._get_relation(subject, predicate, obj)
        if existing:
            existing.reinforce()
            return

        rel = Relation(subject, predicate, obj, confidence)
        self._graph[subject][predicate].append(rel)
        self._reverse[(predicate, str(obj))].append(subject)

        # Store vector if provided
        if vector is not None:
            self.vectors.store(
                vector, label=f"{subject}:{predicate}:{obj}",
                modality="relational",
                metadata={"subject": subject, "predicate": predicate, "obj": obj},
            )

    def add_object_properties(
        self,
        obj_name: str,
        props:    Dict[str, Any],
        vector:   Optional[np.ndarray] = None,
    ) -> None:
        """Bulk-add properties of an object from a dict."""
        for pred, val in props.items():
            if pred in ("name", ):
                continue
            self.add_fact(obj_name, pred, val)
        if vector is not None:
            self.vectors.store(vector, label=obj_name, modality="object",
                               metadata={"props": props})

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_facts(self, subject: str) -> List[Relation]:
        """Return all known facts about a subject."""
        result = []
        for pred_list in self._graph.get(subject, {}).values():
            result.extend(pred_list)
        return sorted(result, key=lambda r: -r.confidence)

    def get_value(
        self,
        subject:    str,
        predicate:  str,
        default:    Any = None,
    ) -> Any:
        """Get the most confident value for subject's predicate."""
        relations = self._graph.get(subject, {}).get(predicate, [])
        if not relations:
            return default
        best = max(relations, key=lambda r: r.confidence)
        return best.obj

    def query(
        self,
        predicate: str,
        value:     Any,
    ) -> List[Tuple[str, float]]:
        """
        Find all subjects with predicate=value.
        Returns [(subject, confidence), ...] sorted by confidence.
        """
        subjects = self._reverse.get((predicate, str(value)), [])
        result = []
        for subj in subjects:
            rels = self._graph.get(subj, {}).get(predicate, [])
            for r in rels:
                if str(r.obj) == str(value):
                    result.append((subj, r.confidence))
        return sorted(result, key=lambda x: -x[1])

    def find_edible(self) -> List[str]:
        """What objects are known to be edible?"""
        return [s for s, _ in self.query("edible", True)]

    def find_dangerous(self) -> List[str]:
        """What objects are known to be non-edible or harmful?"""
        return [s for s, _ in self.query("edible", False)]

    def what_is(self, subject: str) -> str:
        """Generate a natural language description of a concept."""
        facts = self.get_facts(subject)
        if not facts:
            return f"'{subject}': unknown"
        parts = []
        for r in facts[:5]:
            parts.append(f"{r.predicate}={r.obj}")
        return f"'{subject}': " + ", ".join(parts)

    def is_known(self, concept: str) -> bool:
        return concept in self._graph and bool(self._graph[concept])

    def get_category(self, concept: str) -> Optional[str]:
        return self.get_value(concept, "is_a") or self.get_value(concept, "category")

    # ------------------------------------------------------------------
    # Vector-based recall
    # ------------------------------------------------------------------

    def recall_similar(
        self,
        query_vec: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[float, MemoryEntry]]:
        """Recall similar stored concept vectors."""
        return self.vectors.retrieve(query_vec, k=k)

    def store_concept_vector(
        self,
        label: str,
        vector: np.ndarray,
        modality: str = "concept",
    ) -> MemoryEntry:
        return self.vectors.store(vector, label=label, modality=modality)

    # ------------------------------------------------------------------
    # Learning from environment
    # ------------------------------------------------------------------

    def ingest_env_feedback(
        self,
        action: str,
        obj_name: str,
        obj_props: Dict[str, Any],
        success: bool,
        reward: float,
    ) -> None:
        """Automatically extract facts from one environment interaction."""
        # Object properties
        for k, v in obj_props.items():
            if k not in ("name", ):
                self.add_fact(obj_name, k, v)

        # Outcome of this action
        action_key = f"{action}({obj_name})"
        if success and reward >= 0.3:
            self.add_fact(action_key, "outcome", "satisfied")
            self.add_fact(action_key, "reward", round(reward, 2))
        elif success:
            self.add_fact(action_key, "outcome", "partial")
        elif reward <= -0.2:
            self.add_fact(action_key, "outcome", "harmful")
        else:
            self.add_fact(action_key, "outcome", "failed")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def n_facts(self) -> int:
        return sum(
            len(rels)
            for pred_dict in self._graph.values()
            for rels in pred_dict.values()
        )

    def summary(self) -> Dict:
        return {
            "concepts":    len(self._graph),
            "total_facts": self.n_facts(),
            "edible":      self.find_edible(),
            "dangerous":   self.find_dangerous(),
            "vector_mem":  len(self.vectors),
        }

    def __repr__(self) -> str:
        return (f"RelationalMemory(concepts={len(self._graph)}, "
                f"facts={self.n_facts()}, vectors={len(self.vectors)})")
