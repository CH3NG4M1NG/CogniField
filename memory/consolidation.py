"""
memory/consolidation.py
========================
Long-Term Memory Consolidation

Converts many specific experiences into compact, general knowledge.
Runs periodically (like sleep-based memory consolidation in biology).

Three Operations
----------------
1. MERGE
   Group similar memory entries → create one representative "concept" entry.
   e.g. 30 experiences of "ate apple" → one strong "apple_food" entry.

2. STRENGTHEN
   Entries accessed frequently get reinforced (Hebbian: "fire together, wire together").
   Low-activation, rarely accessed entries decay toward removal.

3. PRUNE
   Remove entries below activation threshold.
   Keeps memory lean and prevents interference from stale knowledge.

4. ABSTRACT
   Detect clusters of relational facts that share a common pattern
   → create higher-level abstract entries.
   e.g. apple→edible, bread→edible, water→edible
      → food category entries get generalised "edible=True" rule strengthened.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from ..memory.memory_store import MemoryStore, MemoryEntry
from ..memory.relational_memory import RelationalMemory
from ..latent_space.frequency_space import FrequencySpace


@dataclass
class ConsolidationReport:
    """Summary of one consolidation cycle."""
    merged:     int = 0
    pruned:     int = 0
    strengthened: int = 0
    abstractions: int = 0
    before_size: int = 0
    after_size:  int = 0
    elapsed_ms:  float = 0.0
    timestamp:   float = field(default_factory=time.time)


class MemoryConsolidator:
    """
    Periodically consolidates and prunes vector + relational memory.

    Parameters
    ----------
    vec_memory   : The agent's MemoryStore (vector entries).
    rel_memory   : The agent's RelationalMemory (concept graph).
    space        : Shared FrequencySpace.
    merge_threshold : Cosine similarity above which entries are merged.
    prune_threshold : Activation below which entries are pruned.
    min_merge_cluster : Minimum cluster size to trigger a merge.
    """

    def __init__(
        self,
        vec_memory:        MemoryStore,
        rel_memory:        RelationalMemory,
        space:             FrequencySpace,
        merge_threshold:   float = 0.92,
        prune_threshold:   float = 0.08,
        min_merge_cluster: int   = 3,
        seed:              int   = 42,
    ) -> None:
        self.vec_mem         = vec_memory
        self.rel_mem         = rel_memory
        self.space           = space
        self.merge_threshold = merge_threshold
        self.prune_threshold = prune_threshold
        self.min_merge       = min_merge_cluster
        self._rng            = np.random.default_rng(seed)
        self._reports:       List[ConsolidationReport] = []
        self._cycle_count    = 0

    # ------------------------------------------------------------------
    # Main consolidation cycle
    # ------------------------------------------------------------------

    def consolidate(self, verbose: bool = False) -> ConsolidationReport:
        """
        Run one full consolidation cycle.
        Returns a ConsolidationReport with what was done.
        """
        t0     = time.time()
        report = ConsolidationReport(before_size=len(self.vec_mem))
        self._cycle_count += 1

        # Phase 1: Decay all entries
        self.vec_mem.decay_all()

        # Phase 2: Strengthen frequently accessed entries
        report.strengthened = self._strengthen()

        # Phase 3: Merge near-duplicates
        report.merged = self._merge()

        # Phase 4: Prune weak entries
        report.pruned = self._prune()

        # Phase 5: Abstract relational knowledge
        report.abstractions = self._abstract_relational()

        report.after_size = len(self.vec_mem)
        report.elapsed_ms = (time.time() - t0) * 1000

        self._reports.append(report)

        if verbose:
            print(f"  [Consolidation #{self._cycle_count}] "
                  f"merged={report.merged}, pruned={report.pruned}, "
                  f"strengthened={report.strengthened}, "
                  f"abstract={report.abstractions}, "
                  f"size: {report.before_size}→{report.after_size} "
                  f"({report.elapsed_ms:.0f}ms)")

        return report

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _strengthen(self) -> int:
        """Reinforce entries that have been accessed many times."""
        count = 0
        for entry in self.vec_mem._entries:
            if entry.access_count >= 3:
                entry.reinforce(0.05 * min(entry.access_count, 10))
                count += 1
        return count

    def _merge(self) -> int:
        """
        Cluster similar vector entries and merge each cluster
        into one representative entry.
        """
        if len(self.vec_mem) < self.min_merge * 2:
            return 0

        mat = self.vec_mem._mat
        if mat is None or len(mat) < self.min_merge:
            return 0

        # Find pairs above merge_threshold
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        normed = mat / norms
        # Sample a subset for efficiency
        n = min(200, len(normed))
        idx_sample = self._rng.choice(len(normed), n, replace=False)
        sub = normed[idx_sample]
        sim_matrix = sub @ sub.T   # (n, n)

        merged_indices = set()
        merge_count = 0

        for i in range(n):
            if idx_sample[i] in merged_indices:
                continue
            partners = [idx_sample[j] for j in range(n)
                       if i != j
                       and sim_matrix[i, j] >= self.merge_threshold
                       and idx_sample[j] not in merged_indices]
            if len(partners) < self.min_merge - 1:
                continue

            # Merge: create centroid entry, remove the rest
            group_idx = [idx_sample[i]] + partners[:self.min_merge - 1]
            group_entries = [self.vec_mem._entries[k] for k in group_idx
                            if k < len(self.vec_mem._entries)]
            if len(group_entries) < 2:
                continue

            # Centroid vector
            centroid = np.mean([e.vector for e in group_entries], axis=0)
            centroid = self.space.l2(centroid.astype(np.float32))

            # Best label (highest activation wins)
            best = max(group_entries, key=lambda e: e.activation)
            best.vector = centroid
            best.activation = min(1.0, max(e.activation for e in group_entries) + 0.1)
            best.access_count += sum(e.access_count for e in group_entries[1:])

            # Mark others for removal
            for e in group_entries[1:]:
                e.activation = 0.0

            for k in group_idx:
                merged_indices.add(k)
            merge_count += 1

        # Remove zero-activation entries
        if merge_count > 0:
            self.vec_mem._entries = [e for e in self.vec_mem._entries
                                     if e.activation > 0.0]
            self.vec_mem._dirty = True

        return merge_count

    def _prune(self) -> int:
        """Remove entries below activation threshold."""
        before = len(self.vec_mem._entries)
        self.vec_mem._entries = [
            e for e in self.vec_mem._entries
            if e.activation >= self.prune_threshold
        ]
        removed = before - len(self.vec_mem._entries)
        if removed > 0:
            self.vec_mem._dirty = True
        return removed

    def _abstract_relational(self) -> int:
        """
        Look for patterns in relational memory and strengthen generalisations.

        Example:
          apple→edible=True, bread→edible=True, water→edible=True
          + apple→is_a=food, bread→is_a=food, water→is_a=food
          → strengthen the rule: food→edible=True (already exists or create)
        """
        count = 0

        # Gather edible objects grouped by category
        edible_by_cat: Dict[str, List[str]] = {}
        for obj in self.rel_mem.find_edible():
            cat = self.rel_mem.get_category(obj)
            if cat:
                edible_by_cat.setdefault(cat, []).append(obj)

        # If 2+ objects in same category are edible → reinforce category rule
        for cat, objects in edible_by_cat.items():
            if len(objects) >= 2:
                # Add/reinforce: category → edible = True
                confidence = min(1.0, 0.5 + 0.1 * len(objects))
                self.rel_mem.add_fact(cat, "edible", True, confidence=confidence)
                count += 1

        # Dangerous objects grouped by category
        dangerous_by_cat: Dict[str, List[str]] = {}
        for obj in self.rel_mem.find_dangerous():
            cat = self.rel_mem.get_category(obj)
            if cat:
                dangerous_by_cat.setdefault(cat, []).append(obj)

        for cat, objects in dangerous_by_cat.items():
            if len(objects) >= 2:
                confidence = min(1.0, 0.5 + 0.1 * len(objects))
                self.rel_mem.add_fact(cat, "edible", False, confidence=confidence)
                count += 1

        return count

    # ------------------------------------------------------------------
    # Selective consolidation (on-demand for one entry group)
    # ------------------------------------------------------------------

    def consolidate_concept(self, concept_label: str) -> bool:
        """
        Specifically consolidate all memory entries related to one concept.
        Returns True if any merging happened.
        """
        entries = [e for e in self.vec_mem._entries
                   if concept_label.lower() in e.label.lower()]
        if len(entries) < 2:
            return False

        vecs = np.stack([e.vector for e in entries])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        normed = vecs / norms
        sims = normed @ normed.T

        merged = False
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                if sims[i, j] >= self.merge_threshold:
                    # Merge j into i
                    entries[i].vector = self.space.l2(
                        ((entries[i].vector + entries[j].vector) / 2).astype(np.float32)
                    )
                    entries[i].activation = max(entries[i].activation, entries[j].activation)
                    entries[j].activation = 0.0
                    merged = True

        if merged:
            self.vec_mem._entries = [e for e in self.vec_mem._entries
                                     if e.activation > 0.0]
            self.vec_mem._dirty = True
        return merged

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    def total_pruned(self) -> int:
        return sum(r.pruned for r in self._reports)

    def total_merged(self) -> int:
        return sum(r.merged for r in self._reports)

    def summary(self) -> Dict:
        if not self._reports:
            return {"cycles": 0}
        last = self._reports[-1]
        return {
            "cycles":       self._cycle_count,
            "total_pruned": self.total_pruned(),
            "total_merged": self.total_merged(),
            "last_cycle":   {
                "merged":       last.merged,
                "pruned":       last.pruned,
                "strengthened": last.strengthened,
                "abstractions": last.abstractions,
                "size_change":  last.after_size - last.before_size,
            },
        }

    def __repr__(self) -> str:
        return (f"MemoryConsolidator(cycles={self._cycle_count}, "
                f"pruned={self.total_pruned()}, merged={self.total_merged()})")
