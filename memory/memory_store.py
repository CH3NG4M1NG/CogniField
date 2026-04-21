"""
memory/memory_store.py
======================
Associative Vector Memory

Stores (vector, label, metadata) tuples and supports:
  - Fast similarity-based retrieval
  - Online K-Means clustering (self-organisation)
  - Activation-based forgetting (Hebbian decay)
  - Batch operations

The memory acts as the system's "knowledge base" — everything seen
and validated is recorded here, and can be recalled by any query
vector regardless of modality.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans


@dataclass
class MemoryEntry:
    """A single memory record."""
    id:           str
    vector:       np.ndarray
    label:        str                   # human-readable tag
    modality:     str                   # "text" | "image" | "audio" | "fused"
    activation:   float  = 1.0         # decays over time
    access_count: int    = 0
    timestamp:    float  = field(default_factory=time.time)
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def decay(self, rate: float = 0.003) -> None:
        self.activation = max(0.0, self.activation - rate)

    def reinforce(self, amount: float = 0.15) -> None:
        self.activation = min(1.0, self.activation + amount)


class MemoryStore:
    """
    Associative memory that stores and retrieves latent vectors.

    Parameters
    ----------
    dim             : Vector dimensionality.
    max_size        : Maximum number of stored memories.
    similarity_threshold : Minimum cosine similarity to consider two vectors
                          "the same" (for deduplication).
    decay_rate      : Per-step activation decay.
    prune_threshold : Remove memories with activation below this.
    n_clusters      : Target number of self-organised clusters.
    """

    def __init__(
        self,
        dim: int = 128,
        max_size: int = 10_000,
        similarity_threshold: float = 0.95,
        decay_rate: float = 0.003,
        prune_threshold: float = 0.05,
        n_clusters: int = 16,
        seed: int = 42,
    ) -> None:
        self.dim                  = dim
        self.max_size             = max_size
        self.similarity_threshold = similarity_threshold
        self.decay_rate           = decay_rate
        self.prune_threshold      = prune_threshold
        self.n_clusters           = n_clusters

        self._entries: List[MemoryEntry] = []
        self._matrix: Optional[np.ndarray] = None     # (N, D) cache
        self._dirty  = True
        self._kmeans: Optional[MiniBatchKMeans] = None
        self._rng    = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_matrix(self) -> None:
        if self._entries:
            self._matrix = np.stack(
                [e.vector for e in self._entries], axis=0
            ).astype(np.float32)
        else:
            self._matrix = None
        self._dirty = False

    @property
    def _mat(self) -> Optional[np.ndarray]:
        if self._dirty:
            self._rebuild_matrix()
        return self._matrix

    @staticmethod
    def _l2(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + 1e-8)

    def _sims(self, query: np.ndarray) -> np.ndarray:
        mat = self._mat
        if mat is None:
            return np.array([])
        q = self._l2(query)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        return (mat / norms) @ q

    # ------------------------------------------------------------------
    # STORE
    # ------------------------------------------------------------------

    def store(
        self,
        vector: np.ndarray,
        label: str,
        modality: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
        allow_duplicate: bool = False,
    ) -> MemoryEntry:
        """
        Store a vector in memory.

        If a near-duplicate exists and allow_duplicate=False,
        reinforce the existing entry instead.

        Returns the stored or reinforced entry.
        """
        if not allow_duplicate and len(self._entries) > 0:
            sims = self._sims(vector)
            if len(sims) > 0 and sims.max() >= self.similarity_threshold:
                idx = int(sims.argmax())
                self._entries[idx].reinforce()
                self._entries[idx].access_count += 1
                return self._entries[idx]

        entry = MemoryEntry(
            id=str(uuid.uuid4())[:8],
            vector=self._l2(vector.copy()),
            label=label,
            modality=modality,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        self._dirty = True

        if len(self._entries) > self.max_size:
            self._prune()

        return entry

    def store_batch(
        self,
        vectors: np.ndarray,
        labels: List[str],
        modality: str = "text",
    ) -> List[MemoryEntry]:
        return [
            self.store(v, l, modality)
            for v, l in zip(vectors, labels)
        ]

    # ------------------------------------------------------------------
    # RETRIEVE
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: np.ndarray,
        k: int = 5,
        modality_filter: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> List[Tuple[float, MemoryEntry]]:
        """
        Retrieve the k most similar memories to a query vector.

        Parameters
        ----------
        query            : (dim,) query vector
        k                : Maximum results
        modality_filter  : If set, only return this modality
        min_similarity   : Minimum cosine similarity threshold

        Returns
        -------
        List of (similarity, entry) sorted descending.
        """
        if not self._entries:
            return []

        sims = self._sims(query)
        entries = self._entries

        # Apply modality filter
        if modality_filter:
            mask  = [i for i, e in enumerate(entries)
                     if e.modality == modality_filter]
            sims  = sims[mask] if len(mask) > 0 else np.array([])
            entries = [entries[i] for i in mask]

        if len(sims) == 0:
            return []

        k    = min(k, len(sims))
        top  = np.argpartition(sims, -k)[-k:]
        top  = top[np.argsort(sims[top])[::-1]]

        results = []
        for idx in top:
            sim = float(sims[idx])
            if sim >= min_similarity:
                e = entries[idx]
                e.reinforce(0.02)
                e.access_count += 1
                results.append((sim, e))

        return results

    def retrieve_by_label(self, label: str) -> Optional[MemoryEntry]:
        """Exact label lookup."""
        for e in self._entries:
            if e.label == label:
                return e
        return None

    # ------------------------------------------------------------------
    # CLUSTER
    # ------------------------------------------------------------------

    def cluster(self, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        Run MiniBatch K-Means on stored vectors.
        Returns cluster centroids (K, dim).
        """
        mat = self._mat
        if mat is None or len(mat) < 2:
            return np.zeros((0, self.dim))

        K = min(n_clusters or self.n_clusters, len(mat))
        self._kmeans = MiniBatchKMeans(
            n_clusters=K, n_init=3, random_state=42
        )
        self._kmeans.fit(mat)
        centroids = self._kmeans.cluster_centers_.astype(np.float32)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8
        return centroids / norms

    def get_cluster_label(self, vector: np.ndarray) -> int:
        """Return the nearest cluster index."""
        if self._kmeans is None:
            self.cluster()
        if self._kmeans is None:
            return 0
        return int(self._kmeans.predict(
            vector.reshape(1, -1)
        )[0])

    # ------------------------------------------------------------------
    # MAINTENANCE
    # ------------------------------------------------------------------

    def decay_all(self) -> None:
        """Apply decay to all entries (call once per step)."""
        for e in self._entries:
            e.decay(self.decay_rate)

    def _prune(self) -> int:
        before = len(self._entries)
        self._entries = [
            e for e in self._entries
            if e.activation >= self.prune_threshold
        ]
        removed = before - len(self._entries)
        if removed > 0:
            self._dirty = True
        return removed

    def prune(self) -> int:
        return self._prune()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def labels(self) -> List[str]:
        return [e.label for e in self._entries]

    @property
    def modalities(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in self._entries:
            counts[e.modality] = counts.get(e.modality, 0) + 1
        return counts

    def summary(self) -> Dict:
        if not self._entries:
            return {"size": 0}
        acts = [e.activation for e in self._entries]
        return {
            "size": len(self._entries),
            "mean_activation": float(np.mean(acts)),
            "modalities": self.modalities,
        }

    def __repr__(self) -> str:
        return (f"MemoryStore(size={len(self)}/{self.max_size}, "
                f"dim={self.dim})")
