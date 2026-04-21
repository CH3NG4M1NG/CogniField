"""
latent_space/frequency_space.py
================================
The Unified Frequency Space

This is the heart of CogniField.

Concept
-------
"Frequency space" is NOT about literal Hz.  It is a metaphor for how
the brain might represent meaning: as a continuous field where:

  • Similar meanings are geometrically close   (high cosine similarity)
  • Compositions create new semantic positions (vector arithmetic)
  • Distant points represent incompatible or novel concepts
  • The space has structure: clusters, manifolds, gradients

All encoders (text, image, audio) project into this same D-dimensional
unit hypersphere.  Once there, a text like "I eat apple" and an image
of an apple should be close together.

This module provides:
  - Similarity computation
  - Vector composition (several operators)
  - Projection alignment (Procrustes) across modalities
  - Space visualisation support (PCA to 2D)
  - Analogy queries: A:B :: C:?
"""

from __future__ import annotations

import enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA


class ComposeMode(str, enum.Enum):
    """Vector composition operators."""
    ADD       = "add"        # normalised sum
    MEAN      = "mean"       # mean (centroid)
    GEOMETRIC = "geometric"  # geometric mean (log space)
    WEIGHTED  = "weighted"   # weighted sum
    ANALOGY   = "analogy"    # A:B::C:? → C + (B-A)


class FrequencySpace:
    """
    Unified D-dimensional latent space for all modalities.

    Parameters
    ----------
    dim  : Dimensionality (must match all encoders).
    """

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim
        self._pca: Optional[PCA] = None

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    @staticmethod
    def l2(vec: np.ndarray) -> np.ndarray:
        """L2-normalise a vector onto the unit sphere."""
        n = np.linalg.norm(vec)
        return vec / (n + 1e-8)

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity ∈ [-1, 1].
        Equivalent to dot product of unit vectors.
        """
        return float(np.dot(a, b) /
                     (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    @staticmethod
    def distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Angular distance ∈ [0, 1].
        0 = identical, 1 = opposite.
        """
        cos = FrequencySpace.similarity(a, b)
        return (1.0 - cos) / 2.0

    @staticmethod
    def batch_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Cosine similarity of query against all rows of matrix.
        query  : (D,)
        matrix : (N, D)
        Returns (N,) similarity scores.
        """
        q = query / (np.linalg.norm(query) + 1e-8)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        return (matrix / norms) @ q

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def combine(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
        mode: ComposeMode = ComposeMode.ADD,
        weight_a: float = 0.5,
    ) -> np.ndarray:
        """
        Combine two latent vectors into one.

        Parameters
        ----------
        vec_a, vec_b : Unit vectors in frequency space.
        mode         : Composition operator.
        weight_a     : Weight of vec_a in WEIGHTED mode.

        Returns
        -------
        np.ndarray  shape (dim,), L2-normalised.
        """
        return self.compose([vec_a, vec_b], mode=mode,
                            weights=[weight_a, 1.0 - weight_a])

    def compose(
        self,
        vecs: List[np.ndarray],
        mode: ComposeMode = ComposeMode.MEAN,
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Compose an arbitrary number of latent vectors.

        Parameters
        ----------
        vecs    : List of (dim,) unit vectors.
        mode    : Composition operator.
        weights : Per-vector weights (WEIGHTED mode only).

        Returns
        -------
        np.ndarray  shape (dim,), L2-normalised.
        """
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        if len(vecs) == 1:
            return self.l2(vecs[0].copy())

        mat = np.stack(vecs, axis=0).astype(np.float32)

        if mode == ComposeMode.ADD or mode == ComposeMode.MEAN:
            result = mat.sum(axis=0)
        elif mode == ComposeMode.WEIGHTED:
            w = np.array(weights or [1.0] * len(vecs), dtype=float)
            w /= w.sum()
            result = (mat * w[:, None]).sum(axis=0)
        elif mode == ComposeMode.GEOMETRIC:
            eps = 1e-8
            log_abs = np.log(np.abs(mat) + eps)
            signs   = np.sign(mat.mean(axis=0) + eps)
            result  = signs * np.exp(log_abs.mean(axis=0))
        else:
            result = mat.sum(axis=0)

        return self.l2(result)

    def analogy(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
    ) -> np.ndarray:
        """
        Vector analogy: A is to B as C is to ?
        answer ≈ C + (B - A)

        Parameters
        ----------
        a, b, c : Unit vectors. Query: A:B :: C:?

        Returns
        -------
        np.ndarray  shape (dim,), L2-normalised.
        """
        delta  = b - a
        answer = self.l2(c + delta)
        return answer

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def align_modalities(
        self,
        source_vecs: np.ndarray,
        target_vecs: np.ndarray,
        n_iter: int = 50,
        lr: float = 0.05,
    ) -> np.ndarray:
        """
        Learn a rotation matrix R such that
        source_vecs @ R ≈ target_vecs

        Uses gradient-free Procrustes approximation.

        Parameters
        ----------
        source_vecs : (N, D) - vectors from one modality
        target_vecs : (N, D) - corresponding vectors from another modality

        Returns
        -------
        R : (D, D) rotation matrix
        """
        # Orthogonal Procrustes: minimise ||S R - T||_F
        M  = source_vecs.T @ target_vecs
        U, _, Vt = np.linalg.svd(M)
        R  = U @ Vt
        return R.astype(np.float32)

    # ------------------------------------------------------------------
    # PCA visualisation
    # ------------------------------------------------------------------

    def fit_pca(self, vecs: np.ndarray) -> "FrequencySpace":
        """Fit PCA for 2D visualisation of the space."""
        self._pca = PCA(n_components=2)
        self._pca.fit(vecs)
        return self

    def project_2d(self, vecs: np.ndarray) -> np.ndarray:
        """Project vectors to 2D for plotting. Returns (N, 2)."""
        if self._pca is None:
            self.fit_pca(vecs)
        return self._pca.transform(vecs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def nearest_in_batch(
        self,
        query: np.ndarray,
        pool: np.ndarray,
        k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest vectors in pool to query.
        Returns (indices, similarities) sorted descending.
        """
        sims = self.batch_similarity(query, pool)
        k    = min(k, len(sims))
        idx  = np.argpartition(sims, -k)[-k:]
        idx  = idx[np.argsort(sims[idx])[::-1]]
        return idx, sims[idx]

    def centroid(self, vecs: List[np.ndarray]) -> np.ndarray:
        """Mean of a set of vectors, L2-normalised."""
        return self.l2(np.stack(vecs).mean(axis=0))

    def __repr__(self) -> str:
        return f"FrequencySpace(dim={self.dim})"
