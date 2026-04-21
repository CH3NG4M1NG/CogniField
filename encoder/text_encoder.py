"""
encoder/text_encoder.py
========================
Text → Latent Vector Encoder

Architecture
------------
Rather than a fixed character-frequency mapping, we learn a semantic
embedding using a combination of:

  1. Bag-of-character-ngrams  (character n-grams, n=2..4)
     → captures morphology, subword patterns
  2. Bag-of-words (word unigrams)
     → captures lexical semantics
  3. SVD projection into the shared latent space (dim=D)
     → creates a continuous "frequency field" where similar
       texts are geometrically close

This is the "frequency field" concept made concrete:
  text → learned feature space → D-dim unit vector

The encoder is fitted on a reference corpus and can then
encode any new text zero-shot via the learned projection.
"""

from __future__ import annotations

import re
import math
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class TextEncoder:
    """
    Encodes text strings into fixed-size latent vectors.

    Parameters
    ----------
    dim         : Output dimensionality (latent space dimension).
    ngram_range : Character n-gram range for sub-word features.
    min_df      : Minimum document frequency for vocabulary terms.
    seed        : RNG seed for reproducibility.
    """

    # A small default corpus that ensures the encoder can bootstrap
    # without external data.
    _BOOTSTRAP_CORPUS: List[str] = [
        "the cat sat on the mat",
        "a dog runs in the park",
        "I eat an apple every morning",
        "the red apple fell from the tree",
        "birds fly high in the sky",
        "water flows down the mountain stream",
        "she reads a book by the window",
        "the sun rises in the east",
        "light travels faster than sound",
        "knowledge grows through curiosity",
        "pick up the stone from the ground",
        "move the object to the left",
        "observe the environment carefully",
        "the agent learns from feedback",
        "memory stores and retrieves patterns",
        "frequency space maps meaning to geometry",
        "similar concepts cluster together",
        "novel inputs trigger exploration",
        "reasoning corrects errors over time",
        "multimodal inputs share one space",
        "image shows a bright red apple",
        "audio signal contains spoken words",
        "the system encodes all modalities",
        "latent vectors represent semantic content",
        "cosine similarity measures conceptual distance",
    ]

    def __init__(
        self,
        dim: int = 128,
        ngram_range: Tuple[int, int] = (2, 4),
        min_df: int = 1,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.seed = seed
        self._fitted = False

        # Character n-gram TF-IDF → captures morphology
        self._char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
            max_features=8000,
        )
        # Word-level TF-IDF → captures semantics
        self._word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=min_df,
            sublinear_tf=True,
            max_features=5000,
        )
        # Latent space projection via SVD (like LSA)
        n_components = min(dim, 200)
        self._svd = TruncatedSVD(n_components=n_components, random_state=seed)
        # Final linear projection to exact dim
        rng = np.random.default_rng(seed)
        self._proj: Optional[np.ndarray] = None
        self._n_components = n_components

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, corpus: Optional[List[str]] = None) -> "TextEncoder":
        """
        Fit the encoder on a text corpus.
        Uses the built-in bootstrap corpus if none provided.
        """
        corpus = corpus or []
        full_corpus = list(set(self._BOOTSTRAP_CORPUS + corpus))

        char_matrix = self._char_vec.fit_transform(full_corpus)
        word_matrix = self._word_vec.fit_transform(full_corpus)

        # Horizontally stack char + word features
        from scipy.sparse import hstack
        combined = hstack([char_matrix, word_matrix])

        # Fit SVD
        actual_components = min(self._n_components, combined.shape[1] - 1,
                                combined.shape[0] - 1)
        self._svd.n_components = actual_components
        self._svd.fit(combined)

        # Random final projection to target dim
        rng = np.random.default_rng(self.seed)
        self._proj = rng.standard_normal(
            (actual_components, self.dim)
        ).astype(np.float32)
        # Orthonormalise columns
        Q, _ = np.linalg.qr(self._proj)
        self._proj = Q[:, :self.dim].astype(np.float32) \
            if Q.shape[1] >= self.dim else self._proj

        self._fitted = True
        return self

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        """Fit and encode in one call. Returns (N, dim) matrix."""
        self.fit(corpus)
        return np.stack([self.encode(t) for t in corpus])

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a string into a unit latent vector.

        Parameters
        ----------
        text : str

        Returns
        -------
        np.ndarray  shape (dim,)  float32, L2-normalised
        """
        if not self._fitted:
            self.fit()

        text = self._preprocess(text)

        from scipy.sparse import hstack
        char_feat = self._char_vec.transform([text])
        word_feat = self._word_vec.transform([text])
        combined  = hstack([char_feat, word_feat])

        svd_vec = self._svd.transform(combined)[0].astype(np.float32)
        projected = svd_vec @ self._proj
        return self._l2_norm(projected)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts. Returns (N, dim)."""
        return np.stack([self.encode(t) for t in texts])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(text: str) -> str:
        """Lowercase, collapse whitespace, keep punctuation."""
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _l2_norm(vec: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(vec)
        return vec / (n + 1e-8)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two text encodings."""
        va = self.encode(text_a)
        vb = self.encode(text_b)
        return float(np.dot(va, vb))

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return f"TextEncoder(dim={self.dim}, status={status})"
