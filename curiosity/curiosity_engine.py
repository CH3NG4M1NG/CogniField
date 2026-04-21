"""
curiosity/curiosity_engine.py
==============================
Curiosity-Driven Exploration Engine

Biological analogy: the hippocampus flags unfamiliar stimuli and directs
attention toward them.  We replicate this computationally:

  1. detect_novelty(vector)
     If the vector is far from all known memories → it is novel.
     Novelty score = 1 − max(cosine_similarity with known vectors)

  2. trigger_exploration(vector)
     If novelty ≥ threshold:
       a. Label it as "unknown_XXXXX"
       b. Store in memory with special metadata
       c. Attempt to find the closest known concept (context)
       d. Generate a hypothesis label via analogy
       e. Return an exploration report

  3. Curiosity-weighted learning
     Novel samples get extra weight in the learning signal.
     This is the "intrinsic motivation" component.

The curiosity score feeds into the combined loss:
  total_loss = error_loss + λ_curiosity * (1 - novelty_score)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..latent_space.frequency_space import FrequencySpace, ComposeMode
from ..memory.memory_store import MemoryStore, MemoryEntry


@dataclass
class ExplorationReport:
    """Result of a curiosity-triggered exploration."""
    input_label:      str           # "unknown_XXXXX"
    novelty_score:    float         # how novel the input is [0,1]
    nearest_known:    Optional[str] # closest memory label
    nearest_sim:      float         # similarity to closest known
    hypothesis:       str           # proposed label
    stored_entry:     Optional[MemoryEntry]
    timestamp:        float = field(default_factory=time.time)


class CuriosityEngine:
    """
    Detects novelty and triggers exploration of unknown inputs.

    Parameters
    ----------
    space            : Shared FrequencySpace.
    memory           : Shared MemoryStore.
    novelty_threshold: Minimum novelty score to trigger exploration.
    """

    def __init__(
        self,
        space: Optional[FrequencySpace] = None,
        memory: Optional[MemoryStore] = None,
        novelty_threshold: float = 0.4,
        seed: int = 42,
    ) -> None:
        self.space             = space  if space  is not None else FrequencySpace()
        self.memory            = memory if memory is not None else MemoryStore()
        self.novelty_threshold = novelty_threshold
        self._rng              = np.random.default_rng(seed)
        self._exploration_log: List[ExplorationReport] = []
        self._unknown_counter  = 0

    # ------------------------------------------------------------------
    # Novelty detection
    # ------------------------------------------------------------------

    def detect_novelty(
        self,
        vector: np.ndarray,
        top_k: int = 3,
    ) -> float:
        """
        Compute novelty score ∈ [0, 1].
        0 = already well-known, 1 = completely unknown.

        Parameters
        ----------
        vector : Query latent vector.
        top_k  : Number of nearest memories to check.

        Returns
        -------
        float  novelty score
        """
        if len(self.memory) == 0:
            return 1.0

        recalls = self.memory.retrieve(vector, k=top_k)
        if not recalls:
            return 1.0

        max_sim = max(sim for sim, _ in recalls)
        novelty = 1.0 - float(max_sim)
        return float(np.clip(novelty, 0.0, 1.0))

    def is_novel(self, vector: np.ndarray) -> bool:
        return self.detect_novelty(vector) >= self.novelty_threshold

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------

    def trigger_exploration(
        self,
        vector: np.ndarray,
        raw_input: str = "",
        modality: str = "unknown",
    ) -> ExplorationReport:
        """
        Explore a novel input: store it, find context, form hypothesis.

        Parameters
        ----------
        vector    : Latent vector of the novel input.
        raw_input : Original string/description (for logging).
        modality  : "text" | "image" | "audio"

        Returns
        -------
        ExplorationReport
        """
        novelty = self.detect_novelty(vector)
        self._unknown_counter += 1
        label = f"unknown_{self._unknown_counter:04d}"

        # Find nearest known
        recalls = self.memory.retrieve(vector, k=3)
        nearest_label = None
        nearest_sim   = 0.0
        if recalls:
            nearest_sim, nearest_entry = recalls[0]
            nearest_label = nearest_entry.label

        # Hypothesis: if close enough to something known, name it
        hypothesis = self._form_hypothesis(
            vector, recalls, raw_input, novelty
        )

        # Store with "unknown" metadata
        entry = self.memory.store(
            vector=vector,
            label=label,
            modality=modality,
            metadata={
                "raw_input":    raw_input,
                "novelty":      novelty,
                "hypothesis":   hypothesis,
                "nearest":      nearest_label,
                "nearest_sim":  nearest_sim,
                "explored":     True,
            },
        )

        report = ExplorationReport(
            input_label=label,
            novelty_score=novelty,
            nearest_known=nearest_label,
            nearest_sim=float(nearest_sim),
            hypothesis=hypothesis,
            stored_entry=entry,
        )
        self._exploration_log.append(report)
        return report

    # ------------------------------------------------------------------
    # Hypothesis formation
    # ------------------------------------------------------------------

    def _form_hypothesis(
        self,
        vector: np.ndarray,
        recalls: list,
        raw_input: str,
        novelty: float,
    ) -> str:
        """
        Form a hypothesis label for an unknown input.

        Rules:
        1. If very similar to one known concept → "similar to {label}"
        2. If near two known concepts → "between {a} and {b}"
        3. If completely novel → "new concept [prefix from raw_input]"
        """
        if not recalls:
            prefix = raw_input.split()[:2]
            return "new concept: " + (" ".join(prefix) if prefix else "unknown")

        sim0, e0 = recalls[0]

        if sim0 >= 0.80:
            return f"similar to '{e0.label}'"
        if sim0 >= 0.55:
            if len(recalls) >= 2:
                sim1, e1 = recalls[1]
                return f"between '{e0.label}' and '{e1.label}'"
            return f"variant of '{e0.label}'"

        # Low similarity — inspect if raw input gives a clue
        if raw_input:
            words = raw_input.lower().split()
            content = [w for w in words
                       if len(w) > 3 and w not in
                       {"this", "that", "with", "from", "have"}]
            if content:
                return f"possible new concept (raw: '{content[0]}')"

        return "unknown concept"

    # ------------------------------------------------------------------
    # Curiosity-weighted learning signal
    # ------------------------------------------------------------------

    def curiosity_weight(self, vector: np.ndarray) -> float:
        """
        Return a weight multiplier [1.0, 3.0] for a learning update.
        Novel inputs get more weight → drives exploration.
        """
        novelty = self.detect_novelty(vector)
        # Linear map: novelty 0 → weight 1.0; novelty 1 → weight 3.0
        return 1.0 + 2.0 * novelty

    def batch_novelty(self, vectors: np.ndarray) -> np.ndarray:
        """Compute novelty for a batch. Returns (N,) array."""
        return np.array([self.detect_novelty(v) for v in vectors])

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def n_explorations(self) -> int:
        return len(self._exploration_log)

    def exploration_summary(self) -> Dict:
        if not self._exploration_log:
            return {"n_explorations": 0}
        novelties = [r.novelty_score for r in self._exploration_log]
        sims      = [r.nearest_sim   for r in self._exploration_log]
        return {
            "n_explorations":   self.n_explorations,
            "mean_novelty":     float(np.mean(novelties)),
            "mean_nearest_sim": float(np.mean(sims)),
            "threshold":        self.novelty_threshold,
        }

    def __repr__(self) -> str:
        return (f"CuriosityEngine(threshold={self.novelty_threshold}, "
                f"explorations={self.n_explorations})")
