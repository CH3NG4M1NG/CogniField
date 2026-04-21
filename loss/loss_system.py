"""
loss/loss_system.py
====================
Multi-Signal Loss System

Combines multiple learning signals into a unified loss:

  total_loss = w_error * error_loss
             + w_uncertainty * uncertainty_score
             + w_novelty * novelty_bonus
             + w_structure * structure_penalty

Each signal plays a distinct role:

  error_loss        : How wrong is the output? (reconstruction error)
  uncertainty_score : How confident is the system? (entropy-based)
  novelty_score     : How surprising is the input? (curiosity signal)
  structure_penalty : How grammatically / semantically broken is output?

The weights adapt over time:
  - High recent error → increase error_loss weight
  - High novelty → increase novelty weight (explore more)
  - Low uncertainty → decrease uncertainty weight (already confident)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..latent_space.frequency_space import FrequencySpace


@dataclass
class LossConfig:
    """Configurable weights for the loss system."""
    w_error:       float = 1.0
    w_uncertainty: float = 0.5
    w_novelty:     float = 0.3
    w_structure:   float = 0.2
    # Adaptation rates
    adapt_rate:    float = 0.05
    # Smoothing
    ema_alpha:     float = 0.1


@dataclass
class LossRecord:
    """One loss computation record."""
    error_loss:        float
    uncertainty_score: float
    novelty_score:     float
    structure_penalty: float
    total_loss:        float
    is_novel:          bool


class LossSystem:
    """
    Computes and tracks multi-signal learning loss.

    Parameters
    ----------
    config : LossConfig
    space  : FrequencySpace (for cosine error computation)
    """

    def __init__(
        self,
        config: Optional[LossConfig] = None,
        space: Optional[FrequencySpace] = None,
    ) -> None:
        self.cfg   = config if config is not None else LossConfig()
        self.space = space  if space  is not None else FrequencySpace()

        # Running EMA of each signal
        self._ema_error       = 0.5
        self._ema_uncertainty = 0.5
        self._ema_novelty     = 0.5

        # History
        self._history: List[LossRecord] = []

        # Adaptive weights
        self._w_error       = self.cfg.w_error
        self._w_uncertainty = self.cfg.w_uncertainty
        self._w_novelty     = self.cfg.w_novelty
        self._w_structure   = self.cfg.w_structure

    # ------------------------------------------------------------------
    # Individual signal computation
    # ------------------------------------------------------------------

    def error_loss(
        self,
        predicted_vec: np.ndarray,
        target_vec: np.ndarray,
    ) -> float:
        """
        Cosine error loss ∈ [0, 1].
        0 = perfect prediction, 1 = opposite direction.
        """
        cos = self.space.similarity(predicted_vec, target_vec)
        return float((1.0 - cos) / 2.0)

    def uncertainty_score(
        self,
        scores: List[float],
    ) -> float:
        """
        Uncertainty from a distribution of similarity scores ∈ [0, 1].
        High entropy → high uncertainty.
        """
        if not scores:
            return 0.5
        s = np.array(scores, dtype=float)
        s = np.clip(s, 0.0, 1.0)
        s /= s.sum() + 1e-8
        entropy = -np.sum(s * np.log(s + 1e-8))
        max_entropy = np.log(max(len(s), 2))
        return float(entropy / max_entropy)

    def novelty_score(self, novelty: float) -> float:
        """Novelty signal: novel inputs carry a positive learning bonus."""
        return float(np.clip(novelty, 0.0, 1.0))

    def structure_penalty(self, structure_score: float) -> float:
        """
        Structure penalty ∈ [0, 1].
        Low structure score → high penalty.
        """
        return float(1.0 - np.clip(structure_score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Combined loss
    # ------------------------------------------------------------------

    def compute(
        self,
        predicted_vec: np.ndarray,
        target_vec: np.ndarray,
        candidate_scores: Optional[List[float]] = None,
        novelty: float = 0.0,
        structure_score: float = 1.0,
    ) -> LossRecord:
        """
        Compute the combined loss for one prediction.

        Parameters
        ----------
        predicted_vec     : The system's output vector.
        target_vec        : The expected/correct vector.
        candidate_scores  : Similarity distribution (for uncertainty).
        novelty           : Novelty score from CuriosityEngine.
        structure_score   : Quality from StructureChecker.

        Returns
        -------
        LossRecord
        """
        el = self.error_loss(predicted_vec, target_vec)
        us = self.uncertainty_score(candidate_scores or [0.5])
        ns = self.novelty_score(novelty)
        sp = self.structure_penalty(structure_score)

        total = (self._w_error       * el
               + self._w_uncertainty * us
               + self._w_novelty     * ns
               + self._w_structure   * sp)

        # Normalise by weight sum
        weight_sum = (self._w_error + self._w_uncertainty
                    + self._w_novelty + self._w_structure)
        total /= (weight_sum + 1e-8)

        record = LossRecord(
            error_loss=el,
            uncertainty_score=us,
            novelty_score=ns,
            structure_penalty=sp,
            total_loss=float(total),
            is_novel=novelty >= 0.4,
        )
        self._history.append(record)
        self._update_ema(el, us, ns)
        self._adapt_weights()
        return record

    def compute_batch(
        self,
        predicted_vecs: np.ndarray,
        target_vecs: np.ndarray,
        novelties: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute losses for a batch.
        Returns (per_sample_losses, mean_loss).
        """
        novelties = novelties or [0.0] * len(predicted_vecs)
        losses = [
            self.compute(p, t, novelty=n).total_loss
            for p, t, n in zip(predicted_vecs, target_vecs, novelties)
        ]
        arr = np.array(losses, dtype=np.float32)
        return arr, float(arr.mean())

    # ------------------------------------------------------------------
    # Adaptive weights
    # ------------------------------------------------------------------

    def _update_ema(self, el: float, us: float, ns: float) -> None:
        a = self.cfg.ema_alpha
        self._ema_error       = (1-a) * self._ema_error       + a * el
        self._ema_uncertainty = (1-a) * self._ema_uncertainty + a * us
        self._ema_novelty     = (1-a) * self._ema_novelty     + a * ns

    def _adapt_weights(self) -> None:
        r = self.cfg.adapt_rate
        # High error → upweight error signal
        if self._ema_error > 0.6:
            self._w_error = min(2.0, self._w_error + r)
        elif self._ema_error < 0.3:
            self._w_error = max(0.5, self._w_error - r)

        # High novelty → upweight curiosity
        if self._ema_novelty > 0.5:
            self._w_novelty = min(1.0, self._w_novelty + r)
        else:
            self._w_novelty = max(0.1, self._w_novelty - r)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        if not self._history:
            return {"n_steps": 0}
        totals   = [r.total_loss        for r in self._history]
        errors   = [r.error_loss        for r in self._history]
        novelties = [r.novelty_score    for r in self._history]
        return {
            "n_steps":        len(self._history),
            "mean_total":     round(float(np.mean(totals)), 4),
            "mean_error":     round(float(np.mean(errors)), 4),
            "mean_novelty":   round(float(np.mean(novelties)), 4),
            "current_weights": {
                "error":       round(self._w_error, 3),
                "uncertainty": round(self._w_uncertainty, 3),
                "novelty":     round(self._w_novelty, 3),
                "structure":   round(self._w_structure, 3),
            },
        }

    def __repr__(self) -> str:
        return (f"LossSystem(w_error={self._w_error:.2f}, "
                f"w_novelty={self._w_novelty:.2f})")
