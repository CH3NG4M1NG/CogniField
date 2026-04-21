"""
reasoning/reasoning_engine.py
==============================
Self-Correcting Reasoning Engine

The engine implements a generate → evaluate → correct loop:

  1. generate_solution(input_vec)
     Propose a solution by querying memory + composition.

  2. evaluate_solution(solution_vec, target_vec)
     Score the solution against a target or ground-truth.

  3. detect_error(solution, expected)
     Classify the type of error (semantic mismatch, low confidence, etc.)

  4. Retry loop
     If score < threshold, apply correction strategy and retry.
     Each retry uses a different strategy (portfolio).
     Maximum iterations bounded.

This models early-stage self-improvement: the system doesn't give up
after one wrong answer but explores different reasoning paths.
"""

from __future__ import annotations

import time
import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..latent_space.frequency_space import FrequencySpace, ComposeMode
from ..memory.memory_store import MemoryStore, MemoryEntry


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

class ErrorType(str, enum.Enum):
    NONE              = "none"
    SEMANTIC_MISMATCH = "semantic_mismatch"   # wrong meaning
    LOW_CONFIDENCE    = "low_confidence"      # score near 0.5
    OUT_OF_DISTRIBUTION = "ood"               # far from all memories
    COMPOSITION_FAIL  = "composition_fail"    # composition diverged
    UNKNOWN           = "unknown"


@dataclass
class ReasoningResult:
    """Result of one reasoning iteration."""
    solution_vec:   np.ndarray
    score:          float
    error_type:     ErrorType
    n_retries:      int
    strategy_used:  str
    elapsed_ms:     float
    metadata:       Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.score >= 0.7 and self.error_type == ErrorType.NONE


# ---------------------------------------------------------------------------
# Reasoning Engine
# ---------------------------------------------------------------------------

class ReasoningEngine:
    """
    Generates, evaluates, and self-corrects solutions in latent space.

    Parameters
    ----------
    space      : FrequencySpace instance (shared with all other modules).
    memory     : MemoryStore instance.
    max_retries: Maximum correction iterations.
    threshold  : Minimum score to accept a solution.
    """

    def __init__(
        self,
        space: Optional[FrequencySpace] = None,
        memory: Optional[MemoryStore] = None,
        max_retries: int = 5,
        threshold: float = 0.65,
        seed: int = 42,
    ) -> None:
        self.space       = space  if space  is not None else FrequencySpace()
        self.memory      = memory if memory is not None else MemoryStore()
        self.max_retries = max_retries
        self.threshold   = threshold
        self._rng        = np.random.default_rng(seed)
        self._history:   List[ReasoningResult] = []

        # Strategy portfolio (each is a function that proposes a candidate)
        self._strategies: List[Tuple[str, Callable]] = [
            ("memory_nearest",    self._strategy_memory_nearest),
            ("composition_add",   self._strategy_composition_add),
            ("composition_mean",  self._strategy_composition_mean),
            ("analogy_shift",     self._strategy_analogy_shift),
            ("random_walk",       self._strategy_random_walk),
            ("centroid",          self._strategy_centroid),
        ]
        # Win counts per strategy (for adaptive selection)
        self._strategy_wins = {name: 0 for name, _ in self._strategies}
        self._strategy_tries = {name: 0 for name, _ in self._strategies}

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def generate_solution(
        self,
        input_vec: np.ndarray,
        context_vecs: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Propose an initial solution vector for a given input.

        Strategy: recall top-k memories, compose them with the input.
        """
        recalls = self.memory.retrieve(input_vec, k=3)
        if not recalls:
            return input_vec.copy()

        vecs = [input_vec] + [e.vector for _, e in recalls]
        weights = [2.0] + [sim for sim, _ in recalls]
        return self.space.compose(vecs, mode=ComposeMode.WEIGHTED,
                                  weights=weights)

    def evaluate_solution(
        self,
        solution_vec: np.ndarray,
        target_vec: np.ndarray,
    ) -> float:
        """
        Score a solution against a target. Returns ∈ [0, 1].
        Uses cosine similarity mapped to [0, 1].
        """
        cos = self.space.similarity(solution_vec, target_vec)
        return float((cos + 1.0) / 2.0)   # map [-1,1] → [0,1]

    def detect_error(
        self,
        solution_vec: np.ndarray,
        target_vec: Optional[np.ndarray] = None,
        score: Optional[float] = None,
    ) -> ErrorType:
        """Score >= threshold -> NONE (success), else classify failure type."""
        if np.linalg.norm(solution_vec) < 1e-4:
            return ErrorType.COMPOSITION_FAIL
        if score is not None:
            if score >= self.threshold:
                return ErrorType.NONE
            if score < 0.5:
                return ErrorType.SEMANTIC_MISMATCH
            return ErrorType.LOW_CONFIDENCE
        recalls = self.memory.retrieve(solution_vec, k=1)
        if not recalls:
            return ErrorType.OUT_OF_DISTRIBUTION
        best_sim, _ = recalls[0]
        if best_sim < 0.3:
            return ErrorType.OUT_OF_DISTRIBUTION
        return ErrorType.UNKNOWN

    def reason(
        self,
        input_vec: np.ndarray,
        target_vec: np.ndarray,
        context_vecs: Optional[List[np.ndarray]] = None,
        verbose: bool = False,
    ) -> ReasoningResult:
        """
        Full reasoning loop with retry.

        1. Generate initial solution
        2. Evaluate against target
        3. If below threshold, select strategy and retry
        4. Return best result found

        Parameters
        ----------
        input_vec    : The query/input in latent space.
        target_vec   : The expected output in latent space.
        context_vecs : Optional additional context vectors.
        verbose      : Print retry trace.
        """
        t0 = time.time()
        best_vec   = self.generate_solution(input_vec, context_vecs)
        best_score = self.evaluate_solution(best_vec, target_vec)
        best_strategy = "initial"

        for attempt in range(self.max_retries):
            err = self.detect_error(best_vec, target_vec, best_score)
            if err == ErrorType.NONE:
                break

            # Select strategy (UCB1-style)
            strategy_name, strategy_fn = self._select_strategy()
            self._strategy_tries[strategy_name] += 1

            try:
                candidate = strategy_fn(
                    input_vec, target_vec, context_vecs or []
                )
                score = self.evaluate_solution(candidate, target_vec)
            except Exception:
                score = 0.0
                candidate = best_vec.copy()

            if verbose:
                print(f"    retry {attempt+1}: strategy={strategy_name} "
                      f"score={score:.3f}")

            if score > best_score:
                best_score    = score
                best_vec      = candidate.copy()
                best_strategy = strategy_name
                self._strategy_wins[strategy_name] += 1

        final_error = self.detect_error(best_vec, target_vec, best_score)
        elapsed = (time.time() - t0) * 1000

        result = ReasoningResult(
            solution_vec=best_vec,
            score=best_score,
            error_type=final_error,
            n_retries=attempt + 1,
            strategy_used=best_strategy,
            elapsed_ms=elapsed,
        )
        self._history.append(result)
        return result

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _strategy_memory_nearest(self, inp, tgt, ctx):
        recalls = self.memory.retrieve(tgt, k=1)
        if recalls:
            return recalls[0][1].vector.copy()
        return inp.copy()

    def _strategy_composition_add(self, inp, tgt, ctx):
        vecs = [inp, tgt] + ctx[:2]
        return self.space.compose(vecs, mode=ComposeMode.ADD)

    def _strategy_composition_mean(self, inp, tgt, ctx):
        vecs = [inp] + ctx[:3]
        return self.space.compose(vecs, mode=ComposeMode.MEAN)

    def _strategy_analogy_shift(self, inp, tgt, ctx):
        if len(ctx) >= 2:
            return self.space.analogy(ctx[0], ctx[1], inp)
        return self.space.analogy(inp, tgt, inp)

    def _strategy_random_walk(self, inp, tgt, ctx):
        noise = self._rng.standard_normal(self.space.dim).astype(np.float32)
        noise /= np.linalg.norm(noise) + 1e-8
        return self.space.l2(inp + 0.3 * noise)

    def _strategy_centroid(self, inp, tgt, ctx):
        recalls = self.memory.retrieve(inp, k=5)
        if len(recalls) >= 2:
            vecs = [e.vector for _, e in recalls]
            return self.space.centroid(vecs)
        return inp.copy()

    # ------------------------------------------------------------------
    # Strategy selection (UCB1)
    # ------------------------------------------------------------------

    def _select_strategy(self) -> Tuple[str, Callable]:
        total_tries = sum(self._strategy_tries.values()) + 1
        best_score  = -1.0
        best_idx    = 0

        for i, (name, _) in enumerate(self._strategies):
            tries = self._strategy_tries[name]
            wins  = self._strategy_wins[name]
            if tries == 0:
                return self._strategies[i]
            win_rate = wins / tries
            explore  = (2 * np.log(total_tries) / tries) ** 0.5
            ucb      = win_rate + explore
            if ucb > best_score:
                best_score = ucb
                best_idx   = i

        return self._strategies[best_idx]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def success_rate(self) -> float:
        if not self._history:
            return 0.0
        return sum(r.success for r in self._history) / len(self._history)

    def strategy_report(self) -> List[Dict]:
        return sorted([
            {
                "strategy": n,
                "wins":   self._strategy_wins[n],
                "tries":  self._strategy_tries[n],
                "rate":   round(self._strategy_wins[n] /
                                max(self._strategy_tries[n], 1), 3),
            }
            for n, _ in self._strategies
        ], key=lambda d: -d["rate"])

    def __repr__(self) -> str:
        return (f"ReasoningEngine(max_retries={self.max_retries}, "
                f"threshold={self.threshold}, "
                f"success_rate={self.success_rate():.1%})")
