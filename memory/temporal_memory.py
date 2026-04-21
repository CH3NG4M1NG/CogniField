"""
memory/temporal_memory.py
==========================
Temporal Memory

Tracks long-term patterns over time:
  - repeated (action, target) outcomes
  - drift in belief confidence over time
  - cyclical behaviour patterns
  - which contexts correlate with success/failure

This is distinct from EpisodicMemory (individual episodes) —
TemporalMemory cares about PATTERNS across many episodes.

Key Features
------------
1. PATTERN DETECTION
   If eat(apple) → success 8 out of 10 times: stable pattern.
   If eat(purple_berry) alternates success/failure: unstable.

2. TEMPORAL DRIFT TRACKING
   Monitor whether a belief's confidence trends upward, downward,
   or oscillates over time.

3. RECURRENCE DETECTION
   Detect if the same failure repeats (stuck in a loop).
   Triggers strategy change when recurrence exceeds threshold.

4. CONTEXT CORRELATION
   Link outcomes to contexts (e.g. low novelty + EXPLOIT → high sr).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TemporalPattern:
    """A detected recurring pattern."""
    pattern_type: str        # "stable", "unstable", "improving", "declining"
    key:          str        # what the pattern is about (e.g. "eat(apple)")
    confidence:   float      # how certain we are about this pattern
    n_samples:    int
    success_rate: float
    description:  str
    first_seen:   float = field(default_factory=time.time)
    last_seen:    float = field(default_factory=time.time)


class TemporalMemory:
    """
    Long-term pattern tracker across agent steps.

    Parameters
    ----------
    max_per_key : Maximum outcome records per (action, target) key.
    window      : Sliding window for pattern analysis.
    """

    def __init__(
        self,
        max_per_key: int = 100,
        window:      int = 20,
    ) -> None:
        self.max_per_key = max_per_key
        self.window      = window

        # Raw outcome history: {key → deque of (step, success, reward)}
        self._outcomes: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_per_key)
        )

        # Belief confidence snapshots: {belief_key → [(step, conf)]}
        self._conf_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )

        # Context-outcome pairs for correlation
        self._context_log: deque = deque(maxlen=500)

        # Detected patterns (cached)
        self._patterns: Dict[str, TemporalPattern] = {}

        self._step = 0
        self._recurrence_counts: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        action:  str,
        target:  str,
        success: bool,
        reward:  float,
        step:    int = 0,
        context: Optional[Dict] = None,
    ) -> None:
        """Record one action-target outcome."""
        key = f"{action}({target})"
        self._step = step or self._step + 1
        self._outcomes[key].append((self._step, success, reward))

        # Recurrence: same failure 3+ times in a row
        if not success:
            self._recurrence_counts[key] += 1
        else:
            self._recurrence_counts[key] = 0

        # Context log
        if context:
            self._context_log.append({
                "step": self._step, "key": key,
                "success": success, "reward": reward,
                **context,
            })

    def record_belief_snapshot(
        self,
        belief_key: str,
        confidence: float,
        step:       int = 0,
    ) -> None:
        """Snapshot a belief's confidence for drift analysis."""
        self._conf_history[belief_key].append((step or self._step, confidence))

    # ------------------------------------------------------------------
    # Pattern analysis
    # ------------------------------------------------------------------

    def detect_pattern(self, action: str, target: str) -> Optional[TemporalPattern]:
        """
        Detect the pattern for a specific (action, target) pair.
        Returns None if insufficient data.
        """
        key = f"{action}({target})"
        outcomes = list(self._outcomes[key])
        if len(outcomes) < 4:
            return None

        recent = outcomes[-self.window:]
        sr     = float(np.mean([o[1] for o in recent]))
        n      = len(recent)

        # Classify pattern
        if sr >= 0.75:
            ptype = "stable"
            desc  = f"Consistent success ({sr:.0%})"
        elif sr <= 0.25:
            ptype = "declining"
            desc  = f"Consistent failure ({sr:.0%})"
        else:
            # Check for oscillation
            successes = [int(o[1]) for o in recent]
            alternations = sum(abs(successes[i] - successes[i-1])
                               for i in range(1, len(successes)))
            if alternations >= len(successes) * 0.6:
                ptype = "unstable"
                desc  = f"Oscillating outcome ({alternations} alternations)"
            elif sr > 0.50:
                ptype = "improving"
                desc  = f"Mostly successful ({sr:.0%})"
            else:
                ptype = "declining"
                desc  = f"Mostly failing ({sr:.0%})"

        # Pattern confidence: more samples = more confident
        conf = float(min(0.95, 0.5 + n / (self.window * 2)))

        pattern = TemporalPattern(
            pattern_type=ptype, key=key,
            confidence=conf, n_samples=n,
            success_rate=sr, description=desc,
        )
        self._patterns[key] = pattern
        return pattern

    def detect_all_patterns(self) -> List[TemporalPattern]:
        """Detect patterns for all tracked (action, target) pairs."""
        patterns = []
        for key in list(self._outcomes.keys()):
            parts = key.replace(")", "").split("(")
            if len(parts) == 2:
                action, target = parts
                p = self.detect_pattern(action, target)
                if p:
                    patterns.append(p)
        return patterns

    def is_stuck(self, action: str, target: str, threshold: int = 3) -> bool:
        """True if same (action, target) has failed threshold times in a row."""
        key = f"{action}({target})"
        return self._recurrence_counts[key] >= threshold

    # ------------------------------------------------------------------
    # Drift analysis
    # ------------------------------------------------------------------

    def belief_drift(self, belief_key: str) -> str:
        """
        Characterise how a belief's confidence has been changing.
        Returns 'rising' | 'falling' | 'stable' | 'oscillating' | 'unknown'.
        """
        snaps = list(self._conf_history.get(belief_key, deque()))
        if len(snaps) < 4:
            return "unknown"

        confs   = [s[1] for s in snaps[-10:]]
        trend   = float(np.polyfit(range(len(confs)), confs, 1)[0])
        std_dev = float(np.std(confs))

        if std_dev > 0.12:
            return "oscillating"
        if trend > 0.005:
            return "rising"
        if trend < -0.005:
            return "falling"
        return "stable"

    def mean_confidence(self, belief_key: str) -> float:
        snaps = list(self._conf_history.get(belief_key, deque()))
        if not snaps:
            return 0.5
        return float(np.mean([s[1] for s in snaps[-10:]]))

    # ------------------------------------------------------------------
    # Context correlation
    # ------------------------------------------------------------------

    def best_strategy_for_context(self) -> Optional[str]:
        """
        Given recent context history, which strategy correlates best
        with success? Returns strategy name or None.
        """
        if len(self._context_log) < 8:
            return None

        strat_sr: Dict[str, List[float]] = defaultdict(list)
        for record in list(self._context_log)[-50:]:
            strat = record.get("strategy", "")
            if strat:
                strat_sr[strat].append(float(record.get("success", 0)))

        if not strat_sr:
            return None

        best = max(strat_sr, key=lambda s: np.mean(strat_sr[s])
                   if len(strat_sr[s]) >= 3 else -1)
        return best

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def known_keys(self) -> List[str]:
        return list(self._outcomes.keys())

    def success_rate_for(self, action: str, target: str) -> float:
        key = f"{action}({target})"
        outcomes = list(self._outcomes.get(key, deque()))
        if not outcomes:
            return 0.5
        return float(np.mean([o[1] for o in outcomes[-self.window:]]))

    def summary(self) -> Dict:
        patterns = self.detect_all_patterns()
        by_type  = {}
        for p in patterns:
            by_type[p.pattern_type] = by_type.get(p.pattern_type, 0) + 1
        return {
            "tracked_keys":       len(self._outcomes),
            "n_patterns":         len(patterns),
            "pattern_types":      by_type,
            "belief_keys":        len(self._conf_history),
            "context_records":    len(self._context_log),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"TemporalMemory(keys={s['tracked_keys']}, "
                f"patterns={s['n_patterns']})")
