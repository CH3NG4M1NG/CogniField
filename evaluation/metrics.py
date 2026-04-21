"""
evaluation/metrics.py
======================
Performance and Stability Metrics

Tracks the agent's reliability over time across multiple dimensions:

1. SUCCESS RATE         — action success rate with moving average
2. BELIEF STABILITY     — how stable are beliefs over time?
3. CONSISTENCY SCORE    — how consistent are decisions?
4. ERROR REDUCTION RATE — are mistakes decreasing over time?
5. NOVELTY HANDLING     — how well are unknowns handled?
6. RISK COMPLIANCE      — is the agent avoiding known-dangerous actions?
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StepMetric:
    """Metrics for one agent step."""
    step:              int
    success:           bool
    reward:            float
    belief_confidence: float    # mean confidence across all beliefs
    n_conflicts:       int      # conflicts detected this step
    n_blocks:          int      # risky actions blocked
    novelty:           float
    goal_type:         str
    action:            str
    timestamp:         float = field(default_factory=time.time)


class AgentMetrics:
    """
    Tracks and reports agent performance and stability over time.

    Parameters
    ----------
    window : Rolling window for moving averages.
    """

    def __init__(self, window: int = 50) -> None:
        self.window   = window
        self._steps:  deque = deque(maxlen=window * 4)
        self._n_total = 0

        # Belief stability tracking: snapshot confidence values
        self._belief_snapshots: List[Dict[str, float]] = []

        # Running stats
        self._phase_boundaries: List[int] = [0]

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        step:              int,
        success:           bool,
        reward:            float,
        belief_confidence: float = 0.5,
        n_conflicts:       int   = 0,
        n_blocks:          int   = 0,
        novelty:           float = 0.0,
        goal_type:         str   = "",
        action:            str   = "",
    ) -> None:
        m = StepMetric(
            step=step, success=success, reward=reward,
            belief_confidence=belief_confidence,
            n_conflicts=n_conflicts, n_blocks=n_blocks,
            novelty=novelty, goal_type=goal_type, action=action,
        )
        self._steps.append(m)
        self._n_total += 1

    def snapshot_beliefs(self, belief_confs: Dict[str, float]) -> None:
        """Record a snapshot of belief confidences for stability tracking."""
        self._belief_snapshots.append({**belief_confs, "_ts": time.time()})
        if len(self._belief_snapshots) > 100:
            self._belief_snapshots.pop(0)

    # ------------------------------------------------------------------
    # Computed metrics
    # ------------------------------------------------------------------

    def success_rate(self, window: Optional[int] = None) -> float:
        """Rolling success rate."""
        recent = list(self._steps)[-(window or self.window):]
        if not recent:
            return 0.0
        return float(np.mean([m.success for m in recent]))

    def mean_reward(self, window: Optional[int] = None) -> float:
        recent = list(self._steps)[-(window or self.window):]
        if not recent:
            return 0.0
        return float(np.mean([m.reward for m in recent]))

    def belief_stability(self) -> float:
        """
        Stability of beliefs over recent snapshots.
        1.0 = completely stable, 0.0 = chaotic.

        Measured as 1 - mean_std of confidence values across snapshots.
        """
        if len(self._belief_snapshots) < 2:
            return 1.0

        # Collect common keys across last 5 snapshots
        recent = self._belief_snapshots[-5:]
        all_keys = set.intersection(*[set(s.keys()) - {"_ts"} for s in recent])
        if not all_keys:
            return 1.0

        stds = []
        for key in all_keys:
            vals = [s[key] for s in recent if key in s]
            if len(vals) >= 2:
                stds.append(float(np.std(vals)))

        if not stds:
            return 1.0
        mean_std = float(np.mean(stds))
        return float(np.clip(1.0 - mean_std * 4, 0.0, 1.0))

    def consistency_score(self) -> float:
        """
        How consistent are decisions for the same action type?
        High = same action type tends to have the same outcome.
        """
        steps = list(self._steps)
        if len(steps) < 4:
            return 0.5

        # Group by action type
        action_outcomes: Dict[str, List[bool]] = {}
        for m in steps:
            action_outcomes.setdefault(m.action, []).append(m.success)

        consistencies = []
        for action, outcomes in action_outcomes.items():
            if len(outcomes) < 2:
                continue
            sr  = float(np.mean(outcomes))
            # Consistency = how far from 50/50 (deterministic = 1.0)
            consistencies.append(abs(sr - 0.5) * 2)

        return float(np.mean(consistencies)) if consistencies else 0.5

    def error_reduction_rate(self) -> float:
        """
        Rate at which error rate is declining over time.
        Positive = improving, negative = getting worse.
        """
        steps = list(self._steps)
        if len(steps) < 10:
            return 0.0

        half = len(steps) // 2
        early_sr = float(np.mean([m.success for m in steps[:half]]))
        late_sr  = float(np.mean([m.success for m in steps[half:]]))
        return float(late_sr - early_sr)

    def risk_compliance(self) -> float:
        """
        Fraction of steps where risky actions were appropriately blocked.
        Only meaningful if blocks are recorded.
        """
        steps  = list(self._steps)
        blocks = sum(m.n_blocks for m in steps)
        if blocks == 0:
            return 1.0   # no risky actions attempted
        # We can't directly measure "should have been blocked" without ground truth
        # Proxy: if blocks occurred and success rate is high, agent is being appropriately cautious
        return min(1.0, 0.5 + self.success_rate() * 0.5)

    def conflict_rate(self) -> float:
        """Average conflicts per step (lower = more consistent beliefs)."""
        steps = list(self._steps)
        if not steps:
            return 0.0
        return float(np.mean([m.n_conflicts for m in steps]))

    def mean_belief_confidence(self) -> float:
        steps = list(self._steps)
        if not steps:
            return 0.5
        return float(np.mean([m.belief_confidence for m in steps]))

    # ------------------------------------------------------------------
    # Comprehensive report
    # ------------------------------------------------------------------

    def report(self) -> Dict:
        """Return a comprehensive performance report."""
        steps = list(self._steps)
        if not steps:
            return {"n_steps": 0, "status": "no_data"}

        # Early vs late comparison
        half     = max(1, len(steps) // 2)
        early_sr = float(np.mean([m.success for m in steps[:half]]))
        late_sr  = float(np.mean([m.success for m in steps[half:]]))

        return {
            "n_steps":           self._n_total,
            "window":            len(steps),
            "success_rate":      round(self.success_rate(), 3),
            "mean_reward":       round(self.mean_reward(), 3),
            "belief_stability":  round(self.belief_stability(), 3),
            "consistency_score": round(self.consistency_score(), 3),
            "error_reduction":   round(self.error_reduction_rate(), 3),
            "conflict_rate":     round(self.conflict_rate(), 3),
            "risk_compliance":   round(self.risk_compliance(), 3),
            "mean_belief_conf":  round(self.mean_belief_confidence(), 3),
            "trend":             "improving" if late_sr > early_sr + 0.05
                                 else "declining" if late_sr < early_sr - 0.05
                                 else "stable",
            "early_sr":          round(early_sr, 3),
            "late_sr":           round(late_sr, 3),
        }

    def stability_grade(self) -> str:
        """
        Single letter grade for overall stability.
        A = excellent, B = good, C = acceptable, D = poor, F = failing
        """
        r      = self.report()
        score  = (0.3  * r.get("success_rate", 0)
                + 0.25 * r.get("belief_stability", 0)
                + 0.20 * r.get("consistency_score", 0)
                + 0.15 * max(0, r.get("error_reduction", 0) + 0.5)
                + 0.10 * (1 - min(1, r.get("conflict_rate", 0) * 5)))
        if score >= 0.80: return "A"
        if score >= 0.65: return "B"
        if score >= 0.50: return "C"
        if score >= 0.35: return "D"
        return "F"

    def __repr__(self) -> str:
        r = self.report()
        return (f"AgentMetrics(steps={self._n_total}, "
                f"sr={r.get('success_rate',0):.1%}, "
                f"stability={r.get('belief_stability',0):.3f}, "
                f"grade={self.stability_grade()})")
