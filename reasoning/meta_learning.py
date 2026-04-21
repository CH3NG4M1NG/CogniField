"""
reasoning/meta_learning.py
===========================
Meta-Learning / Self-Improvement Engine

Observes the agent's performance over time and adapts its strategies.
This is the "learning to learn" layer — instead of just learning about
the world, the agent learns about its own reasoning process.

What It Tracks
--------------
  - Per-action success rates
  - Per-strategy win rates (in reasoning engine)
  - Failure patterns (what triggers failures)
  - Planning accuracy (predicted vs actual outcomes)

What It Adapts
--------------
  1. STRATEGY WEIGHTS
     Boost strategies that have worked recently.
     Suppress strategies that consistently fail.

  2. CONFIDENCE CALIBRATION
     Compare confidence predictions with actual outcomes.
     If agent was overconfident → reduce confidence scaling.

  3. PLANNING HORIZON
     If plans with depth > D routinely fail → reduce depth.
     If short plans keep succeeding → try longer ones.

  4. NOVELTY THRESHOLD
     If exploring novel things leads to success → lower threshold.
     If novel things are mostly harmful → raise threshold.

  5. GOAL PRIORITY BIAS
     Learn which goal types tend to succeed in current environment.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PerformanceRecord:
    """One recorded performance observation."""
    step:         int
    action:       str
    success:      bool
    reward:       float
    goal_type:    str
    strategy:     str
    plan_depth:   int
    novelty:      float
    confidence:   float
    timestamp:    float = field(default_factory=time.time)


class MetaLearner:
    """
    Analyses performance history and adapts agent strategies.

    Parameters
    ----------
    history_window  : Number of steps to consider in analyses.
    adapt_rate      : How aggressively to update strategy weights.
    """

    def __init__(
        self,
        history_window: int   = 100,
        adapt_rate:     float = 0.1,
    ) -> None:
        self.window      = history_window
        self.adapt_rate  = adapt_rate

        self._records:  deque = deque(maxlen=history_window)
        self._insights: List[str] = []
        self._cycle     = 0

        # Adaptive parameters (these get tuned over time)
        self.params: Dict[str, float] = {
            "plan_depth_bias":       0.0,    # +/- adjustment to preferred depth
            "novelty_threshold_adj": 0.0,    # adjustment to novelty threshold
            "exploration_rate":      0.5,    # exploration vs exploitation
            "confidence_scale":      1.0,    # multiplier on confidence scores
            "retry_budget":          1.0,    # multiplier on max_retries
        }

        # Strategy performance tracking
        self._strategy_scores: Dict[str, List[float]] = {}
        self._action_success:  Dict[str, List[bool]]  = {}
        self._goal_success:    Dict[str, List[bool]]  = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        step:       int,
        action:     str,
        success:    bool,
        reward:     float,
        goal_type:  str  = "",
        strategy:   str  = "",
        plan_depth: int  = 1,
        novelty:    float = 0.0,
        confidence: float = 0.5,
    ) -> None:
        """Record one performance observation."""
        rec = PerformanceRecord(
            step=step, action=action, success=success,
            reward=reward, goal_type=goal_type,
            strategy=strategy, plan_depth=plan_depth,
            novelty=novelty, confidence=confidence,
        )
        self._records.append(rec)

        # Update rolling trackers
        self._action_success.setdefault(action, []).append(success)
        if len(self._action_success[action]) > 30:
            self._action_success[action].pop(0)

        if goal_type:
            self._goal_success.setdefault(goal_type, []).append(success)

        if strategy:
            score = reward if success else -abs(reward)
            self._strategy_scores.setdefault(strategy, []).append(score)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyse(self) -> Dict:
        """
        Analyse recent performance and return a dict of insights.
        Also updates internal adaptive parameters.
        """
        if len(self._records) < 5:
            return {"status": "insufficient_data", "insights": []}

        self._cycle += 1
        insights = []
        recommendations = {}

        records = list(self._records)
        recent  = records[-20:]

        # Overall success rate
        overall_sr  = np.mean([r.success for r in records])
        recent_sr   = np.mean([r.success for r in recent])

        # Trend: is success rate improving?
        if len(records) >= 20:
            old_sr = np.mean([r.success for r in records[:20]])
            trend  = recent_sr - old_sr
        else:
            trend = 0.0

        if recent_sr < 0.4:
            insights.append(f"Low success rate ({recent_sr:.1%}) — strategies need revision")
            recommendations["increase_retries"] = True
        if trend > 0.1:
            insights.append(f"Success rate improving (+{trend:.1%} over last {len(records)} steps)")
        elif trend < -0.1:
            insights.append(f"Success rate declining ({trend:.1%}) — environment may have changed")

        # Action-specific analysis
        action_sr = {}
        for action, successes in self._action_success.items():
            if len(successes) >= 3:
                sr = np.mean(successes[-10:])  # recent window
                action_sr[action] = sr
                if sr < 0.3:
                    insights.append(f"Action '{action}' has low success ({sr:.1%})")
                    recommendations[f"avoid_{action}"] = True
                elif sr > 0.8:
                    insights.append(f"Action '{action}' is highly reliable ({sr:.1%})")

        # Planning depth analysis
        depths    = [r.plan_depth for r in recent if r.plan_depth > 0]
        depth_sr  = {}
        if depths:
            for d in set(depths):
                d_records = [r for r in recent if r.plan_depth == d]
                if len(d_records) >= 2:
                    depth_sr[d] = np.mean([r.success for r in d_records])
            if depth_sr:
                best_depth = max(depth_sr, key=depth_sr.get)
                if best_depth != 2:
                    recommendations["preferred_depth"] = best_depth

        # Novelty-success correlation
        nov_records = [r for r in recent if r.novelty > 0.3]
        if len(nov_records) >= 3:
            nov_sr = np.mean([r.success for r in nov_records])
            if nov_sr < 0.3:
                insights.append("Exploring novel things often fails — be more cautious")
                recommendations["raise_novelty_threshold"] = True
            elif nov_sr > 0.6:
                insights.append("Exploring novel things tends to succeed — explore more")
                recommendations["lower_novelty_threshold"] = True

        # Confidence calibration
        confident_records = [r for r in recent if r.confidence > 0.7]
        if len(confident_records) >= 3:
            conf_actual_sr = np.mean([r.success for r in confident_records])
            calibration_error = abs(0.75 - conf_actual_sr)  # expected 75% when conf>0.7
            if calibration_error > 0.25:
                if conf_actual_sr < 0.5:
                    insights.append("Overconfident — confidence calibration off")
                    recommendations["reduce_confidence_scale"] = True
                else:
                    insights.append("Well-calibrated or underconfident")

        # Apply parameter updates
        self._update_params(recommendations, insights)

        return {
            "cycle":           self._cycle,
            "overall_sr":      round(float(overall_sr), 3),
            "recent_sr":       round(float(recent_sr), 3),
            "trend":           round(float(trend), 3),
            "action_sr":       {k: round(v, 3) for k, v in action_sr.items()},
            "recommendations": recommendations,
            "insights":        insights,
            "params":          {k: round(v, 3) for k, v in self.params.items()},
        }

    def _update_params(self, recommendations: Dict, insights: List[str]) -> None:
        """Apply adaptive parameter updates from analysis."""
        a = self.adapt_rate

        if recommendations.get("increase_retries"):
            self.params["retry_budget"] = min(2.0, self.params["retry_budget"] + a)

        if recommendations.get("raise_novelty_threshold"):
            self.params["novelty_threshold_adj"] = min(0.3,
                self.params["novelty_threshold_adj"] + a * 0.5)

        if recommendations.get("lower_novelty_threshold"):
            self.params["novelty_threshold_adj"] = max(-0.3,
                self.params["novelty_threshold_adj"] - a * 0.5)

        if recommendations.get("reduce_confidence_scale"):
            self.params["confidence_scale"] = max(0.5,
                self.params["confidence_scale"] - a * 0.3)

        preferred_depth = recommendations.get("preferred_depth")
        if preferred_depth is not None:
            delta = int(preferred_depth) - 2
            self.params["plan_depth_bias"] = float(
                np.clip(self.params["plan_depth_bias"] + a * delta, -2, 2)
            )

    # ------------------------------------------------------------------
    # Strategy-specific performance
    # ------------------------------------------------------------------

    def best_strategy(self, candidates: List[str]) -> str:
        """Return the highest-performing strategy from a list."""
        scores = {}
        for s in candidates:
            if s in self._strategy_scores and self._strategy_scores[s]:
                scores[s] = np.mean(self._strategy_scores[s][-10:])
        if not scores:
            return candidates[0] if candidates else ""
        return max(scores, key=scores.get)

    def strategy_ranking(self) -> List[Tuple[str, float]]:
        """Return strategies sorted by recent performance."""
        results = []
        for s, scores in self._strategy_scores.items():
            if scores:
                results.append((s, float(np.mean(scores[-10:]))))
        return sorted(results, key=lambda x: -x[1])

    # ------------------------------------------------------------------
    # Performance metrics for GoalGenerator
    # ------------------------------------------------------------------

    def performance_metrics(self) -> Dict:
        """Return a performance metrics dict suitable for GoalGenerator."""
        records = list(self._records)
        if not records:
            return {"overall_success_rate": 1.0}

        overall_sr = float(np.mean([r.success for r in records]))
        action_success = {}
        for action, successes in self._action_success.items():
            if successes:
                action_success[action] = float(np.mean(successes[-10:]))

        recent_danger = sum(
            1 for r in records[-20:]
            if r.reward < -0.2
        )

        return {
            "overall_success_rate": overall_sr,
            "action_success":       action_success,
            "recent_danger_count":  recent_danger,
            "n_records":            len(records),
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._records:
            sr = np.mean([r.success for r in self._records])
            return (f"MetaLearner(cycles={self._cycle}, "
                    f"records={len(self._records)}, sr={sr:.1%})")
        return f"MetaLearner(cycles={self._cycle}, records=0)"
