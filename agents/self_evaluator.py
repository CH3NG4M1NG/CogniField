"""
agents/self_evaluator.py
=========================
Self-Evaluator

Periodically grades the agent's own performance across multiple
dimensions and produces a structured weakness report.

Evaluation Dimensions
---------------------
BELIEF_ACCURACY   – confidence calibration vs actual outcomes
GOAL_EFFICIENCY   – goals completed / goals started
EXPLORATION_DEPTH – how many unique objects / predicates explored
RISK_COMPLIANCE   – high-risk actions avoided when confidence was low
COMMUNICATION     – messages sent AND received (bidirectionality)
CONSISTENCY       – no contradictory beliefs
LEARNING_SPEED    – how quickly new beliefs become reliable

Output
------
- Numerical grade per dimension (0–1)
- Overall composite score (0–1)
- Letter grade (A–F)
- Weakness list: dimensions scoring below threshold
- Improvement suggestions for each weakness
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DIMENSIONS = [
    "belief_accuracy",
    "goal_efficiency",
    "exploration_depth",
    "risk_compliance",
    "communication",
    "consistency",
    "learning_speed",
]

DIMENSION_WEIGHTS = {
    "belief_accuracy":    0.25,
    "goal_efficiency":    0.20,
    "exploration_depth":  0.10,
    "risk_compliance":    0.15,
    "communication":      0.10,
    "consistency":        0.10,
    "learning_speed":     0.10,
}

IMPROVEMENT_SUGGESTIONS = {
    "belief_accuracy":   "Run more experiments; apply uncertainty decay more aggressively.",
    "goal_efficiency":   "Use goal conflict resolver to drop low-value goals.",
    "exploration_depth": "Switch to EXPLORE strategy; lower novelty threshold.",
    "risk_compliance":   "Increase risk engine tolerance only for high-confidence targets.",
    "communication":     "Call ensure_bidirectional_comm() every step.",
    "consistency":       "Run consistency audit every 3 steps; apply enforcer.",
    "learning_speed":    "Increase share_beliefs_freq; read shared memory more often.",
}


@dataclass
class EvalReport:
    """One self-evaluation report."""
    step:         int
    scores:       Dict[str, float]      # dimension → score [0,1]
    overall:      float
    grade:        str
    weaknesses:   List[str]
    suggestions:  Dict[str, str]
    notes:        str
    timestamp:    float = field(default_factory=time.time)

    def is_excellent(self) -> bool:
        return self.overall >= 0.85

    def needs_improvement(self) -> bool:
        return len(self.weaknesses) > 0

    def worst_dimension(self) -> Optional[str]:
        if not self.scores:
            return None
        return min(self.scores, key=self.scores.get)


class SelfEvaluator:
    """
    Periodically evaluates agent performance and generates weakness reports.

    Parameters
    ----------
    eval_freq        : Steps between evaluations.
    weakness_threshold : Score below this = weakness.
    """

    def __init__(
        self,
        eval_freq:           int   = 15,
        weakness_threshold:  float = 0.50,
    ) -> None:
        self.eval_freq           = eval_freq
        self.weakness_threshold  = weakness_threshold
        self._reports: List[EvalReport] = []

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(self, step: int, agent) -> Optional[EvalReport]:
        """
        Run one evaluation cycle. Returns EvalReport or None if not due.

        Parameters
        ----------
        step  : Current step number.
        agent : CogniFieldAgentV9 (must have all v9 attributes).
        """
        if step % self.eval_freq != 0:
            return None

        scores = {}

        # 1. Belief accuracy: calibration from meta-cognition
        if hasattr(agent, "meta_cog"):
            s = agent.meta_cog.summary()
            cal = s.get("calibration_score", 0.5)
            over = s.get("overconfident", False)
            scores["belief_accuracy"] = float(cal * (0.7 if over else 1.0))
        else:
            scores["belief_accuracy"] = 0.5

        # 2. Goal efficiency
        if hasattr(agent, "goal_system"):
            gs = agent.goal_system
            completed = getattr(gs, "completed_count", 0)
            total     = completed + getattr(gs, "_failed_count", 0)
            scores["goal_efficiency"] = float(completed / max(total, 1))
        else:
            scores["goal_efficiency"] = 0.5

        # 3. Exploration depth: unique beliefs discovered
        if hasattr(agent, "beliefs"):
            n_beliefs  = len(agent.beliefs)
            n_reliable = len(agent.beliefs.reliable_beliefs())
            scores["exploration_depth"] = float(
                min(1.0, n_reliable / max(n_beliefs, 1))
            )
        else:
            scores["exploration_depth"] = 0.5

        # 4. Risk compliance: blocked/cautious vs total risky actions
        if hasattr(agent, "risk_engine"):
            rp = agent.risk_engine.risk_profile()
            n  = rp.get("n_assessments", 0)
            if n > 0:
                blocked = rp.get("blocked", 0) + rp.get("caution", 0)
                # High risk compliance = few blocks relative to total
                # (if many blocks, agent is attempting too many risky actions)
                risky_rate = blocked / n
                scores["risk_compliance"] = float(max(0, 1.0 - risky_rate))
            else:
                scores["risk_compliance"] = 1.0
        else:
            scores["risk_compliance"] = 1.0

        # 5. Communication bidirectionality
        tx = getattr(agent, "_msgs_sent_total",    0)
        rx = getattr(agent, "_msgs_received_total", 0)
        if tx > 0 and rx > 0:
            scores["communication"] = 1.0
        elif tx > 0 or rx > 0:
            scores["communication"] = 0.5
        else:
            scores["communication"] = 0.0

        # 6. Consistency
        if hasattr(agent, "consistency_engine"):
            audit = agent.consistency_engine.audit()
            n_viol = audit.get("n_violations", 0)
            scores["consistency"] = float(1.0 / (1.0 + n_viol))
        else:
            scores["consistency"] = 1.0

        # 7. Learning speed: how quickly beliefs become reliable
        if hasattr(agent, "beliefs"):
            bs = agent.beliefs
            reliable = len(bs.reliable_beliefs())
            total    = max(len(bs), 1)
            # Also factor in number of steps
            speed = reliable / total * min(1.0, step / 20)
            scores["learning_speed"] = float(min(1.0, speed))
        else:
            scores["learning_speed"] = 0.5

        # Overall weighted score
        overall = float(sum(
            DIMENSION_WEIGHTS.get(d, 0.1) * s
            for d, s in scores.items()
        ))

        # Grade
        grade = self._to_grade(overall)

        # Weaknesses
        weaknesses = [d for d, s in scores.items()
                      if s < self.weakness_threshold]
        suggestions = {d: IMPROVEMENT_SUGGESTIONS[d] for d in weaknesses}

        # Notes
        notes = f"Step {step}: grade={grade}, overall={overall:.3f}"
        if weaknesses:
            notes += f", weak on: {weaknesses}"

        report = EvalReport(
            step=step, scores=scores, overall=overall,
            grade=grade, weaknesses=weaknesses,
            suggestions=suggestions, notes=notes,
        )
        self._reports.append(report)
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_grade(self, score: float) -> str:
        if score >= 0.88: return "A"
        if score >= 0.75: return "B"
        if score >= 0.60: return "C"
        if score >= 0.45: return "D"
        return "F"

    def latest_report(self) -> Optional[EvalReport]:
        return self._reports[-1] if self._reports else None

    def improvement_over_time(self) -> float:
        """Return how much overall score has improved across all reports."""
        if len(self._reports) < 2:
            return 0.0
        return self._reports[-1].overall - self._reports[0].overall

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        if not self._reports:
            return {"n_reports": 0}
        latest = self._reports[-1]
        return {
            "n_reports":     len(self._reports),
            "latest_grade":  latest.grade,
            "latest_overall":round(latest.overall, 3),
            "weaknesses":    latest.weaknesses,
            "improvement":   round(self.improvement_over_time(), 3),
            "scores":        {d: round(s, 3) for d, s in latest.scores.items()},
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"SelfEvaluator(reports={s['n_reports']}, "
                f"grade={s.get('latest_grade','?')}, "
                f"overall={s.get('latest_overall',0):.2f})")
