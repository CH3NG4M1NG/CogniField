"""
core/meta_cognition.py
========================
Meta-Cognition Engine

The agent thinks about its own thinking. It monitors:
  - whether its beliefs are overconfident relative to evidence
  - whether it has recurring blind spots (systematic errors)
  - which reasoning strategies are working vs failing
  - how its performance compares to its own expectations

Meta-Cognitive Processes
-------------------------
1. OVERCONFIDENCE DETECTION
   Compare actual success rate to expected success rate derived
   from belief confidence. If success_rate << mean_confidence,
   the agent is overconfident and must decay beliefs.

2. BIAS DETECTION
   Track per-predicate error rates. If the agent consistently
   mis-predicts "edibility" but is accurate on "fragility",
   it has a domain-specific bias.

3. REFLECTION LOG
   Human-readable self-analysis written after each reflection cycle.
   Contains: "I am overconfident on apple.edible. My predicted
   success was 0.85 but actual was 0.52. Downgrading."

4. EXPECTATION CALIBRATION
   Maintain a calibration curve: at confidence=0.8, what fraction
   of actions actually succeed? A well-calibrated agent has a slope
   near 1.0 on this curve.
"""

from __future__ import annotations

import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ReflectionEntry:
    """One self-reflection record."""
    step:         int
    finding:      str           # what was found
    action_taken: str           # what was done about it
    metric:       str           # which metric triggered it
    before:       float         # value before adjustment
    after:        float         # value after adjustment
    timestamp:    float = field(default_factory=time.time)


@dataclass
class CalibrationPoint:
    """One (confidence_bucket, actual_success_rate) data point."""
    conf_bucket:  float   # e.g. 0.8 = "70–90% confidence bucket"
    n_samples:    int
    n_successes:  int

    @property
    def empirical_rate(self) -> float:
        return self.n_successes / max(self.n_samples, 1)

    @property
    def calibration_error(self) -> float:
        return abs(self.conf_bucket - self.empirical_rate)


class MetaCognitionEngine:
    """
    Self-analysis, bias detection, and strategy adjustment.

    Parameters
    ----------
    overconf_threshold  : If mean_conf - success_rate > this, flag overconfidence.
    bias_threshold      : Domain error rate above this = detected bias.
    reflection_interval : Steps between reflection cycles.
    """

    def __init__(
        self,
        overconf_threshold:  float = 0.20,
        bias_threshold:      float = 0.40,
        reflection_interval: int   = 10,
    ) -> None:
        self.overconf_threshold  = overconf_threshold
        self.bias_threshold      = bias_threshold
        self.reflection_interval = reflection_interval

        # Rolling performance history
        self._step_records: deque = deque(maxlen=200)
        # Per-domain (predicate) history: {predicate → [correct_bool]}
        self._domain_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        # Calibration: {bucket_str → CalibrationPoint}
        self._calibration: Dict[str, CalibrationPoint] = {}
        # Reflection log
        self._reflections: List[ReflectionEntry] = []
        # Strategy adjustments made
        self._adjustments: int = 0
        self._step_count   = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        belief_conf: float,
        predicted:   Any,
        actual:      Any,
        predicate:   str = "",
        step:        int = 0,
    ) -> None:
        """
        Record one outcome for calibration and bias tracking.

        Parameters
        ----------
        belief_conf : Confidence the agent had before acting.
        predicted   : What the agent expected to happen.
        actual      : What actually happened.
        predicate   : Domain (e.g. "edible", "fragile").
        """
        correct = (str(predicted).lower() == str(actual).lower())
        self._step_records.append({
            "step":   step or self._step_count,
            "conf":   belief_conf,
            "correct": correct,
            "predicate": predicate,
        })

        # Domain tracking
        if predicate:
            self._domain_history[predicate].append(correct)

        # Calibration bucket
        bucket = round(belief_conf * 5) / 5   # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        key    = f"{bucket:.1f}"
        if key not in self._calibration:
            self._calibration[key] = CalibrationPoint(bucket, 0, 0)
        cp = self._calibration[key]
        cp.n_samples  += 1
        cp.n_successes += int(correct)

    def record_step(
        self,
        step:        int,
        success:     bool,
        reward:      float,
        mean_conf:   float,
        action:      str = "",
        predicate:   str = "",
    ) -> None:
        """Record one agent step for macro-level analysis."""
        self._step_count = step
        self._step_records.append({
            "step":      step,
            "success":   success,
            "reward":    reward,
            "conf":      mean_conf,
            "correct":   success,
            "predicate": predicate,
            "action":    action,
        })

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def detect_overconfidence(
        self,
        window: int = 20,
    ) -> Tuple[bool, float, float]:
        """
        Detect if agent is systematically overconfident.

        Returns
        -------
        (is_overconfident, mean_conf, actual_sr)
        """
        recent = list(self._step_records)[-window:]
        if len(recent) < 5:
            return False, 0.5, 0.5

        mean_conf = float(np.mean([r["conf"] for r in recent]))
        actual_sr = float(np.mean([r.get("success", r.get("correct", 0.5))
                                    for r in recent]))
        gap = mean_conf - actual_sr
        return gap > self.overconf_threshold, mean_conf, actual_sr

    def detect_biases(self) -> Dict[str, float]:
        """
        Return domains where agent's error rate exceeds bias_threshold.
        {predicate → error_rate}
        """
        biases = {}
        for pred, history in self._domain_history.items():
            if len(history) < 4:
                continue
            error_rate = 1.0 - float(np.mean(list(history)))
            if error_rate > self.bias_threshold:
                biases[pred] = round(error_rate, 3)
        return biases

    def calibration_score(self) -> float:
        """
        Overall calibration: 1.0 = perfect, 0.0 = maximally mis-calibrated.
        Measured as 1 - mean_calibration_error across buckets with data.
        """
        pts = [cp for cp in self._calibration.values() if cp.n_samples >= 3]
        if not pts:
            return 0.5
        errors = [cp.calibration_error for cp in pts]
        return float(1.0 - np.mean(errors))

    def expected_vs_actual(self) -> Tuple[float, float]:
        """Return (mean_confidence, actual_success_rate) from recent history."""
        recent = list(self._step_records)[-30:]
        if not recent:
            return 0.5, 0.5
        mc  = float(np.mean([r["conf"] for r in recent]))
        asr = float(np.mean([r.get("success", r.get("correct", 0.5))
                              for r in recent]))
        return mc, asr

    def performance_trend(self) -> str:
        """Return 'improving' | 'declining' | 'stable' based on recent vs older."""
        recs = list(self._step_records)
        if len(recs) < 10:
            return "unknown"
        half    = len(recs) // 2
        early   = float(np.mean([r.get("success", 0.5) for r in recs[:half]]))
        late    = float(np.mean([r.get("success", 0.5) for r in recs[half:]]))
        if late > early + 0.08:
            return "improving"
        if late < early - 0.08:
            return "declining"
        return "stable"

    # ------------------------------------------------------------------
    # Reflection cycle
    # ------------------------------------------------------------------

    def reflect(
        self,
        step: int,
        adjust_fn: Optional[Callable[[str, float], None]] = None,
    ) -> List[ReflectionEntry]:
        """
        Run one reflection cycle. Detects issues and optionally calls
        adjust_fn(issue_type, magnitude) to make corrections.

        Returns list of reflection entries produced.
        """
        if step % self.reflection_interval != 0:
            return []

        entries = []

        # 1. Overconfidence check
        over, mc, asr = self.detect_overconfidence()
        if over:
            msg = (f"Overconfident: predicted {mc:.3f} but achieved {asr:.3f} "
                   f"(gap={mc-asr:.3f}). Decaying belief confidence.")
            after = mc - (mc - asr) * 0.4
            entry = ReflectionEntry(
                step=step, finding=msg,
                action_taken="belief_decay",
                metric="overconfidence",
                before=mc, after=after,
            )
            entries.append(entry)
            self._adjustments += 1
            if adjust_fn:
                adjust_fn("overconfidence", mc - asr)

        # 2. Bias detection
        biases = self.detect_biases()
        for pred, err_rate in biases.items():
            msg = (f"Domain bias on '{pred}': error_rate={err_rate:.3f}. "
                   f"Reducing confidence on {pred} beliefs.")
            entry = ReflectionEntry(
                step=step, finding=msg,
                action_taken=f"bias_correction:{pred}",
                metric="domain_bias",
                before=1.0 - err_rate, after=1.0 - err_rate * 0.8,
            )
            entries.append(entry)
            self._adjustments += 1
            if adjust_fn:
                adjust_fn(f"bias:{pred}", err_rate)

        # 3. Calibration report
        cal = self.calibration_score()
        if cal < 0.60:
            msg = f"Poor calibration score: {cal:.3f}. Increasing uncertainty."
            entry = ReflectionEntry(
                step=step, finding=msg,
                action_taken="uncertainty_increase",
                metric="calibration",
                before=cal, after=cal + 0.1,
            )
            entries.append(entry)
            if adjust_fn:
                adjust_fn("poor_calibration", 1.0 - cal)

        # 4. Trend report
        trend = self.performance_trend()
        if trend != "unknown":
            msg = f"Performance trend: {trend}."
            entry = ReflectionEntry(
                step=step, finding=msg,
                action_taken="logged",
                metric="trend",
                before=0.0, after=0.0,
            )
            entries.append(entry)

        self._reflections.extend(entries)
        return entries

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def reflection_log(self, n: int = 5) -> List[str]:
        return [r.finding for r in self._reflections[-n:]]

    def summary(self) -> Dict:
        over, mc, asr = self.detect_overconfidence()
        return {
            "steps_recorded":     len(self._step_records),
            "overconfident":      over,
            "mean_confidence":    round(mc, 3),
            "actual_success_rate":round(asr, 3),
            "calibration_score":  round(self.calibration_score(), 3),
            "biases":             self.detect_biases(),
            "performance_trend":  self.performance_trend(),
            "adjustments":        self._adjustments,
            "n_reflections":      len(self._reflections),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"MetaCognition(steps={s['steps_recorded']}, "
                f"overconf={s['overconfident']}, "
                f"cal={s['calibration_score']:.2f}, "
                f"trend={s['performance_trend']})")
