"""
core/deep_thinker.py
======================
Deep Thinking Engine

Performs structured multi-step reasoning without any LLM.

Philosophy
----------
Instead of returning the first plausible answer, DeepThinker runs
N deliberation steps, each refining the reasoning through a different
lens:

  Step 1  KNOWLEDGE CHECK    — what do we actually know?
  Step 2  UNCERTAINTY SCAN   — how confident are we?
  Step 3  CAUSAL REASONING   — what are the causes and effects?
  Step 4  CONSEQUENCE TRACE  — what happens if we act?
  Step 5  CONTRADICTION CHECK — do our beliefs contradict?
  Step 6  RISK EVALUATION    — what could go wrong?
  Step 7  SYNTHESIS          — integrate all steps into final stance

In "fast" mode, only steps 1-2-7 run (minimum viable reasoning).
In "deep" mode, all steps run plus additional iterations if confidence
remains below the threshold after the first pass.

Each step produces a ThoughtRecord: what was examined, what was found,
and how it changed the confidence estimate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ThinkingMode(str, Enum):
    FAST  = "fast"    # 3 steps: knowledge + uncertainty + synthesis
    DEEP  = "deep"    # 7+ steps: full deliberation
    AUTO  = "auto"    # fast if confident, deep if uncertain


class ReasoningStep(str, Enum):
    KNOWLEDGE_CHECK     = "knowledge_check"
    UNCERTAINTY_SCAN    = "uncertainty_scan"
    CAUSAL_REASONING    = "causal_reasoning"
    CONSEQUENCE_TRACE   = "consequence_trace"
    CONTRADICTION_CHECK = "contradiction_check"
    RISK_EVALUATION     = "risk_evaluation"
    SYNTHESIS           = "synthesis"
    SELF_CORRECTION     = "self_correction"


@dataclass
class ThoughtRecord:
    """One reasoning step result."""
    step:        ReasoningStep
    finding:     str           # what was discovered
    confidence:  float         # confidence AFTER this step
    evidence:    float         # evidence weight behind this step
    delta:       float         # confidence change caused by this step
    timestamp:   float = field(default_factory=time.time)


@dataclass
class ThinkingResult:
    """
    Complete deep-thinking output.

    Attributes
    ----------
    decision     : Final decision ("proceed", "avoid", "uncertain", "investigate")
    confidence   : Final confidence [0, 1]
    reasoning    : Ordered list of human-readable reasoning steps
    thoughts     : Detailed ThoughtRecord per step
    n_steps      : Number of deliberation steps executed
    mode         : Which thinking mode was used
    elapsed_ms   : Total thinking time
    safe         : Whether all safety checks passed
    contradictions : Any contradictions found
    """
    decision:        str
    confidence:      float
    reasoning:       List[str]
    thoughts:        List[ThoughtRecord]
    n_steps:         int
    mode:            ThinkingMode
    elapsed_ms:      float
    safe:            bool       = True
    contradictions:  List[str]  = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "decision":       self.decision,
            "confidence":     round(self.confidence, 4),
            "reasoning":      self.reasoning,
            "n_steps":        self.n_steps,
            "mode":           self.mode.value,
            "elapsed_ms":     round(self.elapsed_ms, 1),
            "safe":           self.safe,
            "contradictions": self.contradictions,
        }


class DeepThinker:
    """
    Multi-step structured reasoning engine.

    Parameters
    ----------
    mode               : ThinkingMode (fast/deep/auto)
    min_steps          : Minimum reasoning steps (default 3)
    confidence_target  : Desired confidence before stopping (deep mode)
    max_iterations     : Maximum reasoning passes
    """

    FAST_STEPS = [
        ReasoningStep.KNOWLEDGE_CHECK,
        ReasoningStep.UNCERTAINTY_SCAN,
        ReasoningStep.SYNTHESIS,
    ]

    DEEP_STEPS = [
        ReasoningStep.KNOWLEDGE_CHECK,
        ReasoningStep.UNCERTAINTY_SCAN,
        ReasoningStep.CAUSAL_REASONING,
        ReasoningStep.CONSEQUENCE_TRACE,
        ReasoningStep.CONTRADICTION_CHECK,
        ReasoningStep.RISK_EVALUATION,
        ReasoningStep.SYNTHESIS,
    ]

    def __init__(
        self,
        mode:               ThinkingMode = ThinkingMode.AUTO,
        min_steps:          int   = 3,
        confidence_target:  float = 0.70,
        max_iterations:     int   = 3,
    ) -> None:
        self.mode               = mode
        self.min_steps          = min_steps
        self.confidence_target  = confidence_target
        self.max_iterations     = max_iterations
        self._history:          List[ThinkingResult] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def think(
        self,
        subject:       str,
        predicate:     str,
        belief_system,
        world_model_data: Optional[Dict] = None,
        context:          Optional[Dict] = None,
    ) -> ThinkingResult:
        """
        Run deep thinking on (subject, predicate) using available beliefs.

        Parameters
        ----------
        subject        : Object/concept being reasoned about (e.g. "apple")
        predicate      : Property being evaluated (e.g. "edible")
        belief_system  : BeliefSystem instance
        world_model_data : Optional extra world model data
        context        : Optional context dict (uncertainty level, etc.)

        Returns
        -------
        ThinkingResult
        """
        t0 = time.time()
        ctx = context or {}
        wmd = world_model_data or {}

        # Choose steps
        mode = self._select_mode(subject, predicate, belief_system)
        steps = self.DEEP_STEPS if mode == ThinkingMode.DEEP else self.FAST_STEPS
        # Enforce minimum steps
        if len(steps) < self.min_steps:
            steps = self.DEEP_STEPS[:max(self.min_steps, 3)]

        thoughts:        List[ThoughtRecord] = []
        reasoning:       List[str] = []
        contradictions:  List[str] = []
        confidence       = 0.50   # neutral prior
        safe             = True

        for iteration in range(self.max_iterations):
            for step in steps:
                record = self._execute_step(
                    step, subject, predicate, belief_system,
                    wmd, ctx, confidence
                )
                thoughts.append(record)
                reasoning.append(f"[{step.value}] {record.finding}")
                confidence = record.confidence

                # Contradiction found → immediately flag unsafe
                if step == ReasoningStep.CONTRADICTION_CHECK and record.delta < -0.10:
                    contradictions.append(record.finding)

                # Risk evaluation: if very high risk, override to avoid
                if step == ReasoningStep.RISK_EVALUATION and confidence < 0.25:
                    safe = False

            # Re-iterate in deep mode if still uncertain
            if mode == ThinkingMode.DEEP and confidence >= self.confidence_target:
                break
            if iteration == 0 and mode == ThinkingMode.FAST:
                break

        # Final decision
        decision = self._make_decision(confidence, safe, contradictions, predicate)

        elapsed = (time.time() - t0) * 1000
        result = ThinkingResult(
            decision=decision,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            reasoning=reasoning,
            thoughts=thoughts,
            n_steps=len(thoughts),
            mode=mode,
            elapsed_ms=elapsed,
            safe=safe,
            contradictions=contradictions,
        )
        self._history.append(result)
        return result

    # ------------------------------------------------------------------
    # Step executors
    # ------------------------------------------------------------------

    def _execute_step(
        self,
        step:           ReasoningStep,
        subject:        str,
        predicate:      str,
        belief_system,
        wmd:            Dict,
        ctx:            Dict,
        current_conf:   float,
    ) -> ThoughtRecord:
        """Dispatch to the appropriate step handler."""
        key = f"{subject}.{predicate}"

        if step == ReasoningStep.KNOWLEDGE_CHECK:
            return self._step_knowledge_check(key, subject, predicate,
                                               belief_system, current_conf)
        elif step == ReasoningStep.UNCERTAINTY_SCAN:
            return self._step_uncertainty_scan(key, belief_system, ctx, current_conf)
        elif step == ReasoningStep.CAUSAL_REASONING:
            return self._step_causal_reasoning(subject, predicate, wmd, current_conf)
        elif step == ReasoningStep.CONSEQUENCE_TRACE:
            return self._step_consequence_trace(subject, predicate, belief_system, current_conf)
        elif step == ReasoningStep.CONTRADICTION_CHECK:
            return self._step_contradiction_check(key, belief_system, current_conf)
        elif step == ReasoningStep.RISK_EVALUATION:
            return self._step_risk_evaluation(subject, predicate, belief_system, current_conf)
        elif step == ReasoningStep.SYNTHESIS:
            return self._step_synthesis(key, belief_system, current_conf)
        else:
            return ThoughtRecord(step=step, finding="No handler",
                                 confidence=current_conf, evidence=0, delta=0)

    def _step_knowledge_check(self, key, subject, predicate, bs, conf) -> ThoughtRecord:
        """Check what we actually know about this belief."""
        belief = bs.get(key)

        if belief is None:
            # Check category-level knowledge
            cat_belief = bs.get(f"{subject}.category") or bs.get(f"{subject}.is_a")
            if cat_belief and cat_belief.confidence > 0.6:
                cat_edible = bs.get(f"{cat_belief.value}.{predicate}")
                if cat_edible and cat_edible.confidence > 0.6:
                    new_conf = cat_edible.confidence * 0.75
                    finding  = (f"No direct {key} belief; inferred from category "
                                f"{cat_belief.value}.{predicate}={cat_edible.value} "
                                f"(conf={cat_edible.confidence:.2f})")
                    delta = new_conf - conf
                    return ThoughtRecord(step=ReasoningStep.KNOWLEDGE_CHECK,
                                         finding=finding, confidence=new_conf,
                                         evidence=cat_edible.total_evidence * 0.75,
                                         delta=delta)
            # Completely unknown
            new_conf = 0.20
            finding  = (f"No knowledge found for '{key}'. "
                        f"Subject '{subject}' has {len(bs.beliefs_about(subject))} related beliefs. "
                        f"Defaulting to unknown-safety rule: low confidence.")
            return ThoughtRecord(step=ReasoningStep.KNOWLEDGE_CHECK,
                                  finding=finding, confidence=new_conf,
                                  evidence=0.0, delta=new_conf - conf)

        finding = (f"Found '{key}': value={belief.value}, "
                   f"conf={belief.confidence:.3f}, "
                   f"evidence={belief.total_evidence:.1f}, "
                   f"source={belief.source}")
        delta = belief.confidence - conf
        return ThoughtRecord(step=ReasoningStep.KNOWLEDGE_CHECK,
                              finding=finding, confidence=belief.confidence,
                              evidence=belief.total_evidence, delta=delta)

    def _step_uncertainty_scan(self, key, bs, ctx, conf) -> ThoughtRecord:
        """Evaluate how certain we should be given the uncertainty context."""
        unc_level = ctx.get("uncertainty", "low")
        unc_penalties = {"none": 0.0, "low": 0.02, "medium": 0.06,
                         "high": 0.12, "chaotic": 0.22}
        penalty = unc_penalties.get(unc_level, 0.05)

        belief = bs.get(key)
        age_penalty = 0.0
        if belief:
            age = belief.age_seconds
            if age > 300:   # very old belief
                age_penalty = min(0.10, age / 6000)

        new_conf = max(0.10, conf - penalty - age_penalty)
        finding  = (f"Uncertainty level={unc_level} applies penalty={penalty:.2f}. "
                    f"Age penalty={age_penalty:.2f}. "
                    f"Adjusted confidence: {conf:.3f} → {new_conf:.3f}")
        return ThoughtRecord(step=ReasoningStep.UNCERTAINTY_SCAN,
                              finding=finding, confidence=new_conf,
                              evidence=1.0, delta=new_conf - conf)

    def _step_causal_reasoning(self, subject, predicate, wmd, conf) -> ThoughtRecord:
        """Reason from cause→effect using world model data."""
        causal_chains = wmd.get("causal_chains", {})
        rule = causal_chains.get(f"{subject}.{predicate}")

        if rule:
            cause  = rule.get("cause", "unknown")
            effect = rule.get("effect", "unknown")
            boost  = rule.get("confidence_boost", 0.05)
            new_conf = min(0.95, conf + boost)
            finding  = (f"Causal chain found: {cause} → {effect}. "
                        f"This {'+' if boost > 0 else ''}modifies confidence by {boost:+.2f}")
            return ThoughtRecord(step=ReasoningStep.CAUSAL_REASONING,
                                  finding=finding, confidence=new_conf,
                                  evidence=2.0, delta=boost)

        finding = (f"No causal chain for {subject}.{predicate}. "
                   f"Reasoning from first principles: "
                   f"if {predicate} → effect is unknown → neutral adjustment.")
        return ThoughtRecord(step=ReasoningStep.CAUSAL_REASONING,
                              finding=finding, confidence=conf,
                              evidence=0.5, delta=0.0)

    def _step_consequence_trace(self, subject, predicate, bs, conf) -> ThoughtRecord:
        """Trace downstream consequences of acting on this belief."""
        # Check known effects: if we act and it's wrong, how bad is it?
        toxicity_key = f"{subject}.toxic"
        toxic_b      = bs.get(toxicity_key)

        if toxic_b and toxic_b.value is True and toxic_b.confidence > 0.6:
            # Consequence of eating toxic: catastrophic
            new_conf = min(conf, 0.15)
            finding  = (f"CONSEQUENCE: {subject} is marked toxic "
                        f"(conf={toxic_b.confidence:.2f}). Acting could cause harm. "
                        f"Reducing confidence heavily.")
            return ThoughtRecord(step=ReasoningStep.CONSEQUENCE_TRACE,
                                  finding=finding, confidence=new_conf,
                                  evidence=toxic_b.total_evidence, delta=new_conf - conf)

        # Check historical outcomes
        safe_key = f"{subject}.safe"
        safe_b   = bs.get(safe_key)
        if safe_b and not safe_b.value and safe_b.confidence > 0.6:
            new_conf = min(conf, 0.25)
            finding  = (f"Historical outcomes show {subject} is not safe "
                        f"(conf={safe_b.confidence:.2f})")
            return ThoughtRecord(step=ReasoningStep.CONSEQUENCE_TRACE,
                                  finding=finding, confidence=new_conf,
                                  evidence=safe_b.total_evidence, delta=new_conf - conf)

        finding = (f"Consequence trace for {subject}.{predicate}: "
                   f"no catastrophic downstream effects found. Confidence stable.")
        return ThoughtRecord(step=ReasoningStep.CONSEQUENCE_TRACE,
                              finding=finding, confidence=conf,
                              evidence=1.0, delta=0.0)

    def _step_contradiction_check(self, key, bs, conf) -> ThoughtRecord:
        """Detect contradictions between this belief and related beliefs."""
        belief = bs.get(key)
        if belief is None:
            return ThoughtRecord(step=ReasoningStep.CONTRADICTION_CHECK,
                                  finding="No belief to check for contradictions.",
                                  confidence=conf, evidence=0, delta=0)

        # Check for direct conflicts in belief history
        n_conflicts = bs.n_conflicts
        recent_conflicts = bs.get_conflicts(recency_seconds=300)
        key_conflicts = [c for c in recent_conflicts if c.get("key") == key]

        if key_conflicts:
            penalty   = min(0.15, len(key_conflicts) * 0.05)
            new_conf  = max(0.15, conf - penalty)
            finding   = (f"CONTRADICTION: {len(key_conflicts)} recent conflicts on '{key}'. "
                         f"Values {key_conflicts[0].get('old_value')} vs "
                         f"{key_conflicts[0].get('new_value')}. "
                         f"Confidence penalised by {penalty:.2f}.")
            return ThoughtRecord(step=ReasoningStep.CONTRADICTION_CHECK,
                                  finding=finding, confidence=new_conf,
                                  evidence=float(len(key_conflicts)), delta=-penalty)

        finding = f"No contradictions found for '{key}'. Belief is consistent."
        return ThoughtRecord(step=ReasoningStep.CONTRADICTION_CHECK,
                              finding=finding, confidence=conf,
                              evidence=1.0, delta=0.0)

    def _step_risk_evaluation(self, subject, predicate, bs, conf) -> ThoughtRecord:
        """Evaluate whether the risk of acting is acceptable."""
        danger_keys = [f"{subject}.edible", f"{subject}.toxic",
                       f"{subject}.safe",   f"{subject}.dangerous"]
        max_danger  = 0.0
        danger_source = ""

        for dk in danger_keys:
            b = bs.get(dk)
            if b is None:
                continue
            is_danger = (
                (dk.endswith(".toxic")     and b.value is True)  or
                (dk.endswith(".edible")    and b.value is False) or
                (dk.endswith(".safe")      and b.value is False) or
                (dk.endswith(".dangerous") and b.value is True)
            )
            if is_danger and b.confidence > max_danger:
                max_danger    = b.confidence
                danger_source = dk

        # Also check category-level: material / tool → not edible
        cat_b = bs.get(f"{subject}.category")
        if cat_b and cat_b.value in ("material", "tool", "rock", "metal", "glass"):
            cat_danger = cat_b.confidence * 0.85
            if cat_danger > max_danger:
                max_danger    = cat_danger
                danger_source = f"{subject}.category={cat_b.value}"

        # Lower threshold: danger ≥ 0.45 already blocks proceed
        if max_danger >= 0.45:
            new_conf = min(conf, max(0.10, 0.35 - max_danger * 0.15))
            finding  = (f"DANGER SIGNAL ({danger_source}): strength={max_danger:.2f}. "
                        f"Risk overrides proceed. Confidence → {new_conf:.3f}.")
            return ThoughtRecord(step=ReasoningStep.RISK_EVALUATION,
                                  finding=finding, confidence=new_conf,
                                  evidence=max_danger, delta=new_conf - conf)

        if conf < 0.40:
            finding = (f"MODERATE RISK: confidence {conf:.3f} below safety threshold 0.40. "
                       f"Insufficient knowledge — investigate first.")
        else:
            finding = (f"Risk acceptable: conf={conf:.3f} ≥ 0.40. "
                       f"No danger signals (max={max_danger:.2f}).")

        return ThoughtRecord(step=ReasoningStep.RISK_EVALUATION,
                              finding=finding, confidence=conf,
                              evidence=max(max_danger, 0.5), delta=0.0)

    def _step_synthesis(self, key, bs, conf) -> ThoughtRecord:
        """Integrate all prior steps into a final confidence estimate."""
        belief = bs.get(key)
        parts  = key.split(".")
        subj   = parts[0] if parts else "unknown"

        # Pull in any reinforcing beliefs about the same subject
        related = bs.beliefs_about(subj)
        boosts  = []
        for rb in related:
            if rb.is_reliable and rb.key != key:
                boosts.append(rb.confidence * 0.05)

        total_boost = min(0.10, sum(boosts))
        new_conf    = min(0.95, conf + total_boost)

        ev_str = f"{belief.total_evidence:.1f}" if belief else "0"
        finding = (f"Synthesis: integrated {len(related)} related beliefs. "
                   f"Reinforcement boost={total_boost:.3f}. "
                   f"Evidence={ev_str}. "
                   f"Final confidence: {new_conf:.3f}")
        return ThoughtRecord(step=ReasoningStep.SYNTHESIS,
                              finding=finding, confidence=new_conf,
                              evidence=float(len(related)), delta=total_boost)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _select_mode(self, subject, predicate, bs) -> ThinkingMode:
        """Auto-select mode based on current knowledge state."""
        if self.mode != ThinkingMode.AUTO:
            return self.mode
        key    = f"{subject}.{predicate}"
        belief = bs.get(key)
        # Use fast mode if we already have a reliable belief
        if belief and belief.is_reliable and belief.confidence >= 0.70:
            return ThinkingMode.FAST
        return ThinkingMode.DEEP

    def _make_decision(
        self,
        confidence:     float,
        safe:           bool,
        contradictions: List[str],
        predicate:      str,
    ) -> str:
        if not safe or contradictions:
            return "avoid"
        if confidence >= 0.75:
            return "proceed"
        if confidence >= 0.50:
            return "proceed_with_caution"
        if confidence >= 0.30:
            return "investigate"
        return "avoid"

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        if not self._history:
            return {"runs": 0}
        decisions = {}
        for r in self._history:
            decisions[r.decision] = decisions.get(r.decision, 0) + 1
        return {
            "runs":           len(self._history),
            "mean_steps":     round(float(np.mean([r.n_steps for r in self._history])), 1),
            "mean_confidence":round(float(np.mean([r.confidence for r in self._history])), 3),
            "decisions":      decisions,
        }

    def __repr__(self) -> str:
        return f"DeepThinker(mode={self.mode.value}, runs={len(self._history)})"
