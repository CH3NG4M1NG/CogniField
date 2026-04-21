"""
agent/goal_generator.py
========================
Self-Generated Goal System

Generates goals autonomously from internal signals rather than waiting
for external instructions. This is the engine of self-direction.

Generation Sources
------------------
1. CURIOSITY SIGNAL
   - Unknown concepts detected in memory or environment
   - High novelty scores → "understand X" goals

2. FAILURE ANALYSIS
   - Low success rates on specific action types
   - Repeated failures → "improve X skill" goals

3. ENVIRONMENT OBSERVATIONS
   - Visible objects with unknown properties
   - Low satiation → "find food" goals
   - Unseen areas → "explore region" goals

4. KNOWLEDGE GAPS
   - Relational memory missing key facts
   - Conflicting hypotheses → "resolve X" goals

5. META-GOALS
   - "improve reasoning" when frustration is high
   - "consolidate memory" when fatigue is high

Goal Scoring
------------
Each candidate goal receives a priority score:

  priority = urgency × relevance × feasibility × novelty_bonus

The highest-scoring goals are added to the GoalSystem.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .goals import GoalSystem, GoalType, Goal
from .internal_state import InternalState
from ..memory.relational_memory import RelationalMemory
from ..memory.memory_store import MemoryStore
from ..curiosity.advanced_curiosity import AdvancedCuriosityEngine
from ..world_model.transition_model import TransitionModel
from ..latent_space.frequency_space import FrequencySpace


@dataclass
class GoalCandidate:
    """A proposed goal before it's accepted into the system."""
    label:     str
    goal_type: GoalType
    target:    str
    priority:  float
    source:    str           # "curiosity" | "failure" | "environment" | "meta"
    rationale: str
    created_at: float = field(default_factory=time.time)


class GoalGenerator:
    """
    Generates goals autonomously from memory, curiosity, performance, and state.

    Parameters
    ----------
    goal_system    : The agent's GoalSystem to push goals into.
    rel_memory     : Relational memory for knowledge gap detection.
    vec_memory     : Vector memory for similarity-based queries.
    curiosity      : AdvancedCuriosityEngine for novelty signals.
    world_model    : TransitionModel for failure analysis.
    space          : FrequencySpace for vector encoding.
    enc_fn         : Callable to encode text to a vector.
    max_active_goals: Don't generate more goals if above this.
    """

    def __init__(
        self,
        goal_system:      GoalSystem,
        rel_memory:       RelationalMemory,
        vec_memory:       MemoryStore,
        curiosity:        AdvancedCuriosityEngine,
        world_model:      TransitionModel,
        space:            FrequencySpace,
        enc_fn,                                    # text → np.ndarray
        max_active_goals: int = 6,
        dim:              int = 64,
    ) -> None:
        self.goals         = goal_system
        self.rel_mem       = rel_memory
        self.vec_mem       = vec_memory
        self.curiosity     = curiosity
        self.wm            = world_model
        self.space         = space
        self.enc           = enc_fn
        self.max_active    = max_active_goals
        self.dim           = dim
        self._generated_count = 0
        self._generation_log: List[GoalCandidate] = []

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def generate(
        self,
        internal_state:     InternalState,
        env_observation:    Optional[Dict] = None,
        performance_metrics: Optional[Dict] = None,
        max_new_goals:      int = 3,
    ) -> List[Goal]:
        """
        Generate new goals from all available signals.

        Parameters
        ----------
        internal_state      : Current InternalState of the agent.
        env_observation     : Latest environment observation dict.
        performance_metrics : Dict of action→success_rate, overall_success, etc.
        max_new_goals       : Max goals to add in this cycle.

        Returns
        -------
        List of newly created Goal objects.
        """
        if self.goals.active_count >= self.max_active:
            return []

        candidates: List[GoalCandidate] = []

        # 1. Curiosity-driven goals
        candidates += self._curiosity_goals(internal_state)

        # 2. Failure-driven goals
        if performance_metrics:
            candidates += self._failure_goals(performance_metrics, internal_state)

        # 3. Environment-driven goals
        if env_observation:
            candidates += self._environment_goals(env_observation, internal_state)

        # 4. Knowledge-gap goals
        candidates += self._knowledge_gap_goals(internal_state)

        # 5. Meta-goals (self-improvement)
        candidates += self._meta_goals(internal_state)

        # Rank and deduplicate
        candidates = self._rank_candidates(candidates)
        candidates = self._deduplicate(candidates)

        # Add top-N to goal system
        created = []
        for cand in candidates[:max_new_goals]:
            goal_vec = self.enc(cand.label)
            g = self.goals.add_goal(
                label=cand.label,
                goal_type=cand.goal_type,
                target=cand.target,
                priority=cand.priority,
                goal_vec=goal_vec,
                metadata={"source": cand.source, "rationale": cand.rationale},
            )
            created.append(g)
            self._generation_log.append(cand)
            self._generated_count += 1

        return created

    # ------------------------------------------------------------------
    # Goal generators
    # ------------------------------------------------------------------

    def _curiosity_goals(self, state: InternalState) -> List[GoalCandidate]:
        """Generate goals from curiosity exploration log."""
        candidates = []
        curiosity_scale = state.curiosity

        # Recently explored unknowns with open hypotheses
        for report in self.curiosity._exploration_log[-10:]:
            concept = report.get("concept", "")
            if not concept or self.rel_mem.is_known(concept):
                continue
            priority = float(report.get("priority", 0.4)) * curiosity_scale
            candidates.append(GoalCandidate(
                label=f"understand {concept}",
                goal_type=GoalType.EXPLORE,
                target=concept,
                priority=min(0.9, priority + 0.1),
                source="curiosity",
                rationale=f"Novel concept '{concept}' with {report.get('n_hypotheses',0)} open hypotheses",
            ))

        # Concepts with unconfirmed hypotheses
        open_hyps = [h for h in self.curiosity._hypotheses
                     if h.status == "open" and h.predicate in ("edible", "fragile", "category")]
        for h in open_hyps[:3]:
            candidates.append(GoalCandidate(
                label=f"test {h.subject}.{h.predicate}",
                goal_type=GoalType.LEARN,
                target=h.subject,
                priority=0.5 * curiosity_scale,
                source="curiosity",
                rationale=f"Open hypothesis: {h.subject}.{h.predicate}={h.predicted} (conf={h.confidence:.2f})",
            ))

        return candidates

    def _failure_goals(
        self,
        metrics: Dict,
        state:   InternalState,
    ) -> List[GoalCandidate]:
        """Generate goals from failure pattern analysis."""
        candidates = []

        # Low overall success → improve reasoning
        overall_sr = metrics.get("overall_success_rate", 1.0)
        if overall_sr < 0.5 and state.frustration > 0.3:
            candidates.append(GoalCandidate(
                label="improve reasoning strategies",
                goal_type=GoalType.CUSTOM,
                target="reasoning",
                priority=0.7 * state.frustration,
                source="failure",
                rationale=f"Overall success rate is {overall_sr:.1%} — need better strategies",
            ))

        # Specific action failures
        for action, sr in metrics.get("action_success", {}).items():
            if sr < 0.4:
                candidates.append(GoalCandidate(
                    label=f"improve {action} skill",
                    goal_type=GoalType.LEARN,
                    target=action,
                    priority=0.6 * (1 - sr),
                    source="failure",
                    rationale=f"Action '{action}' succeeds only {sr:.1%} of the time",
                ))

        # Dangerous actions taken (high-penalty events)
        recent_danger = metrics.get("recent_danger_count", 0)
        if recent_danger > 0:
            candidates.append(GoalCandidate(
                label="avoid dangerous actions",
                goal_type=GoalType.AVOID,
                target="dangerous",
                priority=0.85,
                source="failure",
                rationale=f"{recent_danger} dangerous actions taken recently",
            ))

        return candidates

    def _environment_goals(
        self,
        obs:   Dict,
        state: InternalState,
    ) -> List[GoalCandidate]:
        """Generate goals from environment observation."""
        candidates = []

        # Low satiation → find food
        satiation = obs.get("satiation", 1.0)
        if satiation < 0.4:
            edible = self.rel_mem.find_edible()
            if edible:
                candidates.append(GoalCandidate(
                    label=f"eat {edible[0]}",
                    goal_type=GoalType.EAT_OBJECT,
                    target=edible[0],
                    priority=0.6 + 0.4 * (1 - satiation),
                    source="environment",
                    rationale=f"Satiation is low ({satiation:.2f}), known food: {edible[0]}",
                ))
            else:
                candidates.append(GoalCandidate(
                    label="find food",
                    goal_type=GoalType.EXPLORE,
                    target="food",
                    priority=0.7 * (1 - satiation),
                    source="environment",
                    rationale=f"Satiation is low ({satiation:.2f}) and no known food",
                ))

        # Unknown objects visible
        unknown_objs = obs.get("unknown_objects", [])
        for obj in unknown_objs[:2]:
            candidates.append(GoalCandidate(
                label=f"investigate {obj}",
                goal_type=GoalType.EXPLORE,
                target=obj,
                priority=0.55 * state.curiosity,
                source="environment",
                rationale=f"Unknown object '{obj}' visible in environment",
            ))

        # Low health → seek safety
        health = obs.get("health", 1.0)
        if health < 0.5:
            candidates.append(GoalCandidate(
                label="restore health",
                goal_type=GoalType.CUSTOM,
                target="health",
                priority=0.9,
                source="environment",
                rationale=f"Health is critical ({health:.2f})",
            ))

        return candidates

    def _knowledge_gap_goals(self, state: InternalState) -> List[GoalCandidate]:
        """Generate goals from gaps in relational memory."""
        candidates = []

        # Objects with unknown edibility (we know they exist but not if edible)
        summary = self.rel_mem.summary()
        total_concepts = summary.get("concepts", 0)
        edible_known   = len(summary.get("edible", []))
        dangerous_known= len(summary.get("dangerous", []))

        if total_concepts > 0:
            known_edibility_fraction = (edible_known + dangerous_known) / max(total_concepts, 1)
            if known_edibility_fraction < 0.5 and state.curiosity > 0.4:
                candidates.append(GoalCandidate(
                    label="learn object properties",
                    goal_type=GoalType.LEARN,
                    target="objects",
                    priority=0.5 * (1 - known_edibility_fraction),
                    source="knowledge_gap",
                    rationale=f"Only {known_edibility_fraction:.0%} of objects have known edibility",
                ))

        # Rules with low confidence
        low_conf_rules = [r for r in self.wm.get_rules()
                         if r.confidence < 0.5 and r.hit_count + r.miss_count < 3]
        for rule in low_conf_rules[:2]:
            candidates.append(GoalCandidate(
                label=f"verify {rule.action}({rule.object_category}) rule",
                goal_type=GoalType.LEARN,
                target=rule.action,
                priority=0.4,
                source="knowledge_gap",
                rationale=f"Rule {rule.action}({rule.object_category}) has low confidence ({rule.confidence:.2f})",
            ))

        return candidates

    def _meta_goals(self, state: InternalState) -> List[GoalCandidate]:
        """Generate self-improvement / housekeeping goals."""
        candidates = []

        # High fatigue → consolidate
        if state.should_consolidate():
            candidates.append(GoalCandidate(
                label="consolidate memory",
                goal_type=GoalType.CUSTOM,
                target="memory",
                priority=0.4 + 0.3 * state.fatigue,
                source="meta",
                rationale=f"Fatigue is high ({state.fatigue:.2f}), memory consolidation needed",
            ))

        # High frustration → meta-learning
        if state.should_meta_learn():
            candidates.append(GoalCandidate(
                label="revise reasoning strategies",
                goal_type=GoalType.CUSTOM,
                target="strategies",
                priority=0.6 * state.frustration,
                source="meta",
                rationale=f"Frustration is high ({state.frustration:.2f}), strategies need revision",
            ))

        # Exploration when curious and rested
        if state.should_explore_boldly():
            candidates.append(GoalCandidate(
                label="explore environment",
                goal_type=GoalType.EXPLORE,
                target="environment",
                priority=0.5 * state.curiosity,
                source="meta",
                rationale=f"High curiosity ({state.curiosity:.2f}) with low fatigue",
            ))

        return candidates

    # ------------------------------------------------------------------
    # Ranking and deduplication
    # ------------------------------------------------------------------

    def _rank_candidates(self, candidates: List[GoalCandidate]) -> List[GoalCandidate]:
        """Sort candidates by priority descending."""
        return sorted(candidates, key=lambda c: -c.priority)

    def _deduplicate(self, candidates: List[GoalCandidate]) -> List[GoalCandidate]:
        """Remove candidates that overlap with existing active goals."""
        active_labels = {g.label.lower() for g in self.goals._goals if g.is_active}
        active_targets = {g.target.lower() for g in self.goals._goals if g.is_active}

        result = []
        seen_labels = set()
        for cand in candidates:
            label_lower  = cand.label.lower()
            target_lower = cand.target.lower()
            # Skip if essentially same as existing active goal
            if label_lower in active_labels:
                continue
            if target_lower in active_targets and cand.goal_type != GoalType.META if hasattr(GoalType, 'META') else False:
                continue
            if label_lower in seen_labels:
                continue
            seen_labels.add(label_lower)
            result.append(cand)
        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        sources = {}
        for g in self._generation_log:
            sources[g.source] = sources.get(g.source, 0) + 1
        return {
            "total_generated": self._generated_count,
            "by_source":       sources,
            "current_active":  self.goals.active_count,
        }

    def __repr__(self) -> str:
        return (f"GoalGenerator(generated={self._generated_count}, "
                f"active_goals={self.goals.active_count})")
