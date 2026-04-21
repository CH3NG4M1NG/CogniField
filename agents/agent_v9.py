"""
agents/agent_v9.py
===================
CogniField v9 — Adaptive, Self-Reflective Intelligence

Extends AgentV8 with:
  - MetaCognition  (self-analysis, bias detection, calibration)
  - UncertaintyEngine (noisy inputs, confidence decay, partial obs)
  - GoalConflictResolver (trade-off decisions between competing goals)
  - StrategyManager (dynamic explore/exploit/verify/recover switching)
  - TemporalMemory  (long-term pattern detection, drift tracking)
  - SelfEvaluator   (graded performance reports + weakness detection)

v9 Loop (22 steps):
  1.  Observe (with noise injection)
  2.  Apply GroupMind signal + current strategy
  3.  Receive + decode messages
  4.  Read shared memory
  5.  Apply uncertainty decay to beliefs
  6.  Update private memory + temporal memory
  7.  Validate knowledge
  8.  Detect novelty
  9.  Resolve goal conflicts
  10. Select goal (conflict-resolved)
  11. Plan
  12. Simulate
  13. Risk check
  14. Act
  15. Receive feedback (with noise injection)
  16. Update beliefs (noise-aware)
  17. Meta-cognition: record outcome for calibration
  18. Temporal memory: record pattern
  19. Share experience / write shared memory
  20. Strategy evaluation + possible switch
  21. Self-evaluation (periodic)
  22. Reflection cycle (periodic)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .agent_v8 import CogniFieldAgentV8, AgentV8Config, V8Step
from .goal_conflict_resolver import GoalConflictResolver, ResolutionStrategy
from .strategy_manager import StrategyManager, Strategy
from .self_evaluator import SelfEvaluator
from ..core.meta_cognition import MetaCognitionEngine
from ..core.uncertainty_engine import UncertaintyEngine, UncertaintyLevel
from ..memory.temporal_memory import TemporalMemory
from ..agents.group_mind import GroupMind
from ..reasoning.global_consensus import GlobalConsensus
from ..communication.communication_module import CommunicationModule
from ..memory.shared_memory import SharedMemory
from ..core.event_bus import EventBus
from ..planning.cooperation_engine import CooperationEngine


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AgentV9Config(AgentV8Config):
    uncertainty_level:     str   = "medium"   # none/low/medium/high/chaotic
    metacog_interval:      int   = 10
    strategy_eval_freq:    int   = 8
    self_eval_freq:        int   = 15
    temporal_window:       int   = 20
    overconf_threshold:    float = 0.20
    noise_seed_offset:     int   = 100  # offset from agent seed for noise RNG


# ---------------------------------------------------------------------------
# V9 Step record
# ---------------------------------------------------------------------------

@dataclass
class V9Step(V8Step):
    uncertainty_level: str   = "medium"
    strategy:          str   = "explore"
    strategy_switched: bool  = False
    goal_conflicts:    int   = 0
    self_eval_grade:   str   = ""
    reflection_items:  int   = 0
    decay_applied:     int   = 0
    noise_corruptions: int   = 0


# ---------------------------------------------------------------------------
# CogniFieldAgentV9
# ---------------------------------------------------------------------------

class CogniFieldAgentV9(CogniFieldAgentV8):
    """
    v9 — Adaptive self-reflective intelligence agent.
    """

    def __init__(
        self,
        config:      Optional[AgentV9Config] = None,
        env          = None,
        comm_bus:    Optional[CommunicationModule] = None,
        shared_mem:  Optional[SharedMemory]        = None,
        group_mind:  Optional[GroupMind]           = None,
        global_cons: Optional[GlobalConsensus]     = None,
        event_bus:   Optional[EventBus]            = None,
        coop_engine: Optional[CooperationEngine]   = None,
    ) -> None:
        cfg = config or AgentV9Config()
        super().__init__(config=cfg, env=env, comm_bus=comm_bus,
                         shared_mem=shared_mem, group_mind=group_mind,
                         global_cons=global_cons, event_bus=event_bus,
                         coop_engine=coop_engine)

        self.cfg_v9: AgentV9Config = cfg

        # Map string level to enum
        level_map = {l.value: l for l in UncertaintyLevel}
        noise_seed = cfg.seed + cfg.noise_seed_offset

        # v9 components
        self.meta_cog    = MetaCognitionEngine(
            overconf_threshold=cfg.overconf_threshold,
            reflection_interval=cfg.metacog_interval,
        )
        self.uncertainty = UncertaintyEngine(
            level=level_map.get(cfg.uncertainty_level, UncertaintyLevel.MEDIUM),
            seed=noise_seed,
        )
        self.goal_resolver = GoalConflictResolver(
            strategy=ResolutionStrategy.UTILITY_MAXIMISATION
        )
        self.strategy_mgr = StrategyManager(
            initial_strategy=Strategy.EXPLORE,
            eval_freq=cfg.strategy_eval_freq,
        )
        self.temporal_mem = TemporalMemory(window=cfg.temporal_window)
        self.self_eval    = SelfEvaluator(
            eval_freq=cfg.self_eval_freq,
            weakness_threshold=0.50,
        )

        # Auto-detect uncertainty from reward variance
        self._auto_detect_uncertainty_freq = 10

    # ══════════════════════════════════════════════════════════════════
    # Override step() — v9 22-step loop
    # ══════════════════════════════════════════════════════════════════

    def step(
        self,
        text_input:   str = "",
        force_action: Optional[Tuple[str, str]] = None,
        verbose:      Optional[bool] = None,
    ) -> V9Step:
        verbose = verbose if verbose is not None else self.cfg.verbose

        # ── Apply current strategy to agent params ──
        self.strategy_mgr.apply_to_agent(self)

        # ── 5: Uncertainty decay on beliefs ──
        decay_count = 0
        if self._step_count % 3 == 0:
            decay_count = self.uncertainty.decay_all_beliefs(self.beliefs, steps=1)

        # ── 9: Resolve goal conflicts ──
        goal_conflicts = 0
        active_goals = [g for g in self.goal_system._goals if g.is_active]
        if len(active_goals) >= 2:
            decision = self.goal_resolver.resolve(
                active_goals,
                belief_system=self.beliefs,
                internal_state=self.internal_state,
            )
            goal_conflicts = len(decision.conflicts)
            if decision.dropped_goals and verbose:
                for g in decision.dropped_goals[:1]:
                    print(f"      [{self.agent_id}] ⚖ Dropped goal '{g.label}' "
                          f"(conflict resolution)")

        # ── Run v8 step ──
        v8_result = super().step(text_input=text_input,
                                  force_action=force_action,
                                  verbose=verbose)

        action   = v8_result.action_taken or ""
        obj_name = v8_result.action_obj   or ""
        success  = v8_result.env_success or False
        reward   = v8_result.env_reward or 0.0

        # ── 15-b: Inject noise into incoming feedback ──
        noise_corruptions = 0
        if action and obj_name and reward != 0.0:
            noisy_obs = self.uncertainty.corrupt(
                success, confidence=self.internal_state.confidence,
                predicate="edible" if action == "eat" else "",
            )
            if noisy_obs.was_corrupted:
                noise_corruptions += 1
                # Dampen the belief update by corrupting the observation weight
                key = f"{obj_name}.edible"
                if action == "eat" and self.beliefs.get(key):
                    b = self.beliefs.get(key)
                    b.confidence *= noisy_obs.confidence_weight

        # ── 17: Meta-cognition record ──
        self.meta_cog.record_step(
            step=self._step_count,
            success=success,
            reward=reward,
            mean_conf=self.beliefs.summary().get("mean_conf", 0.5),
            action=action,
            predicate="edible" if action == "eat" else "",
        )

        # ── 18: Temporal memory pattern ──
        if action and obj_name:
            self.temporal_mem.record_outcome(
                action, obj_name, success, reward,
                step=self._step_count,
                context={
                    "strategy": self.strategy_mgr.current.value,
                    "uncertainty": self.uncertainty.level.value,
                },
            )
        # Snapshot key beliefs
        for key in ["apple.edible", "stone.edible", "bread.edible"]:
            conf = self.beliefs.get_confidence(key, default=0.5)
            if conf != 0.5:
                self.temporal_mem.record_belief_snapshot(key, conf,
                                                          self._step_count)

        # ── Auto-detect uncertainty ──
        if reward != 0.0:
            self.uncertainty.record_outcome_variance(reward)
        if self._step_count % self._auto_detect_uncertainty_freq == 0:
            self.uncertainty.auto_detect_level()

        # ── 20: Strategy evaluation ──
        strategy_switched = False
        peer_agreement = 0.5
        switch_event = self.strategy_mgr.evaluate(
            self._step_count, peer_agreement=peer_agreement
        )
        if switch_event:
            strategy_switched = True
            if verbose:
                print(f"      [{self.agent_id}] 🔄 Strategy: "
                      f"{switch_event.from_strat.value} → "
                      f"{switch_event.to_strat.value} "
                      f"({switch_event.reason})")
        self.strategy_mgr.record_step(
            self._step_count, success, reward,
            novelty=v8_result.novelty,
        )

        # ── 21: Self-evaluation ──
        self_eval_grade = ""
        eval_report = self.self_eval.evaluate(self._step_count, self)
        if eval_report:
            self_eval_grade = eval_report.grade
            if verbose:
                print(f"      [{self.agent_id}] 📊 Self-eval: "
                      f"grade={eval_report.grade}, "
                      f"weak={eval_report.weaknesses}")

        # ── 22: Reflection ──
        reflection_items = 0

        def adjust(issue_type: str, magnitude: float) -> None:
            """Callback from meta_cog.reflect() to adjust beliefs."""
            if issue_type == "overconfidence":
                for b in list(self.beliefs.all_beliefs()):
                    b.confidence = self.uncertainty.apply_decay(
                        b.confidence, steps=3
                    )
            elif issue_type.startswith("bias:"):
                pred = issue_type.split(":", 1)[1]
                for b in list(self.beliefs.all_beliefs()):
                    if b.key.endswith(f".{pred}"):
                        b.confidence *= 0.85

        reflections = self.meta_cog.reflect(self._step_count, adjust_fn=adjust)
        reflection_items = len(reflections)
        if verbose and reflections:
            for r in reflections[:1]:
                print(f"      [{self.agent_id}] 🪞 Reflection: {r.finding[:60]}")

        s = V9Step(
            # V8 fields
            step=v8_result.step,
            input_text=v8_result.input_text,
            active_goal=v8_result.active_goal,
            action_taken=v8_result.action_taken,
            action_obj=v8_result.action_obj,
            env_success=v8_result.env_success,
            env_reward=v8_result.env_reward,
            novelty=v8_result.novelty,
            risk_decision=v8_result.risk_decision,
            risk_score=v8_result.risk_score,
            belief_updates=v8_result.belief_updates,
            conflicts_found=v8_result.conflicts_found,
            goals_generated=v8_result.goals_generated,
            experiment_run=v8_result.experiment_run,
            consistency_ok=v8_result.consistency_ok,
            elapsed_ms=v8_result.elapsed_ms,
            messages_received=v8_result.messages_received,
            messages_sent=v8_result.messages_sent,
            beliefs_from_peers=v8_result.beliefs_from_peers,
            social_learning=v8_result.social_learning,
            negotiations_run=v8_result.negotiations_run,
            coop_tasks_done=v8_result.coop_tasks_done,
            role_changed=v8_result.role_changed,
            vocab_size=v8_result.vocab_size,
            reputation_updates=v8_result.reputation_updates,
            shared_mem_reads=v8_result.shared_mem_reads,
            shared_mem_writes=v8_result.shared_mem_writes,
            experiences_shared=v8_result.experiences_shared,
            consistency_fixes=v8_result.consistency_fixes,
            global_consensus_applied=v8_result.global_consensus_applied,
            event_reactions=v8_result.event_reactions,
            coord_signal=v8_result.coord_signal,
            # V9 fields
            uncertainty_level=self.uncertainty.level.value,
            strategy=self.strategy_mgr.current.value,
            strategy_switched=strategy_switched,
            goal_conflicts=goal_conflicts,
            self_eval_grade=self_eval_grade,
            reflection_items=reflection_items,
            decay_applied=decay_count,
            noise_corruptions=noise_corruptions,
        )
        return s

    # ══════════════════════════════════════════════════════════════════
    # v9-specific helpers
    # ══════════════════════════════════════════════════════════════════

    def force_strategy(self, strategy: Strategy) -> None:
        """Force the agent into a specific strategy immediately."""
        self.strategy_mgr.evaluate(self._step_count, force=strategy)
        self.strategy_mgr.apply_to_agent(self)

    def set_uncertainty_level(self, level: UncertaintyLevel) -> None:
        self.uncertainty.level = level

    def get_reflection_log(self, n: int = 5) -> List[str]:
        return self.meta_cog.reflection_log(n)

    def is_stuck(self, action: str, target: str) -> bool:
        return self.temporal_mem.is_stuck(action, target, threshold=3)

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════

    def v9_summary(self) -> Dict:
        base = self.v8_summary()
        base.update({
            "meta_cognition":   self.meta_cog.summary(),
            "uncertainty":      self.uncertainty.summary(),
            "goal_conflicts":   self.goal_resolver.summary(),
            "strategy":         self.strategy_mgr.summary(),
            "temporal_memory":  self.temporal_mem.summary(),
            "self_evaluation":  self.self_eval.summary(),
        })
        return base

    def __repr__(self) -> str:
        strat = self.strategy_mgr.current.value
        unc   = self.uncertainty.level.value
        grade = self.self_eval.latest_report().grade if self.self_eval._reports else "?"
        return (f"AgentV9(id={self.agent_id}, role={self.role.value}, "
                f"strategy={strat}, uncertainty={unc}, "
                f"grade={grade})")
