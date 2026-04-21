"""
agent/agent_v4.py
==================
CogniField v4 — Autonomous Self-Improving Cognitive Agent

The 14-step continuous learning loop:

  1.  Observe environment
  2.  Encode to latent space
  3.  Update relational + vector memory
  4.  Detect novelty (curiosity)
  5.  Generate internal goals (self-directed)
  6.  Select active goal
  7.  Plan hierarchically (with subgoal decomposition)
  8.  Simulate plan (imagination before action)
  9.  Execute best action
  10. Receive environment feedback
  11. Update world model + causal graph
  12. Run abstraction (extract general rules)
  13. Analyse performance (meta-learning)
  14. Adapt strategies + internal state
  → Consolidate memory periodically

New in v4 over v3:
  - Self-generated goals (GoalGenerator)
  - InternalState (confidence, curiosity, fatigue, frustration)
  - WorldSimulator (imagination before acting)
  - HierarchicalPlanner (subgoal decomposition)
  - MemoryConsolidator (periodic memory compression)
  - AbstractionEngine (rule induction from experience)
  - MetaLearner (performance tracking + strategy adaptation)
  - Continuous autonomous loop (run_autonomous)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..encoder.text_encoder       import TextEncoder
from ..encoder.image_encoder      import ImageEncoder
from ..latent_space.frequency_space import FrequencySpace, ComposeMode
from ..memory.memory_store        import MemoryStore
from ..memory.relational_memory   import RelationalMemory
from ..memory.consolidation       import MemoryConsolidator
from ..reasoning.reasoning_engine import ReasoningEngine
from ..reasoning.abstraction      import AbstractionEngine
from ..reasoning.meta_learning    import MetaLearner
from ..curiosity.advanced_curiosity import AdvancedCuriosityEngine
from ..loss.loss_system           import LossSystem, LossConfig
from ..world_model.transition_model import TransitionModel
from ..world_model.causal_graph   import CausalGraph
from ..world_model.simulator      import WorldSimulator
from ..planning.planner           import Planner
from ..planning.hierarchical_planner import HierarchicalPlanner
from ..agents.goals                import GoalSystem, GoalType, Goal
from ..agents.goal_generator       import GoalGenerator
from ..agents.internal_state       import InternalState
from ..environment.rich_env       import RichEnv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AgentV4Config:
    dim:                    int   = 64
    novelty_threshold:      float = 0.40
    reasoning_threshold:    float = 0.60
    max_retries:            int   = 4
    memory_size:            int   = 8_000
    decay_rate:             float = 0.002
    plan_depth:             int   = 3
    plan_beam:              int   = 3
    consolidation_interval: int   = 20   # steps between consolidations
    abstraction_interval:   int   = 15   # steps between abstraction runs
    meta_analysis_interval: int   = 10   # steps between meta-learning analyses
    max_active_goals:       int   = 5
    seed:                   int   = 42
    verbose:                bool  = False


# ---------------------------------------------------------------------------
# Step record
# ---------------------------------------------------------------------------

@dataclass
class V4Step:
    step:               int
    input_text:         str
    active_goal:        Optional[str]
    plan_type:          str            # "hierarchical" | "flat" | "none"
    plan_depth:         int
    sim_score:          float          # simulation pre-screening score
    action_taken:       Optional[str]
    action_obj:         Optional[str]
    env_success:        Optional[bool]
    env_reward:         Optional[float]
    novelty:            float
    goals_generated:    int
    abstraction_ran:    bool
    meta_analysis_ran:  bool
    consolidated:       bool
    internal_state:     Dict[str, str]  # snapshot of InternalState
    elapsed_ms:         float


# ---------------------------------------------------------------------------
# CogniFieldAgentV4
# ---------------------------------------------------------------------------

class CogniFieldAgentV4:
    """
    Autonomous self-improving cognitive agent (CogniField v4).

    Parameters
    ----------
    config : AgentV4Config
    env    : RichEnv instance
    """

    def __init__(
        self,
        config: Optional[AgentV4Config] = None,
        env:    Optional[RichEnv]       = None,
    ) -> None:
        self.cfg = config or AgentV4Config()
        cfg = self.cfg

        if cfg.verbose:
            print("  [v4] Initialising CogniField Agent v4...")

        # ── Core foundations ──
        self.space  = FrequencySpace(dim=cfg.dim)
        self.enc    = TextEncoder(dim=cfg.dim, seed=cfg.seed)
        self.enc.fit()

        # ── Memory ──
        self.vec_memory = MemoryStore(
            dim=cfg.dim, max_size=cfg.memory_size,
            decay_rate=cfg.decay_rate, seed=cfg.seed
        )
        self.rel_memory = RelationalMemory(dim=cfg.dim, space=self.space)

        # ── World model + simulator ──
        self.world_model  = TransitionModel(space=self.space, dim=cfg.dim)
        self.causal_graph = CausalGraph()
        self.simulator    = WorldSimulator(
            self.world_model, self.causal_graph, self.space, dim=cfg.dim
        )

        # ── Planning (flat + hierarchical) ──
        self.flat_planner = Planner(
            self.world_model, self.causal_graph, self.space,
            max_depth=cfg.plan_depth, beam_width=cfg.plan_beam, dim=cfg.dim
        )
        self.h_planner = HierarchicalPlanner(
            self.flat_planner, self.simulator, self.space,
            max_depth=cfg.plan_depth, dim=cfg.dim
        )

        # ── Goals ──
        self.goal_system = GoalSystem(max_active=cfg.max_active_goals)

        # ── Internal state ──
        self.internal_state = InternalState()

        # ── Curiosity (advanced) ──
        self.curiosity = AdvancedCuriosityEngine(
            space=self.space,
            rel_memory=self.rel_memory,
            vec_memory=self.vec_memory,
            novelty_threshold=cfg.novelty_threshold,
            dim=cfg.dim, seed=cfg.seed,
        )

        # ── Reasoning ──
        self.reasoning = ReasoningEngine(
            space=self.space, memory=self.vec_memory,
            max_retries=cfg.max_retries,
            threshold=cfg.reasoning_threshold, seed=cfg.seed,
        )

        # ── Abstraction ──
        self.abstraction = AbstractionEngine(
            rel_memory=self.rel_memory,
            world_model=self.world_model,
            causal_graph=self.causal_graph,
            space=self.space,
        )

        # ── Meta-learning ──
        self.meta_learner = MetaLearner(history_window=100, adapt_rate=0.1)

        # ── Memory consolidator ──
        self.consolidator = MemoryConsolidator(
            vec_memory=self.vec_memory,
            rel_memory=self.rel_memory,
            space=self.space,
        )

        # ── Goal generator ──
        self.goal_gen = GoalGenerator(
            goal_system=self.goal_system,
            rel_memory=self.rel_memory,
            vec_memory=self.vec_memory,
            curiosity=self.curiosity,
            world_model=self.world_model,
            space=self.space,
            enc_fn=self.enc.encode,
            max_active_goals=cfg.max_active_goals,
            dim=cfg.dim,
        )

        # ── Loss ──
        self.loss_sys = LossSystem(
            config=LossConfig(w_error=1.0, w_novelty=0.3),
            space=self.space,
        )

        # ── Environment ──
        self.env = env

        # ── State ──
        self._step_count     = 0
        self._step_log:      List[V4Step] = []
        self._current_plan:  Optional[Any] = None
        self._plan_step_idx: int = 0
        self._prev_state_vec: Optional[np.ndarray] = None
        self._last_env_obs:  Optional[Dict] = None

        if cfg.verbose:
            print("  [v4] Agent ready.\n")

    # ══════════════════════════════════════════════════════════════════
    # Main 14-step cycle
    # ══════════════════════════════════════════════════════════════════

    def step(
        self,
        text_input:    str = "",
        force_action:  Optional[Tuple[str, str]] = None,
        verbose:       Optional[bool] = None,
    ) -> V4Step:
        """One full 14-step autonomous cognitive cycle."""
        t0 = time.time()
        verbose = verbose if verbose is not None else self.cfg.verbose
        self.internal_state.tick()
        self._step_count += 1
        S = self._step_count

        # ── 1-2: Observe + Encode ──
        input_vec = (self.enc.encode(text_input)
                     if text_input
                     else (self._prev_state_vec if self._prev_state_vec is not None else self._zero_vec()))

        # ── 3: Update memory ──
        if text_input:
            self.vec_memory.store(
                input_vec, label=text_input[:30], modality="text",
                metadata={"step": S}
            )

        # ── 4: Novelty detection ──
        eff_threshold = self.internal_state.effective_novelty_threshold(
            self.cfg.novelty_threshold
        )
        novelty = self.curiosity.detect_novelty(input_vec, text_input)
        if novelty >= eff_threshold and text_input:
            self.curiosity.explore(text_input, input_vec)
            self.internal_state.on_novel_input(novelty)

        # ── 5: Goal generation (self-directed) ──
        new_goals = 0
        env_obs = self._last_env_obs or {}
        perf    = self.meta_learner.performance_metrics()

        if S % 3 == 0 or self.goal_system.active_count == 0:
            generated = self.goal_gen.generate(
                internal_state=self.internal_state,
                env_observation=env_obs,
                performance_metrics=perf,
                max_new_goals=2,
            )
            new_goals = len(generated)
            if verbose and generated:
                for g in generated:
                    print(f"    🌱 Goal generated: '{g.label}' "
                          f"[{g.goal_type.value}] p={g.priority:.2f} "
                          f"src={g.metadata.get('source','?')}")

        # ── 6: Select goal ──
        active_goal = self.goal_system.select_active_goal()
        goal_label  = active_goal.label if active_goal else "explore"
        goal_vec    = (active_goal.goal_vec
                       if active_goal and active_goal.goal_vec is not None
                       else input_vec)

        if verbose and active_goal:
            print(f"    🎯 Goal: '{goal_label}'")

        # ── 7: Hierarchical planning ──
        plan_type = "none"
        plan_depth = 0
        h_plan = None
        env_available = (self.env.available_objects() if self.env else [])
        env_inventory = (self.env.inventory             if self.env else [])
        state_vec     = (self.env.state_vector()         if self.env else input_vec)

        if goal_label != "explore" and len(env_available) > 0:
            h_plan = self.h_planner.plan_hierarchical(
                goal_label, goal_vec, state_vec,
                env_available, env_inventory
            )
            plan_type  = "hierarchical"
            plan_depth = h_plan.depth

        elif goal_label:
            flat_plan = self.flat_planner.plan(
                goal_label, goal_vec, state_vec,
                env_available, env_inventory
            )
            h_plan     = flat_plan   # use flat plan as fallback
            plan_type  = "flat"
            plan_depth = flat_plan.depth if hasattr(flat_plan, 'depth') else 1

        # ── 8: Simulation pre-screening ──
        sim_score = 0.5
        action_to_take = None
        action_obj     = None

        if h_plan and hasattr(h_plan, 'flat_actions') and h_plan.flat_actions:
            sim_result = self.simulator.simulate(
                state_vec, h_plan.flat_actions[:3], goal_vec
            )
            sim_score = self.simulator._score_simulation(sim_result, goal_vec)
            if verbose:
                print(f"    🔮 Sim score: {sim_score:.3f} "
                      f"(reward={sim_result.total_reward:+.2f})")

            if sim_score > 0.3:
                if h_plan.flat_actions:
                    action_to_take, action_obj = h_plan.flat_actions[0]
        elif h_plan and hasattr(h_plan, 'action_sequence'):
            # Flat Plan fallback
            seq = h_plan.action_sequence
            if seq:
                action_to_take, action_obj = seq[0]
                sim_score = h_plan.total_score if hasattr(h_plan, 'total_score') else 0.5

        # Force override
        if force_action:
            action_to_take, action_obj = force_action

        # ── 9: Execute ──
        env_fb  = None
        if action_to_take and self.env:
            args   = (action_obj,) if action_obj else ()
            env_fb = self.env.step(action_to_take, *args)
            if verbose:
                status = "✓" if env_fb["success"] else "✗"
                print(f"    {status} {action_to_take}({action_obj or ''}) → "
                      f"{env_fb['message'][:55]}  r={env_fb.get('reward',0):+.2f}")
            self._last_env_obs = env_fb

        # ── 10-11: Feedback + World model update ──
        wm_updated = False
        if env_fb and action_to_take:
            wm_updated = self._update_world_model(env_fb, state_vec, action_to_take)
            self._update_relational_memory(env_fb)
            self._update_hypotheses(env_fb)

            # Update internal state
            if env_fb.get("success"):
                self.internal_state.on_success(abs(env_fb.get("reward", 0.3)))
            else:
                self.internal_state.on_failure(abs(env_fb.get("reward", 0.1)))

            # Check goal completion
            if active_goal:
                if self.goal_system.check_goal_satisfied(active_goal, env_fb):
                    self.goal_system.mark_completed(active_goal)
                    self.internal_state.on_goal_completed()
                    if h_plan and hasattr(h_plan, 'flat_actions'):
                        self.h_planner.record_success(
                            goal_label, h_plan.flat_actions
                        )
                    if verbose:
                        print(f"    ✅ Goal COMPLETED: '{goal_label}'")
                elif env_fb.get("reward", 0) < -0.3:
                    self.goal_system.mark_failed(active_goal)

        # ── 12: Abstraction (periodic) ──
        abstracted = False
        if S % self.cfg.abstraction_interval == 0:
            new_rules = self.abstraction.run(verbose=verbose)
            abstracted = bool(new_rules)

        # ── 13: Meta-learning analysis (periodic) ──
        meta_ran = False
        if S % self.cfg.meta_analysis_interval == 0:
            self.meta_learner.record(
                step=S,
                action=action_to_take or "none",
                success=env_fb.get("success", False) if env_fb else False,
                reward=env_fb.get("reward", 0) if env_fb else 0,
                goal_type=active_goal.goal_type.value if active_goal else "",
                plan_depth=plan_depth,
                novelty=novelty,
                confidence=self.internal_state.confidence,
            )
            analysis = self.meta_learner.analyse()
            meta_ran = True
            if verbose and analysis.get("insights"):
                for insight in analysis["insights"][:2]:
                    print(f"    📊 Meta: {insight}")

        elif action_to_take and env_fb:
            # Always record even when not analysing
            self.meta_learner.record(
                step=S,
                action=action_to_take,
                success=env_fb.get("success", False),
                reward=env_fb.get("reward", 0),
                goal_type=active_goal.goal_type.value if active_goal else "",
                plan_depth=plan_depth,
                novelty=novelty,
                confidence=self.internal_state.confidence,
            )

        # ── 14: Memory consolidation (periodic) ──
        consolidated = False
        if S % self.cfg.consolidation_interval == 0:
            if self.internal_state.should_consolidate():
                self.consolidator.consolidate(verbose=verbose)
                self.internal_state.on_consolidation()
                consolidated = True

        # Update previous state
        self._prev_state_vec = (self.env.state_vector() if self.env else input_vec)

        elapsed = (time.time() - t0) * 1000
        record = V4Step(
            step=S,
            input_text=text_input,
            active_goal=goal_label,
            plan_type=plan_type,
            plan_depth=plan_depth,
            sim_score=sim_score,
            action_taken=action_to_take,
            action_obj=action_obj or "",
            env_success=env_fb.get("success") if env_fb else None,
            env_reward=env_fb.get("reward")   if env_fb else None,
            novelty=novelty,
            goals_generated=new_goals,
            abstraction_ran=abstracted,
            meta_analysis_ran=meta_ran,
            consolidated=consolidated,
            internal_state=self.internal_state.summary(),
            elapsed_ms=elapsed,
        )
        self._step_log.append(record)
        return record

    # ══════════════════════════════════════════════════════════════════
    # Continuous autonomous loop
    # ══════════════════════════════════════════════════════════════════

    def run_autonomous(
        self,
        n_steps:          int = 50,
        callback:         Optional[Callable[[V4Step], None]] = None,
        stop_on_goal:     Optional[str] = None,
        verbose:          bool = True,
    ) -> List[V4Step]:
        """
        Run the agent autonomously for n_steps.

        Parameters
        ----------
        n_steps      : Number of steps to run.
        callback     : Optional function called after each step.
        stop_on_goal : Stop early if this goal label is completed.
        verbose      : Print step summaries.
        """
        log = []
        if verbose:
            print(f"\n  {'Step':4s} | {'Goal':28s} | {'Action':12s} | "
                  f"{'Obj':12s} | {'Reward':6s} | {'Nov':5s} | {'New G':5s} | Info")
            print(f"  {'─'*4} | {'─'*28} | {'─'*12} | {'─'*12} | "
                  f"{'─'*6} | {'─'*5} | {'─'*5} | {'─'*20}")

        for _ in range(n_steps):
            s = self.step(verbose=False)
            log.append(s)

            if verbose:
                rew_str = f"{s.env_reward:+.2f}" if s.env_reward is not None else "  N/A"
                info_parts = []
                if s.abstraction_ran:   info_parts.append("abs")
                if s.meta_analysis_ran: info_parts.append("meta")
                if s.consolidated:      info_parts.append("cons")
                if s.novelty >= 0.35:   info_parts.append("⚡")
                info = ",".join(info_parts)
                print(f"  {s.step:4d} | {(s.active_goal or '–')[:28]:28s} | "
                      f"{(s.action_taken or '–')[:12]:12s} | "
                      f"{(s.action_obj or '–')[:12]:12s} | "
                      f"{rew_str:6s} | {s.novelty:.3f} | {s.goals_generated:5d} | {info}")

            if callback:
                callback(s)

            # Early stop
            if stop_on_goal:
                completed = [g.label for g in self.goal_system._completed]
                if any(stop_on_goal in l for l in completed):
                    if verbose:
                        print(f"\n  ✅ Goal '{stop_on_goal}' achieved at step {s.step}!")
                    break

        return log

    # ══════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════

    def _zero_vec(self) -> np.ndarray:
        return np.zeros(self.cfg.dim, dtype=np.float32)

    def _update_world_model(
        self, env_fb: Dict, prev_state: np.ndarray, action: str
    ) -> bool:
        next_state   = env_fb.get("state_vec", prev_state)
        reward       = env_fb.get("reward", 0.0)
        success      = env_fb.get("success", False)
        obj_name     = env_fb.get("object_name", "")
        obj_category = env_fb.get("object_category", "unknown")

        self.world_model.record(
            prev_state, action, next_state,
            reward, success, obj_name, obj_category
        )
        obj_props = env_fb.get("object_props", {})
        self.causal_graph.ingest_feedback(
            action, obj_name,
            {**obj_props, "category": obj_category},
            success, reward
        )
        return True

    def _update_relational_memory(self, env_fb: Dict) -> None:
        obj_name  = env_fb.get("object_name", "")
        obj_props = env_fb.get("object_props", {})
        action    = env_fb.get("action", "")
        success   = env_fb.get("success", False)
        reward    = env_fb.get("reward", 0.0)

        if obj_name and obj_props:
            self.rel_memory.add_object_properties(obj_name, obj_props)
        learned = env_fb.get("learned", "")
        if learned and obj_name:
            is_edible = "not" not in learned.lower()
            self.rel_memory.add_fact(obj_name, "edible", is_edible, 1.0)
        if obj_name and action:
            self.rel_memory.ingest_env_feedback(
                action, obj_name, obj_props, success, reward
            )

    def _update_hypotheses(self, env_fb: Dict) -> None:
        obj_name  = env_fb.get("object_name", "")
        obj_props = env_fb.get("object_props", {})
        if not obj_name:
            return
        for prop, val in obj_props.items():
            self.curiosity.update_hypotheses(obj_name, prop, val)

    # ══════════════════════════════════════════════════════════════════
    # Teaching interface
    # ══════════════════════════════════════════════════════════════════

    def teach(self, text: str, label: str = "",
              props: Optional[Dict] = None) -> None:
        vec = self.enc.encode(text)
        l   = label or text[:30]
        self.vec_memory.store(vec, label=l, modality="text", allow_duplicate=True)
        self.rel_memory.store_concept_vector(l, vec)
        if props:
            self.rel_memory.add_object_properties(l, props, vector=vec)
            for prop, val in props.items():
                self.causal_graph.add_property(l, prop, val)
            if "category" in props:
                self.causal_graph.add_is_a(l, props["category"])

    def add_goal(self, label: str, gtype: GoalType = GoalType.CUSTOM,
                 target: str = "", priority: float = 0.7) -> Goal:
        goal_vec = self.enc.encode(label)
        return self.goal_system.add_goal(
            label=label, goal_type=gtype,
            target=target or label,
            priority=priority, goal_vec=goal_vec,
        )

    # ══════════════════════════════════════════════════════════════════
    # Queries
    # ══════════════════════════════════════════════════════════════════

    def what_is(self, concept: str) -> str:
        return self.rel_memory.what_is(concept)

    def what_can_i_eat(self) -> List[str]:
        return self.rel_memory.find_edible()

    def what_is_dangerous(self) -> List[str]:
        return self.rel_memory.find_dangerous()

    def recall(self, query: str, k: int = 5) -> List[Tuple[float, str]]:
        vec = self.enc.encode(query)
        results = self.vec_memory.retrieve(vec, k=k)
        return [(sim, e.label) for sim, e in results]

    def simulate_action(self, action: str, obj: str = "") -> Dict:
        state = (self.env.state_vector() if self.env
                 else self._prev_state_vec or self._zero_vec())
        return self.simulator.test_hypothesis(action, obj, state)

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════

    def summary(self) -> Dict:
        perf = self.meta_learner.performance_metrics()
        return {
            "steps":                self._step_count,
            "vector_memory":        len(self.vec_memory),
            "relational_facts":     self.rel_memory.n_facts(),
            "world_model_rules":    self.world_model.n_rules,
            "world_transitions":    self.world_model.n_transitions,
            "abstract_rules":       len(self.abstraction._rules),
            "goals_completed":      self.goal_system.completed_count,
            "goals_active":         self.goal_system.active_count,
            "goals_generated":      self.goal_gen._generated_count,
            "curiosity_explorations": self.curiosity.n_explorations,
            "consolidation_cycles": self.consolidator.cycle_count,
            "meta_cycles":          self.meta_learner._cycle,
            "success_rate":         round(perf.get("overall_success_rate", 0), 3),
            "internal_state":       self.internal_state.as_dict(),
            "env":                  self.env.stats() if self.env else None,
        }

    def __repr__(self) -> str:
        return (f"CogniFieldAgentV4(steps={self._step_count}, "
                f"mem={len(self.vec_memory)}, "
                f"rules={self.world_model.n_rules}, "
                f"goals_done={self.goal_system.completed_count})")
