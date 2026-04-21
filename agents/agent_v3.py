"""
agent/agent_v3.py
==================
CogniField v3 — Goal-Driven Planning Cognitive Agent

Full 10-step loop:
  1.  observe       → raw input (text / env observation)
  2.  encode        → latent vector
  3.  update memory → relational + vector memory
  4.  detect novelty → curiosity check
  5.  select goal   → goal system chooses current objective
  6.  plan          → planner generates action sequence using world model
  7.  execute action → environment step
  8.  receive feedback
  9.  update world model + relational memory + hypotheses
  10. repeat

New in v3:
  - WorldModel (transition + causal graph)
  - Planner (multi-step, depth-limited)
  - GoalSystem (persistent objectives)
  - RelationalMemory (typed concept graph)
  - AdvancedCuriosityEngine (hypothesis testing)
  - RichEnv (richer objects, partial observability)
  - Enhanced reasoning (world-logic errors)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..encoder.text_encoder    import TextEncoder
from ..encoder.image_encoder   import ImageEncoder
from ..latent_space.frequency_space import FrequencySpace, ComposeMode
from ..memory.memory_store     import MemoryStore
from ..memory.relational_memory import RelationalMemory
from ..reasoning.reasoning_engine import ReasoningEngine
from ..language.structure_checker import StructureChecker
from ..curiosity.advanced_curiosity import AdvancedCuriosityEngine
from ..loss.loss_system         import LossSystem, LossConfig
from ..world_model.transition_model import TransitionModel
from ..world_model.causal_graph import CausalGraph
from ..planning.planner import Planner, Plan
from ..agents.goals import GoalSystem, GoalType, Goal
from ..environment.rich_env import RichEnv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AgentV3Config:
    dim:                  int   = 64
    novelty_threshold:    float = 0.40
    reasoning_threshold:  float = 0.60
    max_retries:          int   = 4
    memory_size:          int   = 5_000
    decay_rate:           float = 0.003
    plan_depth:           int   = 4
    plan_beam:            int   = 3
    seed:                 int   = 42
    verbose:              bool  = True


# ---------------------------------------------------------------------------
# Step record
# ---------------------------------------------------------------------------

@dataclass
class V3Step:
    step:              int
    input_text:        str
    encoded_vec:       np.ndarray
    active_goal:       Optional[str]
    plan:              Optional[Plan]
    action_taken:      Optional[str]
    action_obj:        Optional[str]
    env_success:       Optional[bool]
    env_reward:        Optional[float]
    novelty:           float
    hypotheses_formed: int
    world_model_updated: bool
    elapsed_ms:        float


# ---------------------------------------------------------------------------
# CogniFieldAgentV3
# ---------------------------------------------------------------------------

class CogniFieldAgentV3:
    """
    Goal-driven, planning-capable cognitive agent (CogniField v3).

    Parameters
    ----------
    config : AgentV3Config
    env    : RichEnv instance
    """

    def __init__(
        self,
        config: Optional[AgentV3Config] = None,
        env:    Optional[RichEnv]       = None,
    ) -> None:
        self.cfg = config or AgentV3Config()
        cfg = self.cfg

        if cfg.verbose:
            print("  [v3] Initialising CogniField Agent v3...")

        # ── Shared foundations ──
        self.space  = FrequencySpace(dim=cfg.dim)
        self.enc    = TextEncoder(dim=cfg.dim, seed=cfg.seed)
        self.img_enc = ImageEncoder(dim=cfg.dim, seed=cfg.seed)
        self.enc.fit()

        # ── Memory (vector + relational) ──
        self.vec_memory = MemoryStore(
            dim=cfg.dim, max_size=cfg.memory_size,
            decay_rate=cfg.decay_rate, seed=cfg.seed
        )
        self.rel_memory = RelationalMemory(dim=cfg.dim, space=self.space)

        # ── World model ──
        self.world_model  = TransitionModel(space=self.space, dim=cfg.dim)
        self.causal_graph = CausalGraph()

        # ── Planning ──
        self.planner = Planner(
            transition_model=self.world_model,
            causal_graph=self.causal_graph,
            space=self.space,
            max_depth=cfg.plan_depth,
            beam_width=cfg.plan_beam,
            dim=cfg.dim,
        )

        # ── Goals ──
        self.goals = GoalSystem()

        # ── Curiosity (advanced) ──
        self.curiosity = AdvancedCuriosityEngine(
            space=self.space,
            rel_memory=self.rel_memory,
            vec_memory=self.vec_memory,
            novelty_threshold=cfg.novelty_threshold,
            dim=cfg.dim,
            seed=cfg.seed,
        )

        # ── Reasoning ──
        self.reasoning = ReasoningEngine(
            space=self.space,
            memory=self.vec_memory,
            max_retries=cfg.max_retries,
            threshold=cfg.reasoning_threshold,
            seed=cfg.seed,
        )

        # ── Language ──
        self.checker = StructureChecker()

        # ── Loss ──
        self.loss_sys = LossSystem(
            config=LossConfig(w_error=1.0, w_novelty=0.3),
            space=self.space,
        )

        # ── Environment ──
        self.env = env

        # ── State ──
        self._step_count    = 0
        self._step_log:     List[V3Step] = []
        self._current_plan: Optional[Plan] = None
        self._plan_step_idx: int = 0
        self._prev_state_vec: Optional[np.ndarray] = None

        if cfg.verbose:
            print("  [v3] Agent ready.\n")

    # ══════════════════════════════════════════════════════════════════
    # Main step loop
    # ══════════════════════════════════════════════════════════════════

    def step(
        self,
        text_input:    str = "",
        goal_override: Optional[str] = None,
        force_action:  Optional[Tuple[str, str]] = None,
        verbose:       Optional[bool] = None,
    ) -> V3Step:
        """
        Full 10-step cognitive cycle.

        Parameters
        ----------
        text_input    : Optional text input to encode and reason about.
        goal_override : Override the active goal with this label.
        force_action  : Force a specific (action, object) pair.
        verbose       : Override config verbose setting.
        """
        t0 = time.time()
        verbose = verbose if verbose is not None else self.cfg.verbose
        self._step_count += 1

        # ── Step 1-2: Observe + Encode ──
        input_vec = (self.enc.encode(text_input)
                     if text_input
                     else (self._prev_state_vec if self._prev_state_vec is not None
                           else self._zero_vec()))

        # ── Step 3: Update memory ──
        if text_input:
            self.vec_memory.store(
                input_vec, label=text_input[:30], modality="text",
                metadata={"step": self._step_count}
            )

        # ── Step 4: Novelty / Curiosity ──
        novelty = self.curiosity.detect_novelty(input_vec, text_input)
        n_hyp   = 0
        if novelty >= self.cfg.novelty_threshold and text_input:
            report = self.curiosity.explore(text_input, input_vec)
            n_hyp  = report.get("n_hypotheses", 0)
            if verbose:
                print(f"    ⚡ Novelty={novelty:.2f} | hypotheses: {n_hyp} | "
                      f"suggest: {report.get('suggested_action','?')}")

        # ── Step 5: Goal selection ──
        active_goal = self._select_goal(goal_override, input_vec)
        goal_label  = active_goal.label if active_goal else "explore"
        goal_vec    = (active_goal.goal_vec if active_goal and active_goal.goal_vec is not None
                       else input_vec)

        if verbose and active_goal:
            print(f"    🎯 Goal: '{goal_label}'")

        # ── Step 6: Plan ──
        env_available = (self.env.available_objects() if self.env else [])
        env_inventory = (self.env.inventory             if self.env else [])
        env_state_vec = (self.env.state_vector()         if self.env else input_vec)

        plan = self._make_plan(
            goal_label, goal_vec, env_state_vec, env_available, env_inventory
        )
        if verbose and plan and not plan.is_empty:
            print(f"    📋 Plan: {plan.describe()}")

        # ── Step 7: Execute ──
        action_taken = None
        action_obj   = None
        env_fb       = None
        wm_updated   = False

        if force_action:
            action_taken, action_obj = force_action
        elif plan and not plan.is_empty:
            # Execute next step from plan
            action_taken, action_obj = self._next_plan_action(plan)

        if action_taken and self.env:
            args = (action_obj,) if action_obj else ()
            env_fb = self.env.step(action_taken, *args)
            if verbose:
                status = "✓" if env_fb["success"] else "✗"
                print(f"    {status} {action_taken}({action_obj or ''}) → "
                      f"{env_fb['message'][:60]}  r={env_fb.get('reward',0):+.2f}")

        # ── Step 8-9: Update world model + memory ──
        if env_fb:
            wm_updated = self._update_world_model(env_fb, env_state_vec, action_taken)
            self._update_relational_memory(env_fb)
            self._update_hypotheses(env_fb)

            # Check if goal satisfied
            if active_goal:
                if self.goals.check_goal_satisfied(active_goal, env_fb):
                    self.goals.mark_completed(active_goal)
                    if verbose:
                        print(f"    ✅ Goal COMPLETED: '{goal_label}'")
                elif env_fb.get("reward", 0) < -0.3:
                    self.goals.mark_failed(active_goal)

        # ── Loss update ──
        env_reward = env_fb.get("reward", 0) if env_fb else 0
        next_state = (self.env.state_vector() if self.env else input_vec)
        self._prev_state_vec = next_state

        elapsed = (time.time() - t0) * 1000
        record = V3Step(
            step=self._step_count,
            input_text=text_input,
            encoded_vec=input_vec,
            active_goal=goal_label,
            plan=plan,
            action_taken=action_taken,
            action_obj=action_obj,
            env_success=env_fb.get("success") if env_fb else None,
            env_reward=env_fb.get("reward")  if env_fb else None,
            novelty=novelty,
            hypotheses_formed=n_hyp,
            world_model_updated=wm_updated,
            elapsed_ms=elapsed,
        )
        self._step_log.append(record)
        return record

    # ══════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════

    def _zero_vec(self) -> np.ndarray:
        return np.zeros(self.cfg.dim, dtype=np.float32)

    def _select_goal(
        self,
        override:   Optional[str],
        input_vec:  np.ndarray,
    ) -> Optional[Goal]:
        """Select the highest-priority active goal."""
        if override:
            # Create temporary goal
            g_vec = self.enc.encode(override)
            g = self.goals.add_goal(
                override, GoalType.CUSTOM,
                target=override, priority=0.9,
                goal_vec=g_vec,
            )
            return g
        return self.goals.select_active_goal()

    def _make_plan(
        self,
        goal_label:    str,
        goal_vec:      np.ndarray,
        state_vec:     np.ndarray,
        available:     List[Tuple[str, str]],
        inventory:     List[str],
    ) -> Optional[Plan]:
        """Generate a plan for the current goal."""
        if not goal_label or goal_label == "explore":
            # No specific goal → observe
            from ..planning.planner import Plan, PlanStep
            dummy_step = self.planner._make_step("observe", "", state_vec, goal_vec)
            return Plan("explore", [dummy_step], dummy_step.score, 1)

        return self.planner.plan(
            goal_label=goal_label,
            goal_vec=goal_vec,
            current_state_vec=state_vec,
            available_objects=available,
            inventory=inventory,
        )

    def _next_plan_action(self, plan: Plan) -> Tuple[str, str]:
        """Get the next action from the current or new plan."""
        if self._current_plan is not plan:
            self._current_plan  = plan
            self._plan_step_idx = 0

        if self._plan_step_idx < len(plan.steps):
            step = plan.steps[self._plan_step_idx]
            self._plan_step_idx += 1
            return step.action, step.object_name
        return "observe", ""

    def _update_world_model(
        self,
        env_fb:     Dict,
        prev_state: np.ndarray,
        action:     str,
    ) -> bool:
        """Record transition in world model from environment feedback."""
        next_state   = env_fb.get("state_vec", prev_state)
        reward       = env_fb.get("reward", 0.0)
        success      = env_fb.get("success", False)
        obj_name     = env_fb.get("object_name", "")
        obj_category = env_fb.get("object_category", "unknown")

        self.world_model.record(
            state_vec=prev_state,
            action=action or "unknown",
            next_state_vec=next_state,
            reward=reward,
            success=success,
            object_name=obj_name,
            object_category=obj_category,
        )

        # Ingest into causal graph
        obj_props = env_fb.get("object_props", {})
        self.causal_graph.ingest_feedback(
            action=action or "unknown",
            object_name=obj_name,
            object_props={**obj_props, "category": obj_category},
            success=success,
            reward=reward,
        )
        return True

    def _update_relational_memory(self, env_fb: Dict) -> None:
        """Extract and store relational facts from environment feedback."""
        obj_name  = env_fb.get("object_name", "")
        obj_props = env_fb.get("object_props", {})
        action    = env_fb.get("action", "")
        success   = env_fb.get("success", False)
        reward    = env_fb.get("reward", 0.0)

        if obj_name and obj_props:
            self.rel_memory.add_object_properties(obj_name, obj_props)

        # Specific learning events
        learned = env_fb.get("learned", "")
        if learned:
            if "edible" in learned and obj_name:
                is_edible = "not" not in learned.lower()
                self.rel_memory.add_fact(obj_name, "edible", is_edible, confidence=1.0)

        if obj_name and action:
            self.rel_memory.ingest_env_feedback(
                action, obj_name, obj_props, success, reward
            )

    def _update_hypotheses(self, env_fb: Dict) -> None:
        """Update open hypotheses based on observed feedback."""
        obj_name  = env_fb.get("object_name", "")
        obj_props = env_fb.get("object_props", {})
        if not obj_name:
            return
        for prop, val in obj_props.items():
            self.curiosity.update_hypotheses(obj_name, prop, val)

    # ══════════════════════════════════════════════════════════════════
    # High-level task interface
    # ══════════════════════════════════════════════════════════════════

    def teach(self, text: str, label: str = "", props: Optional[Dict] = None) -> None:
        """
        Teach the agent a fact about the world via text.
        Encodes, stores in memory, optionally adds relational facts.
        """
        vec = self.enc.encode(text)
        l   = label or text[:30]
        self.vec_memory.store(vec, label=l, modality="text",
                              allow_duplicate=True)
        self.rel_memory.store_concept_vector(l, vec)
        if props:
            self.rel_memory.add_object_properties(l, props, vector=vec)

    def add_goal(
        self,
        label:    str,
        gtype:    GoalType = GoalType.CUSTOM,
        target:   str = "",
        priority: float = 0.7,
    ) -> Goal:
        """Add a goal to the agent's goal system."""
        goal_vec = self.enc.encode(label)
        return self.goals.add_goal(
            label=label, goal_type=gtype,
            target=target or label,
            priority=priority,
            goal_vec=goal_vec,
        )

    def run_episode(
        self,
        n_steps:       int = 20,
        goal_label:    Optional[str] = None,
        verbose:       bool = True,
    ) -> List[V3Step]:
        """
        Run a full episode of n_steps.
        Uses active goals or explores if none set.
        """
        log = []
        for i in range(n_steps):
            # Auto-generate goals from world knowledge if none active
            if self.goals.active_count == 0 and self.env:
                edible = self.rel_memory.find_edible()
                unknown = [o.name for o in self.env.visible_objects()
                           if o.category == "unknown"]
                self.goals.infer_goals_from_context(edible, unknown,
                                                     self.env.inventory)

            s = self.step(
                text_input=goal_label or "",
                goal_override=goal_label,
                verbose=verbose,
            )
            log.append(s)

            # Stop if goal completed
            if goal_label and self.goals.completed_count > 0:
                if any(g.label == goal_label for g in self.goals._completed):
                    break
        return log

    # ══════════════════════════════════════════════════════════════════
    # Recall & queries
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

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════

    def summary(self) -> Dict:
        return {
            "steps":             self._step_count,
            "vector_memory":     len(self.vec_memory),
            "relational_facts":  self.rel_memory.n_facts(),
            "world_model_rules": self.world_model.n_rules,
            "world_transitions": self.world_model.n_transitions,
            "goals":             self.goals.summary(),
            "curiosity":         self.curiosity.summary(),
            "env":               self.env.stats() if self.env else None,
        }

    def __repr__(self) -> str:
        return (f"CogniFieldAgentV3(steps={self._step_count}, "
                f"memory={len(self.vec_memory)}, "
                f"rules={self.world_model.n_rules}, "
                f"goals_done={self.goals.completed_count})")
