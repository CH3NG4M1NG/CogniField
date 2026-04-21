"""
agent/agent_v5.py
==================
CogniField v5 — Stable, Belief-Driven Autonomous Agent

The 18-step continuous learning loop:

  1.  Observe environment
  2.  Encode to latent space
  3.  Update episodic + vector memory
  4.  Validate existing knowledge (periodic)
  5.  Detect novelty
  6.  Assess risk of novelty / unknown
  7.  Generate internal goals (self-directed)
  8.  Select highest-priority goal
  9.  Plan hierarchically
  10. Simulate outcomes (imagination)
  11. Risk evaluation of plan steps
  12. Execute safest action (or experiment)
  13. Receive feedback
  14. Update belief system (Bayesian, multi-evidence)
  15. Resolve conflicts
  16. Check consistency + propagate
  17. Consolidate memory (periodic)
  18. Update metrics + adapt strategies
  → Repeat

Key differences from v4:
  - Every belief update goes through Bayesian aggregation
  - Risk engine gates all non-trivial actions
  - Consistency engine checks before committing beliefs
  - Knowledge validator periodically re-tests beliefs
  - Experiment engine designs structured safe tests
  - Episodic memory tracks experiences with importance
  - Procedural memory stores successful action patterns
  - Stability metrics track reliability over time
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..encoder.text_encoder         import TextEncoder
from ..latent_space.frequency_space import FrequencySpace, ComposeMode
from ..memory.memory_store          import MemoryStore
from ..memory.relational_memory     import RelationalMemory
from ..memory.consolidation         import MemoryConsolidator
from ..memory.episodic_memory       import EpisodicMemoryStore, ProceduralMemoryStore
from ..reasoning.reasoning_engine   import ReasoningEngine
from ..reasoning.abstraction        import AbstractionEngine
from ..reasoning.meta_learning      import MetaLearner
from ..reasoning.conflict_resolver  import ConflictResolver
from ..reasoning.consistency_engine import ConsistencyEngine
from ..reasoning.validation         import KnowledgeValidator
from ..curiosity.advanced_curiosity import AdvancedCuriosityEngine
from ..curiosity.experiment_engine  import ExperimentEngine
from ..world_model.belief_system    import BeliefSystem
from ..world_model.transition_model import TransitionModel
from ..world_model.causal_graph     import CausalGraph
from ..world_model.simulator        import WorldSimulator
from ..planning.planner             import Planner
from ..planning.hierarchical_planner import HierarchicalPlanner
from ..agents.goals                  import GoalSystem, GoalType, Goal
from ..agents.goal_generator         import GoalGenerator
from ..agents.internal_state         import InternalState
from ..agents.risk_engine            import RiskEngine
from ..evaluation.metrics           import AgentMetrics
from ..environment.rich_env         import RichEnv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AgentV5Config:
    dim:                    int   = 64
    novelty_threshold:      float = 0.40
    reasoning_threshold:    float = 0.60
    max_retries:            int   = 4
    memory_size:            int   = 8_000
    decay_rate:             float = 0.002
    plan_depth:             int   = 3
    plan_beam:              int   = 3
    risk_tolerance:         float = 0.35
    consolidation_interval: int   = 20
    abstraction_interval:   int   = 15
    meta_analysis_interval: int   = 10
    validation_interval:    int   = 12
    max_active_goals:       int   = 5
    seed:                   int   = 42
    verbose:                bool  = False


# ---------------------------------------------------------------------------
# Step record
# ---------------------------------------------------------------------------

@dataclass
class V5Step:
    step:             int
    input_text:       str
    active_goal:      Optional[str]
    action_taken:     Optional[str]
    action_obj:       str
    env_success:      Optional[bool]
    env_reward:       Optional[float]
    novelty:          float
    risk_decision:    str          # "proceed" | "caution" | "block" | "experiment"
    risk_score:       float
    belief_updates:   int
    conflicts_found:  int
    goals_generated:  int
    experiment_run:   bool
    consistency_ok:   bool
    elapsed_ms:       float


# ---------------------------------------------------------------------------
# CogniFieldAgentV5
# ---------------------------------------------------------------------------

class CogniFieldAgentV5:
    """
    Stable, belief-driven autonomous agent (CogniField v5).

    Parameters
    ----------
    config : AgentV5Config
    env    : RichEnv instance
    """

    def __init__(
        self,
        config: Optional[AgentV5Config] = None,
        env:    Optional[RichEnv]       = None,
    ) -> None:
        self.cfg = config or AgentV5Config()
        cfg = self.cfg

        if cfg.verbose:
            print("  [v5] Initialising CogniField Agent v5...")

        # ── Core ──
        self.space  = FrequencySpace(dim=cfg.dim)
        self.enc    = TextEncoder(dim=cfg.dim, seed=cfg.seed)
        self.enc.fit()

        # ── Memory (tri-store) ──
        self.vec_memory      = MemoryStore(dim=cfg.dim, max_size=cfg.memory_size,
                                           decay_rate=cfg.decay_rate, seed=cfg.seed)
        self.rel_memory      = RelationalMemory(dim=cfg.dim, space=self.space)
        self.episodic_memory = EpisodicMemoryStore(max_episodes=2000)
        self.procedural_memory = ProceduralMemoryStore(max_procedures=200)

        # ── Belief system (v5 core) ──
        self.beliefs = BeliefSystem(decay_rate=0.001)

        # ── World model ──
        self.world_model  = TransitionModel(space=self.space, dim=cfg.dim)
        self.causal_graph = CausalGraph()
        self.simulator    = WorldSimulator(self.world_model, self.causal_graph,
                                           self.space, dim=cfg.dim)

        # ── Planning ──
        self.flat_planner = Planner(self.world_model, self.causal_graph, self.space,
                                    max_depth=cfg.plan_depth, beam_width=cfg.plan_beam,
                                    dim=cfg.dim)
        self.h_planner = HierarchicalPlanner(self.flat_planner, self.simulator,
                                             self.space, max_depth=cfg.plan_depth,
                                             dim=cfg.dim)

        # ── Goals ──
        self.goal_system = GoalSystem(max_active=cfg.max_active_goals)

        # ── Internal state ──
        self.internal_state = InternalState()

        # ── Curiosity + Experiments ──
        self.curiosity = AdvancedCuriosityEngine(
            space=self.space, rel_memory=self.rel_memory,
            vec_memory=self.vec_memory, novelty_threshold=cfg.novelty_threshold,
            dim=cfg.dim, seed=cfg.seed,
        )
        self.experiment_engine = ExperimentEngine(
            belief_system=self.beliefs,
            simulator=self.simulator,
            curiosity=self.curiosity,
            min_conf_to_act=0.70,
        )

        # ── Reasoning ──
        self.abstraction = AbstractionEngine(
            rel_memory=self.rel_memory, world_model=self.world_model,
            causal_graph=self.causal_graph, space=self.space,
        )
        self.meta_learner = MetaLearner(history_window=100, adapt_rate=0.1)

        # ── v5 Stability systems ──
        self.conflict_resolver  = ConflictResolver()
        self.consistency_engine = ConsistencyEngine(self.beliefs)
        self.knowledge_validator = KnowledgeValidator(
            self.beliefs, self.rel_memory, self.world_model,
            validation_interval=30.0,
        )
        self.risk_engine        = RiskEngine(self.beliefs, risk_tolerance=cfg.risk_tolerance)

        # ── Memory consolidator ──
        self.consolidator = MemoryConsolidator(
            vec_memory=self.vec_memory, rel_memory=self.rel_memory, space=self.space
        )

        # ── Goal generator ──
        self.goal_gen = GoalGenerator(
            goal_system=self.goal_system, rel_memory=self.rel_memory,
            vec_memory=self.vec_memory, curiosity=self.curiosity,
            world_model=self.world_model, space=self.space,
            enc_fn=self.enc.encode, max_active_goals=cfg.max_active_goals,
            dim=cfg.dim,
        )

        # ── Metrics ──
        self.metrics = AgentMetrics(window=50)

        # ── Environment ──
        self.env = env

        # ── State ──
        self._step_count      = 0
        self._step_log:       List[V5Step] = []
        self._prev_state_vec: Optional[np.ndarray] = None
        self._last_env_obs:   Optional[Dict] = None
        self._pending_experiment: Optional[str] = None   # object to experiment on

        if cfg.verbose:
            print("  [v5] Agent ready.\n")

    # ══════════════════════════════════════════════════════════════════
    # Main 18-step cycle
    # ══════════════════════════════════════════════════════════════════

    def step(
        self,
        text_input:   str = "",
        force_action: Optional[Tuple[str, str]] = None,
        verbose:      Optional[bool] = None,
    ) -> V5Step:
        """One full 18-step stable cognitive cycle."""
        t0 = time.time()
        verbose = verbose if verbose is not None else self.cfg.verbose
        self.internal_state.tick()
        self._step_count += 1
        S = self._step_count

        # ── 1-2: Observe + Encode ──
        input_vec = (self.enc.encode(text_input)
                     if text_input
                     else (self._prev_state_vec
                           if self._prev_state_vec is not None
                           else np.zeros(self.cfg.dim, dtype=np.float32)))

        # ── 3: Update episodic + vector memory ──
        if text_input:
            self.vec_memory.store(input_vec, label=text_input[:30], modality="text",
                                  metadata={"step": S})

        # ── 4: Validate knowledge (periodic) ──
        if S % self.cfg.validation_interval == 0:
            self.knowledge_validator.validate_all(verbose=verbose)

        # ── 5: Detect novelty ──
        eff_threshold = self.internal_state.effective_novelty_threshold(self.cfg.novelty_threshold)
        novelty = self.curiosity.detect_novelty(input_vec, text_input)
        if novelty >= eff_threshold and text_input:
            self.curiosity.explore(text_input, input_vec)
            self.internal_state.on_novel_input(novelty)

        # ── 6: Risk assessment of novelty ──
        risk_decision = "proceed"
        risk_score    = 0.0

        # ── 7: Goal generation ──
        new_goals = 0
        env_obs   = self._last_env_obs or {}
        perf      = self.meta_learner.performance_metrics()

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
                    src = g.metadata.get("source", "?")
                    print(f"    🌱 Goal: '{g.label}' [src={src}, p={g.priority:.2f}]")

        # ── 8: Select goal ──
        active_goal = self.goal_system.select_active_goal()
        goal_label  = active_goal.label if active_goal else "explore"
        goal_vec    = (active_goal.goal_vec
                       if active_goal and active_goal.goal_vec is not None
                       else input_vec)

        if verbose and active_goal:
            print(f"    🎯 Goal: '{goal_label}'")

        # ── 9-10: Plan + Simulate ──
        env_available = (self.env.available_objects() if self.env else [])
        env_inventory = (self.env.inventory             if self.env else [])
        state_vec     = (self.env.state_vector()         if self.env else input_vec)

        action_to_take = None
        action_obj     = ""
        sim_score      = 0.5
        experiment_run = False

        # Check if we have a pending experiment
        if self._pending_experiment and self.env:
            target_obj = self._pending_experiment
            exp = self.experiment_engine.design(
                target_obj, "edible",
                state_vec=state_vec,
            )
            safe, reason = self.experiment_engine.is_safe_to_execute(exp)
            if safe:
                action_to_take = exp.action
                action_obj     = target_obj
                risk_decision  = "experiment"
                experiment_run = True
                self._pending_experiment = None
                if verbose:
                    print(f"    🧪 Experiment: {exp.action}({target_obj}) "
                          f"[safety={exp.safety_level.name}]")
            else:
                if verbose:
                    print(f"    🛑 Experiment unsafe: {reason} → using 'inspect' instead")
                action_to_take = "inspect"
                action_obj     = target_obj
                risk_decision  = "caution"
                self._pending_experiment = None

        elif goal_label != "explore" and env_available:
            # Normal planning path
            try:
                h_plan = self.h_planner.plan_hierarchical(
                    goal_label, goal_vec, state_vec, env_available, env_inventory
                )
                if h_plan and h_plan.flat_actions:
                    # ── 11: Risk evaluation ──
                    for act, obj in h_plan.flat_actions[:3]:
                        ra = self.risk_engine.assess(
                            act, obj,
                            agent_confidence=self.internal_state.confidence,
                        )
                        if ra.decision == "block":
                            if verbose:
                                print(f"    🛑 BLOCKED: {act}({obj}) — {ra.reason}")
                            risk_decision = "block"
                            risk_score    = ra.risk_score
                            # Try safer alternative
                            alt = ra.safer_alternative
                            if alt:
                                action_to_take = alt
                                action_obj     = obj
                            break
                        elif ra.decision == "caution":
                            risk_decision = "caution"
                            risk_score    = ra.risk_score

                    if action_to_take is None and h_plan.flat_actions:
                        action_to_take, action_obj = h_plan.flat_actions[0]
                        # Simulate chosen action
                        sim_result = self.simulator.simulate(
                            state_vec, [(action_to_take, action_obj or "")], goal_vec
                        )
                        sim_score = self.simulator._score_simulation(sim_result, goal_vec)
            except Exception:
                action_to_take = "observe"
                action_obj     = ""

        if action_to_take is None:
            action_to_take = "observe"

        if force_action:
            action_to_take, action_obj = force_action

        # ── 12: Execute ──
        env_fb = None
        if action_to_take and self.env:
            args   = (action_obj,) if action_obj else ()
            env_fb = self.env.step(action_to_take, *args)
            if verbose:
                status = "✓" if env_fb["success"] else "✗"
                print(f"    {status} {action_to_take}({action_obj}) → "
                      f"{env_fb['message'][:55]}  r={env_fb.get('reward',0):+.2f}")
            self._last_env_obs = env_fb

        # ── 13: Receive feedback + record episode ──
        belief_updates = 0
        conflicts_found = 0
        consistency_ok  = True

        if env_fb:
            self.episodic_memory.record(
                step=S, action=action_to_take,
                target=action_obj or env_fb.get("object_name", ""),
                outcome="success" if env_fb.get("success") else "failure",
                reward=env_fb.get("reward", 0.0),
                state_vec=state_vec,
                context={"goal": goal_label, "novelty": novelty},
            )

            # ── 14: Belief update (Bayesian) ──
            belief_updates = self._update_beliefs_from_feedback(env_fb)

            # ── 15: Resolve conflicts ──
            conflicts = self.conflict_resolver.scan(self.beliefs)
            conflicts_found = len(conflicts)
            if conflicts and verbose:
                for c in conflicts[:2]:
                    print(f"    ⚡ Conflict resolved: {c.key} "
                          f"({c.strategy.value}) → {c.notes[:50]}")

            # ── 16: Consistency check + propagate ──
            audit = self.consistency_engine.audit()
            consistency_ok = audit["consistent"]
            if not consistency_ok and verbose:
                print(f"    ⚠️  Consistency violation: "
                      f"{audit['violations'][0]['type'] if audit['violations'] else '?'}")

            # Propagate new beliefs
            obj_name = env_fb.get("object_name", "")
            if obj_name:
                self.consistency_engine.propagate(f"{obj_name}.edible")
                self.consistency_engine.propagate(f"{obj_name}.is_a")

            # Update internal state
            if env_fb.get("success"):
                self.internal_state.on_success(abs(env_fb.get("reward", 0.3)))
            else:
                self.internal_state.on_failure(abs(env_fb.get("reward", 0.1)))

            # Check goal satisfaction
            if active_goal:
                if self.goal_system.check_goal_satisfied(active_goal, env_fb):
                    self.goal_system.mark_completed(active_goal)
                    self.internal_state.on_goal_completed()
                    if verbose:
                        print(f"    ✅ Goal COMPLETED: '{goal_label}'")
                elif env_fb.get("reward", 0) < -0.3:
                    self.goal_system.mark_failed(active_goal)

            # Update world model
            self._update_world_model(env_fb, state_vec, action_to_take)

            # Process experiment if one was running
            if experiment_run:
                # Find the experiment we just ran
                pending_exps = [e for e in self.experiment_engine._experiments
                                if e.status == "designed" and e.target == action_obj]
                if not pending_exps:
                    # Find most recent designed experiment
                    all_designed = [e for e in self.experiment_engine._experiments
                                    if e.status == "designed"]
                    if all_designed:
                        exp = all_designed[-1]
                        self.experiment_engine.process_result(exp, env_fb)

        # ── 17: Consolidate memory (periodic) ──
        if S % self.cfg.consolidation_interval == 0:
            if self.internal_state.should_consolidate():
                self.consolidator.consolidate(verbose=False)
                self.internal_state.on_consolidation()

            # Abstraction
            if S % self.cfg.abstraction_interval == 0:
                new_rules = self.abstraction.run(verbose=False)
                # Sync abstract rules to belief system
                for rule in new_rules:
                    if rule.is_strong:
                        self.beliefs.update(
                            f"{rule.subject}.{rule.predicate}",
                            rule.value, source="abstraction",
                            weight=rule.confidence * 0.8,
                        )

        # ── 18: Update metrics + adapt ──
        if action_to_take and env_fb:
            self.meta_learner.record(
                step=S, action=action_to_take,
                success=env_fb.get("success", False),
                reward=env_fb.get("reward", 0),
                goal_type=active_goal.goal_type.value if active_goal else "",
                plan_depth=1, novelty=novelty,
                confidence=self.internal_state.confidence,
            )

        # Record metrics
        mean_conf = self.beliefs.summary().get("mean_conf", 0.5)
        self.metrics.record(
            step=S,
            success=bool(env_fb.get("success") if env_fb else False),
            reward=float(env_fb.get("reward", 0) if env_fb else 0),
            belief_confidence=mean_conf,
            n_conflicts=conflicts_found,
            n_blocks=int(risk_decision == "block"),
            novelty=novelty,
            goal_type=active_goal.goal_type.value if active_goal else "",
            action=action_to_take or "",
        )
        # Snapshot beliefs for stability tracking
        if S % 5 == 0:
            self.metrics.snapshot_beliefs(
                {b.key: b.confidence for b in self.beliefs.reliable_beliefs()}
            )

        # Update prev state
        self._prev_state_vec = (self.env.state_vector() if self.env
                                else input_vec)

        elapsed = (time.time() - t0) * 1000
        record = V5Step(
            step=S,
            input_text=text_input,
            active_goal=goal_label,
            action_taken=action_to_take,
            action_obj=action_obj or "",
            env_success=env_fb.get("success") if env_fb else None,
            env_reward=env_fb.get("reward")   if env_fb else None,
            novelty=novelty,
            risk_decision=risk_decision,
            risk_score=risk_score,
            belief_updates=belief_updates,
            conflicts_found=conflicts_found,
            goals_generated=new_goals,
            experiment_run=experiment_run,
            consistency_ok=consistency_ok,
            elapsed_ms=elapsed,
        )
        self._step_log.append(record)
        return record

    # ══════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════

    def _update_beliefs_from_feedback(self, env_fb: Dict) -> int:
        """Update belief system with Bayesian evidence from feedback. Returns count."""
        count     = 0
        obj_name  = env_fb.get("object_name", "")
        obj_props = env_fb.get("object_props", {})
        action    = env_fb.get("action", "")
        success   = env_fb.get("success", False)
        reward    = env_fb.get("reward", 0.0)

        # Check consistency before updating
        for prop, val in obj_props.items():
            if prop in ("name",):
                continue
            key = f"{obj_name}.{prop}"
            allowed, adj_weight, reason = self.consistency_engine.check_before_update(
                key, val, source="direct_observation"
            )
            if allowed:
                self.beliefs.update(key, val, source="direct_observation",
                                    weight=adj_weight)
                count += 1

        # Specific action outcomes
        if obj_name and action == "eat":
            edible = success and reward >= 0.3
            key    = f"{obj_name}.edible"
            allowed, adj_wt, _ = self.consistency_engine.check_before_update(
                key, edible, source="direct_observation"
            )
            if allowed:
                self.beliefs.update(key, edible, source="direct_observation",
                                    weight=adj_wt * (0.9 if success else 0.8))
                count += 1

        # Learned facts from environment
        learned = env_fb.get("learned", "")
        if learned and obj_name:
            is_edible = "not" not in learned.lower()
            key = f"{obj_name}.edible"
            self.beliefs.update(key, is_edible, source="direct_observation",
                                weight=0.95)
            count += 1

        # Update relational memory in sync
        if obj_name and obj_props:
            self.rel_memory.add_object_properties(obj_name, obj_props)
        if obj_name and action:
            self.rel_memory.ingest_env_feedback(
                action, obj_name, obj_props, success, reward
            )

        return count

    def _update_world_model(self, env_fb: Dict, prev_state: np.ndarray, action: str) -> None:
        next_state   = env_fb.get("state_vec", prev_state)
        reward       = env_fb.get("reward", 0.0)
        success      = env_fb.get("success", False)
        obj_name     = env_fb.get("object_name", "")
        obj_category = env_fb.get("object_category", "unknown")
        obj_props    = env_fb.get("object_props", {})

        self.world_model.record(
            prev_state, action, next_state,
            reward, success, obj_name, obj_category
        )
        self.causal_graph.ingest_feedback(
            action, obj_name,
            {**obj_props, "category": obj_category},
            success, reward
        )

    # ══════════════════════════════════════════════════════════════════
    # Continuous loop
    # ══════════════════════════════════════════════════════════════════

    def run_autonomous(
        self,
        n_steps:      int  = 50,
        verbose:      bool = True,
        stop_on_goal: Optional[str] = None,
    ) -> List[V5Step]:
        """Run autonomously for n_steps."""
        log = []
        if verbose:
            print(f"\n  {'St':4s}|{'Goal':26s}|{'Action':10s}|{'Obj':10s}|"
                  f"{'Rew':6s}|{'Risk':6s}|{'Blf':5s}|{'Stab'}")
            print(f"  {'─'*4}|{'─'*26}|{'─'*10}|{'─'*10}|"
                  f"{'─'*6}|{'─'*6}|{'─'*5}|{'─'*5}")

        for _ in range(n_steps):
            s = self.step(verbose=False)
            log.append(s)

            if verbose:
                rew = f"{s.env_reward:+.2f}" if s.env_reward is not None else "  N/A"
                blf = f"{self.beliefs.summary().get('mean_conf', 0):.2f}"
                stab = self.metrics.stability_grade()
                rd   = s.risk_decision[0].upper()   # P/C/B/E
                print(f"  {s.step:4d}|{(s.active_goal or '–')[:26]:26s}|"
                      f"{(s.action_taken or '–')[:10]:10s}|"
                      f"{(s.action_obj or '–')[:10]:10s}|"
                      f"{rew:6s}|{rd}{s.risk_score:.2f}|{blf}|{stab}")

            if stop_on_goal:
                for g in self.goal_system._completed:
                    if stop_on_goal.lower() in g.label.lower():
                        if verbose:
                            print(f"\n  ✅ Goal achieved at step {s.step}")
                        return log
        return log

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
            for prop, val in props.items():
                # Use consistency check when teaching
                key = f"{l}.{prop}"
                allowed, wt, _ = self.consistency_engine.check_before_update(
                    key, val, source="prior"
                )
                if allowed:
                    self.beliefs.update(key, val, source="prior", weight=wt)
            self.rel_memory.add_object_properties(l, props, vector=vec)
            if "category" in props:
                self.causal_graph.add_is_a(l, props["category"])

    def schedule_experiment(self, obj_name: str) -> None:
        """Schedule a structured experiment on an object."""
        self._pending_experiment = obj_name

    def add_goal(self, label: str, gtype: GoalType = GoalType.CUSTOM,
                 target: str = "", priority: float = 0.7) -> Goal:
        goal_vec = self.enc.encode(label)
        return self.goal_system.add_goal(
            label=label, goal_type=gtype, target=target or label,
            priority=priority, goal_vec=goal_vec,
        )

    # ══════════════════════════════════════════════════════════════════
    # Queries
    # ══════════════════════════════════════════════════════════════════

    def what_is(self, concept: str) -> str:
        facts = self.beliefs.beliefs_about(concept)
        if not facts:
            return f"'{concept}': unknown"
        parts = [f"{b.key.split('.',1)[1]}={b.value}(conf={b.confidence:.2f})"
                 for b in sorted(facts, key=lambda b: -b.confidence)[:5]]
        return f"'{concept}': " + ", ".join(parts)

    def what_can_i_eat(self, min_conf: float = 0.60) -> List[str]:
        return self.beliefs.find_edible(min_conf=min_conf)

    def what_is_dangerous(self, min_conf: float = 0.60) -> List[str]:
        return self.beliefs.find_dangerous(min_conf=min_conf)

    def how_confident(self, subject: str, predicate: str) -> float:
        return self.beliefs.get_confidence(f"{subject}.{predicate}")

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════

    def summary(self) -> Dict:
        perf    = self.meta_learner.performance_metrics()
        metrics = self.metrics.report()
        return {
            "steps":              self._step_count,
            "vector_memory":      len(self.vec_memory),
            "beliefs":            len(self.beliefs),
            "reliable_beliefs":   len(self.beliefs.reliable_beliefs()),
            "relational_facts":   self.rel_memory.n_facts(),
            "episodic_memory":    self.episodic_memory.size,
            "world_model_rules":  self.world_model.n_rules,
            "abstract_rules":     len(self.abstraction._rules),
            "goals_completed":    self.goal_system.completed_count,
            "goals_active":       self.goal_system.active_count,
            "goals_generated":    self.goal_gen._generated_count,
            "experiments":        self.experiment_engine.summary(),
            "conflicts_resolved": len(self.conflict_resolver._resolved),
            "risk_profile":       self.risk_engine.risk_profile(),
            "metrics":            metrics,
            "stability_grade":    self.metrics.stability_grade(),
            "success_rate":       round(perf.get("overall_success_rate", 0), 3),
            "edible_known":       self.what_can_i_eat(),
            "dangerous_known":    self.what_is_dangerous(),
            "internal_state":     self.internal_state.as_dict(),
            "env":                self.env.stats() if self.env else None,
        }

    def __repr__(self) -> str:
        return (f"CogniFieldAgentV5(steps={self._step_count}, "
                f"beliefs={len(self.beliefs)}, "
                f"grade={self.metrics.stability_grade()})")
