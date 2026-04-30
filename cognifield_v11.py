"""
cognifield_v11.py
==================
CogniField v11 — Self-Learning Deep Reasoning System

New in v11 (no LLM required):
  - Learning-first pipeline: check knowledge → learn if missing → then decide
  - Deep thinking mode: multi-step structured deliberation
  - Unknown Safety Rule: if unknown → avoid by default
  - Experience Engine: learn from outcomes automatically
  - World Model v2: object → category → property → effect hierarchy
  - Self-correction: detect + fix wrong beliefs
  - Internal simulation before final decision

Pipeline for think():
  1. Extract subject + predicate from input
  2. CHECK knowledge exists (world model + belief system)
  3. If UNKNOWN → trigger learning mode (safe probe sequence)
  4. Run DeepThinker deliberation (fast or deep mode)
  5. Run internal simulation to validate decision
  6. Apply safety override if risk too high
  7. Return structured decision with full reasoning trace
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import v10 base
from .cognifield_main import CogniField as CogniFieldV10, CogniFieldConfig, _make_response

# v11 new modules
from .core.deep_thinker import DeepThinker, ThinkingMode
from .core.experience_engine import ExperienceEngine
from .core.world_model_v2 import WorldModelV2


# ---------------------------------------------------------------------------
# v11 Config extension
# ---------------------------------------------------------------------------

@dataclass
class CogniFieldV11Config(CogniFieldConfig):
    """
    Extended configuration for v11.

    New fields
    ----------
    thinking_mode       : "fast" | "deep" | "auto"
    min_thinking_steps  : Minimum reasoning steps (default 3)
    learning_first      : Always check knowledge before deciding (default True)
    unknown_safety_rule : Return "avoid" for completely unknown inputs (default True)
    self_correction     : Run periodic self-correction on beliefs (default True)
    correction_interval : Steps between self-correction runs
    sim_before_decide   : Run internal simulation before final decision
    confidence_target   : Minimum confidence to use "proceed" (default 0.65)
    """
    thinking_mode:       str   = "auto"    # fast | deep | auto
    min_thinking_steps:  int   = 3
    learning_first:      bool  = True
    unknown_safety_rule: bool  = True
    self_correction:     bool  = True
    correction_interval: int   = 8
    sim_before_decide:   bool  = True
    confidence_target:   float = 0.65

    @classmethod
    def from_dict(cls, d: Dict) -> "CogniFieldV11Config":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fields})


# ---------------------------------------------------------------------------
# CogniFieldV11
# ---------------------------------------------------------------------------

class CogniFieldV11(CogniFieldV10):
    """
    CogniField v11 — Self-Learning Deep Reasoning.

    Extends v10 with:
      - DeepThinker: multi-step deliberation
      - ExperienceEngine: learns from outcomes
      - WorldModelV2: structured object hierarchy
      - Learning-first pipeline
      - Unknown safety rule
      - Self-correction
    """

    def __init__(
        self,
        config: Optional[Union[Dict, CogniFieldV11Config]] = None,
    ) -> None:
        # Parse config
        if isinstance(config, dict):
            cfg = CogniFieldV11Config.from_dict(config)
        elif isinstance(config, CogniFieldV11Config):
            cfg = config
        else:
            cfg = CogniFieldV11Config()

        # Init v10 base (fleet, event bus, global consensus, etc.)
        super().__init__(config=cfg)
        self._v11_cfg: CogniFieldV11Config = cfg

        # v11 components (one shared instance)
        mode_map = {
            "fast":  ThinkingMode.FAST,
            "deep":  ThinkingMode.DEEP,
            "auto":  ThinkingMode.AUTO,
        }
        self._deep_thinker = DeepThinker(
            mode=mode_map.get(cfg.thinking_mode, ThinkingMode.AUTO),
            min_steps=cfg.min_thinking_steps,
            confidence_target=cfg.confidence_target,
        )
        self._world_model_v2 = WorldModelV2()

        # One ExperienceEngine per agent (shared belief access here)
        # We create a "fleet" experience engine on the primary agent's beliefs
        self._exp_engine = ExperienceEngine(
            belief_system=self._agents[0].beliefs,
            error_penalty=0.15,
            correct_boost=0.04,
        )

        # Internal step counter for self-correction scheduling
        self._v11_steps    = 0
        self._total_outcomes = 0

    # ══════════════════════════════════════════════════════════════════
    # Override think() — v11 learning-first pipeline
    # ══════════════════════════════════════════════════════════════════

    def think(self, input_text: str) -> Dict:
        """
        Learning-first deep-reasoning pipeline.

        Steps:
          1. Parse input → extract subject + predicate
          2. Check knowledge (world model + beliefs)
          3. If unknown → trigger learning probe
          4. Deep thinking (multi-step deliberation)
          5. Pre-decision simulation (if enabled)
          6. Safety override
          7. Return structured result with full trace
        """
        t0 = time.time()
        self._v11_steps += 1

        # ── 1. Parse input ──
        subject, predicate = self._parse_input(input_text)

        # ── 2. Knowledge check ──
        knowledge_state = self._check_knowledge(subject, predicate)

        # ── 3. Learning probe if unknown ──
        reasoning_prefix: List[str] = []
        if knowledge_state == "unknown" and self._v11_cfg.learning_first:
            probe_results = self._learning_probe(subject, predicate)
            reasoning_prefix = probe_results
            # Re-check after probe
            knowledge_state = self._check_knowledge(subject, predicate)

        # ── 4. World model inference ──
        wm_val, wm_conf = self._world_model_v2.infer_property(subject, predicate)
        if wm_val is not None:
            # Sync world model knowledge into agent beliefs
            self._world_model_v2.sync_to_beliefs(
                self._agents[0].beliefs, min_conf=0.60
            )

        # ── 5. Deep thinking ──
        causal = self._world_model_v2.causal_chains(subject, predicate)
        ctx = {
            "uncertainty":    self._v11_cfg.uncertainty,
            "knowledge_state": knowledge_state,
        }
        think_result = self._deep_thinker.think(
            subject, predicate,
            belief_system=self._agents[0].beliefs,
            world_model_data={"causal_chains": causal},
            context=ctx,
        )

        # ── 6. Internal simulation ──
        sim_endorses = True
        if self._v11_cfg.sim_before_decide:
            sim_endorses = self._pre_decision_simulation(subject, predicate,
                                                          think_result.decision)

        # ── 7. Safety override ──
        decision   = think_result.decision
        confidence = think_result.confidence

        if not think_result.safe:
            decision   = "avoid"
            confidence = min(confidence, 0.20)
        elif self._v11_cfg.unknown_safety_rule and knowledge_state == "unknown":
            decision   = "avoid"
            confidence = min(confidence, 0.25)
        elif not sim_endorses and decision == "proceed":
            decision   = "proceed_with_caution"
            confidence = min(confidence, 0.55)
        elif think_result.contradictions:
            decision   = "uncertain"
            confidence = min(confidence, 0.45)

        # Confidence gate
        if decision == "proceed" and confidence < self._v11_cfg.confidence_target:
            decision = "proceed_with_caution"

        # ── 8. Build reasoning chain ──
        reasoning = (
            reasoning_prefix
            + think_result.reasoning
            + ([f"Simulation endorses decision: {sim_endorses}"]
               if self._v11_cfg.sim_before_decide else [])
        )

        # ── 9. Run v10 fleet step for consensus / shared memory ──
        v10_result = super().think(input_text)

        # Merge v11 decision into v10 result (v11 takes precedence)
        v10_result["decision"]        = decision
        v10_result["confidence"]      = round(float(confidence), 4)
        v10_result["reasoning"]       = reasoning
        v10_result["thinking_steps"]  = think_result.n_steps
        v10_result["thinking_mode"]   = think_result.mode.value
        v10_result["knowledge_state"] = knowledge_state
        v10_result["safe"]            = think_result.safe
        v10_result["contradictions"]  = think_result.contradictions
        v10_result["world_model"]     = {
            "inferred_value": wm_val,
            "inferred_conf":  round(wm_conf, 3) if wm_val else None,
        }
        v10_result["elapsed_ms"] = round((time.time() - t0) * 1000, 1)

        # ── 10. Periodic self-correction ──
        if (self._v11_cfg.self_correction
                and self._v11_steps % self._v11_cfg.correction_interval == 0):
            corrections = self._exp_engine.audit_and_correct()
            if corrections:
                v10_result["self_corrections"] = len(corrections)

        return v10_result

    def decide(self, input_text: str) -> Dict:
        """v11 decide: runs deep think + adds action/risk fields."""
        result = self.think(input_text)
        # Add decide-specific fields
        risk    = self._assess_risk_level(result["confidence"])
        result["risk_level"]   = risk
        result["action"]       = self._pick_action(result["decision"], risk)
        result["alternatives"] = self._generate_alternatives(input_text, result)
        return result

    def simulate(self, scenario: str, steps: int = 10) -> Dict:
        """v11 simulate: uses world model to pre-seed the scenario."""
        # Sync world model to all agents before simulation
        for a in self._agents:
            self._world_model_v2.sync_to_beliefs(a.beliefs, min_conf=0.60)
        return super().simulate(scenario, steps)

    # ══════════════════════════════════════════════════════════════════
    # New v11 public methods
    # ══════════════════════════════════════════════════════════════════

    def learn_from_outcome(
        self,
        input_text:  str,
        subject:     str,
        predicate:   str,
        prediction:  Any,
        actual:      Any,
        action:      str   = "",
        reward:      float = 0.0,
    ) -> Dict:
        """
        Update the system based on an observed outcome.

        Call this after any action to close the learning loop.

        Parameters
        ----------
        input_text : Original question/input.
        subject    : Object acted on.
        predicate  : Property that was evaluated.
        prediction : What the system predicted.
        actual     : What actually happened.
        action     : Action taken (eat, pick, etc.).
        reward     : Observed reward (+positive is good).

        Returns
        -------
        Dict with corrections made.
        """
        self._total_outcomes += 1
        corrections = self._exp_engine.learn_from_outcome(
            input_text, subject, predicate,
            prediction, actual, action, reward,
            step=self._v11_steps,
        )
        # Update world model with observed effect
        if action and reward != 0.0:
            effect = "success" if reward > 0 else "failure"
            self._world_model_v2.add_effect(action, subject, effect, reward)

        return {
            "corrections_made": len(corrections),
            "details": [c.reason for c in corrections],
            "rules_derived":    self._exp_engine.derived_rules(),
        }

    def teach(
        self,
        label:      str,
        properties: Dict[str, Any],
        text:       Optional[str] = None,
    ) -> "CogniFieldV11":
        """v11 teach: also updates the world model."""
        super().teach(label, properties, text)
        # Add to world model
        cat = properties.get("category") or properties.get("is_a")
        self._world_model_v2.add_entity(label, category=cat,
                                         properties=properties, confidence=0.85)
        return self

    def world_knowledge(self, entity: str) -> Dict:
        """Return everything known about an entity from the world model."""
        e = self._world_model_v2.get_entity(entity)
        if e is None:
            return {"entity": entity, "known": False}
        effect_label, reward, eff_conf = self._world_model_v2.infer_effect("eat", entity)
        return {
            "entity":     entity,
            "known":      True,
            "category":   e.category,
            "properties": {k: {"value": v, "confidence": e.get_confidence(k)}
                           for k, v in e.properties.items()},
            "eat_effect": {"effect": effect_label, "reward": reward, "conf": eff_conf},
        }

    def self_reflect(self) -> Dict:
        """
        Run a full self-reflection cycle:
          - audit beliefs for systematic errors
          - check deep thinker calibration
          - return findings
        """
        corrections = self._exp_engine.audit_and_correct()
        think_summary = self._deep_thinker.summary()
        exp_summary   = self._exp_engine.summary()
        wm_summary    = self._world_model_v2.summary()

        findings = []
        if corrections:
            findings.append(f"Self-corrected {len(corrections)} beliefs")
        if think_summary.get("mean_confidence", 0.5) > 0.85:
            findings.append("Warning: mean thinking confidence may be overconfident")
        if exp_summary.get("success_rate", 1.0) < 0.40:
            findings.append("Low success rate — recommend switching to VERIFY strategy")

        return {
            "findings":     findings,
            "corrections":  [c.reason for c in corrections],
            "experience":   exp_summary,
            "thinking":     think_summary,
            "world_model":  wm_summary,
        }

    # ══════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════

    def _parse_input(self, text: str) -> Tuple[str, str]:
        """Extract (subject, predicate) from natural language input."""
        text_l = text.lower()

        # Predicate detection
        predicate = "edible"  # default
        safety_words   = ["safe", "eat", "edible", "consume", "food", "edibility"]
        fragile_words  = ["fragile", "break", "brittle", "delicate"]
        heavy_words    = ["heavy", "weight", "lift", "carry"]
        toxic_words    = ["toxic", "poison", "dangerous", "harmful"]

        if any(w in text_l for w in toxic_words):
            predicate = "toxic"
        elif any(w in text_l for w in safety_words):
            predicate = "edible"
        elif any(w in text_l for w in fragile_words):
            predicate = "fragile"
        elif any(w in text_l for w in heavy_words):
            predicate = "heavy"

        # Subject detection: find the main noun
        import re
        # Skip question/verb words
        stopwords = {"is","the","this","that","a","an","can","i","should","would",
                     "could","will","does","do","it","be","to","safe","edible",
                     "eat","what","about","thing","object"}
        words = re.findall(r'\b[a-zA-Z_]+\b', text_l)
        candidates = [w for w in words if w not in stopwords and len(w) > 2]
        subject = candidates[0] if candidates else "unknown"

        return subject, predicate

    def _check_knowledge(self, subject: str, predicate: str) -> str:
        """
        Check knowledge state for a subject.predicate pair.
        Returns: "known" | "partial" | "unknown"
        """
        key = f"{subject}.{predicate}"
        belief = self._agents[0].beliefs.get(key)

        if belief and belief.is_reliable:
            return "known"

        # Check world model
        val, conf = self._world_model_v2.infer_property(subject, predicate)
        if val is not None and conf >= 0.60:
            return "partial"

        # Check if we know the category at least
        cat_b = self._agents[0].beliefs.get(f"{subject}.category")
        if cat_b and cat_b.is_reliable:
            return "partial"

        return "unknown"

    def _learning_probe(self, subject: str, predicate: str) -> List[str]:
        """
        Run a safe learning probe to gather information about an unknown.
        Does NOT act — inspects available knowledge sources.
        Returns reasoning lines.
        """
        results: List[str] = []
        results.append(f"LEARNING MODE: '{subject}.{predicate}' is unknown. "
                       f"Initiating safe knowledge probe.")

        # 1. Check if category is known → infer from category
        for cat in self._world_model_v2.known_categories():
            ents = self._world_model_v2.entities_in_category(cat)
            if subject in ents:
                val, conf = self._world_model_v2.infer_property(subject, predicate)
                if val is not None:
                    results.append(f"Category inference: {subject} is_a {cat} → "
                                   f"{predicate}={val} (conf={conf:.2f})")
                    # Push to beliefs
                    self._agents[0].beliefs.update(
                        f"{subject}.{predicate}", val,
                        source="inference", weight=conf * 0.8,
                        notes="learning_probe_category"
                    )
                    return results

        # 2. Check for similar known objects
        similar = self._find_similar_objects(subject)
        if similar:
            results.append(f"Nearest known analogues: {similar[:3]}. "
                           f"No direct {predicate} inference possible.")

        # 3. Apply unknown safety rule
        results.append(f"No {predicate} knowledge found for '{subject}'. "
                       f"Unknown safety rule applies: conservative estimate.")
        return results

    def _find_similar_objects(self, subject: str) -> List[str]:
        """Find known objects with similar names (prefix matching)."""
        known = list(self._world_model_v2._entities.keys())
        prefix = subject[:3].lower()
        return [k for k in known if k.startswith(prefix) and k != subject]

    def _pre_decision_simulation(
        self,
        subject:   str,
        predicate: str,
        proposed_decision: str,
    ) -> bool:
        """
        Run a quick mental simulation to validate the proposed decision.
        Returns True if simulation endorses it, False if it contradicts.
        """
        if proposed_decision == "avoid":
            return True   # avoiding is always safe

        # Look up expected effect of acting
        action = "eat" if predicate == "edible" else "use"
        effect, reward, conf = self._world_model_v2.infer_effect(action, subject)

        if effect == "unknown" or conf < 0.35:
            # Can't simulate → neither endorse nor block
            return True

        if reward < -0.20 and conf >= 0.60:
            # Simulation predicts harm → contradict proceed
            return False

        return True

    # ══════════════════════════════════════════════════════════════════
    # Status override
    # ══════════════════════════════════════════════════════════════════

    def status(self) -> Dict:
        base = super().status()
        base["version"]       = "11.0"
        base["thinking_mode"] = self._v11_cfg.thinking_mode
        base["learning_first"]= self._v11_cfg.learning_first
        base["v11"] = {
            "thinking":   self._deep_thinker.summary(),
            "experience": self._exp_engine.summary(),
            "world_model":self._world_model_v2.summary(),
        }
        return base

    def __repr__(self) -> str:
        return (f"CogniFieldV11(agents={len(self._agents)}, "
                f"mode={self._v11_cfg.thinking_mode}, "
                f"outcomes={self._total_outcomes})")

    # ══════════════════════════════════════════════════════════════════
    # v11 Part 2 — Embodied Intelligence Layer
    # ══════════════════════════════════════════════════════════════════

    def _ensure_embodied(self) -> None:
        """Lazily initialise embodied components on first use."""
        if hasattr(self, "_interaction_loop"):
            return

        from .agents.body import VirtualBody
        from .agents.action_system import ActionSystem
        from .agents.perception import PerceptionSystem
        from .core.deep_thinker import DeepThinker, ThinkingMode
        from .core.interaction_loop import InteractionLoop

        mode_map = {"fast": ThinkingMode.FAST, "deep": ThinkingMode.DEEP,
                    "auto": ThinkingMode.AUTO}

        self._body          = VirtualBody(seed=self._v11_cfg.seed)
        self._action_system = ActionSystem(
            body=self._body,
            unknown_safety_rule=self._v11_cfg.unknown_safety_rule,
            min_confidence_to_act=0.35,
        )
        self._perception    = PerceptionSystem()
        self._loop_thinker  = DeepThinker(
            mode=mode_map.get(self._v11_cfg.thinking_mode, ThinkingMode.AUTO),
            min_steps=self._v11_cfg.min_thinking_steps,
        )
        self._interaction_loop = InteractionLoop(
            body=self._body,
            action_system=self._action_system,
            perception=self._perception,
            deep_thinker=self._loop_thinker,
            experience_engine=self._exp_engine,
            world_model=self._world_model_v2,
            belief_system=self._agents[0].beliefs,
            unknown_safety=self._v11_cfg.unknown_safety_rule,
            verbose=self._v11_cfg.verbose,
        )

    def act(self, action: str, obj: str, force: bool = False) -> Dict:
        """
        Execute a single physical action.

        Parameters
        ----------
        action : Action string ("eat", "pick", "inspect", "move", "look")
        obj    : Target object or direction for move
        force  : Skip safety validation (use with care)

        Returns
        -------
        Dict with: action, object, status, effect, reward, reason,
                   body_health, body_hunger, body_inventory
        """
        self._ensure_embodied()

        # Build env_fn from world model
        def env_fn(a, o):
            props = {}
            e = self._world_model_v2.get_entity(o)
            if e:
                props = dict(e.properties)
                props["known"] = True
                props["confidence"] = e.get_confidence("edible", 0.5)
            edible_val, edible_conf = self._world_model_v2.infer_property(o, "edible")
            if edible_val is not None:
                props["edible"]     = edible_val
                props["confidence"] = edible_conf
            effect, reward, conf = self._world_model_v2.infer_effect(a, o)
            props["effect"] = effect
            props["reward"] = reward
            return props

        executed, body_result, log_entry = self._action_system.execute(
            action=action,
            target=obj,
            belief_system=self._agents[0].beliefs,
            env_response=env_fn(action, obj),
            force=force,
        )

        # Observe and learn from result
        obs = self._perception.process_body_result(body_result)
        for key, val, conf in obs.belief_updates:
            self._agents[0].beliefs.update(key, val, "direct_observation", conf)
        if action == "eat":
            self.learn_from_outcome(
                f"{action} {obj}", obj, "edible",
                executed, obs.is_success, action, obs.reward
            )
            self._world_model_v2.add_effect(action, obj, obs.effect, obs.reward)

        return {
            "action":      action,
            "object":      obj,
            "status":      body_result.status.value,
            "effect":      body_result.effect,
            "reward":      round(body_result.reward, 3),
            "reason":      body_result.reason,
            "body_health": round(self._body.state.health, 3),
            "body_hunger": round(self._body.state.hunger, 3),
            "body_inventory": list(self._body.state.inventory),
        }

    def step(self, query: str, env_fn=None) -> Dict:
        """
        Run one full THINK → SIMULATE → DECIDE → ACT → OBSERVE → LEARN cycle.

        Parameters
        ----------
        query  : Natural language query/command ("eat apple", "inspect stone")
        env_fn : Optional callable env_fn(action, target) → dict

        Returns
        -------
        Dict with: decision, action, target, result, effect, reward,
                   belief_updates, body, elapsed_ms, and all reasoning fields
        """
        self._ensure_embodied()

        # Build default env_fn from world model if none provided
        if env_fn is None:
            def env_fn(a, o):
                props = {}
                e = self._world_model_v2.get_entity(o)
                if e:
                    props = dict(e.properties)
                    props["known"] = True
                edible_val, edible_conf = self._world_model_v2.infer_property(o, "edible")
                if edible_val is not None:
                    props["edible"]     = edible_val
                    props["confidence"] = edible_conf
                effect, reward, _ = self._world_model_v2.infer_effect(a, o)
                props["effect"] = effect
                props["reward"] = reward
                return props

        episode_step = self._interaction_loop.step(query, env_fn=env_fn)
        return episode_step.to_dict()

    def run_episode(self, queries: List[str], env_fn=None) -> List[Dict]:
        """
        Run a sequence of queries as a complete episode.

        Parameters
        ----------
        queries : List of query strings in order
        env_fn  : Optional environment simulation function

        Returns
        -------
        List of step dicts (same format as step())
        """
        self._ensure_embodied()

        if env_fn is None:
            def env_fn(a, o):
                edible_val, edible_conf = self._world_model_v2.infer_property(o, "edible")
                effect, reward, _ = self._world_model_v2.infer_effect(a, o)
                return {
                    "edible":     edible_val,
                    "confidence": edible_conf,
                    "effect":     effect,
                    "reward":     reward,
                    "known":      edible_val is not None,
                }

        steps = self._interaction_loop.run_episode(queries, env_fn=env_fn)
        return [s.to_dict() for s in steps]

    def body_status(self) -> Dict:
        """Return current body state."""
        self._ensure_embodied()
        return self._body.summary()

    # ══════════════════════════════════════════════════════════════════
    # v11 Part 3 — Game Integration Layer
    # ══════════════════════════════════════════════════════════════════

    def create_game_loop(
        self,
        adapter     = None,
        vision:     bool = False,
        verbose:    bool = False,
    ):
        """
        Create a GameLoop connected to this CogniField instance.

        Parameters
        ----------
        adapter  : GameAdapter instance. If None, uses JavaAdapter(simulation=True).
        vision   : Enable VisionSystem alongside the adapter.
        verbose  : Print each game step.

        Returns
        -------
        GameLoop ready to call .step_from_game() or .run_episode()
        """
        self._ensure_embodied()

        from .game.java_adapter import JavaAdapter
        from .game.survival_goals import SurvivalGoalManager
        from .game.language_learner import LanguageLearner
        from .game.game_loop import GameLoop
        from .vision.vision_system import VisionSystem

        if adapter is None:
            adapter = JavaAdapter(simulation=True, seed=self._v11_cfg.seed)

        if not adapter.connected:
            adapter.connect()

        # Language learner wired to our world model + beliefs
        lang = LanguageLearner(
            world_model=self._world_model_v2,
            belief_system=self._agents[0].beliefs,
        )
        survival = SurvivalGoalManager()
        vis      = VisionSystem(simulation=True,
                                seed=self._v11_cfg.seed) if vision else None

        game_loop = GameLoop(
            adapter=adapter,
            interaction_loop=self._interaction_loop,
            vision=vis,
            language_learner=lang,
            survival_goals=survival,
            verbose=verbose,
        )
        # Cache for reuse
        self._game_loop = game_loop
        return game_loop

    def game_step(self, adapter=None) -> dict:
        """
        Run one full game-adaptive step.

        Parameters
        ----------
        adapter : Optional GameAdapter. If None, uses cached or creates Java sim.

        Returns dict with decision, effect, reward, health, hunger, goal.
        """
        if not hasattr(self, "_game_loop") or self._game_loop is None:
            self.create_game_loop(adapter=adapter)
        elif adapter is not None:
            self._game_loop.adapter = adapter

        gs = self._game_loop.step_from_game()
        lr = gs.loop_result
        return {
            "step":       gs.step,
            "goal":       gs.active_goal.name if gs.active_goal else None,
            "query":      gs.query,
            "decision":   lr.thinking_decision if lr else "none",
            "executed":   lr.action_executed   if lr else False,
            "effect":     lr.effect            if lr else "",
            "reward":     lr.reward            if lr else 0.0,
            "action_sent":gs.action_sent,
            "new_concepts":gs.new_concepts,
            "health":     gs.observation.health,
            "hunger":     gs.observation.hunger,
            "position":   gs.observation.position,
            "elapsed_ms": gs.elapsed_ms,
        }

    def run_game_episode(
        self,
        n_steps:  int  = 10,
        adapter         = None,
        vision:   bool = False,
        verbose:  bool = False,
    ) -> list:
        """
        Run n_steps of the game loop as a complete episode.

        Returns list of step dicts (same format as game_step()).
        """
        loop = self.create_game_loop(adapter=adapter, vision=vision,
                                      verbose=verbose)
        steps = loop.run_episode(n_steps)
        return [s.to_dict() for s in steps]
