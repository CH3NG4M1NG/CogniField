"""
core/interaction_loop.py
=========================
Embodied Interaction Loop

Implements the full cognitive-physical cycle:

  THINK → SIMULATE → DECIDE → ACT → OBSERVE → LEARN

This is the "heartbeat" of the embodied agent. Each call to step()
runs one complete cycle and returns a structured EpisodeStep with
everything that happened: the decision, the action taken, the
observed result, and what was learned.

Design
------
The loop is intentionally thin — it orchestrates the existing
subsystems (DeepThinker, ExperienceEngine, ActionSystem, etc.)
rather than reimplementing their logic.

                      ┌─────────────────────────┐
                      │       InteractionLoop    │
                      │                          │
  input_query ──────► │  1. parse_intent()       │
                      │  2. think()              │◄── DeepThinker
                      │  3. simulate_outcome()   │◄── WorldModelV2
                      │  4. decide_action()      │
                      │  5. act()                │◄── ActionSystem + Body
                      │  6. observe()            │◄── PerceptionSystem
                      │  7. learn()              │◄── ExperienceEngine
                      │                          │
                      │  return EpisodeStep      │
                      └─────────────────────────┘
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .deep_thinker import DeepThinker, ThinkingMode
from .experience_engine import ExperienceEngine
from ..agents.body import VirtualBody, BodyAction
from ..agents.action_system import ActionSystem, ValidationStatus
from ..agents.perception import PerceptionSystem, Observation


# ---------------------------------------------------------------------------
# Episode step result
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStep:
    """
    Complete record of one interaction loop step.

    Every field is always present — downstream code can safely access
    any attribute without key-checking.
    """
    step:              int
    input_query:       str
    intent_action:     str           # parsed action ("eat", "inspect", ...)
    intent_target:     str           # parsed target ("apple", ...)

    # Cognitive phase
    thinking_decision: str           # what deep thinker decided
    thinking_conf:     float
    thinking_steps:    int
    simulated_outcome: str           # what simulation predicted

    # Action phase
    action_validated:  bool          # passed validation?
    action_executed:   bool          # actually ran?
    action_blocked_reason: str       # why it was blocked (or "")

    # Observation phase
    observation_signal:str           # "success" / "failure" / "danger" / etc.
    reward:            float
    effect:            str           # "satisfied", "damage", "observed", ...

    # Learning phase
    belief_updates:    int           # how many beliefs were updated
    corrections_made:  int           # self-corrections triggered

    # Body state after step
    body_health:       float
    body_hunger:       float
    body_inventory:    List[str]

    # Timing
    elapsed_ms:        float
    timestamp:         float = field(default_factory=time.time)

    @property
    def succeeded(self) -> bool:
        return self.observation_signal in ("success", "neutral")

    @property
    def was_blocked(self) -> bool:
        return not self.action_executed and self.action_validated is False

    def to_dict(self) -> Dict:
        return {
            "step":          self.step,
            "query":         self.input_query,
            "action":        self.intent_action,
            "target":        self.intent_target,
            "decision":      self.thinking_decision,
            "confidence":    round(self.thinking_conf, 3),
            "simulated":     self.simulated_outcome,
            "executed":      self.action_executed,
            "blocked":       self.action_blocked_reason or None,
            "signal":        self.observation_signal,
            "effect":        self.effect,
            "reward":        round(self.reward, 3),
            "belief_updates":self.belief_updates,
            "corrections":   self.corrections_made,
            "body": {
                "health":    round(self.body_health, 3),
                "hunger":    round(self.body_hunger, 3),
                "inventory": self.body_inventory,
            },
            "elapsed_ms":    round(self.elapsed_ms, 1),
        }

    def __str__(self) -> str:
        status = "✓" if self.succeeded else ("⊘" if self.was_blocked else "✗")
        return (f"Step {self.step} {status}: "
                f"{self.intent_action}({self.intent_target}) "
                f"→ {self.effect} [{self.thinking_decision} "
                f"{self.thinking_conf:.0%}]")


# ---------------------------------------------------------------------------
# Interaction Loop
# ---------------------------------------------------------------------------

class InteractionLoop:
    """
    Orchestrates the full embodied cognitive loop.

    Parameters
    ----------
    body             : VirtualBody
    action_system    : ActionSystem
    perception       : PerceptionSystem
    deep_thinker     : DeepThinker
    experience_engine: ExperienceEngine
    world_model      : WorldModelV2 (for simulation step)
    belief_system    : BeliefSystem (shared reference)
    unknown_safety   : Block unknown objects
    verbose          : Print each step
    """

    def __init__(
        self,
        body:              VirtualBody,
        action_system:     ActionSystem,
        perception:        PerceptionSystem,
        deep_thinker:      DeepThinker,
        experience_engine: ExperienceEngine,
        world_model       = None,
        belief_system     = None,
        unknown_safety:    bool = True,
        verbose:           bool = False,
    ) -> None:
        self.body       = body
        self.actions    = action_system
        self.perception = perception
        self.thinker    = deep_thinker
        self.exp        = experience_engine
        self.wm         = world_model
        self.bs         = belief_system
        self.unknown_safety = unknown_safety
        self.verbose    = verbose

        self._step_count = 0
        self._history:   List[EpisodeStep] = []

    # ------------------------------------------------------------------
    # Core loop step
    # ------------------------------------------------------------------

    def step(
        self,
        input_query: str,
        env_fn       = None,   # callable(action, target) → dict
    ) -> EpisodeStep:
        # Auto-build env_fn from world model if none provided
        if env_fn is None and self.wm is not None:
            def env_fn(action, target):
                edible_val, edible_conf = self.wm.infer_property(target, "edible")
                effect, reward, conf    = self.wm.infer_effect(action, target)
                return {
                    "edible":     edible_val,
                    "confidence": edible_conf,
                    "effect":     effect,
                    "reward":     reward,
                    "known":      edible_val is not None,
                }
        """
        Run one complete THINK → ACT → LEARN cycle.

        Parameters
        ----------
        input_query : Natural language query or command (e.g. "eat apple")
        env_fn      : Optional callable that simulates the environment.
                      Signature: env_fn(action, target) → dict

        Returns
        -------
        EpisodeStep with full record of the cycle.
        """
        t0 = time.time()
        self._step_count += 1

        # ── 1. Parse intent ──
        intent_action, intent_target = self._parse_intent(input_query)

        if self.verbose:
            print(f"\n  [Step {self._step_count}] Query: '{input_query}'")
            print(f"    Intent: {intent_action}({intent_target})")

        # ── 2. THINK ──
        think_result = self._think_phase(intent_action, intent_target)

        if self.verbose:
            print(f"    Think: {think_result['decision']} "
                  f"({think_result['confidence']:.0%}, "
                  f"{think_result['steps']} steps)")

        # ── 3. SIMULATE ──
        simulated_outcome = self._simulate_phase(intent_action, intent_target,
                                                   think_result["decision"])

        # ── 4. DECIDE ──
        should_act = self._decide_phase(think_result, simulated_outcome,
                                         intent_action, intent_target)

        # ── 5. ACT ──
        executed, body_result, log_entry = self._act_phase(
            intent_action, intent_target,
            should_act=should_act,
            env_fn=env_fn,
        )

        if self.verbose:
            status = "✓" if executed else "⊘"
            print(f"    Act:   {status} {body_result.effect} "
                  f"(reward={body_result.reward:+.2f})")

        # ── 6. OBSERVE ──
        obs = self._observe_phase(body_result)

        # ── 7. LEARN ──
        belief_updates, corrections = self._learn_phase(
            intent_action, intent_target,
            think_result["decision"], obs,
        )

        if self.verbose:
            print(f"    Learn: {belief_updates} belief updates, "
                  f"{corrections} corrections")

        # ── Build step record ──
        elapsed = (time.time() - t0) * 1000
        episode_step = EpisodeStep(
            step=self._step_count,
            input_query=input_query,
            intent_action=intent_action,
            intent_target=intent_target,
            thinking_decision=think_result["decision"],
            thinking_conf=think_result["confidence"],
            thinking_steps=think_result["steps"],
            simulated_outcome=simulated_outcome,
            action_validated=log_entry.validation in (
                ValidationStatus.ALLOWED, ValidationStatus.WARNED
            ),
            action_executed=executed,
            action_blocked_reason=(
                body_result.reason if not executed else ""
            ),
            observation_signal=obs.signal.value,
            reward=obs.reward,
            effect=obs.effect,
            belief_updates=belief_updates,
            corrections_made=corrections,
            body_health=self.body.state.health,
            body_hunger=self.body.state.hunger,
            body_inventory=list(self.body.state.inventory),
            elapsed_ms=elapsed,
        )
        self._history.append(episode_step)
        return episode_step

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _think_phase(self, action: str, target: str) -> Dict:
        """Run DeepThinker on the target."""
        if self.bs is None:
            return {"decision": "unknown", "confidence": 0.3, "steps": 0}

        # Map action to relevant predicate
        predicate_map = {
            "eat":     "edible",
            "pick":    "heavy",
            "inspect": "edible",
        }
        predicate = predicate_map.get(action, "edible")

        ctx = {}
        causal = {}
        if self.wm:
            causal = self.wm.causal_chains(target, predicate)

        result = self.thinker.think(
            target, predicate, self.bs,
            world_model_data={"causal_chains": causal},
            context=ctx,
        )
        return {
            "decision":   result.decision,
            "confidence": result.confidence,
            "steps":      result.n_steps,
            "safe":       result.safe,
        }

    def _simulate_phase(self, action: str, target: str, decision: str) -> str:
        """Use the world model to predict the outcome before acting."""
        if self.wm is None:
            return "unknown"

        effect, reward, conf = self.wm.infer_effect(action, target)
        if conf < 0.30:
            return "uncertain"
        if reward > 0.2:
            return "positive"
        if reward < -0.1:
            return "negative"
        return effect or "neutral"

    def _decide_phase(
        self,
        think_result:      Dict,
        simulated_outcome: str,
        action:            str,
        target:            str,
    ) -> bool:
        """
        Decide whether to execute the action based on thinking + simulation.
        Returns True if we should attempt the action.
        """
        decision = think_result["decision"]
        conf     = think_result["confidence"]

        # Never act if thinking says avoid
        if decision == "avoid":
            return False

        # Never act if simulation predicts harm AND we're confident about it
        if simulated_outcome == "negative" and conf < 0.55:
            return False

        # Unknown safety rule
        if self.unknown_safety and decision in ("investigate", "avoid"):
            return False

        # Only act on consequential actions if confidence is sufficient
        consequential = {"eat", "pick"}
        if action in consequential and conf < 0.35:
            return False

        return True

    def _act_phase(
        self,
        action:    str,
        target:    str,
        should_act:bool,
        env_fn,
    ) -> Tuple[bool, Any, Any]:
        """Execute the action via ActionSystem."""
        # Get environment response if env_fn provided
        env_response = None
        if env_fn and should_act:
            try:
                raw = env_fn(action, target)
                if isinstance(raw, dict):
                    env_response = raw
            except Exception:
                pass

        executed, body_result, log_entry = self.actions.execute(
            action=action,
            target=target,
            belief_system=self.bs,
            env_response=env_response,
            force=not should_act and action in ("inspect", "look"),
        )
        # If should_act was False, override to block
        if not should_act and action not in ("inspect", "look", "wait"):
            from ..agents.body import BodyActionResult, ActionStatus
            body_result = BodyActionResult(
                action=body_result.action,
                object_name=target,
                status=ActionStatus.BLOCKED,
                effect="blocked_by_decision",
                reward=0.0,
                confidence=1.0,
                observations={},
                body_delta={},
                reason=f"Decision system blocked action on {target}.",
            )
            executed = False
            from ..agents.action_system import ActionLogEntry, ValidationStatus
            log_entry = ActionLogEntry(
                step=self._step_count,
                action=action,
                target=target,
                validation=ValidationStatus.BLOCKED,
                executed=False,
                body_result=body_result,
                env_response=env_response,
                outcome="blocked",
                reward=0.0,
                elapsed_ms=0.0,
            )

        return executed, body_result, log_entry

    def _observe_phase(self, body_result: Any) -> Observation:
        """Process body result through perception."""
        return self.perception.process_body_result(body_result)

    def _learn_phase(
        self,
        action:    str,
        target:    str,
        predicted: str,
        obs:       Observation,
    ) -> Tuple[int, int]:
        """Update beliefs and experience engine from observation."""
        n_updates    = 0
        n_corrections = 0

        if self.bs is None:
            return 0, 0

        # Apply belief updates from perception
        for key, val, conf in obs.belief_updates:
            self.bs.update(key, val, source="direct_observation", weight=conf)
            n_updates += 1

        # Update world model with observed effect
        if self.wm and obs.effect not in ("", "blocked", "blocked_by_decision"):
            self.wm.add_effect(action, target, obs.effect, obs.reward)

        # Experience engine learning
        if action in ("eat",):
            # Derive what we predicted for this predicate
            pred_val  = (predicted not in ("avoid", "investigate"))
            actual_val = obs.is_success

            corrections = self.exp.learn_from_outcome(
                input_text=f"{action} {target}",
                subject=target,
                predicate="edible",
                prediction=pred_val,
                actual=actual_val,
                action=action,
                reward=obs.reward,
                step=self._step_count,
            )
            n_corrections = len(corrections)

        return n_updates, n_corrections

    # ------------------------------------------------------------------
    # Intent parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_intent(query: str) -> Tuple[str, str]:
        """
        Parse a natural language query into (action, target).
        e.g. "eat apple" → ("eat", "apple")
        e.g. "inspect the stone" → ("inspect", "stone")
        """
        import re
        q     = query.lower().strip()
        tokens = re.findall(r'\b[a-zA-Z_]+\b', q)

        action_words = {
            "eat": "eat", "consume": "eat", "taste": "eat",
            "pick": "pick", "grab": "pick", "take": "pick",
            "drop": "drop", "put": "drop", "release": "drop",
            "inspect": "inspect", "examine": "inspect", "check": "inspect",
            "look": "look", "scan": "look", "observe": "look",
            "move": "move", "go": "move", "walk": "move",
            "wait": "wait", "rest": "wait",
        }
        stop_words = {"the", "a", "an", "at", "to", "this", "that",
                      "can", "i", "should", "is", "be", "it"}

        action = "inspect"  # default safe action
        target = ""

        for tok in tokens:
            if tok in action_words and action == "inspect":
                action = action_words[tok]
            elif tok not in stop_words and tok not in action_words and len(tok) > 2:
                if not target:
                    target = tok

        if not target:
            target = "environment"

        return action, target

    # ------------------------------------------------------------------
    # Episode runner
    # ------------------------------------------------------------------

    def run_episode(
        self,
        queries:   List[str],
        env_fn     = None,
    ) -> List[EpisodeStep]:
        """
        Run a sequence of queries as a complete episode.

        Parameters
        ----------
        queries : List of input queries in order
        env_fn  : Optional environment simulation function

        Returns
        -------
        List of EpisodeStep records
        """
        steps = []
        for q in queries:
            if not self.body.state.alive:
                print(f"  ⚠ Agent died at step {self._step_count}. Episode ended.")
                break
            s = self.step(q, env_fn)
            steps.append(s)
        return steps

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def recent_steps(self, n: int = 5) -> List[Dict]:
        return [s.to_dict() for s in self._history[-n:]]

    def success_rate(self) -> float:
        if not self._history:
            return 0.0
        return float(np.mean([s.succeeded for s in self._history]))

    def summary(self) -> Dict:
        return {
            "total_steps":  self._step_count,
            "success_rate": round(self.success_rate(), 3),
            "body":         self.body.summary(),
            "perception":   self.perception.summary(),
            "actions":      self.actions.summary(),
        }

    def __repr__(self) -> str:
        return (f"InteractionLoop(steps={self._step_count}, "
                f"sr={self.success_rate():.0%})")
