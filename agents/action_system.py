"""
agents/action_system.py
========================
Action System

The bridge between the agent's cognitive decision and the physical
execution in the environment. Every action passes through here before
it reaches the body or environment.

Responsibilities
----------------
1. VALIDATION
   Check that an action is physically and cognitively safe before
   executing it. Reasons to block:
     - Unknown object + unknown_safety_rule enabled
     - Contradictory belief (we believe it's toxic)
     - Body incapable (hands full, health too low)
     - Action not applicable to this object

2. EXECUTION
   Route the validated action to the body and environment, capture
   the combined result.

3. LOGGING
   Every action — including blocked/invalid ones — is recorded with
   full context for the learning system and audit trail.

4. FAILURE HANDLING
   On action failure, the system records the outcome for the
   experience engine and flags it for belief update.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .body import VirtualBody, BodyAction, BodyActionResult, ActionStatus


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

class ValidationStatus(str, Enum):
    ALLOWED    = "allowed"
    BLOCKED    = "blocked"     # safety rule prevented execution
    WARNED     = "warned"      # allowed but flagged as risky
    INVALID    = "invalid"     # action not possible for this object/state


@dataclass
class ValidationResult:
    status:    ValidationStatus
    reason:    str
    risk_score:float = 0.0     # 0 = safe, 1 = highly dangerous


# ---------------------------------------------------------------------------
# Action log entry
# ---------------------------------------------------------------------------

@dataclass
class ActionLogEntry:
    """Complete record of one action attempt."""
    step:             int
    action:           str
    target:           str
    validation:       ValidationStatus
    executed:         bool
    body_result:      Optional[BodyActionResult]
    env_response:     Optional[Dict]
    outcome:          str           # final outcome label
    reward:           float
    elapsed_ms:       float
    timestamp:        float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "step":       self.step,
            "action":     self.action,
            "target":     self.target,
            "validation": self.validation.value,
            "executed":   self.executed,
            "outcome":    self.outcome,
            "reward":     round(self.reward, 3),
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# ---------------------------------------------------------------------------
# Action System
# ---------------------------------------------------------------------------

class ActionSystem:
    """
    Validates and executes actions, logging every attempt.

    Parameters
    ----------
    body                  : VirtualBody instance
    unknown_safety_rule   : Block actions on completely unknown objects
    min_confidence_to_act : Minimum belief confidence required to act
    max_risk_to_act       : Block actions above this risk score
    """

    # Actions that are always safe to attempt
    SAFE_ACTIONS = {BodyAction.INSPECT, BodyAction.LOOK, BodyAction.WAIT,
                    BodyAction.MOVE}
    # Actions that require knowledge
    KNOWLEDGE_REQUIRED = {BodyAction.EAT}
    # Actions with physical consequences
    CONSEQUENTIAL = {BodyAction.EAT, BodyAction.PICK}

    def __init__(
        self,
        body:                   VirtualBody,
        unknown_safety_rule:    bool  = True,
        min_confidence_to_act:  float = 0.35,
        max_risk_to_act:        float = 0.80,
    ) -> None:
        self.body                 = body
        self.unknown_safety_rule  = unknown_safety_rule
        self.min_confidence       = min_confidence_to_act
        self.max_risk             = max_risk_to_act
        self._log:   List[ActionLogEntry] = []
        self._step   = 0
        self._blocked_count  = 0
        self._executed_count = 0

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def execute(
        self,
        action:         str,
        target:         str,
        belief_system   = None,
        env_response:   Optional[Dict] = None,
        force:          bool = False,
    ) -> Tuple[bool, BodyActionResult, ActionLogEntry]:
        """
        Validate and execute an action.

        Parameters
        ----------
        action        : Action string ("eat", "pick", "inspect", etc.)
        target        : Target object name
        belief_system : BeliefSystem for knowledge lookup
        env_response  : Pre-fetched environment response
        force         : Skip safety checks (use carefully)

        Returns
        -------
        (executed: bool, body_result: BodyActionResult, log_entry: ActionLogEntry)
        """
        t0 = time.time()
        self._step += 1

        body_action = self._parse_action(action)

        # Validate
        if not force:
            validation = self.validate(body_action, target, belief_system,
                                        env_response)
        else:
            validation = ValidationResult(ValidationStatus.ALLOWED, "forced")

        executed    = (validation.status in (ValidationStatus.ALLOWED,
                                              ValidationStatus.WARNED))
        body_result: Optional[BodyActionResult] = None
        outcome     = "blocked"
        reward      = 0.0

        if executed:
            body_result = self.body.act(body_action, target, env_response)
            outcome     = body_result.effect or body_result.status.value
            reward      = body_result.reward
            self._executed_count += 1
        else:
            # Create a synthetic "blocked" result
            body_result = BodyActionResult(
                action=body_action, object_name=target,
                status=ActionStatus.BLOCKED,
                effect="blocked", reward=0.0, confidence=1.0,
                observations={}, body_delta={},
                reason=validation.reason,
            )
            self._blocked_count += 1

        elapsed = (time.time() - t0) * 1000
        entry   = ActionLogEntry(
            step=self._step, action=action, target=target,
            validation=validation.status, executed=executed,
            body_result=body_result, env_response=env_response,
            outcome=outcome, reward=reward, elapsed_ms=elapsed,
        )
        self._log.append(entry)
        return executed, body_result, entry

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        action:       BodyAction,
        target:       str,
        belief_system = None,
        env_response: Optional[Dict] = None,
    ) -> ValidationResult:
        """
        Check whether an action is safe and physically possible.
        Returns ValidationResult with status and reason.
        """
        # Safe actions always pass
        if action in self.SAFE_ACTIONS:
            return ValidationResult(ValidationStatus.ALLOWED, "Safe action.")

        # Body capability checks
        if action == BodyAction.EAT and not self.body.state.alive:
            return ValidationResult(ValidationStatus.INVALID,
                                    "Body is not alive.")
        if action == BodyAction.PICK and not self.body.state.can_pick:
            reason = ("Hands full."
                      if len(self.body.state.inventory) >= self.body.state.max_carry
                      else "Too tired to pick.")
            return ValidationResult(ValidationStatus.INVALID, reason)

        # Knowledge-required actions: check beliefs
        if action in self.KNOWLEDGE_REQUIRED and belief_system is not None:
            key = f"{target}.edible"
            belief = belief_system.get(key)

            # Unknown safety rule: no belief → block
            if belief is None and self.unknown_safety_rule:
                return ValidationResult(
                    ValidationStatus.BLOCKED,
                    f"Unknown object '{target}': no edibility belief. "
                    f"Safety rule blocks action.",
                    risk_score=0.9,
                )

            # Low confidence → warn but allow
            if belief and belief.confidence < self.min_confidence:
                return ValidationResult(
                    ValidationStatus.WARNED,
                    f"Low confidence ({belief.confidence:.2f} < {self.min_confidence}). "
                    f"Proceeding with caution.",
                    risk_score=0.5,
                )

            # Known dangerous → block
            if belief and belief.value is False and belief.confidence >= 0.60:
                return ValidationResult(
                    ValidationStatus.BLOCKED,
                    f"'{target}' is known non-edible "
                    f"(conf={belief.confidence:.2f}). Action blocked.",
                    risk_score=1.0,
                )

            # Check toxic
            toxic_b = belief_system.get(f"{target}.toxic")
            if toxic_b and toxic_b.value is True and toxic_b.confidence >= 0.55:
                return ValidationResult(
                    ValidationStatus.BLOCKED,
                    f"'{target}' is known toxic (conf={toxic_b.confidence:.2f}). "
                    f"Action blocked.",
                    risk_score=1.0,
                )

        # Environmental risk check
        if env_response:
            env_risk = env_response.get("risk_score", 0.0)
            if env_risk >= self.max_risk:
                return ValidationResult(
                    ValidationStatus.BLOCKED,
                    f"Environment risk {env_risk:.2f} ≥ {self.max_risk}. Blocked.",
                    risk_score=env_risk,
                )

        return ValidationResult(ValidationStatus.ALLOWED,
                                 "All checks passed.", risk_score=0.1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_action(action_str: str) -> BodyAction:
        """Convert action string to BodyAction enum."""
        mapping = {
            "eat":     BodyAction.EAT,
            "pick":    BodyAction.PICK,
            "pickup":  BodyAction.PICK,
            "drop":    BodyAction.DROP,
            "inspect": BodyAction.INSPECT,
            "examine": BodyAction.INSPECT,
            "look":    BodyAction.LOOK,
            "scan":    BodyAction.LOOK,
            "move":    BodyAction.MOVE,
            "go":      BodyAction.MOVE,
            "wait":    BodyAction.WAIT,
            "rest":    BodyAction.WAIT,
        }
        return mapping.get(action_str.lower(), BodyAction.WAIT)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def recent_actions(self, n: int = 10) -> List[Dict]:
        return [e.to_dict() for e in self._log[-n:]]

    def success_rate(self) -> float:
        if not self._log:
            return 0.0
        successes = sum(
            1 for e in self._log
            if e.executed and e.reward > 0
        )
        return successes / max(len(self._log), 1)

    def block_rate(self) -> float:
        if not self._log:
            return 0.0
        return self._blocked_count / max(len(self._log), 1)

    def summary(self) -> Dict:
        return {
            "total_actions":   len(self._log),
            "executed":        self._executed_count,
            "blocked":         self._blocked_count,
            "success_rate":    round(self.success_rate(), 3),
            "block_rate":      round(self.block_rate(), 3),
        }

    def __repr__(self) -> str:
        return (f"ActionSystem(actions={len(self._log)}, "
                f"sr={self.success_rate():.0%}, "
                f"blocked={self._blocked_count})")
