"""
reasoning/negotiation_engine.py
================================
Negotiation Engine

Enables agents to resolve belief disagreements through structured
argumentation, without requiring direct experiments.

Negotiation Protocol
--------------------
Two agents hold conflicting beliefs about the same fact:
  Agent A: apple.edible = True  (conf=0.85, evidence=5)
  Agent B: apple.edible = False (conf=0.60, evidence=3)

They exchange a series of arguments:
  Round 1 — A presents evidence: "I ate apple 5 times, all successes"
  Round 1 — B responds: "My category rule: this apple is unusual"
  Round 2 — A rebuts: "Category rule doesn't override direct evidence"
  Round 2 — B concedes partially: "Okay, maybe it's edible, conf=0.65"
  Round 3 — Consensus: apple.edible = True (merged confidence)

Argument Types
--------------
EVIDENCE      – "I have N direct observations supporting X"
ANALOGY       – "Similar object Y has this property"
AUTHORITY     – "Category rule says all food is edible"
COUNTEREXAMPLE– "But I have an exception: this specific case"
CONCESSION    – "I partially accept your argument; updating conf"

Convergence
-----------
Each round, agents update their beliefs toward each other based on:
  - argument strength (evidence count × confidence)
  - trust in the other agent
  - current belief certainty

After max_rounds or when |conf_A - conf_B| < tolerance, negotiation ends.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..world_model.belief_system import Belief, BeliefSystem


class ArgumentType(str, Enum):
    EVIDENCE       = "evidence"
    ANALOGY        = "analogy"
    AUTHORITY      = "authority"
    COUNTEREXAMPLE = "counterexample"
    CONCESSION     = "concession"


@dataclass
class Argument:
    """One argument presented by an agent during negotiation."""
    agent_id:    str
    arg_type:    ArgumentType
    claim:       Any          # what they're claiming (belief value)
    confidence:  float        # how confident they are
    evidence:    float        # evidence units backing this
    rationale:   str          # human-readable explanation
    strength:    float = 0.0  # computed: evidence × confidence

    def __post_init__(self) -> None:
        self.strength = self.evidence * self.confidence


@dataclass
class NegotiationRound:
    """One round of a negotiation session."""
    round_num:     int
    agent_a_arg:   Argument
    agent_b_arg:   Argument
    conf_a_before: float
    conf_b_before: float
    conf_a_after:  float
    conf_b_after:  float
    delta:         float      # |conf_a - conf_b| at end of round


@dataclass
class NegotiationResult:
    """Final result of a negotiation session."""
    key:           str
    agreed_value:  Any
    agreed_conf:   float
    rounds:        int
    converged:     bool
    agent_a_delta: float     # how much A's conf changed
    agent_b_delta: float     # how much B's conf changed
    notes:         str
    history:       List[NegotiationRound] = field(default_factory=list)
    timestamp:     float = field(default_factory=time.time)


class NegotiationEngine:
    """
    Orchestrates belief negotiation between two agents.

    Parameters
    ----------
    max_rounds   : Maximum rounds before forced termination.
    tolerance    : Confidence gap below which negotiation converges.
    learning_rate: How aggressively agents update during negotiation.
    """

    def __init__(
        self,
        max_rounds:    int   = 5,
        tolerance:     float = 0.10,
        learning_rate: float = 0.25,
    ) -> None:
        self.max_rounds    = max_rounds
        self.tolerance     = tolerance
        self.lr            = learning_rate
        self._sessions:    List[NegotiationResult] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def negotiate(
        self,
        key:        str,
        bs_a:       BeliefSystem,
        agent_a_id: str,
        trust_a:    float,
        bs_b:       BeliefSystem,
        agent_b_id: str,
        trust_b:    float,
    ) -> NegotiationResult:
        """
        Run a negotiation between agent A and agent B on belief `key`.

        Parameters
        ----------
        key        : Belief key to negotiate (e.g. "apple.edible").
        bs_a, bs_b : Each agent's BeliefSystem.
        trust_a    : B's trust in A (how much B should update toward A).
        trust_b    : A's trust in B (how much A should update toward B).

        Returns
        -------
        NegotiationResult with convergence info and history.
        """
        belief_a = bs_a.get(key)
        belief_b = bs_b.get(key)

        if belief_a is None and belief_b is None:
            return NegotiationResult(
                key=key, agreed_value=None, agreed_conf=0.5,
                rounds=0, converged=False,
                agent_a_delta=0.0, agent_b_delta=0.0,
                notes="Both agents have no belief on this key."
            )

        # One agent lacks belief → just adopt the other's
        if belief_a is None:
            if belief_b:
                bs_a.update(key, belief_b.value, "inference",
                            weight=trust_b * belief_b.confidence)
            return NegotiationResult(
                key=key, agreed_value=belief_b.value if belief_b else None,
                agreed_conf=belief_b.confidence * trust_b if belief_b else 0.5,
                rounds=1, converged=True,
                agent_a_delta=0.0, agent_b_delta=0.0,
                notes=f"{agent_a_id} had no belief; adopted {agent_b_id}'s."
            )

        if belief_b is None:
            bs_b.update(key, belief_a.value, "inference",
                        weight=trust_a * belief_a.confidence)
            return NegotiationResult(
                key=key, agreed_value=belief_a.value,
                agreed_conf=belief_a.confidence * trust_a,
                rounds=1, converged=True,
                agent_a_delta=0.0, agent_b_delta=0.0,
                notes=f"{agent_b_id} had no belief; adopted {agent_a_id}'s."
            )

        # Both hold beliefs — potentially conflicting
        conf_a_init = belief_a.confidence
        conf_b_init = belief_b.confidence
        history: List[NegotiationRound] = []

        for round_num in range(1, self.max_rounds + 1):
            conf_a_before = belief_a.confidence
            conf_b_before = belief_b.confidence

            # Build arguments
            arg_a = self._build_argument(agent_a_id, belief_a)
            arg_b = self._build_argument(agent_b_id, belief_b)

            # Values agree — just merge confidences
            if Belief._values_agree(belief_a.value, belief_b.value):
                new_conf_a = self._merge_confidence(
                    belief_a.confidence, belief_b.confidence, trust_b, self.lr
                )
                new_conf_b = self._merge_confidence(
                    belief_b.confidence, belief_a.confidence, trust_a, self.lr
                )
                belief_a = bs_a.update(key, belief_a.value, "inference",
                                        weight=new_conf_a)
                belief_b = bs_b.update(key, belief_b.value, "inference",
                                        weight=new_conf_b)
                history.append(NegotiationRound(
                    round_num=round_num,
                    agent_a_arg=arg_a, agent_b_arg=arg_b,
                    conf_a_before=conf_a_before, conf_b_before=conf_b_before,
                    conf_a_after=belief_a.confidence,
                    conf_b_after=belief_b.confidence,
                    delta=0.0,
                ))
                break

            # Values conflict — persuasion
            new_conf_a, new_conf_b = self._persuade(
                belief_a, belief_b, trust_a, trust_b
            )

            belief_a = bs_a.update(key, belief_a.value, "inference",
                                    weight=new_conf_a)
            belief_b = bs_b.update(key, belief_b.value, "inference",
                                    weight=new_conf_b)

            gap = abs(belief_a.confidence - belief_b.confidence)
            history.append(NegotiationRound(
                round_num=round_num,
                agent_a_arg=arg_a, agent_b_arg=arg_b,
                conf_a_before=conf_a_before, conf_b_before=conf_b_before,
                conf_a_after=belief_a.confidence,
                conf_b_after=belief_b.confidence,
                delta=gap,
            ))

            if gap <= self.tolerance:
                break

        # Determine final agreed state
        converged = (history[-1].delta <= self.tolerance
                     if history and not Belief._values_agree(
                         belief_a.value, belief_b.value) else True)

        # Winner: whichever belief has higher confidence
        if belief_a.confidence >= belief_b.confidence:
            agreed_value = belief_a.value
            agreed_conf  = belief_a.confidence
        else:
            agreed_value = belief_b.value
            agreed_conf  = belief_b.confidence

        notes = (
            f"Converged after {len(history)} rounds. "
            f"A: {conf_a_init:.3f}→{belief_a.confidence:.3f}, "
            f"B: {conf_b_init:.3f}→{belief_b.confidence:.3f}"
        )
        if not converged:
            notes += " [did not fully converge]"

        result = NegotiationResult(
            key=key,
            agreed_value=agreed_value,
            agreed_conf=agreed_conf,
            rounds=len(history),
            converged=converged,
            agent_a_delta=belief_a.confidence - conf_a_init,
            agent_b_delta=belief_b.confidence - conf_b_init,
            notes=notes,
            history=history,
        )
        self._sessions.append(result)
        return result

    # ------------------------------------------------------------------
    # Argument construction
    # ------------------------------------------------------------------

    def _build_argument(self, agent_id: str, belief: Belief) -> Argument:
        """Build the strongest available argument for this belief."""
        # Choose argument type based on evidence depth
        if belief.total_evidence >= 4:
            arg_type  = ArgumentType.EVIDENCE
            rationale = (f"{agent_id} has {belief.total_evidence:.1f} evidence units "
                         f"supporting {belief.value}")
        elif belief.source == "inference":
            arg_type  = ArgumentType.ANALOGY
            rationale = f"By analogy with similar objects, {belief.value} seems right"
        elif belief.source in ("abstraction", "consensus"):
            arg_type  = ArgumentType.AUTHORITY
            rationale = f"Category/consensus rule supports {belief.value}"
        else:
            arg_type  = ArgumentType.EVIDENCE
            rationale = f"Direct observation suggests {belief.value}"

        return Argument(
            agent_id=agent_id,
            arg_type=arg_type,
            claim=belief.value,
            confidence=belief.confidence,
            evidence=belief.total_evidence,
            rationale=rationale,
        )

    # ------------------------------------------------------------------
    # Persuasion logic
    # ------------------------------------------------------------------

    def _persuade(
        self,
        belief_a: Belief,
        belief_b: Belief,
        trust_a:  float,
        trust_b:  float,
    ) -> Tuple[float, float]:
        """
        Each agent's confidence moves toward the other if the other is
        more evidentially supported and trusted.
        """
        strength_a = belief_a.total_evidence * belief_a.confidence * trust_b
        strength_b = belief_b.total_evidence * belief_b.confidence * trust_a

        total = strength_a + strength_b + 1e-8
        weight_a = strength_a / total
        weight_b = strength_b / total

        # The stronger argument "wins" by pulling the weaker belief toward it
        # But both update — nobody stays completely rigid
        if weight_a >= weight_b:
            # A is more convincing: A's conf rises, B's falls (but not to zero)
            new_conf_a = min(0.95, belief_a.confidence + self.lr * weight_a * 0.15)
            new_conf_b = max(0.10, belief_b.confidence - self.lr * weight_a * 0.20)
        else:
            new_conf_a = max(0.10, belief_a.confidence - self.lr * weight_b * 0.20)
            new_conf_b = min(0.95, belief_b.confidence + self.lr * weight_b * 0.15)

        return new_conf_a, new_conf_b

    def _merge_confidence(
        self,
        own_conf:  float,
        peer_conf: float,
        trust:     float,
        lr:        float,
    ) -> float:
        """Merge two confidence values weighted by trust."""
        merged = own_conf + lr * trust * (peer_conf - own_conf)
        return float(np.clip(merged, 0.1, 0.95))

    # ------------------------------------------------------------------
    # Batch negotiation
    # ------------------------------------------------------------------

    def negotiate_all_conflicts(
        self,
        bs_a:       BeliefSystem,
        agent_a_id: str,
        trust_a:    float,
        bs_b:       BeliefSystem,
        agent_b_id: str,
        trust_b:    float,
        min_conf_threshold: float = 0.50,
    ) -> List[NegotiationResult]:
        """
        Negotiate on all keys where A and B have conflicting confident beliefs.
        Returns list of results.
        """
        results = []

        # Find conflicting keys
        keys_a = {b.key for b in bs_a.all_beliefs(min_conf=min_conf_threshold)}
        keys_b = {b.key for b in bs_b.all_beliefs(min_conf=min_conf_threshold)}
        common = keys_a & keys_b

        for key in common:
            ba = bs_a.get(key)
            bb = bs_b.get(key)
            if ba is None or bb is None:
                continue
            if Belief._values_agree(ba.value, bb.value):
                continue   # no conflict
            if ba.confidence < min_conf_threshold or bb.confidence < min_conf_threshold:
                continue

            result = self.negotiate(key, bs_a, agent_a_id, trust_a,
                                    bs_b, agent_b_id, trust_b)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        if not self._sessions:
            return {"sessions": 0}
        converged = sum(1 for s in self._sessions if s.converged)
        rounds    = [s.rounds for s in self._sessions]
        return {
            "sessions":         len(self._sessions),
            "converged":        converged,
            "convergence_rate": round(converged / len(self._sessions), 3),
            "mean_rounds":      round(float(np.mean(rounds)), 2),
            "recent":           [
                {"key": s.key, "converged": s.converged,
                 "rounds": s.rounds, "value": s.agreed_value}
                for s in self._sessions[-5:]
            ],
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (f"NegotiationEngine(sessions={s['sessions']}, "
                f"convergence={s.get('convergence_rate',0):.0%})")
