"""
reasoning/consensus_engine.py
================================
Multi-Agent Consensus Engine

When agents hold different beliefs about the same fact, the consensus
engine aggregates them into a single authoritative community belief.

Consensus Strategies
--------------------
1. CONFIDENCE-WEIGHTED VOTE
   Each agent's belief is weighted by their confidence.
   Belief with highest weighted vote wins.

2. EVIDENCE-WEIGHTED VOTE
   Each agent's belief is weighted by their evidence count.
   More observed evidence = more influential.

3. TRUST-WEIGHTED VOTE
   Each agent's belief is weighted by their trust score
   (as seen from the calling agent's TrustSystem).

4. SUPERMAJORITY
   A belief wins only if it holds ≥ 66% of weighted votes.
   Otherwise → "uncertain" and schedule experiment.

Conflict Escalation
-------------------
If no strategy reaches consensus:
  → mark key as "contested"
  → schedule collaborative experiment
  → agents share task of investigating
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..world_model.belief_system import Belief, BeliefSystem


class ConsensusStrategy(str, Enum):
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    EVIDENCE_WEIGHTED   = "evidence_weighted"
    TRUST_WEIGHTED      = "trust_weighted"
    SUPERMAJORITY       = "supermajority"


@dataclass
class AgentVote:
    """One agent's vote on a belief."""
    agent_id:   str
    value:      Any
    confidence: float
    evidence:   float   = 1.0
    trust:      float   = 0.5   # as seen by the consensus caller


@dataclass
class ConsensusResult:
    """Output of one consensus round."""
    key:          str
    value:        Any
    confidence:   float
    strategy:     ConsensusStrategy
    n_votes:      int
    agreement:    float          # fraction of weighted votes for winning value
    contested:    bool
    notes:        str = ""
    timestamp:    float = field(default_factory=time.time)


class ConsensusEngine:
    """
    Aggregates beliefs from multiple agents into a consensus belief.

    Parameters
    ----------
    supermajority_threshold : Fraction of weighted votes needed to win outright.
    min_votes               : Minimum votes required to form consensus.
    """

    def __init__(
        self,
        supermajority_threshold: float = 0.60,
        min_votes:               int   = 2,
    ) -> None:
        self.supermajority  = supermajority_threshold
        self.min_votes      = min_votes
        self._results:      List[ConsensusResult] = []
        self._contested:    List[str] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def reach_consensus(
        self,
        key:      str,
        votes:    List[AgentVote],
        strategy: ConsensusStrategy = ConsensusStrategy.TRUST_WEIGHTED,
    ) -> ConsensusResult:
        """
        Given votes from multiple agents, compute the consensus belief.

        Parameters
        ----------
        key      : The belief key being resolved (e.g. "apple.edible").
        votes    : List of AgentVote objects.
        strategy : Which aggregation strategy to use.

        Returns
        -------
        ConsensusResult
        """
        if len(votes) < self.min_votes:
            return ConsensusResult(
                key=key, value=None, confidence=0.5,
                strategy=strategy, n_votes=len(votes),
                agreement=0.0, contested=True,
                notes=f"Insufficient votes ({len(votes)} < {self.min_votes})",
            )

        # Compute weights per strategy
        if strategy == ConsensusStrategy.CONFIDENCE_WEIGHTED:
            weights = [v.confidence for v in votes]
        elif strategy == ConsensusStrategy.EVIDENCE_WEIGHTED:
            weights = [v.evidence for v in votes]
        elif strategy == ConsensusStrategy.TRUST_WEIGHTED:
            weights = [v.confidence * v.trust for v in votes]
        else:  # SUPERMAJORITY — same as trust-weighted initially
            weights = [v.confidence * v.trust for v in votes]

        total_weight = sum(weights) + 1e-8

        # Tally votes per distinct value
        tally: Dict[str, float] = {}
        for vote, w in zip(votes, weights):
            val_key = str(vote.value).lower()
            tally[val_key] = tally.get(val_key, 0.0) + w

        # Find winning value
        winning_val_str = max(tally, key=tally.get)
        winning_weight  = tally[winning_val_str]
        agreement       = winning_weight / total_weight

        # Map back to Python value
        winning_value = next(
            (v.value for v in votes if str(v.value).lower() == winning_val_str),
            winning_val_str,
        )

        # Supermajority check
        contested = agreement < self.supermajority
        if strategy == ConsensusStrategy.SUPERMAJORITY and contested:
            if key not in self._contested:
                self._contested.append(key)
            notes = (f"No supermajority: top value '{winning_val_str}' "
                     f"has {agreement:.1%} (need {self.supermajority:.0%})")
        else:
            notes = (f"{strategy.value}: '{winning_val_str}' "
                     f"wins {agreement:.1%} of weighted votes")

        # Consensus confidence = agreement × mean confidence of winning side
        winning_votes = [v for v in votes if str(v.value).lower() == winning_val_str]
        mean_winning_conf = float(np.mean([v.confidence for v in winning_votes]))
        consensus_conf = agreement * mean_winning_conf

        result = ConsensusResult(
            key=key,
            value=winning_value if not contested else None,
            confidence=float(np.clip(consensus_conf, 0.0, 0.95)),
            strategy=strategy,
            n_votes=len(votes),
            agreement=agreement,
            contested=contested,
            notes=notes,
        )
        self._results.append(result)
        return result

    # ------------------------------------------------------------------
    # Convenience: build votes from belief systems
    # ------------------------------------------------------------------

    @staticmethod
    def votes_from_beliefs(
        key:             str,
        agent_beliefs:   Dict[str, BeliefSystem],
        trust_scores:    Optional[Dict[str, float]] = None,
    ) -> List[AgentVote]:
        """
        Build a list of AgentVotes from multiple agents' BeliefSystems.

        Parameters
        ----------
        key            : The belief key to collect votes on.
        agent_beliefs  : {agent_id: BeliefSystem} mapping.
        trust_scores   : {agent_id: trust_score} — if None, all trust=0.5.
        """
        votes = []
        for agent_id, bs in agent_beliefs.items():
            belief = bs.get(key)
            if belief is None:
                continue
            trust = (trust_scores or {}).get(agent_id, 0.5)
            votes.append(AgentVote(
                agent_id=agent_id,
                value=belief.value,
                confidence=belief.confidence,
                evidence=belief.total_evidence,
                trust=trust,
            ))
        return votes

    # ------------------------------------------------------------------
    # Merge consensus into a BeliefSystem
    # ------------------------------------------------------------------

    def apply_to_belief_system(
        self,
        result: ConsensusResult,
        target_bs: BeliefSystem,
        source: str = "consensus",
    ) -> None:
        """Apply a ConsensusResult to a BeliefSystem as a belief update."""
        if result.value is None or result.contested:
            return
        target_bs.update(
            result.key, result.value,
            source=source,
            weight=result.confidence,
            notes=f"consensus ({result.n_votes} agents, agreement={result.agreement:.2f})",
        )

    # ------------------------------------------------------------------
    # Contested keys
    # ------------------------------------------------------------------

    def get_contested_keys(self) -> List[str]:
        """Keys that currently have no consensus."""
        return list(self._contested)

    def pop_contested(self) -> Optional[str]:
        """Return next contested key for experimental resolution."""
        return self._contested.pop(0) if self._contested else None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        if not self._results:
            return {"n_results": 0, "contested": len(self._contested)}
        agreements = [r.agreement for r in self._results]
        strategies: Dict[str, int] = {}
        for r in self._results:
            strategies[r.strategy.value] = strategies.get(r.strategy.value, 0) + 1
        return {
            "n_results":      len(self._results),
            "mean_agreement": round(float(np.mean(agreements)), 3),
            "contested":      len(self._contested),
            "by_strategy":    strategies,
            "recent":         [
                {"key": r.key, "value": r.value,
                 "agreement": round(r.agreement, 2), "contested": r.contested}
                for r in self._results[-5:]
            ],
        }

    def __repr__(self) -> str:
        return (f"ConsensusEngine(results={len(self._results)}, "
                f"contested={len(self._contested)})")
