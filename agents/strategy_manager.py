"""
agents/strategy_manager.py
============================
Strategy Manager

Dynamically switches between named strategies based on observed
performance. Implements the explore–exploit tradeoff and more.

Strategies
----------
EXPLORE         : Prioritise novelty; accept higher risk; low confidence
                  threshold for acting.
EXPLOIT         : Focus on known-good actions; conservative; high confidence
                  threshold.
VERIFY          : Before acting, gather more evidence. Caution + experiment.
RECOVER         : After repeated failures, reset priors and restart cautiously.
COOPERATIVE     : Lean heavily on shared beliefs; trust peers more.
INDEPENDENT     : Ignore peer messages; rely only on private beliefs.

Switching Logic
---------------
After every `eval_freq` steps, compute:
  - recent_success_rate
  - recent_novelty
  - consecutive_failures
  - peer_agreement_rate

Then apply switching rules:
  sr < FAIL_THRESHOLD → switch to RECOVER
  sr > WIN_THRESHOLD and novelty < LOW_NOVELTY → switch to EXPLOIT
  sr < MED_THRESHOLD and novelty > HIGH_NOVELTY → switch to EXPLORE
  consecutive_failures >= MAX_FAILURES → RECOVER
  etc.

Each strategy tweaks:
  - novelty_threshold
  - risk_tolerance
  - min_conf_to_share
  - min_trust_to_adopt
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class Strategy(str, Enum):
    EXPLORE      = "explore"
    EXPLOIT      = "exploit"
    VERIFY       = "verify"
    RECOVER      = "recover"
    COOPERATIVE  = "cooperative"
    INDEPENDENT  = "independent"


@dataclass
class StrategyConfig:
    """Parameter settings for one strategy."""
    novelty_threshold:   float
    risk_tolerance:      float
    min_conf_to_share:   float
    min_trust_to_adopt:  float
    share_beliefs_freq:  int
    description:         str


STRATEGY_CONFIGS: Dict[Strategy, StrategyConfig] = {
    Strategy.EXPLORE: StrategyConfig(
        novelty_threshold=0.25,  risk_tolerance=0.45,
        min_conf_to_share=0.60,  min_trust_to_adopt=0.40,
        share_beliefs_freq=2,    description="Maximise novelty; accept risk",
    ),
    Strategy.EXPLOIT: StrategyConfig(
        novelty_threshold=0.55,  risk_tolerance=0.25,
        min_conf_to_share=0.75,  min_trust_to_adopt=0.60,
        share_beliefs_freq=4,    description="Use known-good actions; be conservative",
    ),
    Strategy.VERIFY: StrategyConfig(
        novelty_threshold=0.50,  risk_tolerance=0.20,
        min_conf_to_share=0.80,  min_trust_to_adopt=0.55,
        share_beliefs_freq=3,    description="Gather evidence before acting",
    ),
    Strategy.RECOVER: StrategyConfig(
        novelty_threshold=0.60,  risk_tolerance=0.15,
        min_conf_to_share=0.85,  min_trust_to_adopt=0.65,
        share_beliefs_freq=5,    description="Reset and restart cautiously after failure",
    ),
    Strategy.COOPERATIVE: StrategyConfig(
        novelty_threshold=0.40,  risk_tolerance=0.35,
        min_conf_to_share=0.55,  min_trust_to_adopt=0.35,
        share_beliefs_freq=1,    description="Trust peers heavily; share frequently",
    ),
    Strategy.INDEPENDENT: StrategyConfig(
        novelty_threshold=0.40,  risk_tolerance=0.35,
        min_conf_to_share=0.90,  min_trust_to_adopt=0.80,
        share_beliefs_freq=6,    description="Ignore peers; rely only on own beliefs",
    ),
}


@dataclass
class StrategySwitchEvent:
    """Record of one strategy switch."""
    step:      int
    from_strat: Strategy
    to_strat:   Strategy
    reason:     str
    sr_before:  float
    timestamp:  float = field(default_factory=time.time)


class StrategyManager:
    """
    Tracks performance and switches agent strategy dynamically.

    Parameters
    ----------
    initial_strategy  : Starting strategy.
    eval_freq         : Steps between strategy evaluations.
    fail_threshold    : Success rate below this triggers RECOVER.
    win_threshold     : Success rate above this allows EXPLOIT.
    max_consec_fails  : Consecutive failures before forcing RECOVER.
    """

    def __init__(
        self,
        initial_strategy: Strategy = Strategy.EXPLORE,
        eval_freq:        int      = 8,
        fail_threshold:   float    = 0.25,
        win_threshold:    float    = 0.70,
        max_consec_fails: int      = 4,
    ) -> None:
        self.current   = initial_strategy
        self.eval_freq = eval_freq
        self.fail_thr  = fail_threshold
        self.win_thr   = win_threshold
        self.max_fails = max_consec_fails

        self._history: List[StrategySwitchEvent] = []
        self._step_log: deque = deque(maxlen=100)
        self._consec_failures = 0
        self._steps           = 0

        # Time spent in each strategy
        self._time_in_strategy: Dict[Strategy, int] = {s: 0 for s in Strategy}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        step:    int,
        success: bool,
        reward:  float,
        novelty: float,
    ) -> None:
        self._steps = step
        self._time_in_strategy[self.current] += 1
        self._step_log.append({
            "step": step, "success": success,
            "reward": reward, "novelty": novelty,
        })
        if success:
            self._consec_failures = 0
        else:
            self._consec_failures += 1

    # ------------------------------------------------------------------
    # Evaluation + switching
    # ------------------------------------------------------------------

    def evaluate(
        self,
        step:             int,
        peer_agreement:   float = 0.5,
        force:            Optional[Strategy] = None,
    ) -> Optional[StrategySwitchEvent]:
        """
        Evaluate whether to switch strategy.

        Parameters
        ----------
        step           : Current step number.
        peer_agreement : Fraction of shared beliefs agents agree on.
        force          : Force a specific strategy (ignores heuristics).

        Returns
        -------
        StrategySwitchEvent if strategy changed, else None.
        """
        if step % self.eval_freq != 0 and force is None:
            return None

        recent = list(self._step_log)[-self.eval_freq:]
        if not recent:
            return None

        sr      = float(np.mean([r["success"] for r in recent]))
        nov     = float(np.mean([r["novelty"] for r in recent]))
        new_strat: Optional[Strategy] = force
        reason   = ""

        if force is None:
            # Switching rules (checked in priority order)
            if self._consec_failures >= self.max_fails:
                new_strat = Strategy.RECOVER
                reason    = f"{self._consec_failures} consecutive failures"
            elif sr <= self.fail_thr:
                new_strat = Strategy.RECOVER
                reason    = f"success_rate={sr:.3f} < fail_threshold={self.fail_thr}"
            elif sr > self.win_thr and nov < 0.20:
                new_strat = Strategy.EXPLOIT
                reason    = f"sr={sr:.3f} high, novelty={nov:.3f} low → exploit"
            elif sr > self.win_thr and nov > 0.40:
                new_strat = Strategy.EXPLORE
                reason    = f"sr={sr:.3f} high, novelty={nov:.3f} high → keep exploring"
            elif sr < 0.50 and peer_agreement > 0.75:
                new_strat = Strategy.COOPERATIVE
                reason    = f"sr={sr:.3f} struggling, peers agree → cooperate"
            elif sr < 0.50 and nov > 0.35:
                new_strat = Strategy.VERIFY
                reason    = f"sr={sr:.3f} low, novelty={nov:.3f} high → verify first"

        if new_strat is None or new_strat == self.current:
            return None

        event = StrategySwitchEvent(
            step=step, from_strat=self.current,
            to_strat=new_strat, reason=reason, sr_before=sr,
        )
        self._history.append(event)
        self.current = new_strat
        self._consec_failures = 0   # reset on switch
        return event

    # ------------------------------------------------------------------
    # Applying strategy to agent
    # ------------------------------------------------------------------

    def apply_to_agent(self, agent) -> None:
        """Apply current strategy parameters to an agent object."""
        cfg = STRATEGY_CONFIGS[self.current]
        agent.cfg.novelty_threshold = cfg.novelty_threshold
        agent.cfg.risk_tolerance    = cfg.risk_tolerance
        if hasattr(agent, "risk_engine"):
            agent.risk_engine.risk_tolerance = cfg.risk_tolerance
        if hasattr(agent, "cfg_v6"):
            agent.cfg_v6.min_conf_to_share   = cfg.min_conf_to_share
            agent.cfg_v6.min_trust_to_adopt  = cfg.min_trust_to_adopt
            agent.cfg_v6.share_beliefs_freq  = cfg.share_beliefs_freq

    def get_config(self) -> StrategyConfig:
        return STRATEGY_CONFIGS[self.current]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def recent_success_rate(self, window: int = 10) -> float:
        recent = list(self._step_log)[-window:]
        if not recent:
            return 0.5
        return float(np.mean([r["success"] for r in recent]))

    def switches(self) -> int:
        return len(self._history)

    def summary(self) -> Dict:
        return {
            "current_strategy":  self.current.value,
            "description":       STRATEGY_CONFIGS[self.current].description,
            "switches":          len(self._history),
            "consec_failures":   self._consec_failures,
            "recent_sr":         round(self.recent_success_rate(), 3),
            "time_per_strategy": {s.value: t for s, t in self._time_in_strategy.items() if t > 0},
            "switch_history":    [
                {"step": e.step, "from": e.from_strat.value,
                 "to": e.to_strat.value, "reason": e.reason}
                for e in self._history[-5:]
            ],
        }

    def __repr__(self) -> str:
        return (f"StrategyManager(current={self.current.value}, "
                f"switches={self.switches()}, "
                f"consec_fails={self._consec_failures})")
