"""
game/game_loop.py
==================
Game Interaction Loop

Extends the core InteractionLoop to work with live game adapters.

The game loop:
  1. read_observation(adapter)
  2. vision_analysis(screenshot) [optional]
  3. language_learning(observation)
  4. update_survival_goals(observation)
  5. select_top_goal()
  6. think() + decide() [CogniField cognition]
  7. act_in_game(adapter, action)
  8. observe_result()
  9. learn_from_outcome()

Extra features vs plain InteractionLoop:
  - Integrates adapter, vision system, language learner, survival goals
  - Tracks Minecraft-specific metrics (health/hunger over time)
  - Generates episode summaries for learning analysis
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .base_adapter import GameAdapter, GameObservation, ActionType
from .survival_goals import SurvivalGoalManager, SurvivalGoal
from .language_learner import LanguageLearner
from ..vision.vision_system import VisionSystem
from ..core.interaction_loop import InteractionLoop, EpisodeStep


# ---------------------------------------------------------------------------
# Game step result
# ---------------------------------------------------------------------------

@dataclass
class GameStep:
    """One step of the game loop with full context."""
    step:              int
    observation:       GameObservation
    active_goal:       Optional[SurvivalGoal]
    query:             str
    loop_result:       Optional[EpisodeStep]   # CogniField loop result
    action_sent:       bool
    game_action:       Dict
    new_concepts:      int               # language learning updates
    elapsed_ms:        float
    timestamp:         float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        lr = self.loop_result
        return {
            "step":           self.step,
            "goal":           self.active_goal.name if self.active_goal else None,
            "query":          self.query,
            "decision":       lr.thinking_decision if lr else "none",
            "executed":       lr.action_executed   if lr else False,
            "effect":         lr.effect            if lr else "",
            "reward":         lr.reward            if lr else 0.0,
            "action_sent":    self.action_sent,
            "new_concepts":   self.new_concepts,
            "health":         self.observation.health,
            "hunger":         self.observation.hunger,
            "position":       self.observation.position,
            "elapsed_ms":     round(self.elapsed_ms, 1),
        }


# ---------------------------------------------------------------------------
# Game Loop
# ---------------------------------------------------------------------------

class GameLoop:
    """
    Orchestrates full game interaction cycle.

    Parameters
    ----------
    adapter           : Connected GameAdapter instance.
    interaction_loop  : CogniField InteractionLoop.
    vision            : Optional VisionSystem for screenshot analysis.
    language_learner  : Optional LanguageLearner for concept extraction.
    survival_goals    : Optional SurvivalGoalManager.
    verbose           : Print each step to stdout.
    """

    def __init__(
        self,
        adapter:          GameAdapter,
        interaction_loop: InteractionLoop,
        vision:           Optional[VisionSystem]      = None,
        language_learner: Optional[LanguageLearner]   = None,
        survival_goals:   Optional[SurvivalGoalManager] = None,
        verbose:          bool = False,
    ) -> None:
        self.adapter          = adapter
        self.loop             = interaction_loop
        self.vision           = vision
        self.lang             = language_learner
        self.survival         = survival_goals or SurvivalGoalManager()
        self.verbose          = verbose

        self._step_count      = 0
        self._history:        List[GameStep] = []
        self._health_history: List[float] = []
        self._hunger_history: List[float] = []

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step_from_game(self, screenshot: Optional[bytes] = None) -> GameStep:
        """
        Run one full game loop step.

        Parameters
        ----------
        screenshot : Optional raw screenshot bytes for vision analysis.

        Returns
        -------
        GameStep with all loop results.
        """
        t0 = time.time()
        self._step_count += 1

        # ── 1. Read observation from game ──
        obs = self.adapter.get_observation()
        self._health_history.append(obs.health)
        self._hunger_history.append(obs.hunger)

        # ── 2. Vision analysis (optional) ──
        if self.vision and screenshot:
            reading = self.vision.analyze(screenshot)
            # Override obs values from vision if more precise
            obs.health = reading.health_hearts
            obs.hunger = reading.hunger_drumsticks

        # ── 3. Language learning ──
        new_concepts = 0
        if self.lang:
            concepts = self.lang.process_observation(obs)
            new_concepts = len(concepts)

        # ── 4. Update survival goals ──
        goals = self.survival.update(obs)
        top_goal = self.survival.top_goal()

        # ── 5. Build query from top goal ──
        query = self._goal_to_query(top_goal, obs)

        if self.verbose:
            print(f"  [GameStep {self._step_count}] "
                  f"health={obs.health:.1f}/20 hunger={obs.hunger:.1f}/20 "
                  f"goal={top_goal.name if top_goal else 'none'!r}")

        # ── 6. CogniField loop step ──
        loop_result: Optional[EpisodeStep] = None
        try:
            loop_result = self.loop.step(query, env_fn=None)
        except Exception as e:
            if self.verbose:
                print(f"    [loop error] {e}")

        # ── 7. Send action to game ──
        game_action, action_sent = self._translate_and_send(
            top_goal, loop_result, obs
        )

        # ── 8. Complete goal if executed ──
        if loop_result and loop_result.action_executed and top_goal:
            self.survival.complete_goal(top_goal.name)

        elapsed = (time.time() - t0) * 1000
        game_step = GameStep(
            step=self._step_count,
            observation=obs,
            active_goal=top_goal,
            query=query,
            loop_result=loop_result,
            action_sent=action_sent,
            game_action=game_action,
            new_concepts=new_concepts,
            elapsed_ms=elapsed,
        )
        self._history.append(game_step)

        if self.verbose and loop_result:
            icon = "✓" if loop_result.action_executed else "⊘"
            print(f"    {icon} {loop_result.intent_action}"
                  f"({loop_result.intent_target}) "
                  f"→ {loop_result.effect} "
                  f"[{loop_result.thinking_decision} "
                  f"{loop_result.thinking_conf:.0%}]")

        return game_step

    def run_episode(
        self,
        n_steps: int,
        screenshot_fn=None,
    ) -> List[GameStep]:
        """
        Run n_steps of the game loop.

        Parameters
        ----------
        n_steps       : Number of steps to run.
        screenshot_fn : Optional callable() → bytes for vision input.

        Returns
        -------
        List of GameStep records.
        """
        steps = []
        for _ in range(n_steps):
            screenshot = screenshot_fn() if screenshot_fn else None
            gs = self.step_from_game(screenshot)
            steps.append(gs)
            # Stop if agent died
            if gs.observation.health <= 0:
                if self.verbose:
                    print(f"  ⚠ Agent died at step {self._step_count}.")
                break
        return steps

    # ------------------------------------------------------------------
    # Action translation
    # ------------------------------------------------------------------

    def _goal_to_query(
        self,
        goal: Optional[SurvivalGoal],
        obs:  GameObservation,
    ) -> str:
        """Build a CogniField query from the active survival goal."""
        if goal is None:
            return "inspect environment"
        return goal.to_query()

    def _translate_and_send(
        self,
        goal:        Optional[SurvivalGoal],
        loop_result: Optional[EpisodeStep],
        obs:         GameObservation,
    ) -> tuple:
        """
        Translate the CogniField decision into a game action and send it.
        Returns (game_action_dict, sent_bool).
        """
        if loop_result is None or not loop_result.action_executed:
            # Fall back to safe action: inspect or wait
            action = {"type": ActionType.LOOK}
            return action, self.adapter.send_action(action)

        cf_action = loop_result.intent_action
        cf_target = loop_result.intent_target

        # Map CogniField actions to game actions
        if cf_action == "eat":
            # Convert target name to mc_id if possible
            mc_id = self._name_to_mc_id(cf_target, obs)
            action = {"type": ActionType.USE, "target": mc_id or cf_target}

        elif cf_action == "pick":
            action = {"type": ActionType.USE, "target": cf_target}

        elif cf_action in ("move", "explore", "flee"):
            direction = cf_target if cf_target in (
                "north","south","east","west","forward","back"
            ) else "forward"
            action = {"type": ActionType.MOVE, "direction": direction, "speed": 1.0}

        elif cf_action == "inspect":
            action = {"type": ActionType.LOOK}

        elif cf_action == "break":
            action = {"type": ActionType.ATTACK}

        else:
            action = {"type": ActionType.LOOK}

        sent = self.adapter.send_action(action)
        return action, sent

    @staticmethod
    def _name_to_mc_id(name: str, obs: GameObservation) -> Optional[str]:
        """Try to find the Minecraft ID for a plain name in current inventory."""
        for item in obs.inventory:
            item_name = item.item_id.split(":")[-1].replace("_", " ")
            if name.lower() in item_name.lower():
                return item.item_id
        return None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def mean_health(self) -> float:
        if not self._health_history:
            return 20.0
        return float(np.mean(self._health_history))

    def mean_hunger(self) -> float:
        if not self._hunger_history:
            return 20.0
        return float(np.mean(self._hunger_history))

    def survival_rate(self) -> float:
        """Fraction of steps where agent was alive (health > 0)."""
        if not self._history:
            return 1.0
        alive = sum(1 for s in self._history if s.observation.health > 0)
        return alive / len(self._history)

    def summary(self) -> Dict:
        return {
            "total_steps":    self._step_count,
            "mean_health":    round(self.mean_health(), 1),
            "mean_hunger":    round(self.mean_hunger(), 1),
            "survival_rate":  round(self.survival_rate(), 3),
            "goals_summary":  self.survival.summary(),
            "lang_summary":   self.lang.summary() if self.lang else {},
            "adapter":        self.adapter.summary(),
        }

    def recent_steps(self, n: int = 5) -> List[Dict]:
        return [s.to_dict() for s in self._history[-n:]]

    def __repr__(self) -> str:
        return (f"GameLoop(steps={self._step_count}, "
                f"health={self.mean_health():.1f}, "
                f"sr={self.survival_rate():.0%})")
