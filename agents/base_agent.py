"""
agent/agent.py
==============
CogniField Agent

Integrates all modules into a perceive → encode → reason → act → learn loop.

Loop:
  1. observe(input)   → raw input (text / image / audio)
  2. encode(input)    → latent vector
  3. reason(vec)      → solution vector + error analysis
  4. act(action)      → environment feedback
  5. update(feedback) → memory + loss update
  6. curiosity_check  → explore if novel

The agent maintains a unified latent representation of its world:
texts, images, and environment states all live in the same frequency space,
enabling cross-modal reasoning (e.g., linking "apple" text to apple image).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..encoder.text_encoder   import TextEncoder
from ..encoder.image_encoder  import ImageEncoder
from ..encoder.audio_encoder  import AudioEncoder
from ..latent_space.frequency_space import FrequencySpace, ComposeMode
from ..memory.memory_store    import MemoryStore
from ..reasoning.reasoning_engine import ReasoningEngine
from ..language.structure_checker import StructureChecker
from ..curiosity.curiosity_engine  import CuriosityEngine
from ..loss.loss_system        import LossSystem, LossConfig
from ..environment.simple_env  import SimpleEnv


@dataclass
class AgentConfig:
    """Hyperparameters for the CogniField agent."""
    dim:                int   = 128
    novelty_threshold:  float = 0.4
    reasoning_threshold: float = 0.65
    max_retries:        int   = 5
    memory_size:        int   = 5_000
    decay_rate:         float = 0.003
    seed:               int   = 42


@dataclass
class Perception:
    """Result of encoding one input."""
    raw:      Any
    modality: str
    vector:   np.ndarray
    label:    str


@dataclass
class AgentStep:
    """Record of one agent perceive-reason-act cycle."""
    step:         int
    perception:   Perception
    reasoning_score: float
    action:       Optional[str]
    env_reward:   Optional[float]
    loss:         float
    novel:        bool
    elapsed_ms:   float


class CogniFieldAgent:
    """
    Full CogniField agent integrating all modules.

    Parameters
    ----------
    config : AgentConfig
    env    : SimpleEnv instance (or None to skip environment)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        env: Optional[SimpleEnv] = None,
    ) -> None:
        self.cfg = config or AgentConfig()
        cfg = self.cfg

        print("  Initialising CogniField Agent...")

        # Shared latent space
        self.space = FrequencySpace(dim=cfg.dim)

        # Encoders
        self.text_enc  = TextEncoder(dim=cfg.dim, seed=cfg.seed)
        self.image_enc = ImageEncoder(dim=cfg.dim, seed=cfg.seed)
        self.audio_enc = AudioEncoder(dim=cfg.dim, seed=cfg.seed)

        # Fit text encoder on bootstrap corpus
        self.text_enc.fit()

        # Shared memory
        self.memory = MemoryStore(
            dim=cfg.dim,
            max_size=cfg.memory_size,
            decay_rate=cfg.decay_rate,
            seed=cfg.seed,
        )

        # Reasoning
        self.reasoning = ReasoningEngine(
            space=self.space,
            memory=self.memory,
            max_retries=cfg.max_retries,
            threshold=cfg.reasoning_threshold,
            seed=cfg.seed,
        )

        # Language checker
        self.checker = StructureChecker()

        # Curiosity
        self.curiosity = CuriosityEngine(
            space=self.space,
            memory=self.memory,
            novelty_threshold=cfg.novelty_threshold,
            seed=cfg.seed,
        )

        # Loss
        self.loss_sys = LossSystem(
            config=LossConfig(w_error=1.0, w_novelty=0.3),
            space=self.space,
        )

        # Environment
        self.env = env

        # Agent state
        self._step_count   = 0
        self._step_log:    List[AgentStep] = []
        self._current_vec: Optional[np.ndarray] = None

        # Modality alignment matrices (text → image, text → audio)
        self._align_img: Optional[np.ndarray] = None
        self._align_aud: Optional[np.ndarray] = None

        print("  Agent ready.\n")

    # ------------------------------------------------------------------
    # 1. Observe (encode)
    # ------------------------------------------------------------------

    def observe(
        self,
        raw_input: Any,
        modality: str = "text",
        label: str = "",
    ) -> Perception:
        """
        Encode a raw input into a latent vector.

        Parameters
        ----------
        raw_input : str | np.ndarray | path
        modality  : "text" | "image" | "audio"
        label     : Optional human-readable label.
        """
        if modality == "text":
            vec = self.text_enc.encode(str(raw_input))
            if not label:
                words = str(raw_input).strip().split()
                label = words[0] if words else "text"
        elif modality == "image":
            vec = self.image_enc.encode(raw_input)
            label = label or "image"
        elif modality == "audio":
            vec = self.audio_enc.encode(raw_input)
            label = label or "audio"
        else:
            raise ValueError(f"Unknown modality: {modality}")

        self._current_vec = vec
        return Perception(raw=raw_input, modality=modality,
                          vector=vec, label=label)

    # ------------------------------------------------------------------
    # 2. Reason
    # ------------------------------------------------------------------

    def reason(
        self,
        perception: Perception,
        target_text: Optional[str] = None,
        context: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Run the reasoning loop on a perception.

        Parameters
        ----------
        perception  : From observe().
        target_text : Expected output (if known).
        context     : Additional context texts.

        Returns
        -------
        (solution_vec, score)
        """
        input_vec = perception.vector
        target_vec = (self.text_enc.encode(target_text)
                      if target_text else input_vec)
        ctx_vecs = ([self.text_enc.encode(c) for c in context]
                    if context else [])

        result = self.reasoning.reason(
            input_vec, target_vec, ctx_vecs, verbose=verbose
        )
        return result.solution_vec, result.score

    # ------------------------------------------------------------------
    # 3. Act (environment)
    # ------------------------------------------------------------------

    def act(
        self,
        action: str,
        *args,
    ) -> Optional[Dict]:
        """Execute an action in the environment. Returns feedback."""
        if self.env is None:
            return None
        return self.env.step(action, *args)

    # ------------------------------------------------------------------
    # 4. Update (memory + loss)
    # ------------------------------------------------------------------

    def update(
        self,
        perception: Perception,
        solution_vec: np.ndarray,
        target_text: Optional[str] = None,
        env_feedback: Optional[Dict] = None,
    ) -> float:
        """
        Update memory and compute loss after one step.

        Returns total loss.
        """
        target_vec = (self.text_enc.encode(target_text)
                      if target_text else perception.vector)

        # Check novelty
        novelty = self.curiosity.detect_novelty(perception.vector)
        if novelty >= self.cfg.novelty_threshold:
            self.curiosity.trigger_exploration(
                perception.vector,
                raw_input=str(perception.raw),
                modality=perception.modality,
            )

        # Language check (text only)
        struct_score = 1.0
        if perception.modality == "text":
            report = self.checker.check(str(perception.raw))
            struct_score = report.overall_score

        # Compute loss
        recall_sims = [sim for sim, _ in
                       self.memory.retrieve(perception.vector, k=5)]
        loss_rec = self.loss_sys.compute(
            solution_vec, target_vec,
            candidate_scores=recall_sims,
            novelty=novelty,
            structure_score=struct_score,
        )

        # Store in memory (with curiosity-weighted activation)
        weight = self.curiosity.curiosity_weight(perception.vector)
        self.memory.store(
            perception.vector,
            label=perception.label,
            modality=perception.modality,
            metadata={
                "novelty": novelty,
                "struct_score": struct_score,
                "weight": weight,
            },
        )

        # Incorporate environment reward
        if env_feedback:
            env_reward = env_feedback.get("reward", 0.0)
            if env_reward > 0:
                state_vec = env_feedback.get("state_vec")
                if state_vec is not None:
                    self.memory.store(
                        state_vec,
                        label=f"env_state_step{self._step_count}",
                        modality="env",
                        metadata={"reward": env_reward},
                    )

        return loss_rec.total_loss

    # ------------------------------------------------------------------
    # 5. Curiosity check (standalone)
    # ------------------------------------------------------------------

    def curiosity_check(
        self,
        perception: Perception,
    ) -> Tuple[float, bool]:
        """
        Check novelty of a perception.
        Returns (novelty_score, triggered_exploration).
        """
        novelty = self.curiosity.detect_novelty(perception.vector)
        triggered = False
        if novelty >= self.cfg.novelty_threshold:
            self.curiosity.trigger_exploration(
                perception.vector,
                raw_input=str(perception.raw),
                modality=perception.modality,
            )
            triggered = True
        return novelty, triggered

    # ------------------------------------------------------------------
    # Full step
    # ------------------------------------------------------------------

    def step(
        self,
        raw_input: Any,
        modality: str = "text",
        label: str = "",
        target_text: Optional[str] = None,
        action: Optional[str] = None,
        action_args: tuple = (),
        context: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> AgentStep:
        """
        Full perceive → reason → act → update cycle.

        Returns AgentStep with all metrics.
        """
        t0 = time.time()
        self._step_count += 1

        # Observe
        perc = self.observe(raw_input, modality, label)
        if verbose:
            print(f"  Step {self._step_count}: [{modality}] '{str(raw_input)[:50]}'")

        # Reason
        sol_vec, score = self.reason(perc, target_text, context, verbose)
        if verbose:
            print(f"    reasoning score: {score:.3f}")

        # Act
        env_fb = None
        env_reward = None
        if action and self.env:
            env_fb = self.act(action, *action_args)
            env_reward = env_fb.get("reward")
            if verbose:
                print(f"    action='{action}': {env_fb.get('message','')}")

        # Update
        loss = self.update(perc, sol_vec, target_text, env_fb)

        # Novelty
        novelty = self.curiosity.detect_novelty(perc.vector)

        elapsed = (time.time() - t0) * 1000
        record = AgentStep(
            step=self._step_count,
            perception=perc,
            reasoning_score=score,
            action=action,
            env_reward=env_reward,
            loss=loss,
            novel=novelty >= self.cfg.novelty_threshold,
            elapsed_ms=elapsed,
        )
        self._step_log.append(record)
        return record

    # ------------------------------------------------------------------
    # Cross-modal linking
    # ------------------------------------------------------------------

    def link_modalities(
        self,
        text: str,
        other_vec: np.ndarray,
        other_modality: str,
        n_align_iters: int = 20,
    ) -> float:
        """
        Link a text and another-modality vector.
        Returns their similarity in the shared space.
        If below threshold, attempt alignment.
        """
        t_vec   = self.text_enc.encode(text)
        initial = self.space.similarity(t_vec, other_vec)

        if initial < 0.2:
            # Learn alignment
            if other_modality == "image":
                R = self.space.align_modalities(
                    other_vec.reshape(1, -1),
                    t_vec.reshape(1, -1),
                )
                self._align_img = R
                other_vec = self.space.l2(other_vec @ R)
            elif other_modality == "audio":
                R = self.space.align_modalities(
                    other_vec.reshape(1, -1),
                    t_vec.reshape(1, -1),
                )
                self._align_aud = R
                other_vec = self.space.l2(other_vec @ R)

        sim = self.space.similarity(t_vec, other_vec)

        # Store cross-modal link in memory
        fused = self.space.combine(t_vec, other_vec, mode=ComposeMode.MEAN)
        self.memory.store(
            fused,
            label=f"link:{text[:20]}+{other_modality}",
            modality="fused",
            metadata={"text": text, "other_modality": other_modality,
                      "similarity": sim},
        )
        return sim

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(self, query: str, k: int = 5) -> List[Tuple[float, str]]:
        """Recall top-k memories for a text query."""
        vec = self.text_enc.encode(query)
        results = self.memory.retrieve(vec, k=k)
        return [(sim, e.label) for sim, e in results]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        return {
            "steps":              self._step_count,
            "memory_size":        len(self.memory),
            "reasoning_success":  f"{self.reasoning.success_rate():.1%}",
            "curiosity_explorations": self.curiosity.n_explorations,
            "loss":               self.loss_sys.summary(),
            "env_stats":          self.env.stats() if self.env else None,
        }

    def __repr__(self) -> str:
        return (f"CogniFieldAgent(dim={self.cfg.dim}, "
                f"steps={self._step_count}, "
                f"memory={len(self.memory)})")
