"""
cognifield/cognifield.py
=========================
CogniField — Unified Public Interface

The single entry point for all CogniField capabilities.

Quick start:

    from cognifield import CogniField

    cf = CogniField()
    result = cf.think("Is this berry safe to eat?")
    print(result["decision"])     # "cautious"
    print(result["confidence"])   # 0.62
    print(result["reasoning"])    # ["No prior knowledge...", ...]

Advanced:

    cf = CogniField({
        "agents": 3,
        "uncertainty": "medium",
        "enable_meta": True,
        "strategy": "adaptive",
        "llm": "ollama",              # optional LLM backend
        "llm_model": "llama3",
    })

With remote LLM:

    cf = CogniField({"llm": "api", "llm_api_key": "sk-..."})

All reasoning happens inside CogniField's multi-agent system.
The LLM (if configured) handles natural language I/O only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CogniFieldConfig:
    """
    Configuration for a CogniField instance.

    Parameters
    ----------
    agents       : Number of reasoning agents in the fleet.
    uncertainty  : Environment uncertainty level — none/low/medium/high/chaotic.
    enable_meta  : Enable MetaCognition (self-reflection and bias detection).
    strategy     : Initial reasoning strategy — explore/exploit/verify/recover/adaptive.
    dim          : Latent space dimensionality (higher = more expressive, slower).
    seed         : Random seed for reproducibility.
    llm          : LLM backend — "mock" (default), "ollama", "api".
    llm_model    : Model name for the chosen LLM backend.
    llm_base_url : Custom endpoint URL (Ollama or compatible API).
    llm_api_key  : API key for remote LLM (or set OPENAI_API_KEY env var).
    verbose      : Print internal reasoning steps.
    """
    agents:       int   = 3
    uncertainty:  str   = "low"
    enable_meta:  bool  = True
    strategy:     str   = "adaptive"
    dim:          int   = 64
    seed:         int   = 42
    llm:          str   = "mock"
    llm_model:    str   = "llama3"
    llm_base_url: str   = "http://localhost:11434"
    llm_api_key:  str   = ""
    verbose:      bool  = False

    @classmethod
    def from_dict(cls, d: Dict) -> "CogniFieldConfig":
        """Build config from a plain dict (unknown keys silently ignored)."""
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in fields})


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

def _make_response(
    decision:    str,
    confidence:  float,
    reasoning:   List[str],
    consensus:   Dict,
    meta:        Dict,
    llm_output:  str   = "",
    elapsed_ms:  float = 0.0,
) -> Dict:
    return {
        "decision":    decision,
        "confidence":  round(float(confidence), 4),
        "reasoning":   reasoning,
        "consensus":   consensus,
        "meta":        meta,
        "llm_output":  llm_output,
        "elapsed_ms":  round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# CogniField
# ---------------------------------------------------------------------------

class CogniField:
    """
    Unified cognitive reasoning system.

    Wraps a fleet of v9 agents with global consensus, uncertainty
    modeling, meta-cognition, and optional LLM natural language layer.

    Usage
    -----
    >>> from cognifield import CogniField
    >>> cf = CogniField()
    >>> result = cf.think("Is this berry safe?")
    >>> print(result["decision"])
    """

    def __init__(
        self,
        config: Optional[Union[Dict, CogniFieldConfig]] = None,
    ) -> None:
        # Parse config
        if isinstance(config, dict):
            self._cfg = CogniFieldConfig.from_dict(config)
        elif isinstance(config, CogniFieldConfig):
            self._cfg = config
        else:
            self._cfg = CogniFieldConfig()

        cfg = self._cfg

        # Lazy-import heavy modules to keep import fast
        from .agents.agent_v9 import CogniFieldAgentV9, AgentV9Config
        from .agents.agent_v7 import AgentRole
        from .agents.group_mind import GroupMind, CoordSignal
        from .reasoning.global_consensus import GlobalConsensus
        from .core.event_bus import EventBus
        from .communication.communication_module import CommunicationModule
        from .memory.shared_memory import SharedMemory
        from .planning.cooperation_engine import CooperationEngine
        from .environment.rich_env import RichEnv
        from .llm.base import create_llm_client

        self._env  = RichEnv(seed=cfg.seed)
        self._bus  = CommunicationModule()
        self._sm   = SharedMemory()
        self._eb   = EventBus()
        self._gm   = GroupMind(event_bus=self._eb)
        self._ce   = CooperationEngine()
        self._gc   = GlobalConsensus(self._sm, self._bus, self._eb,
                                     supermajority=0.55)

        # LLM client
        llm_kwargs: Dict[str, Any] = {}
        if cfg.llm == "ollama":
            llm_kwargs = {"model": cfg.llm_model, "base_url": cfg.llm_base_url}
        elif cfg.llm in ("api", "openai"):
            llm_kwargs = {"model": cfg.llm_model, "api_key": cfg.llm_api_key}
        self._llm = create_llm_client(cfg.llm, **llm_kwargs)

        # Build agent fleet
        n_agents = cfg.agents
        roles = [AgentRole.EXPLORER, AgentRole.ANALYST, AgentRole.RISK_MANAGER,
                 AgentRole.PLANNER, AgentRole.GENERALIST]
        self._agents: List = []
        for i in range(n_agents):
            role = roles[i % len(roles)]
            a_cfg = AgentV9Config(
                agent_id=f"cf_{i}",
                role=role,
                dim=cfg.dim,
                seed=cfg.seed + i,
                verbose=cfg.verbose,
                uncertainty_level=cfg.uncertainty,
            )
            agent = CogniFieldAgentV9(
                config=a_cfg,
                env=self._env,
                comm_bus=self._bus,
                shared_mem=self._sm,
                group_mind=self._gm,
                global_cons=self._gc,
                event_bus=self._eb,
                coop_engine=self._ce,
            )
            self._agents.append(agent)
            self._ce.register_agent(agent.agent_id, role.value)

        # Cross-register negotiations
        for i, a in enumerate(self._agents):
            for j, b in enumerate(self._agents):
                if i != j:
                    a.register_for_negotiation(b)

        # Seed with common knowledge priors
        self._seed_knowledge()

        # Step counter + history
        self._step_count   = 0
        self._think_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def think(self, input_text: str) -> Dict:
        """
        Process an input through the full reasoning pipeline.

        The fleet observes the input, updates beliefs, runs consensus,
        and returns a structured decision.

        Parameters
        ----------
        input_text : Natural language question or observation.

        Returns
        -------
        dict with keys: decision, confidence, reasoning, consensus, meta,
                        llm_output, elapsed_ms
        """
        t0 = time.time()
        self._step_count += 1

        # Run one step on each agent
        steps = []
        for agent in self._agents:
            agent.ensure_bidirectional_comm()
            s = agent.step(text_input=input_text, verbose=False)
            steps.append(s)

        # Run global consensus every call
        trust_map = self._build_trust_map()
        ab = {a.agent_id: a.beliefs for a in self._agents}
        gc_results = self._gc.run_round(ab, trust_map)
        self._gc.apply_to_all(ab)

        # Synthesize decision
        decision, confidence, reasoning = self._synthesize(input_text, gc_results)

        # Consensus summary
        auth = self._gc.get_authoritative()
        consensus_out = {
            k: {"value": r.value, "confidence": round(r.confidence, 3),
                "agreement": round(r.agreement, 3)}
            for k, r in list(auth.items())[:6]
        }

        # Meta summary
        meta = self._meta_summary(steps)

        # Optional LLM enhancement
        llm_text = ""
        if self._llm.is_available() and not isinstance(
            self._llm, __import__("cognifield.llm.base", fromlist=["MockLLM"]).MockLLM
        ):
            ctx = {
                "decision": decision, "confidence": confidence,
                "beliefs": {k: {"value": v["value"], "confidence": v["confidence"]}
                            for k, v in consensus_out.items()},
                "consensus_value": decision,
                "uncertainty": self._cfg.uncertainty,
                "strategy": meta.get("strategy", "explore"),
            }
            prompt   = self._llm.format_decision_prompt(input_text, ctx)
            llm_text = self._llm.generate(prompt, max_tokens=256)
        elif isinstance(self._llm,
                        __import__("cognifield.llm.base",
                                   fromlist=["MockLLM"]).MockLLM):
            # Always call mock for consistent output
            ctx = {"decision": decision, "confidence": confidence,
                   "uncertainty": self._cfg.uncertainty}
            prompt   = self._llm.format_decision_prompt(input_text, ctx)
            llm_text = self._llm.generate(prompt, max_tokens=256)

        elapsed = (time.time() - t0) * 1000
        result  = _make_response(
            decision=decision, confidence=confidence,
            reasoning=reasoning, consensus=consensus_out,
            meta=meta, llm_output=llm_text, elapsed_ms=elapsed,
        )
        self._think_history.append({"input": input_text, **result})
        return result

    def decide(self, input_text: str) -> Dict:
        """
        Make a concrete action decision for a given situation.

        More focused than think() — returns a concrete recommended action
        with risk assessment and alternative options.

        Parameters
        ----------
        input_text : A situation description requiring a decision.

        Returns
        -------
        dict including: decision, confidence, reasoning, risk_level,
                        alternatives, consensus, meta, llm_output
        """
        result = self.think(input_text)

        # Add decision-specific fields
        risk_level = self._assess_risk_level(result["confidence"])
        alternatives = self._generate_alternatives(input_text, result)

        result["risk_level"]   = risk_level
        result["alternatives"] = alternatives
        result["action"]       = self._pick_action(result["decision"], risk_level)
        return result

    def simulate(self, scenario: str, steps: int = 10) -> Dict:
        """
        Run a multi-step simulation of a scenario.

        Agents act autonomously for `steps` iterations, then return
        aggregated outcomes: what was learned, what changed, success rate.

        Parameters
        ----------
        scenario : Scenario description (e.g. "foraging in unknown forest").
        steps    : Number of simulation steps (default: 10).

        Returns
        -------
        dict with: outcomes, belief_changes, success_rate, strategy,
                   consensus, meta, llm_output, elapsed_ms
        """
        t0 = time.time()

        self._gm.set_primary_goal(scenario)

        beliefs_before = {
            a.agent_id: len(a.beliefs.reliable_beliefs())
            for a in self._agents
        }
        outcomes: List[str] = []
        total_success = 0
        total_steps   = 0

        for _ in range(steps):
            for agent in self._agents:
                agent.ensure_bidirectional_comm()
                s = agent.step(text_input=scenario, verbose=False)
                total_steps += 1
                if s.env_success:
                    total_success += 1
                if s.action_taken and s.action_obj:
                    outcome = (f"{s.action_taken}({s.action_obj}): "
                               f"{'✓' if s.env_success else '✗'}")
                    if outcome not in outcomes:
                        outcomes.append(outcome)

        # Run final consensus
        trust_map = self._build_trust_map()
        ab = {a.agent_id: a.beliefs for a in self._agents}
        gc_results = self._gc.run_round(ab, trust_map)
        self._gc.apply_to_all(ab)

        beliefs_after = {
            a.agent_id: len(a.beliefs.reliable_beliefs())
            for a in self._agents
        }
        belief_changes = sum(
            beliefs_after[aid] - beliefs_before[aid]
            for aid in beliefs_before
        )

        sr = total_success / max(total_steps, 1)
        strategy = self._agents[0].strategy_mgr.current.value

        # Consensus after simulation
        auth = self._gc.get_authoritative()
        consensus_out = {
            k: {"value": r.value, "confidence": round(r.confidence, 3)}
            for k, r in list(auth.items())[:6]
        }
        meta = self._meta_summary([])

        # LLM narrative
        sim_result = {
            "success_rate": sr,
            "steps": total_steps,
            "outcomes": outcomes[:5],
            "belief_changes": belief_changes,
            "strategy": strategy,
        }
        llm_text = self._llm.generate(
            self._llm.format_simulation_prompt(scenario, sim_result),
            max_tokens=200,
        )

        elapsed = (time.time() - t0) * 1000
        return {
            "scenario":       scenario,
            "steps_run":      total_steps,
            "outcomes":       outcomes[:8],
            "success_rate":   round(sr, 3),
            "belief_changes": belief_changes,
            "strategy":       strategy,
            "consensus":      consensus_out,
            "meta":           meta,
            "llm_output":     llm_text,
            "elapsed_ms":     round(elapsed, 1),
        }

    # ------------------------------------------------------------------
    # Teach / configuration
    # ------------------------------------------------------------------

    def teach(
        self,
        label:      str,
        properties: Dict[str, Any],
        text:       Optional[str] = None,
    ) -> "CogniField":
        """
        Teach all agents a fact about an object or concept.

        Parameters
        ----------
        label      : Concept name (e.g. "apple").
        properties : Dict of properties (e.g. {"edible": True, "category": "food"}).
        text       : Optional free-text description.

        Returns self (fluent).
        """
        desc = text or (label + " " +
                        " ".join(f"{k} {v}" for k, v in properties.items()))
        for agent in self._agents:
            agent.teach(desc, label=label, props=properties)
        return self

    def add_goal(self, goal_label: str, priority: float = 0.8) -> "CogniField":
        """Add a goal to all agents. Returns self (fluent)."""
        from .agents.goals import GoalType
        for agent in self._agents:
            agent.add_goal(goal_label, GoalType.CUSTOM, priority=priority)
        return self

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def status(self) -> Dict:
        """Return a concise system status snapshot."""
        sm_summ = self._sm.summary()
        return {
            "version":        "10.0",
            "agents":         len(self._agents),
            "llm_backend":    str(self._llm),
            "llm_available":  self._llm.is_available(),
            "shared_beliefs": sm_summ.get("entries", 0),
            "think_calls":    len(self._think_history),
            "config": {
                "uncertainty": self._cfg.uncertainty,
                "strategy":    self._cfg.strategy,
                "dim":         self._cfg.dim,
                "enable_meta": self._cfg.enable_meta,
            },
        }

    def get_beliefs(self, min_confidence: float = 0.60) -> Dict:
        """Return authoritative fleet-wide beliefs above min_confidence."""
        return {
            k: {
                "value":      r.value,
                "confidence": round(r.confidence, 3),
                "agreement":  round(r.agreement, 3),
                "version":    r.version,
            }
            for k, r in self._gc.get_authoritative().items()
            if r.confidence >= min_confidence
        }

    def reset(self) -> "CogniField":
        """
        Soft reset: clear history and step counters, keep agent knowledge.
        Returns self (fluent).
        """
        self._step_count = 0
        self._think_history.clear()
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _seed_knowledge(self) -> None:
        """Pre-load common safe knowledge priors into all agents."""
        priors = [
            ("food",     {"edible": True}),
            ("material", {"edible": False}),
            ("tool",     {"edible": False}),
        ]
        for label, props in priors:
            for agent in self._agents:
                agent.teach(label, label=label, props=props)

    def _build_trust_map(self) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        counts: Dict[str, int]   = {}
        for a in self._agents:
            for pid, rec in a.trust._records.items():
                scores[pid] = scores.get(pid, 0) + rec.trust_score
                counts[pid] = counts.get(pid, 0) + 1
        return {pid: scores[pid] / counts[pid] for pid in scores}

    def _synthesize(
        self,
        input_text: str,
        gc_results: Dict,
    ) -> tuple:
        """Derive a decision, confidence, and reasoning chain from GC results."""
        reasoning: List[str] = []

        # Match input against known beliefs, prioritising edibility
        input_lower = input_text.lower()
        target_belief = None

        # Priority 1: subject named in input AND pred == "edible"
        #             (scan .edible keys first regardless of dict order)
        edible_keys = [k for k in gc_results if k.endswith(".edible")]
        for key in edible_keys:
            subj = key.split(".")[0]
            if subj in input_lower:
                target_belief = gc_results[key]
                break
        if target_belief is None:
            for key, rec in gc_results.items():
                parts = key.split(".")
                if len(parts) == 2:
                    subj, pred = parts
                    if subj in input_lower and pred == "edible":
                        target_belief = rec
                        break

        # Priority 2: edibility keyword → use highest-conf edible belief
        if target_belief is None:
            eat_words = {"eat", "safe", "edible", "consume", "poison", "berry"}
            if any(w in input_lower for w in eat_words):
                edible_recs = {k: r for k, r in gc_results.items()
                               if k.endswith(".edible")}
                if edible_recs:
                    for key, rec in edible_recs.items():
                        subj = key.split(".")[0]
                        if subj in input_lower:
                            target_belief = rec
                            break
                    if target_belief is None:
                        target_belief = max(edible_recs.values(),
                                            key=lambda r: r.confidence)

        # Priority 3: any subject match
        if target_belief is None:
            for key, rec in gc_results.items():
                parts = key.split(".")
                if len(parts) == 2 and parts[0] in input_lower:
                    target_belief = rec
                    break

        # Fallback: highest-confidence authoritative belief
        if target_belief is None and gc_results:
            target_belief = max(gc_results.values(),
                                key=lambda r: r.confidence, default=None)

        # Build decision
        if target_belief is None:
            decision   = "insufficient_data"
            confidence = 0.30
            reasoning.append("No relevant beliefs found for this input.")
            reasoning.append("Recommend gathering more observations.")
        elif target_belief.contested:
            decision   = "uncertain"
            confidence = 0.40
            reasoning.append(f"Belief {target_belief.key} is contested.")
            reasoning.append("Agents disagree — more evidence needed.")
        elif target_belief.value is True:
            decision   = "proceed"
            confidence = target_belief.confidence
            reasoning.append(f"{target_belief.key} = {target_belief.value} "
                             f"(conf={target_belief.confidence:.3f})")
            reasoning.append(f"Agreement across {target_belief.n_agents} agents: "
                             f"{target_belief.agreement:.0%}")
        elif target_belief.value is False:
            decision   = "avoid"
            confidence = target_belief.confidence
            reasoning.append(f"{target_belief.key} = {target_belief.value} "
                             f"(conf={target_belief.confidence:.3f})")
            reasoning.append("Evidence strongly indicates this is not recommended.")
        else:
            decision   = str(target_belief.value)
            confidence = target_belief.confidence
            reasoning.append(f"Consensus: {target_belief.key} = {target_belief.value}")

        # Add strategy context
        for agent in self._agents[:1]:
            strategy = agent.strategy_mgr.current.value
            unc      = agent.uncertainty.level.value
            reasoning.append(f"Active strategy: {strategy}, uncertainty: {unc}")

        return decision, float(confidence), reasoning

    def _meta_summary(self, steps: List) -> Dict:
        """Aggregate meta information from agent steps."""
        meta: Dict[str, Any] = {
            "agents":        len(self._agents),
            "think_calls":   self._step_count,
            "gc_rounds":     self._gc.summary()["rounds"],
            "events_fired":  self._eb.summary()["total_events"],
        }
        if steps:
            meta["strategy"]    = steps[0].strategy if steps else "unknown"
            meta["uncertainty"] = steps[0].uncertainty_level if steps else "unknown"
        if self._cfg.enable_meta:
            meta_reports = []
            for agent in self._agents:
                mc = agent.meta_cog.summary()
                if mc.get("n_reflections", 0) > 0:
                    meta_reports.append({
                        "agent":       agent.agent_id,
                        "calibration": mc.get("calibration_score", 0.5),
                        "trend":       mc.get("performance_trend", "unknown"),
                    })
            meta["meta_cognition"] = meta_reports
        return meta

    def _assess_risk_level(self, confidence: float) -> str:
        if confidence >= 0.80: return "low"
        if confidence >= 0.60: return "medium"
        if confidence >= 0.40: return "high"
        return "critical"

    def _generate_alternatives(
        self, input_text: str, result: Dict
    ) -> List[str]:
        alts = []
        if result["decision"] == "avoid":
            alts = ["gather more evidence", "observe without acting",
                    "consult additional sources"]
        elif result["decision"] == "proceed":
            alts = ["proceed cautiously", "verify first", "take partial action"]
        elif result["decision"] in ("uncertain", "insufficient_data"):
            alts = ["experiment safely", "ask peers", "wait and observe"]
        return alts

    def _pick_action(self, decision: str, risk_level: str) -> str:
        table = {
            ("proceed",           "low"):      "act",
            ("proceed",           "medium"):   "act_with_caution",
            ("proceed",           "high"):     "verify_first",
            ("avoid",             "low"):      "do_not_act",
            ("avoid",             "medium"):   "do_not_act",
            ("avoid",             "high"):     "emergency_stop",
            ("uncertain",         "low"):      "gather_evidence",
            ("uncertain",         "medium"):   "experiment_safely",
            ("insufficient_data", "critical"): "observe_only",
        }
        return table.get((decision, risk_level), "consult_expert")

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"CogniField(agents={len(self._agents)}, "
                f"llm={self._llm!r}, "
                f"uncertainty={self._cfg.uncertainty})")
