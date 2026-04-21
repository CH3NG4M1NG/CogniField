# Changelog

All notable changes are documented here.

## [10.0.0] — Production Release

### Added
- `CogniField` unified entry point class (`cognifield_main.py`)
- `CogniFieldConfig` structured configuration dataclass
- `.think()`, `.decide()`, `.simulate()`, `.teach()`, `.status()` API
- Structured JSON response schema (decision, confidence, reasoning, consensus, meta)
- `llm/` package with `OllamaClient`, `APIClient`, `MockLLM`, `create_llm_client()`
- `api/server.py` Flask REST server with `/health`, `/think`, `/decide`, `/simulate`, `/beliefs`, `/teach`
- `cli/__main__.py` — `python -m cognifield "question"` CLI
- `setup.py` + `pyproject.toml` for `pip install cognifield`
- `docs/QUICKSTART.md`, `docs/API_REFERENCE.md`, `docs/LLM_INTEGRATION.md`, `docs/ARCHITECTURE.md`
- Examples: `basic_usage.py`, `decision_demo.py`, `api_client.py`, `ollama_integration.py`
- `tests/test_v10.py` — 115 tests covering all new v10 modules

### Changed
- `README.md` rewritten as professional framework documentation
- `__init__.py` exports `CogniField` and `CogniFieldConfig` at package root
- `requirements.txt` updated with Flask as optional dependency

---

## [9.0.0] — Adaptive Self-Reflective Intelligence

### Added
- `core/meta_cognition.py` — overconfidence detection, domain bias tracking, calibration curves
- `core/uncertainty_engine.py` — 5 noise levels, belief decay, partial observability, auto-detection
- `agents/goal_conflict_resolver.py` — resource/value/priority conflict detection + resolution
- `agents/strategy_manager.py` — 6 strategies with dynamic switching rules
- `memory/temporal_memory.py` — pattern detection, belief drift, recurrence detection
- `agents/self_evaluator.py` — 7-dimension graded performance reports
- `agents/agent_v9.py` — 22-step adaptive loop
- `tests/test_v9.py` — 131 tests

---

## [8.0.0] — Collective Intelligence

### Added
- `core/event_bus.py` — pub/sub with 13 event types
- `reasoning/global_consensus.py` — fleet-wide belief aggregation + broadcast
- `agents/group_mind.py` — shared goals, coordination signals, experience sharing
- `agents/agent_v8.py` — 20-step collective loop with bidirectional communication
- `tests/test_v8.py` — 90 tests

---

## [7.0.0] — Social Intelligence + Repository Cleanup

### Added
- `communication/language_layer.py` — semantic token vocabulary, evolving shared language
- `reasoning/negotiation_engine.py` — belief argumentation protocol
- `planning/cooperation_engine.py` — role-fitness task assignment
- `memory/social_memory.py` — per-peer interaction history
- `agents/agent_v7.py` — 18-step social loop
- `tests/test_v7.py` — 122 tests

### Changed
- Merged `agent/` into `agents/`; renamed `agent.py` → `base_agent.py`
- Moved version READMEs to `docs/`
- Added `.gitignore`

---

## [6.0.0] — Multi-Agent Communication

### Added
- `communication/communication_module.py` — typed message bus
- `memory/shared_memory.py` — community knowledge store
- `agents/trust_system.py` — per-peer trust scoring
- `reasoning/consensus_engine.py` — multi-agent voting
- `agents/agent_v6.py` — multi-agent aware loop
- `agents/agent_manager.py` — fleet coordinator
- `tests/test_v6.py` — 142 tests

---

## [5.0.0] — Stable Beliefs

### Added
- `world_model/belief_system.py` — Bayesian belief management
- `reasoning/conflict_resolver.py` — contradiction resolution (5 strategies)
- `reasoning/consistency_engine.py` — belief gate-keeping + propagation
- `reasoning/validation.py` — periodic knowledge re-verification
- `curiosity/experiment_engine.py` — safety-ladder experiments
- `agent/risk_engine.py` — probabilistic action risk scoring
- `memory/episodic_memory.py` + `ProceduralMemoryStore`
- `evaluation/metrics.py` — stability grade A–F
- `tests/test_v5.py` — 144 tests

---

## [4.0.0] — Self-Direction

### Added
- `agent/goal_generator.py` — self-generated goals
- `agent/internal_state.py` — confidence/curiosity/fatigue/frustration
- `world_model/simulator.py` — forward imagination
- `planning/hierarchical_planner.py` — subgoal decomposition
- `reasoning/abstraction.py` — rule induction
- `reasoning/meta_learning.py` — performance adaptation
- `memory/consolidation.py` — memory compression
- `tests/test_v4.py` — 123 tests

---

## [3.0.0] — World Model

### Added
- `world_model/transition_model.py` — (state, action) → next_state
- `world_model/causal_graph.py` — causal relationships
- `planning/planner.py` — beam-search planner
- `agent/goals.py` — goal lifecycle
- `memory/relational_memory.py` — typed concept graph
- `curiosity/advanced_curiosity.py` — hypothesis generation
- `environment/rich_env.py` — objects with hidden properties
- `tests/test_v3.py` — 113 tests

---

## [2.0.0] — Foundation

### Added
- `encoder/text_encoder.py`, `image_encoder.py`
- `latent_space/frequency_space.py`
- `memory/memory_store.py`
- `reasoning/reasoning_engine.py`
- `environment/simple_env.py`
- `tests/test_all.py` — 64 tests
