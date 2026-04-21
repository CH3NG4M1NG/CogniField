# CogniField Architecture

## Overview

CogniField is built in layers. Each version added one new capability
on top of all the previous ones. Every layer is backward-compatible.

```
┌─────────────────────────────────────────────────────────────┐
│  PUBLIC API  (cognifield.CogniField)                         │
│  .think()  .decide()  .simulate()  .teach()  .status()      │
├─────────────────────────────────────────────────────────────┤
│  INTEGRATIONS                                                │
│  REST API (Flask)  ·  CLI  ·  LLM layer (Ollama/OpenAI)    │
├─────────────────────────────────────────────────────────────┤
│  ADAPTIVE LAYER  (v9)                                        │
│  MetaCognition  ·  UncertaintyEngine  ·  StrategyManager    │
│  GoalConflictResolver  ·  TemporalMemory  ·  SelfEvaluator  │
├─────────────────────────────────────────────────────────────┤
│  COLLECTIVE LAYER  (v8)                                      │
│  EventBus  ·  GlobalConsensus  ·  GroupMind                 │
├─────────────────────────────────────────────────────────────┤
│  SOCIAL LAYER  (v6–v7)                                       │
│  CommunicationBus  ·  TrustSystem  ·  NegotiationEngine     │
│  CooperationEngine  ·  LanguageLayer  ·  SocialMemory       │
├─────────────────────────────────────────────────────────────┤
│  STABLE BELIEF CORE  (v5)                                    │
│  BeliefSystem  ·  ConsistencyEngine  ·  ConflictResolver    │
│  RiskEngine  ·  ExperimentEngine  ·  KnowledgeValidator     │
├─────────────────────────────────────────────────────────────┤
│  WORLD MODEL  (v3–v4)                                        │
│  TransitionModel  ·  CausalGraph  ·  WorldSimulator         │
│  HierarchicalPlanner  ·  GoalSystem  ·  MetaLearner         │
├─────────────────────────────────────────────────────────────┤
│  FOUNDATION  (v2)                                            │
│  TextEncoder  ·  FrequencySpace  ·  MemoryStore             │
│  RelationalMemory  ·  ReasoningEngine                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Data flow: `cf.think("Is apple safe?")`

```
1. CogniField.think(text)
   │
   ├── For each agent in fleet:
   │     agent.step(text)
   │       ├── Encode text → latent vector
   │       ├── Apply GroupMind coordination signal
   │       ├── Read from SharedMemory (community knowledge)
   │       ├── Apply uncertainty decay to beliefs
   │       ├── Detect novelty
   │       ├── Resolve goal conflicts
   │       ├── Select goal, plan, simulate, risk-check
   │       ├── Act in environment
   │       ├── Update beliefs (Bayesian)
   │       ├── Record to temporal memory
   │       ├── Evaluate strategy (possibly switch)
   │       └── Run meta-cognition reflection (periodic)
   │
   ├── GlobalConsensus.run_round(all_agent_beliefs)
   │     ├── Collect votes from all agents
   │     ├── Weight by: confidence × evidence × trust
   │     ├── Find supermajority winner
   │     ├── Write authoritative belief to SharedMemory
   │     ├── Broadcast CONSENSUS event to all agents
   │     └── Return GlobalBeliefRecord per key
   │
   ├── _synthesize(input_text, gc_results)
   │     ├── Match input words against known beliefs
   │     ├── Prioritise safety predicates (edible > category)
   │     └── Return decision, confidence, reasoning[]
   │
   ├── LLM.generate(structured_prompt) [if configured]
   │     └── Return natural language explanation
   │
   └── Return structured response dict
```

---

## Agent architecture

Each `CogniFieldAgentV9` is a complete cognitive unit:

```
CogniFieldAgentV9
  │
  ├── Private knowledge
  │     ├── BeliefSystem          (Bayesian, versioned)
  │     ├── MemoryStore           (vector similarity)
  │     ├── RelationalMemory      (concept graph)
  │     ├── EpisodicMemory        (time-tagged experiences)
  │     └── TemporalMemory        (long-term patterns)
  │
  ├── Reasoning
  │     ├── ConsistencyEngine     (belief gate-keeping)
  │     ├── ConflictResolver      (contradiction resolution)
  │     ├── NegotiationEngine     (peer argumentation)
  │     └── MetaCognitionEngine   (self-analysis)
  │
  ├── Planning
  │     ├── HierarchicalPlanner   (subgoal decomposition)
  │     ├── WorldSimulator        (forward imagination)
  │     ├── GoalConflictResolver  (competing goals)
  │     └── RiskEngine            (action safety)
  │
  ├── Adaptation
  │     ├── StrategyManager       (explore/exploit/recover)
  │     ├── UncertaintyEngine     (noise + decay)
  │     ├── MetaLearner           (performance tracking)
  │     └── SelfEvaluator         (weakness detection)
  │
  └── Social
        ├── TrustSystem           (peer reputation)
        ├── LanguageLayer         (semantic vocabulary)
        ├── SocialMemory          (interaction history)
        └── InternalState         (confidence/curiosity/fatigue)
```

---

## Shared infrastructure

```
Fleet shared resources (one per CogniField instance)
  │
  ├── SharedMemory          Community belief store (versioned)
  ├── CommunicationModule   Typed message bus (BELIEF/WARNING/QUESTION/...)
  ├── GlobalConsensus       Fleet-wide vote aggregation
  ├── GroupMind             Shared goals + coordination signals
  ├── EventBus              Pub/sub (13 event types)
  ├── CooperationEngine     Task assignment + load balancing
  └── RichEnv               Shared environment
```

---

## Belief confidence lifecycle

```
Unknown object:     confidence = 0.25  (prior)
After hypothesis:   confidence = 0.52  (weak inference)
After 1 experiment: confidence = 0.68  (low evidence)
After 3 tests:      confidence = 0.82  (reliable)
After consensus:    confidence = 0.87  (fleet agrees)
After 10+ tests:    confidence = 0.95  (authoritative)

Under HIGH uncertainty: confidence decays toward 0.30/step
Under CHAOTIC:          confidence decays toward 0.40/step
MetaCognition detects overconfidence and triggers decay if
actual_success_rate << predicted_confidence.
```

---

## Strategy switching

```
StrategyManager monitors recent performance and switches:

Current          Trigger              Switch to
─────────────    ─────────────────    ──────────────
EXPLORE          sr > 0.70, nov < 0.2 EXPLOIT
EXPLORE          sr ≤ 0.25            RECOVER
EXPLORE          4 consec failures    RECOVER
EXPLOIT          sr drops < 0.40      VERIFY
VERIFY           peers agree > 0.75   COOPERATIVE
RECOVER          sr improves > 0.50   EXPLORE
```

---

## Module map

```
cognifield/
  cognifield_main.py     Public CogniField class
  __init__.py            Package exports
  __main__.py            python -m cognifield entry point
  setup.py / pyproject   Packaging

  core/
    event_bus.py         EventBus pub/sub
    meta_cognition.py    Self-analysis, calibration
    uncertainty_engine.py Noise, decay, partial observability

  agents/
    agent_v9.py          Top-level agent (22-step loop)
    agent_v8.py          Collective intelligence extensions
    agent_v7.py          Social intelligence extensions
    agent_v6.py          Multi-agent communication extensions
    agent_v5.py          Stable belief core
    group_mind.py        Shared goals + coordination
    strategy_manager.py  Dynamic strategy switching
    goal_conflict_resolver.py  Competing goal arbitration
    self_evaluator.py    Performance grading
    trust_system.py      Peer reputation
    agent_manager.py     Fleet coordinator (v6)

  memory/
    shared_memory.py     Community knowledge store
    temporal_memory.py   Long-term pattern tracking
    social_memory.py     Interaction history
    episodic_memory.py   Time-tagged experiences
    memory_store.py      Vector similarity

  reasoning/
    global_consensus.py  Fleet-wide voting
    negotiation_engine.py  Peer argumentation
    consensus_engine.py  Multi-agent voting
    belief_system.py     Bayesian belief management
    consistency_engine.py  Belief gate-keeping

  llm/
    base.py              OllamaClient, APIClient, MockLLM

  api/
    server.py            Flask REST server

  cli/
    __main__.py          CLI interface
```
