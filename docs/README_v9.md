# CogniField v9 — Adaptive Self-Reflective Intelligence

## What v9 adds to v8

| v8 State | v9 Solution |
|----------|-------------|
| Fixed confidence regardless of evidence | `UncertaintyEngine` — noise injection + belief decay toward floor |
| No self-analysis | `MetaCognitionEngine` — overconfidence + bias detection + calibration curve |
| All goals treated equally | `GoalConflictResolver` — resource/value/priority conflict detection + 3 resolution strategies |
| One strategy for all situations | `StrategyManager` — dynamic EXPLORE/EXPLOIT/VERIFY/RECOVER/COOPERATIVE switching |
| Short-term memory only | `TemporalMemory` — pattern detection, drift tracking, recurrence detection |
| No performance grading | `SelfEvaluator` — 7-dimension A–F grade with improvement suggestions |

## Module quick reference

### MetaCognitionEngine
```python
mc = MetaCognitionEngine(overconf_threshold=0.15, reflection_interval=10)
mc.record_step(step, success, reward, mean_conf, predicate="edible")

over, mc_val, asr = mc.detect_overconfidence()   # True if conf - sr > 0.15
biases = mc.detect_biases()                       # {"edible": 0.62} if 62% error rate
reflections = mc.reflect(step, adjust_fn=cb)      # fires callback to fix beliefs
```

### UncertaintyEngine
```python
ue = UncertaintyEngine(UncertaintyLevel.MEDIUM, seed=42)
obs = ue.corrupt(True, confidence=0.85, predicate="edible")
# NoisyObservation(observed_value=False, was_corrupted=True, confidence_weight=0.74)

ue.decay_all_beliefs(belief_system, steps=1)   # decay toward floor=0.20
ue.auto_detect_level()                          # LOW→MEDIUM→HIGH→CHAOTIC from variance
ue.consensus_supermajority(base=0.55)           # 0.60 at MEDIUM, 0.73 at CHAOTIC
```

### GoalConflictResolver
```python
gcr = GoalConflictResolver(strategy=ResolutionStrategy.UTILITY_MAXIMISATION)
conflicts = gcr.detect_conflicts(active_goals)   # resource / value / priority
decision  = gcr.resolve(active_goals, belief_system, internal_state)
# decision.chosen_goals, decision.dropped_goals, decision.utility_score
```

### StrategyManager
```python
sm = StrategyManager(eval_freq=8, fail_threshold=0.25, win_threshold=0.70)
sm.record_step(step, success, reward, novelty)
event = sm.evaluate(step)   # returns StrategySwitchEvent or None
# Triggers: sr<=0.25 → RECOVER | sr>0.70+low_nov → EXPLOIT | 4 fails → RECOVER
```

### TemporalMemory
```python
tm = TemporalMemory(window=20)
tm.record_outcome("eat","apple", success=True, reward=0.5, step=N)
pattern = tm.detect_pattern("eat","apple")
# TemporalPattern(type="stable", sr=1.00, n=18, confidence=0.95)
tm.belief_drift("apple.edible")    # "rising" | "falling" | "stable" | "oscillating"
tm.is_stuck("eat","mystery", threshold=3)   # True after 3 consecutive failures
```

### SelfEvaluator
```python
se = SelfEvaluator(eval_freq=15, weakness_threshold=0.50)
report = se.evaluate(step=15, agent=agent)
# EvalReport(grade="B", overall=0.72, weaknesses=["communication"],
#   suggestions={"communication": "Call ensure_bidirectional_comm() every step."})
```
