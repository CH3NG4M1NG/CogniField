# CogniField — 5-Minute Quickstart

## Step 1: Install

```bash
pip install numpy scipy scikit-learn Pillow
# Clone or download the package, then:
pip install -e .         # or: pip install cognifield
```

## Step 2: Basic reasoning in 3 lines

```python
from cognifield import CogniField

cf = CogniField()
print(cf.think("Is this berry safe?"))
```

Output:
```json
{
  "decision": "insufficient_data",
  "confidence": 0.25,
  "reasoning": ["No relevant beliefs found for this input.", "..."]
}
```

The system starts with no world knowledge — you teach it.

## Step 3: Teach facts, then reason

```python
cf = CogniField()

# Teach using fluent chaining
(cf
 .teach("apple",  {"edible": True,  "category": "food"})
 .teach("stone",  {"edible": False, "category": "material"})
 .teach("bread",  {"edible": True,  "category": "food"})
 .teach("mushroom_x", {"edible": False, "toxic": True})
)

# Now reason
print(cf.think("Can I eat the apple?")["decision"])       # proceed
print(cf.think("Is the stone safe to eat?")["decision"])  # avoid
print(cf.think("Should I eat mushroom_x?")["decision"])   # avoid
```

## Step 4: Get action decisions with risk assessment

```python
decision = cf.decide("Should I eat the apple?")

print(decision["decision"])     # proceed
print(decision["action"])       # act
print(decision["risk_level"])   # low
print(decision["alternatives"]) # ["proceed cautiously", ...]
```

## Step 5: Run a simulation

```python
sim = cf.simulate("foraging for food in an unknown environment", steps=10)

print(sim["success_rate"])    # 0.8
print(sim["strategy"])        # "explore"
print(sim["outcomes"][:3])    # ["pick(apple): ✓", "eat(apple): ✓", ...]
```

## Step 6: Use the CLI

```bash
# Quick question
python -m cognifield "Is this mushroom safe?"

# Decision mode
python -m cognifield "Should I eat the red berry?" --mode decide

# Quiet output
python -m cognifield "Is apple safe?" --quiet
# Output: proceed  (confidence=75%)
```

## Step 7: Start the REST API

```bash
python -m cognifield.api.server
```

Then in another terminal:
```bash
curl -X POST http://localhost:8000/think \
     -H "Content-Type: application/json" \
     -d '{"input": "Is this berry safe?"}'
```

## Step 8: Add an LLM (optional)

```bash
# Install Ollama from https://ollama.ai
ollama pull llama3
```

```python
cf = CogniField({"llm": "ollama", "llm_model": "llama3"})
result = cf.think("Is this berry safe?")
print(result["llm_output"])  # Natural language from llama3
```

That's it. For more details see the [full README](../README.md).
