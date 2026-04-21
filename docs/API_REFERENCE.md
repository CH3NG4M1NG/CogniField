# CogniField v10 — API Reference

## `CogniField` class

```python
from cognifield import CogniField
```

### Constructor

```python
CogniField(config=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `dict \| CogniFieldConfig \| None` | Optional configuration. See [Configuration](#configuration). |

---

### `think(input_text)` → `dict`

Run the full multi-agent reasoning pipeline on an input.

**Parameters**
- `input_text` (str): A question, observation, or statement.

**Returns** `dict`:

| Field | Type | Description |
|-------|------|-------------|
| `decision` | str | `"proceed"`, `"avoid"`, `"uncertain"`, `"insufficient_data"`, or a direct value |
| `confidence` | float | Confidence score [0.0, 1.0] |
| `reasoning` | list[str] | Step-by-step reasoning chain |
| `consensus` | dict | Authoritative fleet beliefs with confidence + agreement |
| `meta` | dict | System metadata: agents, strategy, uncertainty, events |
| `llm_output` | str | Natural language explanation (if LLM configured) |
| `elapsed_ms` | float | Processing time in milliseconds |

**Example**
```python
result = cf.think("Is the apple safe to eat?")
# {
#   "decision": "proceed",
#   "confidence": 0.87,
#   "reasoning": ["apple.edible = True (conf=0.874)", "Agreement: 100%"],
#   "consensus": {"apple.edible": {"value": true, "confidence": 0.874}},
#   "meta": {"agents": 3, "strategy": "exploit"},
#   "elapsed_ms": 8.4
# }
```

---

### `decide(input_text)` → `dict`

Make a concrete action decision. Returns all `think()` fields plus:

| Field | Type | Description |
|-------|------|-------------|
| `action` | str | Concrete recommended action (e.g. `"act"`, `"do_not_act"`, `"verify_first"`) |
| `risk_level` | str | `"low"`, `"medium"`, `"high"`, `"critical"` |
| `alternatives` | list[str] | Alternative actions to consider |

**Actions map**

| Decision | Risk | Action |
|----------|------|--------|
| proceed | low | `act` |
| proceed | medium | `act_with_caution` |
| proceed | high | `verify_first` |
| avoid | any | `do_not_act` |
| uncertain | low | `gather_evidence` |
| uncertain | medium | `experiment_safely` |
| insufficient_data | critical | `observe_only` |

---

### `simulate(scenario, steps=10)` → `dict`

Run a multi-step autonomous simulation. Returns:

| Field | Type | Description |
|-------|------|-------------|
| `scenario` | str | The input scenario |
| `steps_run` | int | Total agent steps executed |
| `success_rate` | float | Fraction of successful actions |
| `belief_changes` | int | Net increase in reliable beliefs |
| `strategy` | str | Active strategy at end |
| `outcomes` | list[str] | Sample action outcomes |
| `consensus` | dict | Final authoritative beliefs |
| `llm_output` | str | LLM narrative (if configured) |

---

### `teach(label, properties, text=None)` → `CogniField`

Inject world knowledge into the agent fleet.

**Parameters**
- `label` (str): Concept name (e.g. `"apple"`, `"stone"`).
- `properties` (dict): Property map (e.g. `{"edible": True, "category": "food"}`).
- `text` (str, optional): Free-text description.

**Returns** `self` (fluent chaining).

**Example**
```python
cf.teach("apple",  {"edible": True,  "category": "food"})
  .teach("poison", {"edible": False, "toxic": True})
```

---

### `add_goal(goal_label, priority=0.8)` → `CogniField`

Add a goal to all agents in the fleet.

**Returns** `self`.

---

### `get_beliefs(min_confidence=0.60)` → `dict`

Return authoritative fleet beliefs above the confidence threshold.

```python
beliefs = cf.get_beliefs(min_confidence=0.70)
# {
#   "apple.edible": {"value": true, "confidence": 0.874, "agreement": 1.0, "version": 3},
#   "stone.edible": {"value": false, "confidence": 0.851, "agreement": 0.87, "version": 2}
# }
```

---

### `status()` → `dict`

Return a concise system status snapshot.

```python
cf.status()
# {
#   "version": "10.0",
#   "agents": 3,
#   "llm_backend": "MockLLM()",
#   "llm_available": true,
#   "shared_beliefs": 12,
#   "think_calls": 5,
#   "config": {"uncertainty": "low", "strategy": "adaptive"}
# }
```

---

### `reset()` → `CogniField`

Clear call history and step counters. Agent knowledge is preserved.

---

## `CogniFieldConfig`

```python
from cognifield import CogniFieldConfig
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agents` | int | 3 | Number of reasoning agents (1–10) |
| `uncertainty` | str | `"low"` | `none` / `low` / `medium` / `high` / `chaotic` |
| `enable_meta` | bool | True | Enable meta-cognition |
| `strategy` | str | `"adaptive"` | Initial strategy |
| `dim` | int | 64 | Latent space dimensionality |
| `seed` | int | 42 | Random seed |
| `llm` | str | `"mock"` | LLM backend: `mock` / `ollama` / `api` |
| `llm_model` | str | `"llama3"` | Model name |
| `llm_base_url` | str | `"http://localhost:11434"` | Ollama or compatible API URL |
| `llm_api_key` | str | `""` | API key (or set `OPENAI_API_KEY`) |
| `verbose` | bool | False | Print internal steps |

---

## REST API

### `GET /health`

```json
{
  "status": "ok",
  "version": "10.0",
  "agents": 3,
  "llm": "MockLLM()",
  "config": {"uncertainty": "low", "strategy": "adaptive"}
}
```

### `POST /think`

Request: `{"input": "Is this safe?"}`
Response: same as `CogniField.think()`.

### `POST /decide`

Request: `{"input": "Should I eat this?"}`
Response: same as `CogniField.decide()`.

### `POST /simulate`

Request: `{"scenario": "foraging in forest", "steps": 10}`
Response: same as `CogniField.simulate()`.

### `POST /teach`

Request:
```json
{
  "label": "apple",
  "properties": {"edible": true, "category": "food"},
  "text": "apple is a red edible fruit"
}
```
Response: `{"status": "taught", "label": "apple", "properties": {...}}`

### `GET /beliefs?min_confidence=0.60`

Response:
```json
{
  "beliefs": {"apple.edible": {"value": true, "confidence": 0.874}},
  "n_beliefs": 4,
  "min_confidence": 0.6
}
```

---

## LLM Clients

```python
from cognifield.llm.base import create_llm_client, OllamaClient, APIClient, MockLLM

# Factory (recommended)
client = create_llm_client("ollama", model="llama3")
client = create_llm_client("api",    api_key="sk-...", model="gpt-4o-mini")
client = create_llm_client("mock")

# Direct construction
ollama = OllamaClient(model="llama3", base_url="http://localhost:11434")
api    = APIClient(api_key="sk-...", model="gpt-4o-mini")
mock   = MockLLM()

# Interface
client.is_available()              # bool
client.generate(prompt, max_tokens=512)  # str
```

---

## CLI Reference

```
python -m cognifield [INPUT] [OPTIONS]

Arguments:
  INPUT                   Question or scenario text

Options:
  --mode    think|decide|simulate   Operation mode (default: think)
  --steps   INT                     Simulation steps (default: 8)
  --agents  INT                     Number of agents (default: 3)
  --uncertainty  none|low|medium|high|chaotic
  --llm     mock|ollama|api
  --llm-model   STR                 Model name
  --llm-url     STR                 Ollama base URL
  --llm-key     STR                 API key
  --quiet                           decision + confidence only
  --json                            Raw JSON output
  --status                          Print system status
  --version                         Print version
```
