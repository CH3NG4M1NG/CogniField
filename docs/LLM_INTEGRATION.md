# LLM Integration Guide

CogniField's LLM layer is optional. Without an LLM, the system provides
structured decisions and reasoning chains. With an LLM, it adds natural
language explanations and can accept more conversational input.

## How it works

```
User input  →  CogniField reasons  →  LLM formats output
               (structured logic)      (natural language)
```

**CogniField always makes the decision. The LLM formats the explanation.**

The LLM receives:
- The user's question
- CogniField's decision and confidence
- The active beliefs from the agent fleet
- The uncertainty level and strategy

It returns a 2–3 sentence natural language explanation.

---

## Option A — Local LLM with Ollama

Best for: privacy, no API costs, offline use.

### Setup

```bash
# 1. Install Ollama
# macOS:   brew install ollama
# Linux:   curl -fsSL https://ollama.ai/install.sh | sh
# Windows: https://ollama.ai/download

# 2. Pull a model (pick one)
ollama pull llama3          # Good balance of speed and quality
ollama pull mistral         # Fast, good reasoning
ollama pull phi3            # Small, very fast
ollama pull llama3:70b      # Best quality, needs 40GB RAM
ollama pull gemma2          # Google's open model

# 3. Ollama auto-starts when you run a model
# Or start manually: ollama serve
```

### Use with CogniField

```python
from cognifield import CogniField

cf = CogniField({
    "llm":         "ollama",
    "llm_model":   "llama3",
    "llm_base_url":"http://localhost:11434",  # default
})

result = cf.think("Is this mushroom safe to eat?")
print(result["decision"])     # "avoid" (from CogniField)
print(result["llm_output"])   # Natural language from llama3
```

### CLI with Ollama

```bash
python -m cognifield "Is this mushroom safe?" --llm ollama
python -m cognifield "Is this mushroom safe?" --llm ollama --llm-model mistral
```

### API server with Ollama

```bash
python -m cognifield.api.server --llm ollama --llm-model llama3
```

---

## Option B — Remote API (OpenAI-compatible)

Best for: highest quality, no local hardware needed.

### OpenAI

```python
import os
from cognifield import CogniField

cf = CogniField({
    "llm":         "api",
    "llm_model":   "gpt-4o-mini",   # or gpt-4o, gpt-3.5-turbo
    "llm_api_key": os.environ["OPENAI_API_KEY"],
    # llm_base_url defaults to "https://api.openai.com/v1"
})
```

### Together AI

```python
cf = CogniField({
    "llm":          "api",
    "llm_model":    "meta-llama/Llama-3-70b-chat-hf",
    "llm_api_key":  os.environ["TOGETHER_API_KEY"],
    "llm_base_url": "https://api.together.xyz/v1",
})
```

### Groq (very fast inference)

```python
cf = CogniField({
    "llm":          "api",
    "llm_model":    "llama3-70b-8192",
    "llm_api_key":  os.environ["GROQ_API_KEY"],
    "llm_base_url": "https://api.groq.com/openai/v1",
})
```

### Mistral AI

```python
cf = CogniField({
    "llm":          "api",
    "llm_model":    "mistral-medium",
    "llm_api_key":  os.environ["MISTRAL_API_KEY"],
    "llm_base_url": "https://api.mistral.ai/v1",
})
```

### Environment variable (recommended)

```bash
# Set once in your shell / .env
export OPENAI_API_KEY=sk-...
export COGNIFIELD_LLM_KEY=sk-...   # alternative env var
```

```python
# Key picked up automatically
cf = CogniField({"llm": "api"})
```

---

## Option C — No LLM (default)

If you don't configure an LLM, CogniField uses `MockLLM` internally.
`MockLLM` generates a basic structured response without a real model.
All reasoning still happens correctly — you just don't get a
natural-language narrative.

```python
cf = CogniField()   # MockLLM by default — fully functional
result = cf.think("Is this safe?")
print(result["decision"])    # The real decision (from CogniField)
print(result["llm_output"])  # Basic mock text (not from a real model)
```

---

## Checking availability

```python
from cognifield.llm.base import OllamaClient, create_llm_client

# Check Ollama
client = OllamaClient(model="llama3")
if client.is_available():
    print("Ollama is running")
    resp = client.generate("Say hello in one word")
    print(resp)
else:
    print("Ollama not found — start with: ollama serve")
```

---

## Custom LLM backend

You can plug in any LLM by subclassing `BaseLLMClient`:

```python
from cognifield.llm.base import BaseLLMClient

class MyCustomLLM(BaseLLMClient):
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        # Call your own LLM here
        response = my_llm_api.chat(prompt)
        return response.text

    def is_available(self) -> bool:
        return True  # always available

# Use it
from cognifield import CogniField
from cognifield.cognifield_main import CogniFieldConfig

cf = CogniField()
cf._llm = MyCustomLLM()   # inject directly
```

---

## Prompt format

CogniField builds a structured prompt that looks like:

```
You are a reasoning assistant integrated with CogniField,
a structured cognitive AI system.

User question: Is this mushroom safe to eat?

CogniField analysis:
- Decision: avoid
- Confidence: 0.85
- Key beliefs: mushroom_x.edible=False(0.85), mushroom_x.toxic=True(0.80)
- Multi-agent consensus: avoid
- Uncertainty level: low
- Active strategy: exploit

Based on this structured analysis, provide a clear, concise answer
to the user's question in 2-3 sentences. Be direct about the confidence level.
```

You can override `format_decision_prompt()` on any client to customise this.
