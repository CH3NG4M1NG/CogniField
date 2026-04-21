"""
llm/base.py
============
Base LLM Client Interface

Defines the contract all LLM backends must satisfy:
  - generate(prompt) -> str
  - is_available() -> bool

Two built-in backends:
  OllamaClient  – talks to a local Ollama server
  APIClient     – OpenAI-compatible HTTP API
  MockLLM       – in-process stub (for testing / no-LLM mode)
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Optional


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """Minimal interface every LLM backend must implement."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate text from a prompt. Returns the model's response."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend is reachable."""

    def format_decision_prompt(self, input_text: str, context: dict) -> str:
        """
        Build a structured prompt from user input + CogniField context.
        Subclasses may override for model-specific formatting.
        """
        beliefs_summary = ", ".join(
            f"{k}={v['value']}({v['confidence']:.2f})"
            for k, v in list(context.get("beliefs", {}).items())[:5]
        )
        consensus_summary = context.get("consensus_value", "unknown")

        return f"""You are a reasoning assistant integrated with CogniField,
a structured cognitive AI system.

User question: {input_text}

CogniField analysis:
- Decision: {context.get('decision', 'unknown')}
- Confidence: {context.get('confidence', 0.5):.2f}
- Key beliefs: {beliefs_summary or 'none'}
- Multi-agent consensus: {consensus_summary}
- Uncertainty level: {context.get('uncertainty', 'medium')}
- Active strategy: {context.get('strategy', 'explore')}

Based on this structured analysis, provide a clear, concise answer
to the user's question in 2-3 sentences. Be direct about the confidence level."""

    def format_simulation_prompt(self, scenario: str, result: dict) -> str:
        outcomes = result.get("outcomes", [])
        outcome_text = "; ".join(outcomes[:3]) if outcomes else "no outcomes recorded"
        return f"""You are interpreting a cognitive simulation result.

Scenario: {scenario}

Simulation outcomes:
- Success rate: {result.get('success_rate', 0):.1%}
- Steps run: {result.get('steps', 0)}
- Key outcomes: {outcome_text}
- Belief changes: {result.get('belief_changes', 0)} beliefs updated
- Strategy used: {result.get('strategy', 'explore')}

Summarize what this simulation tells us about the scenario in 2-3 sentences."""


# ---------------------------------------------------------------------------
# Ollama Client
# ---------------------------------------------------------------------------

class OllamaClient(BaseLLMClient):
    """
    Connects to a locally running Ollama server.

    Install:  https://ollama.ai
    Start:    ollama serve
    Pull:     ollama pull llama3

    Parameters
    ----------
    model    : Ollama model name (default: llama3).
    base_url : Ollama API base URL.
    timeout  : HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model:    str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout:  int = 30,
    ) -> None:
        self.model    = model
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """POST to Ollama /api/generate and return the response text."""
        payload = json.dumps({
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
                return data.get("response", "").strip()
        except (urllib.error.URLError, json.JSONDecodeError, Exception) as e:
            return f"[Ollama error: {e}]"

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3):
                return True
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"OllamaClient(model={self.model}, url={self.base_url})"


# ---------------------------------------------------------------------------
# API Client (OpenAI-compatible)
# ---------------------------------------------------------------------------

class APIClient(BaseLLMClient):
    """
    OpenAI-compatible HTTP API client.
    Works with OpenAI, Together AI, Groq, Mistral, or any compatible endpoint.

    Parameters
    ----------
    api_key   : API key (or set OPENAI_API_KEY / COGNIFIELD_LLM_KEY env var).
    base_url  : API base URL (default: OpenAI).
    model     : Model name.
    timeout   : HTTP timeout.
    """

    DEFAULT_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key:  Optional[str] = None,
        base_url: str = DEFAULT_URL,
        model:    str = "gpt-4o-mini",
        timeout:  int = 30,
    ) -> None:
        self.api_key  = (api_key
                         or os.environ.get("OPENAI_API_KEY")
                         or os.environ.get("COGNIFIELD_LLM_KEY", ""))
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.timeout  = timeout

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if not self.api_key:
            return "[APIClient: no API key configured]"

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[APIClient error: {e}]"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def __repr__(self) -> str:
        masked = f"{self.api_key[:8]}..." if self.api_key else "no-key"
        return f"APIClient(model={self.model}, key={masked})"


# ---------------------------------------------------------------------------
# Mock LLM (testing / no-LLM mode)
# ---------------------------------------------------------------------------

class MockLLM(BaseLLMClient):
    """
    In-process stub that generates structured responses without a real model.
    Used when no LLM is configured, or during testing.
    """

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        # Extract key signals from prompt for a deterministic-ish response
        prompt_lower = prompt.lower()

        if "confidence" in prompt_lower:
            # Look for a confidence value in the prompt
            import re
            m = re.search(r"confidence[:\s]+([0-9.]+)", prompt_lower)
            conf = float(m.group(1)) if m else 0.5
            level = "high" if conf > 0.7 else "moderate" if conf > 0.4 else "low"
            return (f"Based on the structured analysis, the system has {level} confidence "
                    f"in this assessment (confidence={conf:.2f}). "
                    f"The multi-agent consensus supports this conclusion.")

        if "safe" in prompt_lower or "eat" in prompt_lower or "edible" in prompt_lower:
            return ("The cognitive analysis suggests exercising caution. "
                    "Without sufficient evidence, the system recommends "
                    "further investigation before acting.")

        if "simulation" in prompt_lower or "scenario" in prompt_lower:
            return ("The simulation reveals consistent patterns in the scenario. "
                    "Multiple agents converged on similar outcomes, "
                    "suggesting reliable predictability in this context.")

        return ("The CogniField system has processed this input through "
                "its multi-agent reasoning pipeline. "
                "The result reflects the current state of the agent fleet's beliefs.")

    def is_available(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "MockLLM()"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_llm_client(
    backend: str = "mock",
    **kwargs,
) -> BaseLLMClient:
    """
    Create an LLM client by backend name.

    Parameters
    ----------
    backend : "ollama" | "api" | "mock" | "none"
    **kwargs: Passed to the client constructor.

    Returns
    -------
    BaseLLMClient instance.
    """
    backend = backend.lower()
    if backend == "ollama":
        return OllamaClient(**kwargs)
    elif backend in ("api", "openai"):
        return APIClient(**kwargs)
    elif backend in ("mock", "none", ""):
        return MockLLM()
    else:
        raise ValueError(f"Unknown LLM backend: {backend!r}. "
                         f"Use 'ollama', 'api', or 'mock'.")
