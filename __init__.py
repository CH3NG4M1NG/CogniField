"""
CogniField v10 — Adaptive Multi-Agent Cognitive Intelligence Framework
=======================================================================

Quick start:

    from cognifield import CogniField

    cf = CogniField()
    result = cf.think("Is this berry safe to eat?")
    print(result["decision"])

With LLM:

    cf = CogniField({"llm": "ollama", "llm_model": "llama3"})
    # or
    cf = CogniField({"llm": "api", "llm_api_key": "sk-..."})
"""

__version__ = "10.0.0"
__author__  = "CogniField Project"

from .cognifield_main import CogniField, CogniFieldConfig

__all__ = ["CogniField", "CogniFieldConfig", "__version__"]
