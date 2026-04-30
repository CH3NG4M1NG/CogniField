"""
CogniField v11 — Self-Learning Adaptive Multi-Agent Cognitive Intelligence
===========================================================================

Quick start (v11 — self-learning, no LLM required):

    from cognifield import CogniField

    cf = CogniField()
    result = cf.think("Is this berry safe to eat?")
    print(result["decision"])      # "avoid" — unknown safety rule
    print(result["thinking_steps"]) # 7+ — deep deliberation

Teach facts, then reason:

    cf.teach("apple",  {"edible": True,  "category": "food"})
    cf.teach("stone",  {"edible": False, "category": "material"})

    result = cf.think("Is apple safe?")
    print(result["decision"])   # "proceed"

Learn from outcomes (self-improving):

    cf.learn_from_outcome("ate apple", "apple", "edible",
                           prediction=True, actual=True,
                           action="eat", reward=0.5)

With LLM (optional):

    cf = CogniField({"llm": "ollama", "llm_model": "llama3"})
"""

__version__ = "11.0.0"
__author__  = "CogniField Project"

# v11 is the default — self-learning deep reasoning
from .cognifield_v11 import CogniFieldV11 as CogniField, CogniFieldV11Config as CogniFieldConfig

# Keep v10 accessible for backward compatibility
from .cognifield_main import (
    CogniField    as CogniFieldV10,
    CogniFieldConfig as CogniFieldV10Config,
)

__all__ = [
    "CogniField",
    "CogniFieldConfig",
    "CogniFieldV10",
    "CogniFieldV10Config",
    "__version__",
]
