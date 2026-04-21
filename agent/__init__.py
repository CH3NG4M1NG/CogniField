"""Compatibility shim — agent/ merged into agents/ in v7."""
from ..agents.base_agent import CogniFieldAgent, AgentConfig
from ..agents.goals import GoalSystem, GoalType, Goal
from ..agents.internal_state import InternalState
from ..agents.risk_engine import RiskEngine
from ..agents.goal_generator import GoalGenerator
