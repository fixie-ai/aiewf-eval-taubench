"""User simulator for multi-turn agent evaluation.

The user simulator plays the role of a customer interacting with the agent.
It follows the task instruction and responds naturally to agent questions,
providing confirmations, corrections, and additional information as needed.
"""
from .llm_user import LLMUserSimulator, UserSimulatorConfig

__all__ = ["LLMUserSimulator", "UserSimulatorConfig"]

