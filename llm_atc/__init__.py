# llm_atc/__init__.py
"""
LLM-ATC-HAL: Embodied LLM Air Traffic Controller with Safety Metrics
====================================================================

A research framework for evaluating Large Language Model performance
in safety-critical air traffic control scenarios with hallucination
detection and mitigation.
"""

__version__ = "0.1.0"
__author__ = "LLM-ATC-HAL Research Team"

# Import modules explicitly to avoid F403 star import issues
from . import agents, memory, metrics, tools

# Core exports
__all__ = [
    "agents",
    "memory",
    "metrics",
    "tools",
]
