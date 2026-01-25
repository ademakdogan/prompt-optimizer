"""
Models module for Prompt Optimizer.

This module provides Pydantic models for:
- Error feedback structures
- Optimization results
- Generated prompts
"""

from prompt_optimizer.models.schemas import (
    PIIEntity,
    PIIResponse,
    ErrorFeedback,
    OptimizationResult,
    PromptHistory,
    MentorPromptRequest,
    GeneratedPrompt,
)

__all__ = [
    "PIIEntity",
    "PIIResponse",
    "ErrorFeedback",
    "OptimizationResult",
    "PromptHistory",
    "MentorPromptRequest",
    "GeneratedPrompt",
]
