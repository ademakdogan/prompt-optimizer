"""
Models module for Prompt Optimizer.

This module provides Pydantic models for:
- PII entity representation
- API responses
- Error feedback structures
- Optimization results
- Dynamic field descriptions
"""

from prompt_optimizer.models.schemas import (
    FieldDescription,
    TargetResult,
    PIIEntity,
    PIIResponse,
    ErrorFeedback,
    OptimizationResult,
    PromptHistory,
    MentorPromptRequest,
    GeneratedPrompt,
)

__all__ = [
    "FieldDescription",
    "TargetResult",
    "PIIEntity",
    "PIIResponse",
    "ErrorFeedback",
    "OptimizationResult",
    "PromptHistory",
    "MentorPromptRequest",
    "GeneratedPrompt",
]

