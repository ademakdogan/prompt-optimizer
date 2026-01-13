"""
Models module for Prompt Optimizer.

This module provides Pydantic models for:
- PII entity representation
- API responses
- Error feedback structures
- Optimization results
"""

from prompt_optimizer.models.schemas import (
    PIIEntity,
    PIIResponse,
    ErrorFeedback,
    OptimizationResult,
    PromptHistory,
)

__all__ = [
    "PIIEntity",
    "PIIResponse",
    "ErrorFeedback",
    "OptimizationResult",
    "PromptHistory",
]
