"""
Models module for Prompt Optimizer.

This module provides Pydantic models for:
- Error feedback structures
- Optimization results
- Generated prompts
- Extraction schema
"""

from prompt_optimizer.models.schemas import (
    ErrorFeedback,
    OptimizationResult,
    PromptHistory,
    MentorPromptRequest,
    GeneratedPrompt,
)
from prompt_optimizer.models.agent_model import (
    ExtractionSchema,
    generate_default_prompt,
    get_schema_field_descriptions,
)

__all__ = [
    "ErrorFeedback",
    "OptimizationResult",
    "PromptHistory",
    "MentorPromptRequest",
    "GeneratedPrompt",
    "ExtractionSchema",
    "generate_default_prompt",
    "get_schema_field_descriptions",
]


