"""
Core module for Prompt Optimizer.

This module provides core functionality for:
- Response evaluation against ground truth
- Prompt optimization loop
"""

from prompt_optimizer.core.evaluator import Evaluator, EvaluationResult
from prompt_optimizer.core.optimizer import PromptOptimizer

__all__ = ["Evaluator", "EvaluationResult", "PromptOptimizer"]
