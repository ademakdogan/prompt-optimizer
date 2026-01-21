"""
Result persistence utilities for saving optimization results.

This module provides utilities for saving and loading
optimization results to/from JSON files.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from prompt_optimizer.models import OptimizationResult
from prompt_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


def save_results(
    results: list[OptimizationResult],
    output_path: str | Path,
    include_errors: bool = True,
) -> None:
    """
    Save optimization results to a JSON file.

    Args:
        results: List of optimization results.
        output_path: Path to save the results.
        include_errors: Whether to include error details.

    Examples:
        >>> from prompt_optimizer.models import OptimizationResult
        >>> results = [OptimizationResult(...)]
        >>> save_results(results, "results.json")
    """
    output_path = Path(output_path)
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "total_iterations": len(results),
        "best_accuracy": max(r.accuracy for r in results) if results else 0.0,
        "results": [],
    }
    
    for result in results:
        result_data = {
            "iteration": result.iteration,
            "prompt": result.prompt,
            "accuracy": result.accuracy,
            "total_samples": result.total_samples,
            "correct_samples": result.correct_samples,
        }
        
        if include_errors:
            result_data["errors"] = [
                {
                    "field_name": e.field_name,
                    "source_text": e.source_text[:200],
                    "agent_answer": e.agent_answer,
                    "ground_truth": e.ground_truth,
                }
                for e in result.errors
            ]
        
        data["results"].append(result_data)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved results to {output_path}")


def load_results(input_path: str | Path) -> dict:
    """
    Load optimization results from a JSON file.

    Args:
        input_path: Path to the results file.

    Returns:
        dict: Loaded results data.

    Raises:
        FileNotFoundError: If the file does not exist.

    Examples:
        >>> data = load_results("results.json")
        >>> data["total_iterations"]
        3
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_best_prompt(results: list[OptimizationResult]) -> Optional[str]:
    """
    Get the prompt with the best accuracy.

    Args:
        results: List of optimization results.

    Returns:
        Optional[str]: The best prompt, or None if no results.

    Examples:
        >>> results = [result1, result2, result3]
        >>> best_prompt = get_best_prompt(results)
        >>> len(best_prompt) > 0
        True
    """
    if not results:
        return None
    
    best = max(results, key=lambda r: r.accuracy)
    return best.prompt
