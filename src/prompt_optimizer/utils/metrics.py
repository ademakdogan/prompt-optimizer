"""
Metrics utilities for tracking optimization performance.

This module provides utilities for calculating and tracking
various performance metrics during optimization.
"""

from dataclasses import dataclass
from typing import Optional

from prompt_optimizer.models import OptimizationResult
from prompt_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationMetrics:
    """
    Aggregated metrics from optimization run.

    Attributes:
        total_iterations: Number of iterations completed.
        best_accuracy: Highest accuracy achieved.
        best_iteration: Iteration number with best accuracy.
        final_accuracy: Accuracy in the final iteration.
        accuracy_improvement: Change from first to last iteration.
        avg_accuracy: Average accuracy across all iterations.
        total_errors: Total unique error types encountered.
    """

    total_iterations: int
    best_accuracy: float
    best_iteration: int
    final_accuracy: float
    accuracy_improvement: float
    avg_accuracy: float
    total_errors: int


def calculate_metrics(results: list[OptimizationResult]) -> Optional[OptimizationMetrics]:
    """
    Calculate aggregated metrics from optimization results.

    Args:
        results: List of optimization results.

    Returns:
        Optional[OptimizationMetrics]: Calculated metrics, or None if no results.

    Examples:
        >>> from prompt_optimizer.models import OptimizationResult
        >>> results = [result1, result2]
        >>> metrics = calculate_metrics(results)
        >>> metrics.best_accuracy
        0.95
    """
    if not results:
        return None
    
    best_result = max(results, key=lambda r: r.accuracy)
    
    return OptimizationMetrics(
        total_iterations=len(results),
        best_accuracy=best_result.accuracy,
        best_iteration=best_result.iteration,
        final_accuracy=results[-1].accuracy,
        accuracy_improvement=results[-1].accuracy - results[0].accuracy,
        avg_accuracy=sum(r.accuracy for r in results) / len(results),
        total_errors=sum(len(r.errors) for r in results),
    )


def format_metrics_report(metrics: OptimizationMetrics) -> str:
    """
    Format metrics as a human-readable report.

    Args:
        metrics: The optimization metrics.

    Returns:
        str: Formatted report string.

    Examples:
        >>> metrics = OptimizationMetrics(...)
        >>> report = format_metrics_report(metrics)
        >>> "Best accuracy" in report
        True
    """
    return f"""
╔══════════════════════════════════════════════════════════╗
║                 OPTIMIZATION METRICS                     ║
╠══════════════════════════════════════════════════════════╣
║  Total iterations:      {metrics.total_iterations:>4}                              ║
║  Best accuracy:         {metrics.best_accuracy:>6.2%} (iteration {metrics.best_iteration})           ║
║  Final accuracy:        {metrics.final_accuracy:>6.2%}                            ║
║  Accuracy improvement:  {metrics.accuracy_improvement:>+6.2%}                            ║
║  Average accuracy:      {metrics.avg_accuracy:>6.2%}                            ║
║  Total error types:     {metrics.total_errors:>4}                              ║
╚══════════════════════════════════════════════════════════╝
"""
