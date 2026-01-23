"""
Unit tests for metrics utilities.
"""

import pytest

from prompt_optimizer.models import OptimizationResult
from prompt_optimizer.utils.metrics import (
    OptimizationMetrics,
    calculate_metrics,
    format_metrics_report,
)


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_empty_results(self) -> None:
        """Test with empty results list."""
        result = calculate_metrics([])
        assert result is None

    def test_single_result(self) -> None:
        """Test with single result."""
        results = [
            OptimizationResult(
                iteration=1,
                prompt="Test",
                accuracy=0.85,
                total_samples=10,
                correct_samples=8.5,
                errors=[],
            ),
        ]
        
        metrics = calculate_metrics(results)
        
        assert metrics is not None
        assert metrics.total_iterations == 1
        assert metrics.best_accuracy == 0.85
        assert metrics.best_iteration == 1
        assert metrics.final_accuracy == 0.85
        assert metrics.accuracy_improvement == 0.0

    def test_multiple_results(self) -> None:
        """Test with multiple results."""
        results = [
            OptimizationResult(
                iteration=1,
                prompt="Test 1",
                accuracy=0.70,
                total_samples=10,
                correct_samples=7,
                errors=[],
            ),
            OptimizationResult(
                iteration=2,
                prompt="Test 2",
                accuracy=0.90,
                total_samples=10,
                correct_samples=9,
                errors=[],
            ),
            OptimizationResult(
                iteration=3,
                prompt="Test 3",
                accuracy=0.85,
                total_samples=10,
                correct_samples=8.5,
                errors=[],
            ),
        ]
        
        metrics = calculate_metrics(results)
        
        assert metrics is not None
        assert metrics.total_iterations == 3
        assert metrics.best_accuracy == 0.90
        assert metrics.best_iteration == 2
        assert metrics.final_accuracy == 0.85
        assert metrics.accuracy_improvement == 0.15  # 0.85 - 0.70


class TestFormatMetricsReport:
    """Tests for format_metrics_report function."""

    def test_format_report(self) -> None:
        """Test formatting metrics report."""
        metrics = OptimizationMetrics(
            total_iterations=3,
            best_accuracy=0.95,
            best_iteration=2,
            final_accuracy=0.90,
            accuracy_improvement=0.10,
            avg_accuracy=0.85,
            total_errors=5,
        )
        
        report = format_metrics_report(metrics)
        
        assert "OPTIMIZATION METRICS" in report
        assert "Best accuracy" in report
        assert "95.00%" in report
        assert "iteration 2" in report
