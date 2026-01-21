"""
Unit tests for persistence utilities.
"""

from pathlib import Path

import pytest

from prompt_optimizer.models import OptimizationResult, ErrorFeedback
from prompt_optimizer.utils.persistence import (
    save_results,
    load_results,
    get_best_prompt,
)


class TestSaveResults:
    """Tests for save_results function."""

    def test_save_empty_results(self, tmp_path: Path) -> None:
        """Test saving empty results list."""
        output_file = tmp_path / "results.json"
        save_results([], output_file)
        
        assert output_file.exists()
        data = load_results(output_file)
        assert data["total_iterations"] == 0
        assert data["best_accuracy"] == 0.0

    def test_save_results_with_data(self, tmp_path: Path) -> None:
        """Test saving results with data."""
        results = [
            OptimizationResult(
                iteration=1,
                prompt="Test prompt",
                accuracy=0.85,
                total_samples=10,
                correct_samples=8.5,
                errors=[],
            ),
            OptimizationResult(
                iteration=2,
                prompt="Better prompt",
                accuracy=0.95,
                total_samples=10,
                correct_samples=9.5,
                errors=[],
            ),
        ]
        
        output_file = tmp_path / "results.json"
        save_results(results, output_file)
        
        data = load_results(output_file)
        assert data["total_iterations"] == 2
        assert data["best_accuracy"] == 0.95

    def test_save_results_with_errors(self, tmp_path: Path) -> None:
        """Test saving results with error details."""
        error = ErrorFeedback(
            field_name="EMAIL",
            prompt="Test",
            source_text="Contact test@email.com",
            agent_answer="No email found",
            ground_truth="test@email.com",
        )
        
        results = [
            OptimizationResult(
                iteration=1,
                prompt="Test prompt",
                accuracy=0.5,
                total_samples=2,
                correct_samples=1,
                errors=[error],
            ),
        ]
        
        output_file = tmp_path / "results.json"
        save_results(results, output_file, include_errors=True)
        
        data = load_results(output_file)
        assert len(data["results"][0]["errors"]) == 1
        assert data["results"][0]["errors"][0]["field_name"] == "EMAIL"


class TestLoadResults:
    """Tests for load_results function."""

    def test_load_nonexistent_file(self) -> None:
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_results("/nonexistent/path.json")


class TestGetBestPrompt:
    """Tests for get_best_prompt function."""

    def test_get_best_prompt_empty(self) -> None:
        """Test getting best prompt from empty list."""
        result = get_best_prompt([])
        assert result is None

    def test_get_best_prompt_single(self) -> None:
        """Test getting best prompt from single result."""
        results = [
            OptimizationResult(
                iteration=1,
                prompt="Only prompt",
                accuracy=0.75,
                total_samples=10,
                correct_samples=7.5,
                errors=[],
            ),
        ]
        
        result = get_best_prompt(results)
        assert result == "Only prompt"

    def test_get_best_prompt_multiple(self) -> None:
        """Test getting best prompt from multiple results."""
        results = [
            OptimizationResult(
                iteration=1,
                prompt="Prompt 1",
                accuracy=0.75,
                total_samples=10,
                correct_samples=7.5,
                errors=[],
            ),
            OptimizationResult(
                iteration=2,
                prompt="Best prompt",
                accuracy=0.95,
                total_samples=10,
                correct_samples=9.5,
                errors=[],
            ),
            OptimizationResult(
                iteration=3,
                prompt="Prompt 3",
                accuracy=0.85,
                total_samples=10,
                correct_samples=8.5,
                errors=[],
            ),
        ]
        
        result = get_best_prompt(results)
        assert result == "Best prompt"
