"""
Unit tests for evaluator module.
"""

import pytest

from prompt_optimizer.core.evaluator import Evaluator, EvaluationResult


class TestEvaluator:
    """Tests for Evaluator class."""

    def test_evaluate_perfect_match(self) -> None:
        """Test evaluation with perfect match."""
        evaluator = Evaluator()
        
        response = {"firstname": "John", "email": "test@email.com"}
        ground_truth = {"firstname": "John", "email": "test@email.com"}
        
        result = evaluator.evaluate_single(response, ground_truth)
        
        assert result.is_correct is True
        assert result.accuracy == 1.0
        assert len(result.missing_fields) == 0
        assert len(result.extra_fields) == 0

    def test_evaluate_missing_field(self) -> None:
        """Test evaluation with missing field."""
        evaluator = Evaluator()
        
        response = {}  # Empty
        ground_truth = {"firstname": "John"}
        
        result = evaluator.evaluate_single(response, ground_truth)
        
        assert result.is_correct is False
        assert result.accuracy == 0.0
        assert len(result.missing_fields) == 1
        assert "firstname" in result.missing_fields

    def test_evaluate_extra_field(self) -> None:
        """Test evaluation with extra (false positive) field."""
        evaluator = Evaluator()
        
        response = {"email": "test@email.com"}
        ground_truth = {}  # Empty
        
        result = evaluator.evaluate_single(response, ground_truth)
        
        assert result.is_correct is False
        assert len(result.extra_fields) == 1

    def test_evaluate_partial_match(self) -> None:
        """Test evaluation with partial match."""
        evaluator = Evaluator()
        
        response = {"firstname": "John"}
        ground_truth = {"firstname": "John", "email": "test@email.com"}
        
        result = evaluator.evaluate_single(response, ground_truth)
        
        assert result.is_correct is False
        assert result.accuracy == 0.5
        assert len(result.missing_fields) == 1

    def test_evaluate_batch(self) -> None:
        """Test batch evaluation."""
        evaluator = Evaluator()
        
        response1 = {"firstname": "John"}
        ground_truth1 = {"firstname": "John"}
        
        response2 = {}
        ground_truth2 = {"email": "test@email.com"}
        
        accuracy, results = evaluator.evaluate_batch(
            [response1, response2],
            [ground_truth1, ground_truth2],
        )
        
        assert len(results) == 2
        assert results[0].is_correct is True
        assert results[1].is_correct is False
        assert accuracy == 0.5

    def test_evaluate_batch_mismatched_lengths(self) -> None:
        """Test batch evaluation with mismatched lengths raises error."""
        evaluator = Evaluator()
        
        with pytest.raises(ValueError, match="Mismatched lengths"):
            evaluator.evaluate_batch(
                [{}],
                [{}, {}],
            )

    def test_collect_errors(self) -> None:
        """Test error collection for mentor feedback."""
        evaluator = Evaluator()
        
        response = {}
        ground_truth = {"firstname": "John"}
        
        # evaluate_single now returns prediction and ground_truth
        result = evaluator.evaluate_single(response, ground_truth)
        
        assert result.prediction == {}
        assert result.ground_truth == {"firstname": "John"}

    def test_get_error_summary(self) -> None:
        """Test error summary generation."""
        evaluator = Evaluator()
        
        result = EvaluationResult(
            is_correct=False,
            accuracy=0.5,
            missing_fields=["firstname", "email"],
            extra_fields=[],
            wrong_values=[],
            prediction={},
            ground_truth={"firstname": "John", "email": "test@mail.com"},
        )
        
        summary = evaluator.get_error_summary([result])
        
        assert summary["MISSING_firstname"] == 1
        assert summary["MISSING_email"] == 1


class TestEvaluationResult:
    """Tests for EvaluationResult named tuple."""

    def test_evaluation_result_creation(self) -> None:
        """Test creating an EvaluationResult."""
        result = EvaluationResult(
            is_correct=True,
            accuracy=1.0,
            missing_fields=[],
            extra_fields=[],
            wrong_values=[],
            prediction={"firstname": "John"},
            ground_truth={"firstname": "John"},
        )
        
        assert result.is_correct is True
        assert result.accuracy == 1.0
        assert result.prediction == {"firstname": "John"}
