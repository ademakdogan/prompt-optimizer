"""
Unit tests for evaluator module.
"""

import pytest

from prompt_optimizer.core.evaluator import Evaluator, EvaluationResult
from prompt_optimizer.models import TargetResult, ErrorFeedback


class TestEvaluator:
    """Tests for Evaluator class."""

    def test_evaluate_perfect_match(self) -> None:
        """Test evaluation with perfect match."""
        evaluator = Evaluator()
        
        response = TargetResult(firstname="John", email="test@email.com")
        ground_truth = TargetResult(firstname="John", email="test@email.com")
        
        result = evaluator.evaluate_single(response, ground_truth)
        
        assert result.is_correct is True
        assert result.accuracy == 1.0
        assert len(result.missing_fields) == 0
        assert len(result.extra_fields) == 0

    def test_evaluate_missing_field(self) -> None:
        """Test evaluation with missing field."""
        evaluator = Evaluator()
        
        response = TargetResult()  # Empty
        ground_truth = TargetResult(firstname="John")
        
        result = evaluator.evaluate_single(response, ground_truth)
        
        assert result.is_correct is False
        assert result.accuracy == 0.0
        assert len(result.missing_fields) == 1
        assert "firstname" in result.missing_fields

    def test_evaluate_extra_field(self) -> None:
        """Test evaluation with extra (false positive) field."""
        evaluator = Evaluator()
        
        response = TargetResult(email="test@email.com")
        ground_truth = TargetResult()  # Empty
        
        result = evaluator.evaluate_single(response, ground_truth)
        
        assert result.is_correct is False
        assert len(result.extra_fields) == 1

    def test_evaluate_partial_match(self) -> None:
        """Test evaluation with partial match."""
        evaluator = Evaluator()
        
        response = TargetResult(firstname="John")
        ground_truth = TargetResult(firstname="John", email="test@email.com")
        
        result = evaluator.evaluate_single(response, ground_truth)
        
        assert result.is_correct is False
        assert result.accuracy == 0.5
        assert len(result.missing_fields) == 1

    def test_evaluate_batch(self) -> None:
        """Test batch evaluation."""
        evaluator = Evaluator()
        
        response1 = TargetResult(firstname="John")
        ground_truth1 = TargetResult(firstname="John")
        
        response2 = TargetResult()
        ground_truth2 = TargetResult(email="test@email.com")
        
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
                [TargetResult()],
                [TargetResult(), TargetResult()],
            )

    def test_collect_errors(self) -> None:
        """Test error collection for mentor feedback."""
        evaluator = Evaluator()
        
        response = TargetResult()
        ground_truth = TargetResult(firstname="John")
        
        errors = evaluator.collect_errors(
            responses=[response],
            ground_truths=[ground_truth],
            source_texts=["Hello John"],
            prompt="Extract PII",
        )
        
        assert len(errors) == 1
        assert errors[0].field_name == "firstname"

    def test_collect_errors_unique_by_field(self) -> None:
        """Test that errors are grouped by field type."""
        evaluator = Evaluator()
        
        # Two samples with same missing field type
        response1 = TargetResult()
        response2 = TargetResult()
        
        ground_truth1 = TargetResult(email="test1@email.com")
        ground_truth2 = TargetResult(email="test2@email.com")
        
        errors = evaluator.collect_errors(
            responses=[response1, response2],
            ground_truths=[ground_truth1, ground_truth2],
            source_texts=["Email: test1@email.com", "Email: test2@email.com"],
            prompt="Extract PII",
        )
        
        # Should only have one unique error for email
        assert len(errors) == 1
        assert errors[0].field_name == "email"

    def test_get_error_summary(self) -> None:
        """Test error summary generation."""
        evaluator = Evaluator()
        
        result = EvaluationResult(
            is_correct=False,
            accuracy=0.5,
            missing_fields=["firstname", "email"],
            extra_fields=[],
            wrong_values=[],
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
        )
        
        assert result.is_correct is True
        assert result.accuracy == 1.0
