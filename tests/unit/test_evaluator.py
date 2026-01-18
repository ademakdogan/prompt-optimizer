"""
Unit tests for evaluator module.
"""

import pytest

from prompt_optimizer.core.evaluator import Evaluator, EvaluationResult
from prompt_optimizer.models import PIIEntity, PIIResponse, ErrorFeedback


class TestEvaluator:
    """Tests for Evaluator class."""

    def test_evaluate_perfect_match(self) -> None:
        """Test evaluation with perfect match."""
        evaluator = Evaluator()
        
        entity = PIIEntity(value="John", label="FIRSTNAME", start=0, end=4)
        response = PIIResponse(entities=[entity], masked_text="[FIRSTNAME]")
        ground_truth = PIIResponse(entities=[entity], masked_text="[FIRSTNAME]")
        
        result = evaluator.evaluate_single(response, ground_truth)
        
        assert result.is_correct is True
        assert result.accuracy == 1.0
        assert len(result.missing_entities) == 0
        assert len(result.extra_entities) == 0

    def test_evaluate_missing_entity(self) -> None:
        """Test evaluation with missing entity."""
        evaluator = Evaluator()
        
        response = PIIResponse(entities=[], masked_text="")
        ground_truth = PIIResponse(
            entities=[PIIEntity(value="John", label="FIRSTNAME", start=0, end=4)],
            masked_text="[FIRSTNAME]",
        )
        
        result = evaluator.evaluate_single(response, ground_truth)
        
        assert result.is_correct is False
        assert result.accuracy == 0.0
        assert len(result.missing_entities) == 1
        assert result.missing_entities[0].label == "FIRSTNAME"

    def test_evaluate_extra_entity(self) -> None:
        """Test evaluation with extra (false positive) entity."""
        evaluator = Evaluator()
        
        response = PIIResponse(
            entities=[PIIEntity(value="test", label="EMAIL", start=0, end=4)],
            masked_text="[EMAIL]",
        )
        ground_truth = PIIResponse(entities=[], masked_text="")
        
        result = evaluator.evaluate_single(response, ground_truth)
        
        assert result.is_correct is False
        assert len(result.extra_entities) == 1

    def test_evaluate_partial_match(self) -> None:
        """Test evaluation with partial match."""
        evaluator = Evaluator()
        
        response = PIIResponse(
            entities=[
                PIIEntity(value="John", label="FIRSTNAME", start=0, end=4),
            ],
            masked_text="[FIRSTNAME] test",
        )
        ground_truth = PIIResponse(
            entities=[
                PIIEntity(value="John", label="FIRSTNAME", start=0, end=4),
                PIIEntity(value="test@email.com", label="EMAIL", start=10, end=24),
            ],
            masked_text="[FIRSTNAME] [EMAIL]",
        )
        
        result = evaluator.evaluate_single(response, ground_truth)
        
        assert result.is_correct is False
        assert result.accuracy == 0.5
        assert len(result.missing_entities) == 1

    def test_evaluate_batch(self) -> None:
        """Test batch evaluation."""
        evaluator = Evaluator()
        
        # Create two responses and ground truths
        response1 = PIIResponse(
            entities=[PIIEntity(value="John", label="FIRSTNAME", start=0, end=4)],
            masked_text="[FIRSTNAME]",
        )
        ground_truth1 = PIIResponse(
            entities=[PIIEntity(value="John", label="FIRSTNAME", start=0, end=4)],
            masked_text="[FIRSTNAME]",
        )
        
        response2 = PIIResponse(entities=[], masked_text="")
        ground_truth2 = PIIResponse(
            entities=[PIIEntity(value="test@email.com", label="EMAIL", start=0, end=14)],
            masked_text="[EMAIL]",
        )
        
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
                [PIIResponse()],
                [PIIResponse(), PIIResponse()],
            )

    def test_collect_errors(self) -> None:
        """Test error collection for mentor feedback."""
        evaluator = Evaluator()
        
        response = PIIResponse(entities=[], masked_text="")
        ground_truth = PIIResponse(
            entities=[PIIEntity(value="John", label="FIRSTNAME", start=0, end=4)],
            masked_text="[FIRSTNAME]",
        )
        
        errors = evaluator.collect_errors(
            responses=[response],
            ground_truths=[ground_truth],
            source_texts=["Hello John"],
            prompt="Extract PII",
        )
        
        assert len(errors) == 1
        assert errors[0].field_name == "FIRSTNAME"

    def test_collect_errors_unique_by_field(self) -> None:
        """Test that errors are grouped by field type."""
        evaluator = Evaluator()
        
        # Two samples with same missing field type
        response1 = PIIResponse(entities=[], masked_text="")
        response2 = PIIResponse(entities=[], masked_text="")
        
        ground_truth1 = PIIResponse(
            entities=[PIIEntity(value="test1@email.com", label="EMAIL", start=0, end=15)],
            masked_text="[EMAIL]",
        )
        ground_truth2 = PIIResponse(
            entities=[PIIEntity(value="test2@email.com", label="EMAIL", start=0, end=15)],
            masked_text="[EMAIL]",
        )
        
        errors = evaluator.collect_errors(
            responses=[response1, response2],
            ground_truths=[ground_truth1, ground_truth2],
            source_texts=["Email: test1@email.com", "Email: test2@email.com"],
            prompt="Extract PII",
        )
        
        # Should only have one unique error for EMAIL
        assert len(errors) == 1
        assert errors[0].field_name == "EMAIL"

    def test_get_error_summary(self) -> None:
        """Test error summary generation."""
        evaluator = Evaluator()
        
        result = EvaluationResult(
            is_correct=False,
            accuracy=0.5,
            missing_entities=[
                PIIEntity(value="John", label="FIRSTNAME", start=0, end=4),
                PIIEntity(value="test@email.com", label="EMAIL", start=5, end=19),
            ],
            extra_entities=[],
            wrong_labels=[],
        )
        
        summary = evaluator.get_error_summary([result])
        
        assert summary["MISSING_FIRSTNAME"] == 1
        assert summary["MISSING_EMAIL"] == 1

    def test_strict_position_mode(self) -> None:
        """Test evaluator in strict position mode."""
        evaluator = Evaluator(strict_position=True)
        
        # Same value and label but different positions
        response = PIIResponse(
            entities=[PIIEntity(value="John", label="FIRSTNAME", start=0, end=4)],
            masked_text="[FIRSTNAME]",
        )
        ground_truth = PIIResponse(
            entities=[PIIEntity(value="John", label="FIRSTNAME", start=10, end=14)],
            masked_text="[FIRSTNAME]",
        )
        
        result = evaluator.evaluate_single(response, ground_truth)
        
        # Should be incorrect because positions don't match
        assert result.is_correct is False


class TestEvaluationResult:
    """Tests for EvaluationResult named tuple."""

    def test_evaluation_result_creation(self) -> None:
        """Test creating an EvaluationResult."""
        result = EvaluationResult(
            is_correct=True,
            accuracy=1.0,
            missing_entities=[],
            extra_entities=[],
            wrong_labels=[],
        )
        
        assert result.is_correct is True
        assert result.accuracy == 1.0
