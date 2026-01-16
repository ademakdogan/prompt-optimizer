"""
Evaluator module for comparing agent responses with ground truth.

This module provides functionality to evaluate PII extraction accuracy
by comparing agent responses against ground truth labels.
"""

from collections import defaultdict
from typing import NamedTuple

from prompt_optimizer.models import ErrorFeedback, PIIEntity, PIIResponse
from prompt_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationResult(NamedTuple):
    """
    Result of evaluating a single sample.

    Attributes:
        is_correct: Whether the extraction was fully correct.
        accuracy: Accuracy score for this sample (0.0 to 1.0).
        missing_entities: Entities in ground truth but not in response.
        extra_entities: Entities in response but not in ground truth.
        wrong_labels: Entities with incorrect labels.
    """
    
    is_correct: bool
    accuracy: float
    missing_entities: list[PIIEntity]
    extra_entities: list[PIIEntity]
    wrong_labels: list[tuple[PIIEntity, PIIEntity]]  # (predicted, expected)


class Evaluator:
    """
    Evaluator for comparing agent responses with ground truth.

    Provides methods to compare individual samples and aggregate
    results across multiple samples.

    Examples:
        >>> evaluator = Evaluator()
        >>> response = PIIResponse(entities=[
        ...     PIIEntity(value="John", label="FIRSTNAME", start=0, end=4)
        ... ], masked_text="[FIRSTNAME] test")
        >>> ground_truth = PIIResponse(entities=[
        ...     PIIEntity(value="John", label="FIRSTNAME", start=0, end=4)
        ... ], masked_text="[FIRSTNAME] test")
        >>> result = evaluator.evaluate_single(response, ground_truth)
        >>> result.is_correct
        True
    """

    def __init__(self, strict_position: bool = False) -> None:
        """
        Initialize the evaluator.

        Args:
            strict_position: If True, also compare start/end positions.
                           If False, only compare value and label.
        """
        self.strict_position = strict_position
        logger.debug(f"Initialized Evaluator (strict_position={strict_position})")

    def evaluate_single(
        self,
        response: PIIResponse,
        ground_truth: PIIResponse,
    ) -> EvaluationResult:
        """
        Evaluate a single response against ground truth.

        Args:
            response: The agent's response.
            ground_truth: The expected ground truth.

        Returns:
            EvaluationResult: Detailed evaluation result.

        Examples:
            >>> evaluator = Evaluator()
            >>> response = PIIResponse(entities=[], masked_text="")
            >>> ground_truth = PIIResponse(entities=[
            ...     PIIEntity(value="test@email.com", label="EMAIL", start=0, end=14)
            ... ], masked_text="[EMAIL]")
            >>> result = evaluator.evaluate_single(response, ground_truth)
            >>> result.is_correct
            False
            >>> len(result.missing_entities)
            1
        """
        response_entities = {self._entity_key(e): e for e in response.entities}
        truth_entities = {self._entity_key(e): e for e in ground_truth.entities}
        
        response_keys = set(response_entities.keys())
        truth_keys = set(truth_entities.keys())
        
        # Find missing, extra, and common entities
        missing_keys = truth_keys - response_keys
        extra_keys = response_keys - truth_keys
        
        missing_entities = [truth_entities[k] for k in missing_keys]
        extra_entities = [response_entities[k] for k in extra_keys]
        
        # Check for wrong labels (same value but different label)
        wrong_labels = []
        for resp_entity in response.entities:
            for truth_entity in ground_truth.entities:
                if (
                    resp_entity.value.lower() == truth_entity.value.lower()
                    and resp_entity.label != truth_entity.label
                ):
                    wrong_labels.append((resp_entity, truth_entity))
        
        # Calculate accuracy
        total_entities = len(ground_truth.entities)
        if total_entities == 0:
            # No entities expected - correct if also no response entities
            accuracy = 1.0 if len(response.entities) == 0 else 0.0
        else:
            correct_count = len(truth_keys & response_keys)
            accuracy = correct_count / total_entities
        
        is_correct = (
            len(missing_entities) == 0
            and len(extra_entities) == 0
            and len(wrong_labels) == 0
        )
        
        return EvaluationResult(
            is_correct=is_correct,
            accuracy=accuracy,
            missing_entities=missing_entities,
            extra_entities=extra_entities,
            wrong_labels=wrong_labels,
        )

    def evaluate_batch(
        self,
        responses: list[PIIResponse],
        ground_truths: list[PIIResponse],
    ) -> tuple[float, list[EvaluationResult]]:
        """
        Evaluate a batch of responses.

        Args:
            responses: List of agent responses.
            ground_truths: List of corresponding ground truths.

        Returns:
            tuple: (overall_accuracy, list of individual results)

        Raises:
            ValueError: If responses and ground_truths have different lengths.

        Examples:
            >>> evaluator = Evaluator()
            >>> responses = [PIIResponse(entities=[], masked_text="")]
            >>> ground_truths = [PIIResponse(entities=[], masked_text="")]
            >>> accuracy, results = evaluator.evaluate_batch(responses, ground_truths)
            >>> accuracy
            1.0
        """
        if len(responses) != len(ground_truths):
            raise ValueError(
                f"Mismatched lengths: {len(responses)} responses vs "
                f"{len(ground_truths)} ground truths"
            )
        
        results = []
        total_accuracy = 0.0
        
        for response, ground_truth in zip(responses, ground_truths):
            result = self.evaluate_single(response, ground_truth)
            results.append(result)
            total_accuracy += result.accuracy
        
        overall_accuracy = total_accuracy / len(results) if results else 0.0
        
        logger.info(
            f"Batch evaluation: {overall_accuracy:.2%} accuracy "
            f"({sum(r.is_correct for r in results)}/{len(results)} correct)"
        )
        
        return overall_accuracy, results

    def collect_errors(
        self,
        responses: list[PIIResponse],
        ground_truths: list[PIIResponse],
        source_texts: list[str],
        prompt: str,
    ) -> list[ErrorFeedback]:
        """
        Collect unique errors for mentor feedback.

        Args:
            responses: List of agent responses.
            ground_truths: List of ground truths.
            source_texts: List of source text inputs.
            prompt: The prompt used for extraction.

        Returns:
            list[ErrorFeedback]: List of unique error feedback entries.

        Examples:
            >>> evaluator = Evaluator()
            >>> responses = [PIIResponse(entities=[], masked_text="")]
            >>> ground_truths = [PIIResponse(entities=[
            ...     PIIEntity(value="test@email.com", label="EMAIL", start=0, end=14)
            ... ], masked_text="[EMAIL]")]
            >>> errors = evaluator.collect_errors(
            ...     responses, ground_truths, ["Contact: test@email.com"], "Extract PII"
            ... )
            >>> len(errors) > 0
            True
        """
        # Group errors by field type to get unique examples
        errors_by_field: dict[str, ErrorFeedback] = {}
        
        for response, ground_truth, source_text in zip(
            responses, ground_truths, source_texts
        ):
            result = self.evaluate_single(response, ground_truth)
            
            # Missing entities
            for entity in result.missing_entities:
                if entity.label not in errors_by_field:
                    errors_by_field[entity.label] = ErrorFeedback(
                        field_name=entity.label,
                        prompt=prompt,
                        source_text=source_text,
                        agent_answer=f"Missing: {entity.label} not detected",
                        ground_truth=f"{entity.label}: {entity.value}",
                    )
            
            # Extra entities (false positives)
            for entity in result.extra_entities:
                field_key = f"FP_{entity.label}"
                if field_key not in errors_by_field:
                    errors_by_field[field_key] = ErrorFeedback(
                        field_name=f"FALSE_POSITIVE_{entity.label}",
                        prompt=prompt,
                        source_text=source_text,
                        agent_answer=f"Incorrectly detected: {entity.value} as {entity.label}",
                        ground_truth="No such entity exists",
                    )
            
            # Wrong labels
            for predicted, expected in result.wrong_labels:
                field_key = f"WRONG_{expected.label}"
                if field_key not in errors_by_field:
                    errors_by_field[field_key] = ErrorFeedback(
                        field_name=f"WRONG_LABEL_{expected.label}",
                        prompt=prompt,
                        source_text=source_text,
                        agent_answer=f"Labeled as {predicted.label}: {predicted.value}",
                        ground_truth=f"Should be {expected.label}: {expected.value}",
                    )
        
        errors = list(errors_by_field.values())
        logger.info(f"Collected {len(errors)} unique error types")
        return errors

    def get_error_summary(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, int]:
        """
        Get a summary of error counts by field type.

        Args:
            results: List of evaluation results.

        Returns:
            dict[str, int]: Count of errors per field type.

        Examples:
            >>> evaluator = Evaluator()
            >>> # Assuming some evaluation results with missing EMAIL entities
            >>> summary = evaluator.get_error_summary([])
            >>> isinstance(summary, dict)
            True
        """
        summary: dict[str, int] = defaultdict(int)
        
        for result in results:
            for entity in result.missing_entities:
                summary[f"MISSING_{entity.label}"] += 1
            for entity in result.extra_entities:
                summary[f"FALSE_POSITIVE_{entity.label}"] += 1
            for _, expected in result.wrong_labels:
                summary[f"WRONG_LABEL_{expected.label}"] += 1
        
        return dict(summary)

    def _entity_key(self, entity: PIIEntity) -> tuple:
        """
        Create a comparison key for an entity.

        Args:
            entity: The PII entity.

        Returns:
            tuple: Comparison key based on settings.
        """
        if self.strict_position:
            return (entity.value.lower(), entity.label, entity.start, entity.end)
        else:
            return (entity.value.lower(), entity.label)
