"""
Evaluator module for comparing agent responses with ground truth.

This module provides utilities for evaluating the accuracy of
data extraction and collecting errors for mentor feedback.
"""

from typing import Any, NamedTuple

from prompt_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationResult(NamedTuple):
    """
    Result of evaluating a single response.

    Attributes:
        is_correct: Whether all fields were correctly extracted.
        accuracy: Fraction of correctly extracted fields (0.0 to 1.0).
        missing_fields: Fields in ground truth but not in response.
        extra_fields: Fields in response but not in ground truth.
        wrong_values: Fields with incorrect values.
        prediction: The agent's prediction dict.
        ground_truth: The expected ground truth dict.
    """

    is_correct: bool
    accuracy: float
    missing_fields: list[str]
    extra_fields: list[str]
    wrong_values: list[tuple[str, str, str]]  # (field, response_value, truth_value)
    prediction: dict[str, Any]
    ground_truth: dict[str, Any]


class Evaluator:
    """
    Evaluator for comparing agent responses with ground truth.

    Examples:
        >>> evaluator = Evaluator()
        >>> response = {"firstname": "John", "email": "john@email.com"}
        >>> ground_truth = {"firstname": "John", "email": "john@email.com"}
        >>> result = evaluator.evaluate_single(response, ground_truth)
        >>> result.is_correct
        True
    """

    def __init__(self, case_sensitive: bool = False) -> None:
        """
        Initialize the evaluator.

        Args:
            case_sensitive: Whether to use case-sensitive comparison.
        """
        self.case_sensitive = case_sensitive

    def evaluate_single(
        self,
        response: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> EvaluationResult:
        """
        Evaluate a single response against ground truth.

        Args:
            response: The agent's response as dict.
            ground_truth: The expected correct result as dict.

        Returns:
            EvaluationResult: Detailed evaluation result.

        Examples:
            >>> evaluator = Evaluator()
            >>> result = evaluator.evaluate_single(
            ...     {"firstname": "John"},
            ...     {"firstname": "John", "email": "test@mail.com"}
            ... )
            >>> result.is_correct
            False
        """
        response_keys = set(response.keys())
        truth_keys = set(ground_truth.keys())
        
        # Find missing and extra fields
        missing_fields = list(truth_keys - response_keys)
        extra_fields = list(response_keys - truth_keys)
        
        # Check for wrong values in common fields
        wrong_values = []
        for field in response_keys & truth_keys:
            resp_val = str(response[field])
            truth_val = str(ground_truth[field])
            
            if not self.case_sensitive:
                resp_val_cmp = resp_val.lower()
                truth_val_cmp = truth_val.lower()
            else:
                resp_val_cmp = resp_val
                truth_val_cmp = truth_val
            
            if resp_val_cmp != truth_val_cmp:
                wrong_values.append((field, resp_val, truth_val))
        
        # Calculate accuracy
        total_fields = len(truth_keys)
        if total_fields == 0:
            accuracy = 1.0 if len(response_keys) == 0 else 0.0
        else:
            correct_count = len(truth_keys & response_keys) - len(wrong_values)
            accuracy = max(0, correct_count) / total_fields
        
        is_correct = (
            len(missing_fields) == 0
            and len(extra_fields) == 0
            and len(wrong_values) == 0
        )
        
        return EvaluationResult(
            is_correct=is_correct,
            accuracy=accuracy,
            missing_fields=missing_fields,
            extra_fields=extra_fields,
            wrong_values=wrong_values,
            prediction=response,
            ground_truth=ground_truth,
        )

    def evaluate_batch(
        self,
        responses: list[dict[str, Any]],
        ground_truths: list[dict[str, Any]],
    ) -> tuple[float, list[EvaluationResult]]:
        """
        Evaluate a batch of responses against ground truths.

        Args:
            responses: List of agent responses.
            ground_truths: List of expected correct results.

        Returns:
            tuple: (average_accuracy, list_of_evaluation_results)

        Raises:
            ValueError: If lengths don't match.

        Examples:
            >>> evaluator = Evaluator()
            >>> results = [{"firstname": "John"}]
            >>> truths = [{"firstname": "John"}]
            >>> accuracy, evals = evaluator.evaluate_batch(results, truths)
            >>> accuracy
            1.0
        """
        if len(responses) != len(ground_truths):
            raise ValueError(
                f"Mismatched lengths: {len(responses)} responses vs "
                f"{len(ground_truths)} ground truths"
            )
        
        eval_results = []
        for response, truth in zip(responses, ground_truths):
            result = self.evaluate_single(response, truth)
            eval_results.append(result)
        
        if eval_results:
            avg_accuracy = sum(r.accuracy for r in eval_results) / len(eval_results)
        else:
            avg_accuracy = 0.0
        
        logger.info(
            f"Batch evaluation: {avg_accuracy:.2%} accuracy "
            f"({sum(1 for r in eval_results if r.is_correct)}/{len(eval_results)} correct)"
        )
        
        return avg_accuracy, eval_results

    def get_error_summary(
        self,
        eval_results: list[EvaluationResult],
    ) -> dict[str, int]:
        """
        Generate error summary from evaluation results.

        Args:
            eval_results: List of evaluation results.

        Returns:
            dict: Mapping of error types to counts.

        Examples:
            >>> evaluator = Evaluator()
            >>> result = EvaluationResult(
            ...     is_correct=False,
            ...     accuracy=0.5,
            ...     missing_fields=["email"],
            ...     extra_fields=[],
            ...     wrong_values=[],
            ...     prediction={},
            ...     ground_truth={"email": "test@mail.com"}
            ... )
            >>> summary = evaluator.get_error_summary([result])
            >>> summary["MISSING_email"]
            1
        """
        summary: dict[str, int] = {}
        
        for result in eval_results:
            for field in result.missing_fields:
                key = f"MISSING_{field}"
                summary[key] = summary.get(key, 0) + 1
            
            for field in result.extra_fields:
                key = f"EXTRA_{field}"
                summary[key] = summary.get(key, 0) + 1
            
            for field, _, _ in result.wrong_values:
                key = f"WRONG_{field}"
                summary[key] = summary.get(key, 0) + 1
        
        return summary
