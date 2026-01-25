"""
Optimizer module for prompt optimization loop.

This module provides the main optimization loop that:
1. Processes data with current prompt
2. Evaluates against ground truth
3. Collects failed predictions for mentor feedback
4. Generates improved prompts with updated field descriptions
"""

from typing import Any

from prompt_optimizer.api import AgentModel, MentorModel
from prompt_optimizer.api.mentor import IterationHistory, FailedPrediction
from prompt_optimizer.config import get_settings
from prompt_optimizer.core.evaluator import Evaluator, EvaluationResult
from prompt_optimizer.models import OptimizationResult
from prompt_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class PromptOptimizer:
    """
    Main optimizer class for prompt optimization loop.

    Coordinates the agent model, mentor model, and evaluator
    to iteratively improve prompts for data extraction.
    The mentor also updates field descriptions to improve schema understanding.

    Attributes:
        agent: The agent model for processing data.
        mentor: The mentor model for generating prompts.
        evaluator: The evaluator for comparing responses.
        window_size: Number of historical iterations to send to mentor.
        loop_count: Total number of optimization iterations.
        field_descriptions: Current field descriptions from mentor.

    Examples:
        >>> optimizer = PromptOptimizer()
        >>> results = optimizer.optimize(
        ...     data=[("Sample text", {"key": "value"})],
        ...     initial_prompt="Extract key information"
        ... )
        >>> len(results) > 0
        True
    """

    def __init__(
        self,
        agent_model: str | None = None,
        mentor_model: str | None = None,
        window_size: int | None = None,
        loop_count: int | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Initialize the prompt optimizer.

        Args:
            agent_model: Model for agent. Uses settings if None.
            mentor_model: Model for mentor. Uses settings if None.
            window_size: History window size. Uses settings if None.
            loop_count: Number of iterations. Uses settings if None.
            api_key: OpenRouter API key. Uses settings if None.
        """
        settings = get_settings()
        
        self.window_size = window_size or settings.window_size
        self.loop_count = loop_count or settings.loop_count
        
        self.agent = AgentModel(model=agent_model, api_key=api_key)
        self.mentor = MentorModel(model=mentor_model, api_key=api_key)
        self.evaluator = Evaluator()
        
        self.history: list[IterationHistory] = []
        self.field_descriptions: dict[str, str] = {}
        
        logger.info(
            f"Initialized PromptOptimizer: "
            f"window_size={self.window_size}, loop_count={self.loop_count}"
        )

    def optimize(
        self,
        data: list[tuple[str, dict[str, Any]]],
        initial_prompt: str | None = None,
    ) -> list[OptimizationResult]:
        """
        Run the optimization loop.

        Args:
            data: List of (source_text, ground_truth) tuples.
            initial_prompt: Starting prompt. If None, mentor generates one.

        Returns:
            list[OptimizationResult]: Results from each iteration.

        Examples:
            >>> optimizer = PromptOptimizer()
            >>> data = [("Contact john@email.com", {"email": "john@email.com"})]
            >>> results = optimizer.optimize(data)
            >>> len(results) == optimizer.loop_count
            True
        """
        logger.info(f"Starting optimization with {len(data)} samples")
        
        # Get initial prompt if not provided
        current_prompt = initial_prompt
        if current_prompt is None:
            logger.info("No initial prompt provided, generating from mentor")
            current_prompt = self._generate_initial_prompt(data)
        
        results: list[OptimizationResult] = []
        
        for iteration in range(1, self.loop_count + 1):
            logger.info(f"\n{'='*50}\nIteration {iteration}/{self.loop_count}\n{'='*50}")
            logger.info(f"Current prompt:\n{current_prompt}")
            
            if self.field_descriptions:
                logger.info(f"Current field descriptions: {list(self.field_descriptions.keys())}")
            
            # Update agent with current field descriptions
            if self.field_descriptions:
                self.agent.update_field_descriptions(self.field_descriptions)
            
            # Process all data with current prompt
            responses = self._process_all_data(data, current_prompt)
            
            # Evaluate responses
            source_texts = [text for text, _ in data]
            ground_truths = [gt for _, gt in data]
            
            accuracy, eval_results = self.evaluator.evaluate_batch(
                responses, ground_truths
            )
            
            accuracy_percent = accuracy * 100
            logger.info(f"Iteration {iteration} accuracy: {accuracy_percent:.1f}%")
            
            # Build iteration history with failed predictions
            failed_predictions = self._collect_failed_predictions(
                eval_results, source_texts
            )
            
            iteration_history = IterationHistory(
                iteration=iteration,
                prompt=current_prompt,
                prompt_accuracy=accuracy_percent,
                failed_predictions=failed_predictions,
            )
            self.history.append(iteration_history)
            
            # Store result
            result = OptimizationResult(
                iteration=iteration,
                prompt=current_prompt,
                accuracy=accuracy,
                total_samples=len(data),
                correct_samples=sum(1 for r in eval_results if r.is_correct),
                errors=[],  # Deprecated, using failed_predictions now
                field_descriptions=self.field_descriptions.copy(),
            )
            results.append(result)
            
            # Log result
            self._log_iteration_result(result, len(failed_predictions))
            
            # If perfect accuracy, stop early
            if accuracy >= 1.0:
                logger.info("Perfect accuracy achieved! Stopping optimization.")
                break
            
            # Generate improved prompt for next iteration (if not last)
            if iteration < self.loop_count and failed_predictions:
                current_prompt = self._generate_improved_prompt(current_prompt)
        
        # Final summary
        self._log_final_summary(results)
        
        return results

    def _generate_initial_prompt(
        self,
        data: list[tuple[str, dict[str, Any]]],
    ) -> str:
        """
        Generate initial prompt using mentor model.

        Args:
            data: Sample data for prompt generation.

        Returns:
            str: Generated initial prompt.
        """
        # Use first sample for initial prompt generation
        sample_text, ground_truth = data[0]
        
        generated = self.mentor.generate_initial_prompt(
            sample_data=sample_text,
            ground_truth=ground_truth,
            current_field_descriptions=self.field_descriptions,
        )
        
        logger.info(f"Generated initial prompt:\n{generated.prompt}")
        logger.info(f"Reasoning: {generated.reasoning}")
        
        # Update field descriptions from mentor
        if generated.field_descriptions:
            self.field_descriptions.update(generated.field_descriptions)
            logger.info(f"Field descriptions updated: {list(generated.field_descriptions.keys())}")
        
        return generated.prompt

    def _generate_improved_prompt(self, current_prompt: str) -> str:
        """
        Generate improved prompt based on history.

        Args:
            current_prompt: The current prompt being used.

        Returns:
            str: Improved prompt.
        """
        # Apply window size to history
        history_window = self.history[-self.window_size:] if self.history else []
        
        generated = self.mentor.generate_prompt(
            history=history_window,
            current_prompt=current_prompt,
            current_field_descriptions=self.field_descriptions,
        )
        
        logger.info(f"Generated improved prompt:\n{generated.prompt}")
        logger.info(f"Reasoning: {generated.reasoning}")
        
        # Update field descriptions from mentor
        if generated.field_descriptions:
            self.field_descriptions.update(generated.field_descriptions)
            logger.info(f"Field descriptions updated: {list(generated.field_descriptions.keys())}")
        
        return generated.prompt

    def _process_all_data(
        self,
        data: list[tuple[str, dict[str, Any]]],
        prompt: str,
    ) -> list[dict[str, Any]]:
        """
        Process all data samples with the agent.

        Args:
            data: List of (source_text, ground_truth) tuples.
            prompt: The prompt to use.

        Returns:
            list[dict]: Agent responses for all samples.
        """
        responses = []
        
        for i, (source_text, _) in enumerate(data):
            logger.debug(f"Processing sample {i+1}/{len(data)}")
            
            try:
                response = self.agent.process_data(
                    prompt=prompt,
                    data=source_text,
                )
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing sample {i+1}: {e}")
                # Return empty response on error
                responses.append({})
        
        return responses

    def _collect_failed_predictions(
        self,
        eval_results: list[EvaluationResult],
        source_texts: list[str],
    ) -> list[FailedPrediction]:
        """
        Collect failed predictions from evaluation results.

        Args:
            eval_results: List of evaluation results.
            source_texts: List of source texts.

        Returns:
            list[FailedPrediction]: List of failed predictions.
        """
        failed = []
        
        for result, source_text in zip(eval_results, source_texts):
            if not result.is_correct:
                failed.append(FailedPrediction(
                    source_text=source_text,
                    ground_truth=result.ground_truth,
                    predict=result.prediction,
                ))
        
        return failed

    def _log_iteration_result(
        self,
        result: OptimizationResult,
        failed_count: int,
    ) -> None:
        """
        Log the result of an iteration.

        Args:
            result: The optimization result.
            failed_count: Number of failed predictions.
        """
        logger.info(f"""
--- Iteration {result.iteration} Summary ---
Accuracy: {result.accuracy:.2%}
Correct samples: {result.correct_samples}/{result.total_samples}
Failed predictions: {failed_count}
Prompt length: {len(result.prompt)} chars
Field descriptions: {len(result.field_descriptions)}
""")

    def _log_final_summary(self, results: list[OptimizationResult]) -> None:
        """
        Log final optimization summary.

        Args:
            results: All optimization results.
        """
        if not results:
            logger.warning("No optimization results to summarize")
            return
        
        best_result = max(results, key=lambda r: r.accuracy)
        
        logger.info(f"""
{'='*60}
OPTIMIZATION COMPLETE
{'='*60}
Total iterations: {len(results)}
Best accuracy: {best_result.accuracy:.2%} (iteration {best_result.iteration})
Final accuracy: {results[-1].accuracy:.2%}

Accuracy progression:
""")
        for result in results:
            bar = "█" * int(result.accuracy * 20)
            logger.info(f"  Iteration {result.iteration}: {result.accuracy:.2%} {bar}")
        
        logger.info(f"\nBest prompt (iteration {best_result.iteration}):")
        logger.info(best_result.prompt)
        
        if best_result.field_descriptions:
            logger.info(f"\nFinal field descriptions:")
            for field, desc in best_result.field_descriptions.items():
                logger.info(f"  {field}: {desc}")
