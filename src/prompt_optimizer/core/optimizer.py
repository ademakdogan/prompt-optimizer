"""
Optimizer module for prompt optimization loop.

This module provides the main optimization loop that:
1. Processes data with current prompt
2. Evaluates against ground truth
3. Collects errors for mentor feedback
4. Generates improved prompts with updated field descriptions
"""

from prompt_optimizer.api import AgentModel, MentorModel
from prompt_optimizer.config import get_settings
from prompt_optimizer.core.evaluator import Evaluator, EvaluationResult
from prompt_optimizer.models import (
    ErrorFeedback,
    OptimizationResult,
    TargetResult,
    PromptHistory,
)
from prompt_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class PromptOptimizer:
    """
    Main optimizer class for prompt optimization loop.

    Coordinates the agent model, mentor model, and evaluator
    to iteratively improve prompts for PII extraction.
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
        >>> # Run optimization with test data
        >>> results = optimizer.optimize(
        ...     data=[("Sample text with john@email.com", target_result)],
        ...     initial_prompt="Extract PII entities"
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
        
        self.history: list[PromptHistory] = []
        self.field_descriptions: dict[str, str] = {}
        
        logger.info(
            f"Initialized PromptOptimizer: "
            f"window_size={self.window_size}, loop_count={self.loop_count}"
        )

    def optimize(
        self,
        data: list[tuple[str, TargetResult]],
        initial_prompt: str | None = None,
        schema_description: str = "",
    ) -> list[OptimizationResult]:
        """
        Run the optimization loop.

        Args:
            data: List of (source_text, target_result) tuples.
            initial_prompt: Starting prompt. If None, mentor generates one.
            schema_description: Description of expected output schema.

        Returns:
            list[OptimizationResult]: Results from each iteration.

        Examples:
            >>> optimizer = PromptOptimizer()
            >>> data = [("Contact john@email.com", target_result)]
            >>> results = optimizer.optimize(data)
            >>> len(results) == optimizer.loop_count
            True
        """
        logger.info(f"Starting optimization with {len(data)} samples")
        
        # Get initial prompt if not provided
        current_prompt = initial_prompt
        if current_prompt is None:
            logger.info("No initial prompt provided, generating from sample data")
            current_prompt = self._generate_initial_prompt(data, schema_description)
        
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
            responses = self._process_all_data(data, current_prompt, schema_description)
            
            # Evaluate responses
            source_texts = [text for text, _ in data]
            ground_truths = [gt for _, gt in data]
            
            accuracy, eval_results = self.evaluator.evaluate_batch(
                responses, ground_truths
            )
            
            logger.info(f"Iteration {iteration} accuracy: {accuracy:.2%}")
            
            # Collect errors
            errors = self.evaluator.collect_errors(
                responses, ground_truths, source_texts, current_prompt
            )
            
            # Get error summary for history
            error_summary = self.evaluator.get_error_summary(eval_results)
            
            # Store result
            result = OptimizationResult(
                iteration=iteration,
                prompt=current_prompt,
                accuracy=accuracy,
                total_samples=len(data),
                correct_samples=sum(r.accuracy for r in eval_results),
                errors=errors,
                field_descriptions=self.field_descriptions.copy(),
            )
            results.append(result)
            
            # Update history
            history_entry = PromptHistory(
                iteration=iteration,
                prompt=current_prompt,
                accuracy=accuracy,
                error_summary=error_summary,
                field_descriptions=self.field_descriptions.copy(),
            )
            self.history.append(history_entry)
            
            # Log result
            self._log_iteration_result(result)
            
            # If perfect accuracy, stop early
            if accuracy >= 1.0:
                logger.info("Perfect accuracy achieved! Stopping optimization.")
                break
            
            # Generate improved prompt for next iteration (if not last)
            if iteration < self.loop_count and errors:
                current_prompt = self._generate_improved_prompt(
                    current_prompt, errors, schema_description
                )
        
        # Final summary
        self._log_final_summary(results)
        
        return results

    def _generate_initial_prompt(
        self,
        data: list[tuple[str, TargetResult]],
        schema_description: str,
    ) -> str:
        """
        Generate initial prompt using mentor model.

        Args:
            data: Sample data for prompt generation.
            schema_description: Description of expected schema.

        Returns:
            str: Generated initial prompt.
        """
        # Use first sample for initial prompt generation
        sample_text, target_result = data[0]
        
        # Convert target result to JSON-like format
        gt_str = self._format_target_result(target_result)
        
        generated = self.mentor.generate_initial_prompt(
            sample_data=sample_text,
            ground_truth=gt_str,
            schema_description=schema_description,
            current_field_descriptions=self.field_descriptions,
        )
        
        logger.info(f"Generated initial prompt:\n{generated.prompt}")
        logger.info(f"Reasoning: {generated.reasoning}")
        
        # Update field descriptions from mentor
        if generated.field_descriptions:
            self.field_descriptions.update(generated.field_descriptions)
            logger.info(f"Field descriptions updated: {list(generated.field_descriptions.keys())}")
        
        return generated.prompt

    def _generate_improved_prompt(
        self,
        current_prompt: str,
        errors: list[ErrorFeedback],
        schema_description: str,
    ) -> str:
        """
        Generate improved prompt based on errors.

        Args:
            current_prompt: The current prompt being used.
            errors: Errors from current iteration.
            schema_description: Description of expected schema.

        Returns:
            str: Improved prompt.
        """
        # Apply window size to history
        history_window = self.history[-self.window_size:] if self.history else []
        
        generated = self.mentor.generate_prompt(
            errors=errors,
            history=history_window,
            current_prompt=current_prompt,
            schema_description=schema_description,
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
        data: list[tuple[str, TargetResult]],
        prompt: str,
        schema_description: str,
    ) -> list[TargetResult]:
        """
        Process all data samples with the agent.

        Args:
            data: List of (source_text, target_result) tuples.
            prompt: The prompt to use.
            schema_description: Schema description.

        Returns:
            list[TargetResult]: Agent responses for all samples.
        """
        responses = []
        
        for i, (source_text, _) in enumerate(data):
            logger.debug(f"Processing sample {i+1}/{len(data)}")
            
            try:
                response = self.agent.process_data(
                    prompt=prompt,
                    data=source_text,
                    schema_description=schema_description,
                )
                responses.append(response)
            except Exception as e:
                logger.error(f"Error processing sample {i+1}: {e}")
                # Return empty response on error
                responses.append(TargetResult())
        
        return responses

    def _format_target_result(self, target_result: TargetResult) -> str:
        """
        Format target result for display.

        Args:
            target_result: The target result to format.

        Returns:
            str: JSON-like formatted string.
        """
        fields = {
            k: v for k, v in target_result.model_dump().items() if v is not None
        }
        
        lines = ["{"]
        for field, value in fields.items():
            lines.append(f'  "{field}": "{value}",')
        lines.append("}")
        
        return "\n".join(lines)

    def _log_iteration_result(self, result: OptimizationResult) -> None:
        """
        Log the result of an iteration.

        Args:
            result: The optimization result.
        """
        logger.info(f"""
--- Iteration {result.iteration} Summary ---
Accuracy: {result.accuracy:.2%}
Correct samples: {result.correct_samples:.1f}/{result.total_samples}
Error types: {len(result.errors)}
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
