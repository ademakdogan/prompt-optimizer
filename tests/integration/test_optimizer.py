"""
Integration tests for the optimizer module.
"""

from unittest.mock import MagicMock, patch

import pytest

from prompt_optimizer.core.evaluator import Evaluator
from prompt_optimizer.models import TargetResult, GeneratedPrompt


class TestEvaluatorIntegration:
    """Integration tests for Evaluator."""

    def test_full_evaluation_workflow(self) -> None:
        """Test complete evaluation workflow."""
        evaluator = Evaluator()
        
        # Sample data pairs
        responses = [
            TargetResult(firstname="John", email="john@email.com"),
            TargetResult(firstname="Jane"),
            TargetResult(email="test@mail.com"),
        ]
        
        ground_truths = [
            TargetResult(firstname="John", email="john@email.com"),
            TargetResult(firstname="Jane", email="jane@email.com"),
            TargetResult(email="test@mail.com", phonenumber="555-1234"),
        ]
        
        source_texts = [
            "Contact John at john@email.com",
            "Jane can be reached at jane@email.com",
            "Call 555-1234 or email test@mail.com",
        ]
        
        # Run batch evaluation
        accuracy, results = evaluator.evaluate_batch(responses, ground_truths)
        
        # First sample is perfect
        assert results[0].is_correct is True
        
        # Others have errors
        assert results[1].is_correct is False
        assert results[2].is_correct is False
        
        # Get error summary
        summary = evaluator.get_error_summary(results)
        assert "MISSING_email" in summary
        assert "MISSING_phonenumber" in summary
        
        # Collect errors for mentor
        errors = evaluator.collect_errors(
            responses, ground_truths, source_texts, "Extract PII"
        )
        
        assert len(errors) >= 1
        assert any(e.field_name == "email" for e in errors)


class TestOptimizerIntegration:
    """Integration tests for PromptOptimizer with mocked API."""

    @patch("prompt_optimizer.core.optimizer.MentorModel")
    @patch("prompt_optimizer.core.optimizer.AgentModel")
    def test_optimization_loop_iteration_count(
        self,
        mock_agent_cls: MagicMock,
        mock_mentor_cls: MagicMock,
    ) -> None:
        """Test that optimizer runs correct number of iterations."""
        from prompt_optimizer.core.optimizer import PromptOptimizer
        
        # Setup mocks
        mock_agent = MagicMock()
        mock_agent.process_data.return_value = TargetResult()  # Empty response
        mock_agent.update_field_descriptions = MagicMock()
        mock_agent_cls.return_value = mock_agent
        
        mock_mentor = MagicMock()
        mock_mentor.generate_initial_prompt.return_value = GeneratedPrompt(
            prompt="Extract PII",
            reasoning="Initial prompt",
            field_descriptions={},
        )
        mock_mentor.generate_prompt.return_value = GeneratedPrompt(
            prompt="Improved prompt",
            reasoning="Based on errors",
            field_descriptions={},
        )
        mock_mentor_cls.return_value = mock_mentor
        
        # Create optimizer with 2 loops
        optimizer = PromptOptimizer(loop_count=2)
        
        data = [
            ("Test text", TargetResult(firstname="John")),
        ]
        
        results = optimizer.optimize(data)
        
        # Should run 2 iterations (loop_count)
        assert len(results) == 2

    @patch("prompt_optimizer.core.optimizer.MentorModel")
    @patch("prompt_optimizer.core.optimizer.AgentModel")
    def test_window_size_limits_history(
        self,
        mock_agent_cls: MagicMock,
        mock_mentor_cls: MagicMock,
    ) -> None:
        """Test that window size limits history sent to mentor."""
        from prompt_optimizer.core.optimizer import PromptOptimizer
        
        # Setup mocks
        mock_agent = MagicMock()
        mock_agent.process_data.return_value = TargetResult()
        mock_agent.update_field_descriptions = MagicMock()
        mock_agent_cls.return_value = mock_agent
        
        mock_mentor = MagicMock()
        mock_mentor.generate_initial_prompt.return_value = GeneratedPrompt(
            prompt="Extract PII",
            reasoning="Initial",
            field_descriptions={},
        )
        mock_mentor.generate_prompt.return_value = GeneratedPrompt(
            prompt="Improved prompt",
            reasoning="Based on errors",
            field_descriptions={},
        )
        mock_mentor_cls.return_value = mock_mentor
        
        # Create optimizer with window_size=2 and 4 loops
        optimizer = PromptOptimizer(window_size=2, loop_count=4)
        
        data = [
            ("Test text", TargetResult(email="test@email.com")),
        ]
        
        optimizer.optimize(data)
        
        # Check last call to generate_prompt
        last_call = mock_mentor.generate_prompt.call_args
        if last_call:
            history_arg = last_call.kwargs.get("history", [])
            # History should be limited to window_size
            assert len(history_arg) <= 2

    @patch("prompt_optimizer.core.optimizer.MentorModel")
    @patch("prompt_optimizer.core.optimizer.AgentModel")
    def test_perfect_accuracy_stops_early(
        self,
        mock_agent_cls: MagicMock,
        mock_mentor_cls: MagicMock,
    ) -> None:
        """Test that optimization stops early on perfect accuracy."""
        from prompt_optimizer.core.optimizer import PromptOptimizer
        
        # Setup mocks - agent returns perfect match
        mock_agent = MagicMock()
        mock_agent.process_data.return_value = TargetResult(firstname="John")
        mock_agent.update_field_descriptions = MagicMock()
        mock_agent_cls.return_value = mock_agent
        
        mock_mentor = MagicMock()
        mock_mentor.generate_initial_prompt.return_value = GeneratedPrompt(
            prompt="Extract PII",
            reasoning="Initial",
            field_descriptions={},
        )
        mock_mentor_cls.return_value = mock_mentor
        
        optimizer = PromptOptimizer(loop_count=5)
        
        data = [
            ("John is here", TargetResult(firstname="John")),
        ]
        
        results = optimizer.optimize(data)
        
        # Should stop after first iteration due to 100% accuracy
        assert len(results) == 1
        assert results[0].accuracy == 1.0
