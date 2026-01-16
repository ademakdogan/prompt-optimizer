"""
Main entry point for Prompt Optimizer.

Run with: python -m prompt_optimizer
"""

import argparse
import sys
from pathlib import Path

from prompt_optimizer.config import get_settings
from prompt_optimizer.core import PromptOptimizer
from prompt_optimizer.data import load_test_data
from prompt_optimizer.utils.logging import setup_logging, get_logger


def main() -> int:
    """
    Main entry point for the prompt optimizer.

    Returns:
        int: Exit code (0 for success, 1 for error).

    Examples:
        >>> # Run from command line
        >>> # python -m prompt_optimizer --data resources/pii/english_pii_43k.jsonl
    """
    parser = argparse.ArgumentParser(
        description="Optimize prompts for PII extraction using mentor-agent architecture"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="resources/pii/english_pii_43k.jsonl",
        help="Path to data file (JSONL or JSON)",
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples to use (default: 5)",
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Initial prompt (if not provided, mentor will generate one)",
    )
    
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="History window size for mentor (default: from settings)",
    )
    
    parser.add_argument(
        "--loops",
        type=int,
        default=None,
        help="Number of optimization loops (default: from settings)",
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    settings = get_settings()
    log_level = args.log_level or settings.log_level
    setup_logging(log_level)
    
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("PROMPT OPTIMIZER")
    logger.info("=" * 60)
    
    # Check for API key
    if not settings.openrouter_api_key:
        logger.error(
            "OpenRouter API key not found. "
            "Please set OPENROUTER_API_KEY in your .env file."
        )
        return 1
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return 1
    
    logger.info(f"Loading {args.samples} samples from {data_path}")
    
    try:
        data = load_test_data(data_path, limit=args.samples)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1
    
    logger.info(f"Loaded {len(data)} samples")
    
    # Create optimizer
    optimizer = PromptOptimizer(
        window_size=args.window_size,
        loop_count=args.loops,
    )
    
    # Run optimization
    logger.info("Starting optimization loop...")
    
    try:
        results = optimizer.optimize(
            data=data,
            initial_prompt=args.prompt,
        )
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Print final results
    if results:
        best = max(results, key=lambda r: r.accuracy)
        logger.info(f"\n{'='*60}")
        logger.info("BEST PROMPT")
        logger.info("=" * 60)
        logger.info(f"Accuracy: {best.accuracy:.2%}")
        logger.info(f"Iteration: {best.iteration}")
        logger.info(f"\n{best.prompt}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
