# Prompt Optimizer

A Python tool for optimizing prompts using a mentor-agent architecture via OpenRouter.

## Overview

This project implements an automated prompt optimization loop where:
1. An **Agent** model processes data with a given prompt, producing structured PII extraction results
2. Responses are evaluated against ground truth using configurable accuracy metrics
3. Errors are collected and analyzed by field type
4. A **Mentor** model analyzes errors and generates improved prompts
5. The loop repeats for a configurable number of iterations

## Features

- **Mentor-Agent Architecture**: Uses two models for optimization
- **Structured Responses**: Uses instructor library for Pydantic-validated responses
- **Window Size**: Configurable history tracking for mentor context
- **PII Extraction**: Specialized for extracting Personally Identifiable Information
- **Comprehensive Logging**: Track prompts and accuracy at each iteration

## Installation

```bash
# Clone and install with uv
uv sync --all-extras
```

## Configuration

Create a `.env` file with:

```env
OPENROUTER_API_KEY=your-api-key-here
AGENT_MODEL=openai/gpt-4.1-nano
MENTOR_MODEL=openai/gpt-4.1-nano
WINDOW_SIZE=2
LOOP_COUNT=3
LOG_LEVEL=INFO
```

## Usage

### Command Line

```bash
# Run with default settings
uv run python -m prompt_optimizer

# Run with custom parameters
uv run python -m prompt_optimizer \
    --data resources/test_data.json \
    --samples 5 \
    --loops 3 \
    --window-size 2 \
    --log-level DEBUG
```

### Python API

```python
from prompt_optimizer.core import PromptOptimizer
from prompt_optimizer.data import load_test_data

# Load data
data = load_test_data("resources/test_data.json", limit=5)

# Create optimizer
optimizer = PromptOptimizer(
    window_size=2,
    loop_count=3,
)

# Run optimization
results = optimizer.optimize(
    data=data,
    initial_prompt="Extract all PII entities from the text.",
)

# Get best result
best = max(results, key=lambda r: r.accuracy)
print(f"Best accuracy: {best.accuracy:.2%}")
print(f"Best prompt: {best.prompt}")
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run unit tests only
uv run pytest tests/unit/ -v

# Run with coverage
uv run pytest tests/ --cov=prompt_optimizer
```

## Project Structure

```
prompt-optimizer/
├── src/prompt_optimizer/
│   ├── api/          # OpenRouter client, agent, mentor
│   ├── config/       # Settings management
│   ├── core/         # Evaluator and optimizer
│   ├── data/         # Data loading utilities
│   ├── models/       # Pydantic schemas
│   └── utils/        # Logging utilities
├── tests/
│   ├── unit/         # Unit tests
│   ├── integration/  # Integration tests
│   └── e2e/          # End-to-end tests
└── resources/        # Test data
```

## License

MIT
