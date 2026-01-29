# Prompt Optimizer

A Python tool for automatically optimizing prompts for structured data extraction using a mentor-agent architecture via OpenRouter.

## Overview

Prompt Optimizer implements an intelligent, iterative optimization loop that improves extraction prompts through continuous feedback. The system uses two AI models working together:

```
┌─────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION LOOP                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐     ┌─────────┐     ┌───────────┐     ┌─────────┐ │
│  │  Data   │────▶│  Agent  │────▶│ Evaluator │────▶│ Mentor  │ │
│  │ Samples │     │  Model  │     │           │     │  Model  │ │
│  └─────────┘     └─────────┘     └───────────┘     └────┬────┘ │
│                       ▲                                  │      │
│                       │         Improved Prompt          │      │
│                       └──────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Agent Model**: Processes input data with the current prompt and extracts structured information
2. **Evaluator**: Compares extractions against ground truth, calculating accuracy and identifying errors
3. **Mentor Model**: Analyzes failed predictions and generates an improved prompt
4. **Loop**: Repeats for a configurable number of iterations until optimal accuracy is achieved

### Intelligent Best Result Selection

When multiple iterations achieve the same accuracy, the optimizer automatically selects the prompt with the **fewest tokens**. This ensures:
- More efficient API usage
- Faster inference times
- Lower costs per extraction

## Features

- 🔄 **Mentor-Agent Architecture**: Two-model system for intelligent prompt improvement
- 📊 **Structured Responses**: Uses instructor library for Pydantic-validated JSON outputs
- 📈 **Progress Tracking**: Visual metrics table with accuracy progression
- 🎯 **Smart Selection**: Picks shortest prompt when accuracy is tied
- 🔧 **Flexible Schema**: Define custom extraction fields via `ExtractionSchema`
- 📝 **Comprehensive Logging**: Track all prompts and accuracies in `mentor_prompts.txt`
- 💾 **Auto-Save**: Best prompt automatically saved to `final_prompt.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/prompt-optimizer.git
cd prompt-optimizer

# Install with uv
uv sync --all-extras
```

## Configuration

Create a `.env` file in the project root:

```env
# Required
OPENROUTER_API_KEY=your-api-key-here

# Optional - Model Selection
AGENT_MODEL=openai/gpt-4.1-nano      # Model for data extraction
MENTOR_MODEL=openai/gpt-4.1-nano     # Model for prompt improvement

# Optional - Optimization Parameters
WINDOW_SIZE=2                         # History iterations sent to mentor
LOOP_COUNT=3                          # Number of optimization iterations
LOG_LEVEL=INFO                        # DEBUG, INFO, WARNING, ERROR
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPENROUTER_API_KEY` | Required | Your OpenRouter API key |
| `AGENT_MODEL` | `openai/gpt-4.1-nano` | Model used for data extraction |
| `MENTOR_MODEL` | `openai/gpt-4.1-nano` | Model used for prompt improvement |
| `WINDOW_SIZE` | `2` | Number of previous iterations shown to mentor |
| `LOOP_COUNT` | `3` | Maximum optimization iterations |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Usage

### Command Line Interface

```bash
# Run with default settings
uv run python -m prompt_optimizer

# Run with custom data file and sample count
uv run python -m prompt_optimizer \
    --data resources/test_data.json \
    --samples 10

# Run with custom initial prompt
uv run python -m prompt_optimizer \
    --prompt "Extract name, email, and phone from the text. Return as JSON."

# Run with all options
uv run python -m prompt_optimizer \
    --data resources/test_data.json \
    --samples 5 \
    --prompt "Extract all fields from the text." \
    --loops 5 \
    --window-size 3 \
    --output results.json \
    --log-level DEBUG
```

### CLI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data` | string | `resources/test_data.json` | Path to data file (JSONL or JSON) |
| `--samples` | int | `5` | Number of samples to use for optimization |
| `--prompt` | string | None | Initial prompt (auto-generated if not provided) |
| `--loops` | int | From settings | Number of optimization iterations |
| `--window-size` | int | From settings | History window size for mentor |
| `--output` | string | None | Output file path to save results (JSON) |
| `--log-level` | string | From settings | Logging level (DEBUG/INFO/WARNING/ERROR) |

### Python API

```python
from prompt_optimizer.core import PromptOptimizer
from prompt_optimizer.data import load_test_data

# Load your data
data = load_test_data("resources/test_data.json", limit=10)

# Create optimizer with custom settings
optimizer = PromptOptimizer(
    window_size=2,   # How much history mentor sees
    loop_count=5,    # Number of optimization iterations
)

# Option 1: Let the system generate initial prompt from schema
results = optimizer.optimize(data=data)

# Option 2: Provide your own initial prompt
my_prompt = """
Extract the following information from the text:
- firstname: First name of the person
- email: Email address
- phonenumber: Phone number

Return as JSON. Only include fields found in the text.
"""
results = optimizer.optimize(data=data, initial_prompt=my_prompt)

# Access best result (automatically selected by accuracy, then token count)
best = results[0] if results else None
print(f"Best accuracy: {best.accuracy:.2%}")
print(f"Best prompt: {best.prompt}")
```

### Customizing the Extraction Schema

Edit `src/prompt_optimizer/models/agent_model.py` to define your extraction fields:

```python
class ExtractionSchema(BaseModel):
    """Define fields to extract from your data."""
    
    # Personal Information
    firstname: str | None = Field(None, description="First name of the person")
    lastname: str | None = Field(None, description="Last name of the person")
    email: str | None = Field(None, description="Email address")
    
    # Add your custom fields
    company: str | None = Field(None, description="Company name")
    product: str | None = Field(None, description="Product mentioned")
    sentiment: str | None = Field(None, description="Sentiment: positive/negative/neutral")
```

## Output Files

### `final_prompt.txt`
The best performing prompt is automatically saved here after each optimization run:

```
# Final Optimized Prompt
# Best accuracy achieved: 97.5%
# Iteration: 3
# Generated: 2024-01-14 10:30:45

Extract the following information from the text...
```

### `mentor_prompts.txt`
Complete log of all mentor interactions during optimization:
- System prompts sent to mentor
- Iteration history with failed predictions
- Generated improved prompts

## Optimization Metrics

After each run, you'll see a detailed metrics table:

```
╔════════════════════════════════════════════════════════════╗
║ OPTIMIZATION METRICS                                       ║
╠════════════════════════════════════════════════════════════╣
║  Total iterations:      3                                  ║
║  Best accuracy:         97.50% (iter 2, ~450 tokens)       ║
║  Final accuracy:        95.00%                             ║
║  Accuracy improvement:  +35.50%                            ║
║  Average accuracy:      82.17%                             ║
╠════════════════════════════════════════════════════════════╣
║  ITERATION DETAILS                                         ║
╠════════════╦══════════════╦════════════════════════════════╣
║ Iteration  ║   Accuracy   ║ Progress                       ║
╠════════════╬══════════════╬════════════════════════════════╣
║     1      ║      60.0%   ║ ████████████░░░░░░░░           ║
║     2      ║      97.5%   ║ ███████████████████░           ║
║     3      ║      95.0%   ║ ███████████████████░           ║
╚════════════╩══════════════╩════════════════════════════════╝
```

**Note**: When iterations 2 and 3 both had 97.5% accuracy, iteration 2 was selected because it had fewer tokens (~450 vs ~620).

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run unit tests only
uv run pytest tests/unit/ -v

# Run integration tests
uv run pytest tests/integration/ -v

# Run with coverage report
uv run pytest tests/ --cov=prompt_optimizer --cov-report=html
```

## Project Structure

```
prompt-optimizer/
├── src/prompt_optimizer/
│   ├── api/              # OpenRouter client, agent, mentor models
│   │   ├── client.py     # OpenRouter API client with instructor
│   │   ├── agent.py      # Agent model for data extraction
│   │   └── mentor.py     # Mentor model for prompt improvement
│   ├── config/           # Settings management
│   │   └── settings.py   # Pydantic settings with env support
│   ├── core/             # Core optimization logic
│   │   ├── evaluator.py  # Accuracy calculation and error analysis
│   │   └── optimizer.py  # Main optimization loop
│   ├── data/             # Data loading utilities
│   │   └── loader.py     # JSON/JSONL data loader
│   ├── models/           # Pydantic schemas
│   │   ├── agent_model.py # ExtractionSchema definition
│   │   └── schemas.py    # Result and feedback models
│   └── utils/            # Utilities
│       ├── logging.py    # Logging configuration
│       ├── metrics.py    # Metrics calculation
│       └── persistence.py # Result saving/loading
├── tests/
│   ├── unit/             # Unit tests (74 tests)
│   ├── integration/      # Integration tests
│   └── e2e/              # End-to-end tests
├── resources/            # Sample test data
├── final_prompt.txt      # Best prompt from last run
├── mentor_prompts.txt    # Mentor interaction logs
└── .env                  # Configuration (create from .env.example)
```

## How Best Result is Selected

The optimizer uses a two-tier selection strategy:

1. **Primary**: Highest accuracy (correct extractions / total fields)
2. **Tiebreaker**: Lowest token count (shorter prompt preferred)

This ensures that when multiple prompts achieve the same accuracy, the most efficient one is chosen, reducing:
- API costs
- Latency
- Context window usage

```python
# Internal selection logic
sorted_results = sorted(
    results,
    key=lambda r: (-r.accuracy, token_count(r.prompt))
)
best = sorted_results[0]  # Highest accuracy, then fewest tokens
```

## Supported Models

Any model available on OpenRouter can be used. Popular choices:

| Model | Speed | Quality | Cost |
|-------|-------|---------|------|
| `openai/gpt-4.1-nano` | Fast | Good | Low |
| `openai/gpt-4.1-mini` | Medium | Better | Medium |
| `google/gemini-2.5-flash-lite` | Fast | Good | Low |
| `anthropic/claude-3-haiku` | Fast | Good | Low |

## License

MIT
