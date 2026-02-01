# Prompt Optimizer

**Stop struggling with prompt engineering. Let AI optimize your prompts automatically.**

Generate accurate extraction prompts directly from your labeled data. Improve your existing prompts and eliminate manual fine-tuning.

**Prompt Optimizer** uses a mentor-agent architecture to automatically generate, refine, and optimize prompts for your specific use case. Simply provide your labeled examples, define your output schema, and let the system discover the optimal prompt through iterative learning.

### Why Use This?

- 🎯 **Automatic Prompt Discovery**: Don't know where to start? The system generates an initial prompt based on your data
- 📈 **Continuous Improvement**: Each iteration learns from mistakes and produces better prompts
- ⚡ **Token Efficiency**: When accuracy is tied, the shortest (cheapest) prompt wins
- 🔄 **Works for Any Domain**: From NER to calculations, entity extraction to data transformation

---

## How It Works

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

1. **Agent Model**: Processes input data with the current prompt and extracts structured information
2. **Evaluator**: Compares extractions against ground truth, calculating accuracy and identifying errors
3. **Mentor Model**: Analyzes failed predictions and generates an improved prompt
4. **Loop**: Repeats for a configurable number of iterations until optimal accuracy is achieved

---

## Quick Start

### Option 1: Local Installation

```bash
# Clone and install
git clone https://github.com/yourusername/prompt-optimizer.git
cd prompt-optimizer
uv sync --all-extras

# Configure
echo "OPENROUTER_API_KEY=your-api-key" > .env

# Run
uv run python -m prompt_optimizer --data resources/test_mapping.json --samples 5
```

### Option 2: Docker

```bash
# Clone and configure
git clone https://github.com/yourusername/prompt-optimizer.git
cd prompt-optimizer
echo "OPENROUTER_API_KEY=your-api-key" > .env

# Build and run
make build
make optimize DATA=resources/test_mapping.json SAMPLES=5 LOOPS=3
```

### Makefile Commands

| Command | Description |
|---------|-------------|
| `make build` | Build Docker image |
| `make run` | Run container (shows help) |
| `make optimize` | Run optimization with defaults |
| `make test` | Run tests in Docker |
| `make shell` | Open shell in container |
| `make clean` | Remove container and image |
| `make help` | Show all available commands |

**Custom optimization:**
```bash
make optimize DATA=resources/my_data.json SAMPLES=10 LOOPS=5
```

---

## Example Datasets

The project includes two example datasets demonstrating different problem types:

### 1. Named Entity Recognition (`test_pii.json`)

This dataset demonstrates **entity extraction** from unstructured text. The AI must identify and extract specific pieces of information (names, emails, coordinates, etc.) from natural language.

```json
{
  "source_text": "Dear Mr. Vandervort, congratulations on turning 68! Your unique ID is 0Zr2bcG1X9Ub. Visit us at 609 Gorczany Pass.",
  "target_result": {
    "prefix": "Mr.",
    "lastname": "Vandervort", 
    "age": "68",
    "username": "0Zr2bcG1X9Ub",
    "street": "609 Gorczany Pass"
  }
}
```

**Characteristics:**
- Text is unstructured natural language
- Fields must be **recognized and extracted** from context
- Values are copied verbatim from source text
- Typical for: NER, PII detection, document parsing, CV/resume extraction

### 2. Calculation & Mapping (`test_mapping.json`)

This dataset demonstrates **data transformation** where the AI must not only extract but also **calculate derived values** from input fields.

```json
{
  "source_text": "\"name\": 'TechSolutions Inc', \"gross\": 1000, \"commission_rate\": 0.1, \"vat\": 180",
  "target_result": {
    "client_name": "TechSolutions Inc",
    "total_gross": 1180,
    "total_mid_gross": 1280
  }
}
```

**Characteristics:**
- Input is semi-structured (key-value pairs)
- Fields require **mathematical calculations**:
  - `total_gross = gross + vat`
  - `total_mid_gross = total_gross + (commission_rate × gross)`
- The AI must learn the formulas from examples
- Typical for: Financial calculations, data transformation, ETL pipelines

---

## How to Use with Your Own Data

### Step 1: Prepare Your Dataset

Create a JSON file with your labeled examples in the following format:

```json
[
  {
    "source_text": "Your input text here...",
    "target_result": {
      "field1": "expected_value1",
      "field2": "expected_value2"
    }
  },
  // ... more examples
]
```

Save it to `resources/your_dataset.json`.

### Step 2: Define Your Schema

Open `src/prompt_optimizer/models/agent_model.py` and update the `ExtractionSchema` class to match your `target_result` fields:

```python
class ExtractionSchema(BaseModel):
    # ═══════════════════════════════════════════════════════════════
    # 🔧 USER CUSTOMIZATION SECTION - MODIFY THIS FOR YOUR DATASET
    # ═══════════════════════════════════════════════════════════════
    
    # Define fields matching your target_result keys:
    field1: str  # Required field
    field2: Optional[str] = None  # Optional field
    field3: Optional[float] = None  # Optional numeric field
```

**Tips:**
- Use `str` for required fields, `Optional[str]` for optional ones
- Use `float` or `int` for numeric values
- Add `Field(description="...")` to provide hints to the AI

### Step 3: Run Optimization

```bash
uv run python -m prompt_optimizer \
    --data resources/your_dataset.json \
    --samples 10 \
    --loops 5
```

### Step 4: Get Your Optimized Prompt

After optimization completes, find your prompt in:
- **`final_prompt.txt`**: The best-performing prompt
- **`mentor_prompts.txt`**: Full history of all iterations

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Required | Your OpenRouter API key |
| `AGENT_MODEL` | `openai/gpt-4.1-nano` | Model for data extraction |
| `MENTOR_MODEL` | `openai/gpt-4.1-nano` | Model for prompt improvement |
| `WINDOW_SIZE` | `2` | History iterations shown to mentor |
| `LOOP_COUNT` | `3` | Maximum optimization iterations |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

### CLI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--data` | string | `resources/test_data.json` | Path to your dataset |
| `--samples` | int | `5` | Number of samples to use |
| `--prompt` | string | None | Initial prompt (auto-generated if not provided) |
| `--loops` | int | From settings | Number of optimization iterations |
| `--window-size` | int | From settings | History window for mentor |
| `--output` | string | None | Save results to JSON file |
| `--log-level` | string | From settings | DEBUG/INFO/WARNING/ERROR |

---

## Python API

```python
from prompt_optimizer.core import PromptOptimizer
from prompt_optimizer.data import load_test_data

# Load your labeled data
data = load_test_data("resources/your_dataset.json", limit=10)

# Create optimizer
optimizer = PromptOptimizer(window_size=2, loop_count=5)

# Option 1: Let the system generate initial prompt
results = optimizer.optimize(data=data)

# Option 2: Improve an existing prompt
my_prompt = """
Extract client information and calculate totals.
Return as JSON with: client_name, total_gross, total_mid_gross
"""
results = optimizer.optimize(data=data, initial_prompt=my_prompt)

# The best prompt is automatically saved to final_prompt.txt
```

---

## Optimization Metrics

After each run, you'll see a detailed metrics table:

```
╔════════════════════════════════════════════════════════════╗
║ OPTIMIZATION METRICS                                       ║
╠════════════════════════════════════════════════════════════╣
║  Total iterations:      4                                  ║
║  Best accuracy:         93.33% (iter 3, ~199 tokens)       ║
║  Final accuracy:        56.67%                             ║
║  Accuracy improvement:  +18.33%                            ║
║  Average accuracy:      56.67%                             ║
╠════════════════════════════════════════════════════════════╣
║  ITERATION DETAILS                                         ║
╠════════════╦══════════════╦════════════════════════════════╣
║ Iteration  ║   Accuracy   ║ Progress                       ║
╠════════════╬══════════════╬════════════════════════════════╣
║     1      ║      38.3%   ║ ███████░░░░░░░░░░░░░           ║
║     2      ║      38.3%   ║ ███████░░░░░░░░░░░░░           ║
║     3      ║      93.3%   ║ ██████████████████░░           ║
║     4      ║      56.7%   ║ ███████████░░░░░░░░░           ║
╚════════════╩══════════════╩════════════════════════════════╝
```

### Smart Best Result Selection

When multiple iterations achieve the same accuracy, the optimizer automatically selects the prompt with the **fewest tokens**. This ensures:
- Lower API costs
- Faster inference
- Reduced context window usage

---

## Project Structure

```
prompt-optimizer/
├── src/prompt_optimizer/
│   ├── api/              # OpenRouter client, agent, mentor
│   ├── config/           # Settings management
│   ├── core/             # Optimizer loop and evaluator
│   ├── data/             # Data loading utilities
│   ├── models/           # Pydantic schemas (edit agent_model.py!)
│   └── utils/            # Logging, metrics, persistence
├── resources/            # Example datasets
│   ├── test_pii.json     # NER/Entity extraction example
│   └── test_mapping.json # Calculation/mapping example
├── tests/                # Unit, integration, e2e tests
├── final_prompt.txt      # Best prompt from last run
└── mentor_prompts.txt    # Full optimization history
```

---

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=prompt_optimizer --cov-report=html
```

---

## Supported Models

Any model on OpenRouter can be used:

| Model | Speed | Quality | Cost |
|-------|-------|---------|------|
| `openai/gpt-4.1-nano` | Fast | Good | Low |
| `openai/gpt-4.1-mini` | Medium | Better | Medium |
| `google/gemini-2.5-flash-lite` | Fast | Good | Low |
| `anthropic/claude-3-haiku` | Fast | Good | Low |

---

## License

MIT
