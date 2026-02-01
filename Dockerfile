# ============================================================================
# Prompt Optimizer - Dockerfile
# ============================================================================
FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Set working directory
WORKDIR /app

# Copy all project files
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY resources/ ./resources/

# Install dependencies and project
RUN uv sync --frozen --no-dev

# Create output directories
RUN mkdir -p /app/output

# Set PATH to include virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Default command
CMD ["python", "-m", "prompt_optimizer", "--help"]
