# ============================================================================
# Prompt Optimizer - Makefile
# ============================================================================
# Quick commands for Docker operations and local development
# ============================================================================

.PHONY: help build run stop clean logs shell test test-local optimize \
        docker-build docker-run docker-stop docker-clean docker-logs \
        docker-shell docker-test docker-optimize rebuild

# Default target
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║              PROMPT OPTIMIZER - MAKEFILE                     ║"
	@echo "╠══════════════════════════════════════════════════════════════╣"
	@echo "║  Docker Commands:                                            ║"
	@echo "║    make build          - Build Docker image                  ║"
	@echo "║    make run            - Run container (shows help)          ║"
	@echo "║    make optimize       - Run optimization with defaults      ║"
	@echo "║    make stop           - Stop running container              ║"
	@echo "║    make clean          - Remove container and image          ║"
	@echo "║    make logs           - Show container logs                 ║"
	@echo "║    make shell          - Open shell in container             ║"
	@echo "║    make rebuild        - Clean, rebuild, and run             ║"
	@echo "║                                                              ║"
	@echo "║  Testing:                                                    ║"
	@echo "║    make test           - Run tests in Docker                 ║"
	@echo "║    make test-local     - Run tests locally with uv           ║"
	@echo "║                                                              ║"
	@echo "║  Custom Optimization:                                        ║"
	@echo "║    make optimize DATA=path/to/data.json SAMPLES=10 LOOPS=5   ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""

# ============================================================================
# Variables (can be overridden from command line)
# ============================================================================
IMAGE_NAME := prompt-optimizer
CONTAINER_NAME := prompt-optimizer
DATA ?= resources/test_mapping.json
SAMPLES ?= 5
LOOPS ?= 3
PROMPT ?= 

# ============================================================================
# Docker Commands
# ============================================================================

# Build the Docker image
build:
	@echo "🔨 Building Docker image..."
	docker build -t $(IMAGE_NAME) .

# Build test image
build-test:
	@echo "🔨 Building test Docker image..."
	docker build -f Dockerfile.test -t $(IMAGE_NAME)-test .

# Run the container (shows help by default)
run: build
	@echo "🚀 Running container..."
	docker run --rm \
		--name $(CONTAINER_NAME) \
		--env-file .env \
		-v $(PWD)/resources:/app/resources:ro \
		-v $(PWD)/output:/app/output \
		-v $(PWD)/final_prompt.txt:/app/final_prompt.txt \
		-v $(PWD)/mentor_prompts.txt:/app/mentor_prompts.txt \
		$(IMAGE_NAME) \
		python -m prompt_optimizer --help

# Run optimization with configurable parameters
optimize: build
	@echo "🎯 Running optimization..."
	@echo "   Data: $(DATA)"
	@echo "   Samples: $(SAMPLES)"
	@echo "   Loops: $(LOOPS)"
	@touch $(PWD)/final_prompt.txt $(PWD)/mentor_prompts.txt
	docker run --rm \
		--name $(CONTAINER_NAME)-run \
		--env-file .env \
		-v $(PWD)/resources:/app/resources:ro \
		-v $(PWD)/output:/app/output \
		-v $(PWD)/final_prompt.txt:/app/final_prompt.txt \
		-v $(PWD)/mentor_prompts.txt:/app/mentor_prompts.txt \
		$(IMAGE_NAME) \
		python -m prompt_optimizer \
			--data $(DATA) \
			--samples $(SAMPLES) \
			--loops $(LOOPS) \
			$(if $(PROMPT),--prompt "$(PROMPT)",)

# Stop running container
stop:
	@echo "🛑 Stopping container..."
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker stop $(CONTAINER_NAME)-run 2>/dev/null || true
	-docker stop $(CONTAINER_NAME)-test 2>/dev/null || true

# Remove container and image
clean: stop
	@echo "🧹 Cleaning up..."
	-docker rm $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm $(CONTAINER_NAME)-run 2>/dev/null || true
	-docker rm $(CONTAINER_NAME)-test 2>/dev/null || true
	-docker rmi $(IMAGE_NAME) 2>/dev/null || true
	-docker rmi $(IMAGE_NAME)-test 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Show container logs
logs:
	docker logs -f $(CONTAINER_NAME)

# Open shell in container
shell: build
	@echo "🐚 Opening shell in container..."
	docker run --rm -it \
		--name $(CONTAINER_NAME)-shell \
		--env-file .env \
		-v $(PWD)/resources:/app/resources:ro \
		-v $(PWD)/output:/app/output \
		$(IMAGE_NAME) \
		/bin/bash

# Rebuild everything from scratch
rebuild: clean build
	@echo "✅ Rebuild complete"

# ============================================================================
# Testing Commands
# ============================================================================

# Run tests in Docker
test: build-test
	@echo "🧪 Running tests in Docker..."
	docker run --rm \
		--name $(CONTAINER_NAME)-test \
		$(IMAGE_NAME)-test \
		python -m pytest tests/ -v --tb=short

# Run tests locally with uv
test-local:
	@echo "🧪 Running tests locally..."
	uv run pytest tests/ -v --tb=short

# ============================================================================
# Development Commands
# ============================================================================

# Install dependencies locally
install:
	uv sync --all-extras

# Format code
format:
	uv run ruff format src/ tests/

# Lint code
lint:
	uv run ruff check src/ tests/

# Run type checking
typecheck:
	uv run mypy src/

# ============================================================================
# Docker Compose Commands (alternative to direct docker commands)
# ============================================================================

compose-build:
	docker compose build

compose-up:
	docker compose up -d prompt-optimizer

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f

compose-test:
	docker compose --profile test run --rm test

compose-optimize:
	docker compose --profile run run --rm optimize
