"""
Test configuration and fixtures for Prompt Optimizer.
"""

import json
from pathlib import Path
from typing import Generator

import pytest

from prompt_optimizer.models import TargetResult


@pytest.fixture
def sample_target_result() -> TargetResult:
    """
    Create a sample TargetResult for testing.

    Returns:
        TargetResult: A test result with PII data.
    """
    return TargetResult(
        firstname="John",
        email="test@example.com",
        phonenumber="555-1234",
    )


@pytest.fixture
def sample_source_text() -> str:
    """
    Create sample source text for testing.

    Returns:
        str: Source text with PII.
    """
    return "John contact: test@example.com phone: 555-1234"


@pytest.fixture
def sample_data_entry() -> dict:
    """
    Create a sample data entry as loaded from JSON.

    Returns:
        dict: A sample data entry.
    """
    return {
        "source_text": "Contact John at john@email.com for details.",
        "target_result": {
            "firstname": "John",
            "email": "john@email.com",
        },
    }


@pytest.fixture
def temp_jsonl_file(tmp_path: Path, sample_data_entry: dict) -> Generator[Path, None, None]:
    """
    Create a temporary JSONL file for testing.

    Args:
        tmp_path: Pytest temporary path fixture.
        sample_data_entry: Sample data entry.

    Yields:
        Path: Path to the temporary file.
    """
    file_path = tmp_path / "test_data.jsonl"
    
    # Write multiple entries
    with open(file_path, "w") as f:
        for i in range(5):
            entry = sample_data_entry.copy()
            entry["id"] = i
            f.write(json.dumps(entry) + "\n")
    
    yield file_path


@pytest.fixture
def mock_settings(monkeypatch):
    """
    Mock settings for testing without real API key.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-api-key")
    monkeypatch.setenv("AGENT_MODEL", "test-model")
    monkeypatch.setenv("MENTOR_MODEL", "test-model")
    monkeypatch.setenv("WINDOW_SIZE", "2")
    monkeypatch.setenv("LOOP_COUNT", "3")
