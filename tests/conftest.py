"""
Test configuration and fixtures for Prompt Optimizer.
"""

import json
from pathlib import Path
from typing import Generator

import pytest

from prompt_optimizer.models import PIIEntity, PIIResponse


@pytest.fixture
def sample_pii_entity() -> PIIEntity:
    """
    Create a sample PIIEntity for testing.

    Returns:
        PIIEntity: A test entity with email data.
    """
    return PIIEntity(
        value="test@example.com",
        label="EMAIL",
        start=10,
        end=26,
    )


@pytest.fixture
def sample_pii_response(sample_pii_entity: PIIEntity) -> PIIResponse:
    """
    Create a sample PIIResponse for testing.

    Args:
        sample_pii_entity: A sample entity.

    Returns:
        PIIResponse: A test response.
    """
    return PIIResponse(
        entities=[sample_pii_entity],
        masked_text="Contact: [EMAIL] for info.",
    )


@pytest.fixture
def sample_ground_truth() -> PIIResponse:
    """
    Create sample ground truth for testing.

    Returns:
        PIIResponse: Ground truth with multiple entity types.
    """
    return PIIResponse(
        entities=[
            PIIEntity(value="John", label="FIRSTNAME", start=0, end=4),
            PIIEntity(value="test@example.com", label="EMAIL", start=20, end=36),
            PIIEntity(value="555-1234", label="PHONENUMBER", start=45, end=53),
        ],
        masked_text="[FIRSTNAME] contact: [EMAIL] phone: [PHONENUMBER]",
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
    Create a sample data entry as loaded from JSONL.

    Returns:
        dict: A sample data entry.
    """
    return {
        "source_text": "Contact John at john@email.com for details.",
        "target_text": "Contact [FIRSTNAME] at [EMAIL] for details.",
        "privacy_mask": [
            {"value": "John", "label": "FIRSTNAME", "start": 8, "end": 12},
            {"value": "john@email.com", "label": "EMAIL", "start": 16, "end": 30},
        ],
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
