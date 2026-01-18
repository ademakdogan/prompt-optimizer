"""
Unit tests for data loader module.
"""

import json
from pathlib import Path

import pytest

from prompt_optimizer.data import DataLoader, load_test_data
from prompt_optimizer.models import PIIEntity, PIIResponse


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_init_with_valid_file(self, temp_jsonl_file: Path) -> None:
        """Test initialization with valid file."""
        loader = DataLoader(temp_jsonl_file)
        
        assert loader.file_path == temp_jsonl_file

    def test_init_with_invalid_file(self) -> None:
        """Test initialization with non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            DataLoader("/nonexistent/path/file.jsonl")

    def test_load_samples_jsonl(self, temp_jsonl_file: Path) -> None:
        """Test loading samples from JSONL file."""
        loader = DataLoader(temp_jsonl_file)
        samples = loader.load_samples()
        
        assert len(samples) == 5
        assert all("source_text" in s for s in samples)

    def test_load_samples_with_limit(self, temp_jsonl_file: Path) -> None:
        """Test loading samples with limit."""
        loader = DataLoader(temp_jsonl_file)
        samples = loader.load_samples(limit=3)
        
        assert len(samples) == 3

    def test_load_json_file(self, tmp_path: Path) -> None:
        """Test loading samples from JSON file."""
        json_file = tmp_path / "test.json"
        data = [
            {"source_text": "Test 1", "target_text": "[TEST]"},
            {"source_text": "Test 2", "target_text": "[TEST]"},
        ]
        
        with open(json_file, "w") as f:
            json.dump(data, f)
        
        loader = DataLoader(json_file)
        samples = loader.load_samples()
        
        assert len(samples) == 2

    def test_parse_ground_truth(self, sample_data_entry: dict) -> None:
        """Test parsing ground truth from sample."""
        result = DataLoader.parse_ground_truth(sample_data_entry)
        
        assert isinstance(result, PIIResponse)
        assert len(result.entities) == 2
        assert result.entities[0].label == "FIRSTNAME"
        assert result.entities[1].label == "EMAIL"

    def test_parse_ground_truth_empty(self) -> None:
        """Test parsing ground truth with no entities."""
        sample = {
            "source_text": "No PII here",
            "target_text": "No PII here",
            "privacy_mask": [],
        }
        
        result = DataLoader.parse_ground_truth(sample)
        
        assert len(result.entities) == 0

    def test_get_source_text(self, sample_data_entry: dict) -> None:
        """Test extracting source text."""
        text = DataLoader.get_source_text(sample_data_entry)
        
        assert text == sample_data_entry["source_text"]

    def test_get_source_text_missing(self) -> None:
        """Test extracting source text when missing."""
        text = DataLoader.get_source_text({})
        
        assert text == ""


class TestLoadTestData:
    """Tests for load_test_data convenience function."""

    def test_load_test_data(self, temp_jsonl_file: Path) -> None:
        """Test load_test_data function."""
        data = load_test_data(temp_jsonl_file, limit=3)
        
        assert len(data) == 3
        assert all(isinstance(item, tuple) for item in data)
        assert all(len(item) == 2 for item in data)

    def test_load_test_data_returns_tuples(self, temp_jsonl_file: Path) -> None:
        """Test that load_test_data returns (source_text, ground_truth) tuples."""
        data = load_test_data(temp_jsonl_file, limit=1)
        
        source_text, ground_truth = data[0]
        
        assert isinstance(source_text, str)
        assert isinstance(ground_truth, PIIResponse)
