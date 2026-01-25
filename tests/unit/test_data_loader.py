"""
Unit tests for data loader module.
"""

import json
from pathlib import Path

import pytest

from prompt_optimizer.data import DataLoader, load_test_data
from prompt_optimizer.models import TargetResult


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
            {"source_text": "Test 1", "target_result": {"firstname": "John"}},
            {"source_text": "Test 2", "target_result": {"email": "test@mail.com"}},
        ]
        
        with open(json_file, "w") as f:
            json.dump(data, f)
        
        loader = DataLoader(json_file)
        samples = loader.load_samples()
        
        assert len(samples) == 2

    def test_parse_target_result(self) -> None:
        """Test parsing target_result from sample."""
        sample = {
            "source_text": "Contact John at john@email.com",
            "target_result": {
                "firstname": "John",
                "email": "john@email.com"
            }
        }
        result = DataLoader.parse_target_result(sample)
        
        assert isinstance(result, TargetResult)
        assert result.firstname == "John"
        assert result.email == "john@email.com"

    def test_parse_target_result_empty(self) -> None:
        """Test parsing target_result with empty data."""
        sample = {
            "source_text": "No PII here",
            "target_result": {},
        }
        
        result = DataLoader.parse_target_result(sample)
        
        assert result.firstname is None
        assert result.email is None

    def test_get_source_text(self) -> None:
        """Test extracting source text."""
        sample = {"source_text": "Contact John at john@email.com"}
        text = DataLoader.get_source_text(sample)
        
        assert text == sample["source_text"]

    def test_get_source_text_missing(self) -> None:
        """Test extracting source text when missing."""
        text = DataLoader.get_source_text({})
        
        assert text == ""


class TestLoadTestData:
    """Tests for load_test_data convenience function."""

    def test_load_test_data(self, tmp_path: Path) -> None:
        """Test load_test_data function."""
        json_file = tmp_path / "test.json"
        data = [
            {"source_text": "Test 1", "target_result": {"firstname": "John"}},
            {"source_text": "Test 2", "target_result": {"email": "test@mail.com"}},
            {"source_text": "Test 3", "target_result": {"phonenumber": "555-1234"}},
        ]
        
        with open(json_file, "w") as f:
            json.dump(data, f)
        
        loaded_data = load_test_data(json_file, limit=3)
        
        assert len(loaded_data) == 3
        assert all(isinstance(item, tuple) for item in loaded_data)
        assert all(len(item) == 2 for item in loaded_data)

    def test_load_test_data_returns_tuples(self, tmp_path: Path) -> None:
        """Test that load_test_data returns (source_text, target_result) tuples."""
        json_file = tmp_path / "test.json"
        data = [
            {"source_text": "Hello John", "target_result": {"firstname": "John"}},
        ]
        
        with open(json_file, "w") as f:
            json.dump(data, f)
        
        loaded_data = load_test_data(json_file, limit=1)
        
        source_text, target_result = loaded_data[0]
        
        assert isinstance(source_text, str)
        assert isinstance(target_result, TargetResult)
