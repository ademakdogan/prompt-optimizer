"""
Data loader module for Prompt Optimizer.

This module provides utilities for loading and parsing data
from various formats (JSONL, JSON) for PII extraction tasks.
"""

import json
from pathlib import Path
from typing import Iterator

from prompt_optimizer.models import PIIEntity, PIIResponse
from prompt_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Data loader for PII extraction datasets.

    Supports loading from JSONL and JSON formats, with methods
    to parse ground truth into structured format.

    Examples:
        >>> loader = DataLoader("resources/pii/english_pii_43k.jsonl")
        >>> samples = loader.load_samples(limit=5)
        >>> len(samples) == 5
        True
    """

    def __init__(self, file_path: str | Path) -> None:
        """
        Initialize the data loader.

        Args:
            file_path: Path to the data file (JSONL or JSON).

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Initialized DataLoader with file: {file_path}")

    def load_samples(self, limit: int | None = None) -> list[dict]:
        """
        Load samples from the data file.

        Args:
            limit: Maximum number of samples to load. If None, loads all.

        Returns:
            list[dict]: List of sample dictionaries.

        Examples:
            >>> loader = DataLoader("test_data.jsonl")
            >>> samples = loader.load_samples(limit=10)
            >>> all("source_text" in s for s in samples)
            True
        """
        samples = []
        
        if self.file_path.suffix == ".jsonl":
            samples = list(self._load_jsonl(limit))
        elif self.file_path.suffix == ".json":
            samples = self._load_json(limit)
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")
        
        logger.info(f"Loaded {len(samples)} samples from {self.file_path}")
        return samples

    def _load_jsonl(self, limit: int | None = None) -> Iterator[dict]:
        """
        Load samples from a JSONL file.

        Args:
            limit: Maximum number of samples to load.

        Yields:
            dict: Individual sample dictionaries.
        """
        count = 0
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if limit is not None and count >= limit:
                    break
                
                line = line.strip()
                if line:
                    yield json.loads(line)
                    count += 1

    def _load_json(self, limit: int | None = None) -> list[dict]:
        """
        Load samples from a JSON file.

        Args:
            limit: Maximum number of samples to load.

        Returns:
            list[dict]: List of sample dictionaries.
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data[:limit] if limit else data
        else:
            return [data]

    @staticmethod
    def parse_ground_truth(sample: dict) -> PIIResponse:
        """
        Parse a sample's ground truth into a PIIResponse.

        Args:
            sample: A sample dictionary containing privacy_mask and target_text.

        Returns:
            PIIResponse: Structured ground truth response.

        Examples:
            >>> sample = {
            ...     "privacy_mask": [
            ...         {"value": "John", "label": "FIRSTNAME", "start": 0, "end": 4}
            ...     ],
            ...     "target_text": "[FIRSTNAME] works here"
            ... }
            >>> result = DataLoader.parse_ground_truth(sample)
            >>> result.entities[0].label
            'FIRSTNAME'
        """
        entities = []
        
        privacy_mask = sample.get("privacy_mask", [])
        for mask in privacy_mask:
            entity = PIIEntity(
                value=mask.get("value", ""),
                label=mask.get("label", ""),
                start=mask.get("start", 0),
                end=mask.get("end", 0),
            )
            entities.append(entity)
        
        masked_text = sample.get("target_text", "")
        
        return PIIResponse(entities=entities, masked_text=masked_text)

    @staticmethod
    def get_source_text(sample: dict) -> str:
        """
        Extract source text from a sample.

        Args:
            sample: A sample dictionary.

        Returns:
            str: The source text for analysis.

        Examples:
            >>> sample = {"source_text": "Contact John at john@email.com"}
            >>> DataLoader.get_source_text(sample)
            'Contact John at john@email.com'
        """
        return sample.get("source_text", "")


def load_test_data(
    file_path: str | Path,
    limit: int = 5,
) -> list[tuple[str, PIIResponse]]:
    """
    Convenience function to load test data with ground truth.

    Args:
        file_path: Path to the data file.
        limit: Number of samples to load.

    Returns:
        list[tuple[str, PIIResponse]]: List of (source_text, ground_truth) tuples.

    Examples:
        >>> data = load_test_data("resources/pii/english_pii_43k.jsonl", limit=5)
        >>> len(data) == 5
        True
        >>> isinstance(data[0][1], PIIResponse)
        True
    """
    loader = DataLoader(file_path)
    samples = loader.load_samples(limit)
    
    result = []
    for sample in samples:
        source_text = DataLoader.get_source_text(sample)
        ground_truth = DataLoader.parse_ground_truth(sample)
        result.append((source_text, ground_truth))
    
    return result
