"""
Data module for Prompt Optimizer.

This module provides utilities for:
- Loading data from JSONL and JSON files
- Parsing ground truth into structured format
"""

from prompt_optimizer.data.loader import DataLoader, load_test_data

__all__ = ["DataLoader", "load_test_data"]
