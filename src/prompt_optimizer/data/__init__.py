"""
Data module for Prompt Optimizer.

This module provides data handling utilities for:
- Loading data from various formats (JSONL, JSON)
- Parsing ground truth values
"""

from prompt_optimizer.data.loader import DataLoader, load_test_data

__all__ = ["DataLoader", "load_test_data"]
