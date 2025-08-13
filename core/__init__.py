"""
Core module for the BenchmarkEngine framework.

This module provides the main interfaces and engine for ML model benchmarking.
"""

from .dataset_registry import DatasetConfig, DatasetRegistry, TaskType
from .engine import BenchmarkEngine
from .interfaces import BaseDataset, BaseMetric, BaseModelAdapter, DataType, OutputType
from .types import BenchmarkConfig, ModelType

__all__ = [
    # Core engine
    "BenchmarkEngine",
    # Interfaces
    "BaseModelAdapter",
    "BaseMetric",
    "BaseDataset",
    "DataType",
    "OutputType",
    # Dataset registry
    "DatasetRegistry",
    "DatasetConfig",
    "TaskType",
    # Configuration
    "BenchmarkConfig",
    "ModelType",
]
