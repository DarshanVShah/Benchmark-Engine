"""
Core module for the BenchmarkEngine framework.

This module provides the main interfaces and engine for ML model benchmarking.
"""

from .interfaces import (
    BaseModelAdapter,
    BaseMetric, 
    BaseDataset,
    DataType,
    OutputType
)

from .engine import BenchmarkEngine

from .dataset_registry import (
    DatasetRegistry,
    DatasetConfig,
    TaskType
)

from .types import BenchmarkConfig

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
    "BenchmarkConfig"
]
