"""
Core module for the BenchmarkEngine framework.

This module provides the main interfaces and engine for ML model benchmarking.
"""

from .dataset_registry import DatasetConfig, DatasetRegistry, TaskType
from .engine import BenchmarkEngine
from .emotion_standardization import StandardizedEmotions, standardized_emotions, map_emotion_to_standard
from .emotion_converter import EmotionOutputConverter, emotion_converter, convert_emotion_output, get_emotion_summary, get_emotion_analysis
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
    # Emotion standardization
    "StandardizedEmotions",
    "standardized_emotions",
    "map_emotion_to_standard",
    # Emotion conversion
    "EmotionOutputConverter",
    "emotion_converter",
    "convert_emotion_output",
    "get_emotion_summary",
    "get_emotion_analysis",
]
