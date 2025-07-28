"""
Plugin implementations for the BenchmarkEngine framework.

This package contains concrete implementations of:
- Model adapters (HuggingFace, TensorFlow Lite, ONNX, etc.)
- Metrics (accuracy, latency, quantization degradation, etc.)
- Datasets (various dataset loaders)
"""

from .dummy_adapter import DummyModelAdapter
from .dummy_metric import DummyAccuracyMetric
from .dummy_dataset import DummyDataset

__all__ = [
    'DummyModelAdapter',
    'DummyAccuracyMetric', 
    'DummyDataset'
] 