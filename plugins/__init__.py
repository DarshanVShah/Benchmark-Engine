"""
Plugin implementations for the BenchmarkEngine framework.

This package contains concrete implementations of:
- Model adapters (HuggingFace, TensorFlow Lite, ONNX, etc.)
- Metrics (accuracy, latency, quantization degradation, etc.)
- Datasets (various dataset loaders)
"""

from .dummy_adapter import DummyModelAdapter
from .huggingface_adapter import HuggingFaceAdapter

# Import metrics from metrics folder
from metrics.dummy_metric import DummyAccuracyMetric
from metrics.accuracy_metric import AccuracyMetric

# Import datasets from datasets folder
from datasets.dummy_dataset import DummyDataset
from datasets.text_dataset import TextDataset

__all__ = [
    # Dummy plugins (for testing)
    'DummyModelAdapter',
    'DummyAccuracyMetric', 
    'DummyDataset',
    
    # Real plugins
    'HuggingFaceAdapter',
    'TextDataset',
    'AccuracyMetric'
] 