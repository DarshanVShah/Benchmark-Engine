"""
Plugin implementations for the BenchmarkEngine framework.

This package contains concrete implementations of:
- Model adapters (HuggingFace, TensorFlow Lite, ONNX, etc.)
"""

from .dummy_adapter import DummyModelAdapter
from .huggingface_adapter import HuggingFaceAdapter
from .tflite_adapter import TensorFlowLiteAdapter

__all__ = [
    # Model adapters
    'DummyModelAdapter',
    'HuggingFaceAdapter',
    'TensorFlowLiteAdapter',
] 