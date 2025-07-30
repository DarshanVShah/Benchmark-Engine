"""
Dataset implementations for the BenchmarkEngine framework.

This package contains dataset loaders for various data types:
- Text datasets (NLP, sentiment analysis)
- Image datasets (computer vision, classification)
- Audio datasets (speech recognition, audio classification)
- Custom datasets (user-defined data loaders)
"""

from .dummy_dataset import DummyDataset
from .text_dataset import TextDataset
from .huggingface_dataset import HuggingFaceDataset

__all__ = [
    'DummyDataset',
    'TextDataset',
    'HuggingFaceDataset'
] 