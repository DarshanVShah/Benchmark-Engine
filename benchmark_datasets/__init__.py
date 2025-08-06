"""
Dataset implementations for the BenchmarkEngine framework.

"""

from .dummy_dataset import DummyDataset
from .huggingface_dataset import HuggingFaceDataset
from .template_dataset import TemplateDataset
from .text_dataset import TextDataset

__all__ = ["DummyDataset", "TextDataset", "HuggingFaceDataset", "TemplateDataset"]
