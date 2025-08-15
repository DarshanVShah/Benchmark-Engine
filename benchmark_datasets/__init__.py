"""
Benchmark Datasets

This package provides standard benchmark datasets for testing models.
"""

from .template_dataset import TemplateDataset
from .simple_text_dataset import SimpleTextDataset
from .huggingface_dataset import HuggingFaceDataset
from .dummy_dataset import DummyDataset
from .goemotions_dataset import GoEmotionsDataset
from .emotion_2018_dataset import Emotion2018Dataset
from .imdb_sentiment_dataset import IMDBDataset

__all__ = [
    "TemplateDataset",
    "SimpleTextDataset", 
    "HuggingFaceDataset",
    "DummyDataset",
    "GoEmotionsDataset",
    "Emotion2018Dataset",
    "IMDBDataset"
]
