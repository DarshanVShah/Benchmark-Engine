"""
IMDB Sentiment Dataset

This module provides a proper implementation for the IMDB sentiment dataset.
This is a binary sentiment classification dataset (positive/negative).
"""

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

from core import BaseDataset, DataType


class IMDBDataset(BaseDataset):
    """
    IMDB sentiment dataset implementation.
    
    Dataset format: CSV with columns [text, sentiment]
    - text: Movie review text
    - sentiment: Binary sentiment label (positive/negative or 1/0)
    
    Task: Binary sentiment classification
    """

    def __init__(self):
        self.data = []
        self.dataset_loaded = False
        self.dataset_path = ""
        self.max_length = 512
        self.num_classes = 2
        self.sentiment_labels = ["negative", "positive"]

    @property
    def output_type(self) -> DataType:
        """IMDB dataset outputs text."""
        return DataType.TEXT

    def load(self, dataset_path: str, config: Dict[str, Any]) -> bool:
        """
        Load IMDB dataset from CSV file.
        
        Args:
            dataset_path: Path to the IMDB CSV file
            config: Configuration dict (max_length, etc.)
        """
        try:
            # Check if file exists
            if not os.path.exists(dataset_path):
                print(f"IMDB dataset file not found: {dataset_path}")
                return False
            
            # Store configuration
            self.max_length = config.get("max_length", 512)
            self.dataset_path = dataset_path
            
            # Load CSV data
            self._load_csv(dataset_path)
            
            if self.data:
                self.dataset_loaded = True
                print(f"IMDB dataset loaded: {len(self.data)} samples")
                return True
            else:
                print("No data loaded from IMDB dataset")
                return False
                
        except Exception as e:
            print(f"Error loading IMDB dataset: {e}")
            return False

    def _load_csv(self, file_path: str):
        """Load CSV file with IMDB format."""
        self.data = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    text = row.get("text", "").strip()
                    sentiment = row.get("sentiment", "").strip()
                    
                    # Skip empty rows
                    if not text or not sentiment:
                        continue
                    
                    # Convert sentiment to numeric label
                    try:
                        if sentiment.lower() in ["positive", "1", "pos"]:
                            label = 1
                        elif sentiment.lower() in ["negative", "0", "neg"]:
                            label = 0
                        else:
                            # Try to parse as integer
                            label = int(sentiment)
                            if label not in [0, 1]:
                                continue
                        sentiment_name = self.sentiment_labels[label]
                    except ValueError:
                        continue
                    
                    # Truncate text if needed
                    if len(text) > self.max_length:
                        text = text[:self.max_length]
                    
                    self.data.append({
                        "text": text,
                        "sentiment": sentiment_name,
                        "label": label
                    })
                            
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            self.data = []

    def get_sample(self, index: int) -> Optional[Dict[str, Any]]:
        """Get a sample by index."""
        if 0 <= index < len(self.data):
            return self.data[index]
        return None

    def get_text(self, index: int) -> Optional[str]:
        """Get text for a sample."""
        sample = self.get_sample(index)
        return sample["text"] if sample else None

    def get_sentiment(self, index: int) -> Optional[str]:
        """Get sentiment name for a sample."""
        sample = self.get_sample(index)
        return sample["sentiment"] if sample else None

    def get_label(self, index: int) -> Optional[int]:
        """Get numeric label for a sample."""
        sample = self.get_sample(index)
        return sample["label"] if sample else None

    def get_all_samples(self) -> List[Dict[str, Any]]:
        """Get all samples."""
        return self.data.copy()

    def get_sample_count(self) -> int:
        """Get total number of samples."""
        return len(self.data)

    def get_sentiment_distribution(self) -> Dict[str, int]:
        """Get distribution of sentiments in the dataset."""
        distribution = {"positive": 0, "negative": 0}
        
        for sample in self.data:
            sentiment = sample["sentiment"]
            distribution[sentiment] += 1
        
        return distribution

    def get_samples_by_sentiment(self, sentiment: str) -> List[Dict[str, Any]]:
        """Get all samples for a specific sentiment."""
        return [sample for sample in self.data if sample["sentiment"] == sentiment]

    def get_positive_samples(self) -> List[Dict[str, Any]]:
        """Get all positive samples."""
        return self.get_samples_by_sentiment("positive")

    def get_negative_samples(self) -> List[Dict[str, Any]]:
        """Get all negative samples."""
        return self.get_samples_by_sentiment("negative")

    def get_random_sample(self) -> Optional[Dict[str, Any]]:
        """Get a random sample from the dataset."""
        import random
        if self.data:
            return random.choice(self.data)
        return None

    def validate_dataset(self) -> bool:
        """Validate that the dataset is properly loaded and formatted."""
        if not self.dataset_loaded:
            return False
        
        if not self.data:
            return False
        
        # Check that all samples have required fields
        for sample in self.data:
            if "text" not in sample or "sentiment" not in sample or "label" not in sample:
                return False
            
            if not isinstance(sample["text"], str):
                return False
            
            if not isinstance(sample["label"], int) or sample["label"] not in [0, 1]:
                return False
            
            if sample["sentiment"] not in self.sentiment_labels:
                return False
        
        return True

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded dataset."""
        if not self.dataset_loaded:
            return {"error": "Dataset not loaded"}
        
        sentiment_dist = self.get_sentiment_distribution()
        positive_count = len(self.get_positive_samples())
        negative_count = len(self.get_negative_samples())
        
        return {
            "name": "IMDB Sentiment",
            "path": self.dataset_path,
            "total_samples": len(self.data),
            "num_classes": self.num_classes,
            "max_length": self.max_length,
            "sentiment_labels": self.sentiment_labels,
            "sentiment_distribution": sentiment_dist,
            "positive_samples": positive_count,
            "negative_samples": negative_count,
            "validation": self.validate_dataset()
        }
