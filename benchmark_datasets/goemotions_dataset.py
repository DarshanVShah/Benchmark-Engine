"""
GoEmotions Dataset

This module provides a proper implementation for the GoEmotions dataset.
GoEmotions is a large-scale emotion dataset with 27 emotions from Reddit.
"""

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

from core.interfaces import BaseDataset, DataType


class GoEmotionsDataset(BaseDataset):
    """
    GoEmotions dataset implementation.
    
    Dataset format: TSV with columns [text, label]
    - text: Reddit comment text
    - label: Numeric emotion label (0-26)
    
    Emotions: 27 different emotions including joy, anger, fear, etc.
    """

    def __init__(self):
        self.data = []
        self.dataset_loaded = False
        self.dataset_path = ""
        self.max_length = 512
        self.num_classes = 27
        self.emotion_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral"
        ]

    @property
    def output_type(self) -> DataType:
        """GoEmotions dataset outputs text."""
        return DataType.TEXT

    def load(self, dataset_path: str, config: Dict[str, Any]) -> bool:
        """
        Load GoEmotions dataset from TSV file.
        
        Args:
            dataset_path: Path to the GoEmotions TSV file
            config: Configuration dict (max_length, etc.)
        """
        try:
            # Check if file exists
            if not os.path.exists(dataset_path):
                print(f"GoEmotions dataset file not found: {dataset_path}")
                return False
            
            # Store configuration
            self.max_length = config.get("max_length", 512)
            self.dataset_path = dataset_path
            
            # Load TSV data
            self._load_tsv(dataset_path)
            
            if self.data:
                self.dataset_loaded = True
                print(f"GoEmotions dataset loaded: {len(self.data)} samples")
                return True
            else:
                print("No data loaded from GoEmotions dataset")
                return False
                
        except Exception as e:
            print(f"Error loading GoEmotions dataset: {e}")
            return False

    def _load_tsv(self, file_path: str):
        """Load TSV file with GoEmotions format."""
        self.data = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                
                for row_num, row in enumerate(reader):
                    if len(row) >= 2:
                        text = row[0].strip()
                        label_str = row[1].strip()
                        
                        # Skip empty rows
                        if not text or not label_str:
                            continue
                        
                        # Convert label to integer
                        try:
                            label = int(label_str)
                            if 0 <= label < self.num_classes:
                                # Truncate text if needed
                                if len(text) > self.max_length:
                                    text = text[:self.max_length]
                                
                                self.data.append({
                                    "text": text,
                                    "label": label,
                                    "emotion": self.emotion_labels[label]
                                })
                        except ValueError:
                            # Skip rows with invalid labels
                            continue
                            
        except Exception as e:
            print(f"Error reading TSV file: {e}")
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

    def get_label(self, index: int) -> Optional[int]:
        """Get numeric label for a sample."""
        sample = self.get_sample(index)
        return sample["label"] if sample else None

    def get_emotion(self, index: int) -> Optional[str]:
        """Get emotion name for a sample."""
        sample = self.get_sample(index)
        return sample["emotion"] if sample else None

    def get_all_samples(self) -> List[Dict[str, Any]]:
        """Get all samples."""
        return self.data.copy()

    def get_sample_count(self) -> int:
        """Get total number of samples."""
        return len(self.data)

    def get_emotion_distribution(self) -> Dict[str, int]:
        """Get distribution of emotions in the dataset."""
        distribution = {}
        for sample in self.data:
            emotion = sample["emotion"]
            distribution[emotion] = distribution.get(emotion, 0) + 1
        return distribution

    def get_samples_by_emotion(self, emotion: str) -> List[Dict[str, Any]]:
        """Get all samples for a specific emotion."""
        return [sample for sample in self.data if sample["emotion"] == emotion]

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
            if "text" not in sample or "label" not in sample or "emotion" not in sample:
                return False
            
            if not isinstance(sample["text"], str) or not isinstance(sample["label"], int):
                return False
            
            if not (0 <= sample["label"] < self.num_classes):
                return False
        
        return True

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded dataset."""
        if not self.dataset_loaded:
            return {"error": "Dataset not loaded"}
        
        emotion_dist = self.get_emotion_distribution()
        
        return {
            "name": "GoEmotions",
            "path": self.dataset_path,
            "total_samples": len(self.data),
            "num_classes": self.num_classes,
            "max_length": self.max_length,
            "emotion_distribution": emotion_dist,
            "emotion_labels": self.emotion_labels,
            "validation": self.validate_dataset()
        }

    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the expected input shape for models."""
        return (self.max_length,)

    def get_samples(self, num_samples: Optional[int] = None) -> List[Any]:
        """Get samples from the dataset (for compatibility)."""
        if num_samples is None:
            return self.data.copy()
        return self.data[:num_samples]

    def get_samples_with_targets(
        self, num_samples: Optional[int] = None
    ) -> List[Tuple[Any, Any]]:
        """Get (input, target) pairs for evaluation."""
        samples = self.get_samples(num_samples)
        return [(sample["text"], sample["label"]) for sample in samples]

