"""
2018-E-c-En Emotion Dataset

This module provides a proper implementation for the 2018-E-c-En emotion dataset.
This is a multi-label emotion classification dataset with 11 emotions.
"""

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

from core import BaseDataset, DataType


class Emotion2018Dataset(BaseDataset):
    """
    2018-E-c-En emotion dataset implementation.
    
    Dataset format: TSV with columns [ID, Tweet, anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust]
    - ID: Tweet identifier
    - Tweet: Text content
    - anger, anticipation, etc.: Binary emotion labels (0 or 1)
    
    Emotions: 11 different emotions in multi-label format
    """

    def __init__(self):
        self.data = []
        self.dataset_loaded = False
        self.dataset_path = ""
        self.max_length = 512
        self.num_classes = 11
        self.emotion_labels = [
            "anger", "anticipation", "disgust", "fear", "joy", 
            "love", "optimism", "pessimism", "sadness", "surprise", "trust"
        ]

    @property
    def output_type(self) -> DataType:
        """2018-E-c-En dataset outputs text."""
        return DataType.TEXT

    def load(self, dataset_path: str, config: Dict[str, Any]) -> bool:
        """
        Load 2018-E-c-En dataset from TSV file.
        
        Args:
            dataset_path: Path to the 2018-E-c-En TSV file
            config: Configuration dict (max_length, etc.)
        """
        try:
            # Check if file exists
            if not os.path.exists(dataset_path):
                print(f"2018-E-c-En dataset file not found: {dataset_path}")
                return False
            
            # Store configuration
            self.max_length = config.get("max_length", 512)
            self.dataset_path = dataset_path
            
            # Load TSV data
            self._load_tsv(dataset_path)
            
            if self.data:
                self.dataset_loaded = True
                print(f"2018-E-c-En dataset loaded: {len(self.data)} samples")
                return True
            else:
                print("No data loaded from 2018-E-c-En dataset")
                return False
                
        except Exception as e:
            print(f"Error loading 2018-E-c-En dataset: {e}")
            return False

    def _load_tsv(self, file_path: str):
        """Load TSV file with 2018-E-c-En format."""
        self.data = []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                
                for row_num, row in enumerate(reader):
                    if len(row) >= 13:  # ID + Tweet + 11 emotions
                        tweet_id = row[0].strip()
                        text = row[1].strip()
                        
                        # Skip empty rows
                        if not text:
                            continue
                        
                        # Parse emotion labels (columns 2-12)
                        emotion_labels = []
                        for i in range(2, 13):
                            try:
                                label = int(row[i].strip())
                                emotion_labels.append(label)
                            except (ValueError, IndexError):
                                emotion_labels.append(0)
                        
                        # Ensure we have exactly 11 emotion labels
                        if len(emotion_labels) == 11:
                            # Truncate text if needed
                            if len(text) > self.max_length:
                                text = text[:self.max_length]
                            
                            # Create emotion mapping
                            emotion_dict = {}
                            for i, label in enumerate(emotion_labels):
                                emotion_dict[self.emotion_labels[i]] = label
                            
                            self.data.append({
                                "id": tweet_id,
                                "text": text,
                                "emotions": emotion_dict,
                                "emotion_vector": emotion_labels,
                                "has_emotions": any(emotion_labels)
                            })
                            
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

    def get_emotions(self, index: int) -> Optional[Dict[str, int]]:
        """Get emotion dictionary for a sample."""
        sample = self.get_sample(index)
        return sample["emotions"] if sample else None

    def get_emotion_vector(self, index: int) -> Optional[List[int]]:
        """Get emotion vector for a sample."""
        sample = self.get_sample(index)
        return sample["emotion_vector"] if sample else None

    def get_emotion_names(self, index: int) -> Optional[List[str]]:
        """Get list of emotion names that are present (label=1)."""
        sample = self.get_sample(index)
        if not sample:
            return None
        
        present_emotions = []
        for emotion, label in sample["emotions"].items():
            if label == 1:
                present_emotions.append(emotion)
        return present_emotions

    def get_all_samples(self) -> List[Dict[str, Any]]:
        """Get all samples."""
        return self.data.copy()

    def get_sample_count(self) -> int:
        """Get total number of samples."""
        return len(self.data)

    def get_emotion_distribution(self) -> Dict[str, int]:
        """Get distribution of emotions in the dataset."""
        distribution = {emotion: 0 for emotion in self.emotion_labels}
        
        for sample in self.data:
            for emotion, label in sample["emotions"].items():
                if label == 1:
                    distribution[emotion] += 1
        
        return distribution

    def get_samples_by_emotion(self, emotion: str) -> List[Dict[str, Any]]:
        """Get all samples that have a specific emotion."""
        return [sample for sample in self.data if sample["emotions"].get(emotion, 0) == 1]

    def get_multi_label_samples(self) -> List[Dict[str, Any]]:
        """Get samples that have multiple emotions."""
        return [sample for sample in self.data if sum(sample["emotion_vector"]) > 1]

    def get_single_label_samples(self) -> List[Dict[str, Any]]:
        """Get samples that have only one emotion."""
        return [sample for sample in self.data if sum(sample["emotion_vector"]) == 1]

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
            if "text" not in sample or "emotions" not in sample or "emotion_vector" not in sample:
                return False
            
            if not isinstance(sample["text"], str):
                return False
            
            if not isinstance(sample["emotions"], dict) or not isinstance(sample["emotion_vector"], list):
                return False
            
            if len(sample["emotion_vector"]) != self.num_classes:
                return False
        
        return True

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded dataset."""
        if not self.dataset_loaded:
            return {"error": "Dataset not loaded"}
        
        emotion_dist = self.get_emotion_distribution()
        multi_label_count = len(self.get_multi_label_samples())
        single_label_count = len(self.get_single_label_samples())
        
        return {
            "name": "2018-E-c-En",
            "path": self.dataset_path,
            "total_samples": len(self.data),
            "num_classes": self.num_classes,
            "max_length": self.max_length,
            "emotion_labels": self.emotion_labels,
            "emotion_distribution": emotion_dist,
            "multi_label_samples": multi_label_count,
            "single_label_samples": single_label_count,
            "validation": self.validate_dataset()
        }
