"""
Text dataset for testing HuggingFace models in the BenchmarkEngine framework.

This dataset generates synthetic text data with classification labels,
perfect for testing text classification models.
"""

import random
from typing import Dict, Any, List, Optional, Tuple
from core import BaseDataset


class TextDataset(BaseDataset):
    """
    A text dataset that generates synthetic text samples with classification labels.
    
    This is useful for testing HuggingFace models without requiring
    actual text dataset files.
    """
    
    def __init__(self):
        self.dataset_loaded = False
        self.dataset_name = "TextDataset"
        self.num_samples = 0
        self.num_classes = 2  # Binary classification by default
        self.max_length = 100  # Maximum text length
        
        # Sample texts for different sentiment/classification tasks
        self.positive_texts = [
            "This product is amazing and works perfectly!",
            "I love this service, it's exactly what I needed.",
            "Great experience, highly recommend to everyone.",
            "Outstanding quality and excellent customer support.",
            "This exceeded my expectations completely.",
            "Fantastic performance and reliable results.",
            "Wonderful user interface and smooth operation.",
            "Excellent value for money, worth every penny.",
            "Superb attention to detail and craftsmanship.",
            "Outstanding features and intuitive design."
        ]
        
        self.negative_texts = [
            "This is terrible, completely disappointed.",
            "Waste of money, doesn't work at all.",
            "Poor quality and bad customer service.",
            "Avoid this product, it's a scam.",
            "Horrible experience, would not recommend.",
            "Broken functionality and slow performance.",
            "Difficult to use and confusing interface.",
            "Overpriced for what you actually get.",
            "Falls apart quickly, very low quality.",
            "Frustrating to use and unreliable."
        ]
        
    def load(self, dataset_path: str) -> bool:
        """
        Simulate loading a text dataset.
        
        Args:
            dataset_path: Path to the dataset (ignored in dummy implementation)
            
        Returns:
            True if "loaded" successfully
        """
        print(f"Loading text dataset from {dataset_path}...")
        
        # Simulate dataset metadata
        self.dataset_name = f"TextDataset-{random.randint(100, 999)}"
        self.num_samples = random.randint(50, 200)
        self.dataset_loaded = True
        
        print(f"  Text dataset loaded: {self.dataset_name} ({self.num_samples} samples)")
        print(f"  Classes: {self.num_classes} (binary classification)")
        print(f"  Max length: {self.max_length} characters")
        
        return True
    
    def get_samples(self, num_samples: Optional[int] = None) -> List[Any]:
        """
        Generate synthetic text samples (for compatibility).
        
        Args:
            num_samples: Number of samples to generate (None = all available)
            
        Returns:
            List of synthetic text samples
        """
        # For compatibility, return samples without targets
        samples_with_targets = self.get_samples_with_targets(num_samples)
        return [sample for sample, _ in samples_with_targets]
    
    def get_samples_with_targets(self, num_samples: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """
        Generate synthetic text samples with classification targets.
        
        Args:
            num_samples: Number of samples to generate (None = all available)
            
        Returns:
            List of (sample, target) tuples for text classification
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        # Determine how many samples to generate
        samples_to_generate = num_samples if num_samples is not None else self.num_samples
        samples_to_generate = min(samples_to_generate, self.num_samples)
        
        # Generate synthetic text samples with targets
        samples_with_targets = []
        for i in range(samples_to_generate):
            # Randomly choose positive or negative sentiment
            is_positive = random.choice([True, False])
            
            if is_positive:
                # Positive sentiment (class 1)
                text = random.choice(self.positive_texts)
                target = 1
            else:
                # Negative sentiment (class 0)
                text = random.choice(self.negative_texts)
                target = 0
            
            # Create sample with metadata
            sample = {
                "id": f"text_{i:04d}",
                "text": text,
                "length": len(text),
                "metadata": {
                    "source": "synthetic_text_dataset",
                    "sentiment": "positive" if is_positive else "negative",
                    "timestamp": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
                }
            }
            
            samples_with_targets.append((sample, target))
        
        return samples_with_targets
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Return metadata about the loaded dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        return {
            "name": self.dataset_name,
            "type": "text",
            "num_samples": self.num_samples,
            "num_classes": self.num_classes,
            "max_length": self.max_length,
            "loaded": self.dataset_loaded,
            "description": "Synthetic text dataset for sentiment analysis and text classification",
            "task": "text-classification",
            "labels": ["negative", "positive"]  # Binary classification labels
        }
    
    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Return the expected input shape for models.
        
        Returns:
            Tuple representing the input shape (for text, this is variable)
        """
        # For text data, the shape is variable, but we can specify max length
        return (self.max_length,) 