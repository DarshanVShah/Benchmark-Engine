"""
Dummy dataset for testing the BenchmarkEngine framework.

This dataset generates synthetic data without requiring
actual dataset files.
"""

import random
from typing import Dict, Any, List, Optional
from core import BaseDataset


class DummyDataset(BaseDataset):
    """
    A dummy dataset that generates synthetic data.
    
    This is useful for testing the framework without requiring
    actual dataset files.
    """
    
    def __init__(self):
        self.dataset_loaded = False
        self.dataset_name = "DummyDataset"
        self.num_samples = 0
        self.sample_size = 10  # Number of features per sample
        
    def load(self, dataset_path: str) -> bool:
        """
        Simulate loading a dataset.
        
        Args:
            dataset_path: Path to the dataset (ignored in dummy implementation)
            
        Returns:
            True if "loaded" successfully
        """
        print(f"Loading dummy dataset from {dataset_path}...")
        
        # Simulate dataset metadata
        self.dataset_name = f"DummyDataset-{random.randint(100, 999)}"
        self.num_samples = random.randint(50, 200)
        self.dataset_loaded = True
        
        print(f"✓ Dummy dataset loaded: {self.dataset_name} ({self.num_samples} samples)")
        return True
    
    def get_samples(self, num_samples: Optional[int] = None) -> List[Any]:
        """
        Generate synthetic samples.
        
        Args:
            num_samples: Number of samples to generate (None = all available)
            
        Returns:
            List of synthetic samples
        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        
        # Determine how many samples to generate
        samples_to_generate = num_samples if num_samples is not None else self.num_samples
        samples_to_generate = min(samples_to_generate, self.num_samples)
        
        # Generate synthetic samples
        samples = []
        for i in range(samples_to_generate):
            # Create a synthetic sample with random features
            sample = {
                "id": f"sample_{i:04d}",
                "features": [random.uniform(0, 1) for _ in range(self.sample_size)],
                "label": random.randint(0, 9),  # 10-class classification
                "metadata": {
                    "source": "dummy",
                    "timestamp": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
                }
            }
            samples.append(sample)
        
        return samples
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Return metadata about the loaded dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        return {
            "name": self.dataset_name,
            "type": "dummy",
            "num_samples": self.num_samples,
            "sample_size": self.sample_size,
            "loaded": self.dataset_loaded,
            "description": "Synthetic dataset for testing the BenchmarkEngine framework"
        } 