"""
Simple Text Dataset for demonstrating the refactored framework.

This dataset provides text samples for classification tasks.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
from core import BaseDataset, DataType


class SimpleTextDataset(BaseDataset):
    """
    Simple text dataset for classification tasks.
    
    This dataset demonstrates the new architecture where:
    - Dataset declares its output_type (TEXT)
    - Model adapters must declare their input_type (TEXT)
    - Framework validates compatibility before running
    """

    def __init__(self):
        self.data = []
        self.dataset_loaded = False
        self.dataset_path = ""
        self.name = "SimpleTextDataset"

    @property
    def output_type(self) -> DataType:
        """This dataset outputs text data."""
        return DataType.TEXT

    def load(self, dataset_path: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load text samples from a file.
        
        Args:
            dataset_path: Path to text file (one sample per line)
            config: Optional configuration (not used for simple dataset)
        """
        try:
            print(f"Loading simple text dataset: {dataset_path}")
            
            if not os.path.exists(dataset_path):
                print(f"  Error: Dataset file not found: {dataset_path}")
                return False
            
            # Load text samples (one per line)
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.data = [line.strip() for line in f if line.strip()]
            
            self.dataset_path = dataset_path
            self.dataset_loaded = True
            
            print(f"  ✓ Dataset loaded successfully")
            print(f"  Samples: {len(self.data)}")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Failed to load dataset: {e}")
            return False

    def get_samples(self, num_samples: Optional[int] = None) -> List[Any]:
        """Get text samples from the dataset."""
        if not self.dataset_loaded:
            return []
        
        samples = self.data
        if num_samples:
            samples = samples[:num_samples]
        
        return samples

    def get_samples_with_targets(self, num_samples: Optional[int] = None) -> List[Tuple[Any, Any]]:
        """
        Get (input, target) pairs for evaluation.
        
        For this simple dataset, we'll create dummy targets.
        In a real implementation, you'd load actual targets from the dataset.
        """
        samples = self.get_samples(num_samples)
        
        # Create dummy targets (class 0 for all samples)
        # In a real implementation, you'd load actual targets
        targets = [0] * len(samples)
        
        return list(zip(samples, targets))

    def get_dataset_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded dataset."""
        return {
            "name": self.name,
            "path": self.dataset_path,
            "num_samples": len(self.data),
            "loaded": self.dataset_loaded,
            "output_type": self.output_type.value
        }

    def get_input_shape(self) -> Tuple[int, ...]:
        """Return the expected input shape for models."""
        # For text data, we don't have a fixed shape
        # The model adapter will handle tokenization
        return (1,)  # Single text input
