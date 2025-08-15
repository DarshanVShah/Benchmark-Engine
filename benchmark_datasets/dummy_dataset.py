"""
Dummy dataset for testing the BenchmarkEngine framework.

This dataset generates synthetic data without requiring
actual dataset files.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from core.interfaces import BaseDataset


class DummyDataset(BaseDataset):
    """
    A dummy dataset that generates synthetic data.

    """

    def __init__(self):
        self.dataset_loaded = False
        self.dataset_name = "DummyDataset"
        self.num_samples = 0
        self.sample_size = 10  # Number of features per sample

    def load(self, dataset_path: str) -> bool:
        """
        Simulate loading a dataset.

        """
        print(f"Loading dummy dataset from {dataset_path}...")

        # Simulate dataset metadata
        self.dataset_name = f"DummyDataset-{random.randint(100, 999)}"
        self.num_samples = random.randint(50, 200)
        self.dataset_loaded = True

        print(f"Dummy dataset loaded: {self.dataset_name} ({self.num_samples} samples)")
        return True

    def get_samples(self, num_samples: Optional[int] = None) -> List[Any]:
        """
        Generate synthetic samples (for compatibility).

        """
        # For compatibility, return samples without targets
        samples_with_targets = self.get_samples_with_targets(num_samples)
        return [sample for sample, _ in samples_with_targets]

    def get_samples_with_targets(
        self, num_samples: Optional[int] = None
    ) -> List[Tuple[Any, Any]]:
        """
        Generate synthetic samples with targets.

        """
        if not self.dataset_loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        # Determine how many samples to generate
        samples_to_generate = (
            num_samples if num_samples is not None else self.num_samples
        )
        samples_to_generate = min(samples_to_generate, self.num_samples)

        # Generate synthetic samples with targets
        samples_with_targets = []
        for i in range(samples_to_generate):
            # Create a synthetic sample with random features
            sample = {
                "id": f"sample_{i:04d}",
                "features": [random.uniform(0, 1) for _ in range(self.sample_size)],
                "metadata": {
                    "source": "dummy",
                    "timestamp": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                },
            }

            # Create a synthetic target (10-class classification)
            target = random.randint(0, 9)

            samples_with_targets.append((sample, target))

        return samples_with_targets

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Return metadata about the loaded dataset.

        """
        return {
            "name": self.dataset_name,
            "type": "dummy",
            "num_samples": self.num_samples,
            "sample_size": self.sample_size,
            "loaded": self.dataset_loaded,
            "description": "Synthetic dataset for testing the BenchmarkEngine framework",
        }

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Return the expected input shape for models.

        """
        # For dummy dataset, return a simple shape
        return (self.sample_size,)
