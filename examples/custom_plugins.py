"""
Custom plugins example demonstrating how to extend the BenchmarkEngine framework.

This example shows how to create custom:
1. Model adapters
2. Metrics
3. Datasets

Following the framework's abstract interfaces.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
import numpy as np
from typing import Dict, Any, List, Optional
from core import BenchmarkEngine, BaseModelAdapter, BaseMetric, BaseDataset


class CustomHuggingFaceAdapter(BaseModelAdapter):
    """
    Example of a custom HuggingFace model adapter.
    
    This shows how to implement a real model adapter that could
    work with actual HuggingFace models.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = "CustomHuggingFace"
        
    def load(self, model_path: str) -> bool:
        """Load a HuggingFace model (simulated)."""
        print(f"Loading HuggingFace model from {model_path}...")
        
        # In a real implementation, you would do:
        # from transformers import AutoModel, AutoTokenizer
        # self.model = AutoModel.from_pretrained(model_path)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # For demo purposes, we'll simulate loading
        time.sleep(0.2)
        self.model_name = f"CustomHF-{random.randint(1000, 9999)}"
        self.model = "simulated_model"  # Mark as loaded
        
        print(f"✓ HuggingFace model loaded: {self.model_name}")
        return True
    
    def run(self, inputs: Any) -> Any:
        """Run inference with HuggingFace model (simulated)."""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        # Simulate tokenization and inference
        time.sleep(0.015)  # Simulate 15ms inference
        
        # Generate dummy predictions
        if isinstance(inputs, dict) and "text" in inputs:
            return f"hf_prediction_{random.randint(1, 100)}"
        else:
            return f"hf_prediction_{random.randint(1, 100)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return HuggingFace model information."""
        return {
            "name": self.model_name,
            "type": "huggingface",
            "framework": "transformers",
            "parameters": random.randint(100, 1000),  # Millions
            "loaded": self.model is not None
        }


class CustomLatencyMetric(BaseMetric):
    """
    Example of a custom latency metric.
    
    This metric calculates various latency statistics.
    """
    
    def __init__(self):
        self.metric_name = "CustomLatency"
        
    def calculate(self, predictions: Any, targets: Any, **kwargs) -> Dict[str, float]:
        """Calculate latency metrics."""
        # In a real implementation, you would extract timing data
        # from the benchmark results or measure it directly
        
        # For demo, generate realistic latency metrics
        avg_latency = random.uniform(10, 50)  # ms
        p50_latency = avg_latency + random.uniform(-5, 5)
        p95_latency = avg_latency + random.uniform(10, 30)
        p99_latency = avg_latency + random.uniform(20, 50)
        
        return {
            "avg_latency_ms": round(avg_latency, 2),
            "p50_latency_ms": round(p50_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "p99_latency_ms": round(p99_latency, 2),
            "min_latency_ms": round(avg_latency - random.uniform(5, 15), 2),
            "max_latency_ms": round(avg_latency + random.uniform(20, 40), 2)
        }
    
    def get_name(self) -> str:
        return self.metric_name


class CustomImageDataset(BaseDataset):
    """
    Example of a custom image dataset loader.
    
    This shows how to implement a dataset that could load
    actual image files.
    """
    
    def __init__(self):
        self.dataset_loaded = False
        self.dataset_name = "CustomImageDataset"
        self.num_samples = 0
        self.image_size = (224, 224)
        
    def load(self, dataset_path: str) -> bool:
        """Load image dataset (simulated)."""
        print(f"Loading image dataset from {dataset_path}...")
        
        # In a real implementation, you would:
        # - Scan the directory for image files
        # - Load image metadata
        # - Set up data loading pipeline
        
        time.sleep(0.1)
        self.dataset_name = f"CustomImage-{random.randint(100, 999)}"
        self.num_samples = random.randint(100, 500)
        self.dataset_loaded = True
        
        print(f"✓ Image dataset loaded: {self.dataset_name} ({self.num_samples} images)")
        return True
    
    def get_samples(self, num_samples: Optional[int] = None) -> List[Any]:
        """Generate synthetic image samples."""
        if not self.dataset_loaded:
            raise RuntimeError("Dataset not loaded")
        
        samples_to_generate = num_samples if num_samples is not None else self.num_samples
        samples_to_generate = min(samples_to_generate, self.num_samples)
        
        samples = []
        for i in range(samples_to_generate):
            # Simulate image data
            sample = {
                "id": f"image_{i:04d}",
                "path": f"/path/to/image_{i:04d}.jpg",
                "size": self.image_size,
                "channels": 3,
                "label": random.randint(0, 999),  # ImageNet-style labels
                "metadata": {
                    "format": "JPEG",
                    "source": "custom_image_dataset"
                }
            }
            samples.append(sample)
        
        return samples
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return image dataset information."""
        return {
            "name": self.dataset_name,
            "type": "image",
            "num_samples": self.num_samples,
            "image_size": self.image_size,
            "channels": 3,
            "loaded": self.dataset_loaded,
            "description": "Custom image dataset for computer vision benchmarks"
        }


def main():
    """Demonstrate custom plugins."""
    
    print("🔧 Custom Plugins Demo")
    print("=" * 50)
    
    # Create engine
    engine = BenchmarkEngine()
    
    # Register custom plugins
    print("\n📦 Registering custom plugins...")
    engine.register_adapter("huggingface", CustomHuggingFaceAdapter)
    engine.register_metric("latency", CustomLatencyMetric)
    engine.register_dataset("images", CustomImageDataset)
    
    # Load model and dataset
    print("\n🔧 Loading custom model and dataset...")
    engine.load_model("huggingface", "bert-base-uncased")
    engine.load_dataset("images", "/path/to/image/dataset")
    
    # Add custom metrics
    print("\n📊 Adding custom metrics...")
    engine.add_metric("latency")
    
    # Run benchmark
    print("\n⚡ Running benchmark with custom plugins...")
    results = engine.run_benchmark(num_samples=15, warmup_runs=2)
    
    # Display results
    print("\n📈 Results:")
    engine.print_results()
    
    # Export results
    print("\n💾 Exporting results...")
    engine.export_results("custom_benchmark_results.json", format="json")
    
    print("\n✅ Custom plugins benchmark completed!")
    print("\nThis demonstrates how easy it is to extend the framework!")
    print("You can create your own adapters, metrics, and datasets")
    print("by implementing the abstract interfaces.")


if __name__ == "__main__":
    main() 