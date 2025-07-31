"""
DistilBERT SST-2 Benchmark Example

This example benchmarks the DistilBERT model fine-tuned on SST-2 (Stanford Sentiment Treebank)
for sentiment analysis. The model achieves 91.1% accuracy on the dev set.

Model: distilbert-base-uncased-finetuned-sst-2-english
Dataset: SST-2 (Stanford Sentiment Treebank)
Task: Binary sentiment classification (positive/negative)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BenchmarkEngine
from plugins import HuggingFaceAdapter
from metrics import AccuracyMetric, ClassificationMetric
from benchmark_datasets import HuggingFaceDataset


def main():
    """Benchmark DistilBERT SST-2 model for sentiment analysis."""
    

    print("Model: distilbert-base-uncased-finetuned-sst-2-english")
    
    # Create the benchmark engine
    engine = BenchmarkEngine()
    
    # Register HuggingFace components
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_metric("accuracy", AccuracyMetric)
    engine.register_dataset("sst2", HuggingFaceDataset)
    
    # Configure benchmark parameters
    engine.configure_benchmark({
        "num_samples": 100,  # Reasonable sample size for evaluation
        "warmup_runs": 5,    # More warmup for stable results
        "batch_size": 1,      # Single sample inference
        "precision": "fp32",  # Full precision
        "device": "cpu"       # Use CPU for demo (change to "cuda" for GPU)
    })
    
    # Load SST-2 dataset
    engine.load_dataset("sst2", "glue")
    
    # Add accuracy metric
    engine.add_metric("accuracy")
    
    # Load DistilBERT SST-2 model
    engine.load_model(
        "huggingface", 
        "distilbert-base-uncased-finetuned-sst-2-english",
        {
            "task_type": "text-classification",
            "max_length": 128,  # SST-2 typical length
            "truncation": True,
            "padding": True
        }
    )
    
    # Run benchmark
    results = engine.run_benchmark()
    
    # Display results
    engine.print_results()
    
    # Export results
    engine.export_results("distilbert_sst2_results.json", format="json")
    engine.export_results("distilbert_sst2_results.md", format="markdown")
    




if __name__ == "__main__":
    main()
    
    print("DistilBERT SST-2 Benchmark Complete!")
