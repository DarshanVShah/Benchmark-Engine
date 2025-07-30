"""
HuggingFace benchmark example demonstrating real model integration.

This example shows how to:
1. Use the HuggingFace adapter with real models
2. Work with text datasets for classification
3. Calculate real accuracy metrics
4. Configure models for different devices/precision
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BenchmarkEngine
from plugins import HuggingFaceAdapter
from benchmark_datasets import TextDataset
from metrics import AccuracyMetric


def main():
    """Run a HuggingFace benchmark with real models."""
    
    print("HuggingFace Benchmark Demo")
    
    # Create the benchmark engine
    engine = BenchmarkEngine()
    
    # Register HuggingFace components
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_metric("accuracy", AccuracyMetric)
    engine.register_dataset("text", TextDataset)
    
    # Configure benchmark parameters
    engine.configure_benchmark({
        "num_samples": 10,  # Small sample for demo
        "warmup_runs": 2,
        "batch_size": 1,
        "precision": "fp32",
        "device": "cpu"  # Use CPU for demo
    })
    
    # Load text dataset
    engine.load_dataset("text", "synthetic_text_data")
    
    # Add accuracy metric
    engine.add_metric("accuracy")
    
    # Test with different HuggingFace models
    models_to_test = [
        {
            "name": "BERT Base",
            "path": "bert-base-uncased",
            "config": {"task_type": "text-classification"}
        },
        {
            "name": "DistilBERT", 
            "path": "distilbert-base-uncased",
            "config": {"task_type": "text-classification"}
        }
    ]
    
    for model_config in models_to_test:
        print(f"\n Testing {model_config['name']}...")
        
        # Load model with configuration
        engine.load_model(
            "huggingface", 
            model_config["path"],
            model_config["config"]
        )
        
        # Run benchmark
        print(f"\n Running benchmark on {model_config['name']}...")
        results = engine.run_benchmark()
        
        # Display results
        print(f"\n Results for {model_config['name']}:")
        engine.print_results()
        
        # Export results
        output_name = model_config["name"].lower().replace(" ", "_")
        engine.export_results(f"{output_name}_results.json", format="json")
        engine.export_results(f"{output_name}_results.md", format="markdown")
        
        print(f"\n {model_config['name']} benchmark completed!")
    
    print("\n All HuggingFace benchmarks completed!")


def demo_model_comparison():
    """Demonstrate comparing multiple HuggingFace models."""
    
    print("\n Model Comparison Demo")
    
    engine = BenchmarkEngine()
    
    # Register components
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_metric("accuracy", AccuracyMetric)
    engine.register_dataset("text", TextDataset)
    
    # Load dataset
    engine.load_dataset("text", "synthetic_text_data")
    engine.add_metric("accuracy")
    
    # Define models to compare
    model_configs = [
        {
            "adapter": "huggingface",
            "path": "bert-base-uncased",
            "config": {"task_type": "text-classification"}
        },
        {
            "adapter": "huggingface", 
            "path": "distilbert-base-uncased",
            "config": {"task_type": "text-classification"}
        }
    ]
    
    # Compare models
    comparison_results = engine.compare_models(
        model_configs=model_configs,
        dataset_name="text",
        metrics=["accuracy"]
    )
    
    # Display comparison
    print("\n Model Comparison Results:")
    for model_name, result in comparison_results.items():
        print(f"\n{model_name}:")
        model_info = result["results"]["model_info"]
        metrics = result["results"]["metrics"]["Accuracy"]
        timing = result["results"]["timing"]
        
        print(f"  Model: {model_info['name']}")
        print(f"  Parameters: {model_info.get('parameters', 'Unknown'):,}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Avg Inference: {timing['average_inference_time']:.4f}s")
        print(f"  Throughput: {timing['throughput']:.2f} samples/s")
    
    # Export comparison
    engine.export_results("model_comparison.json", format="json")
    print("\n Comparison results exported to model_comparison.json")


if __name__ == "__main__":
    try:
        main()
        demo_model_comparison()
    except Exception as e:
        print(f"\nError running HuggingFace benchmark: {e}")
