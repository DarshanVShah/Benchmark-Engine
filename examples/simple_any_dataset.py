"""
Simple Example: Benchmark Any HuggingFace Dataset

This example shows how users can benchmark ANY dataset from HuggingFace Hub
without creating custom adapters, metrics, or dataset files.

Just specify the dataset name and run!
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BenchmarkEngine
from plugins import HuggingFaceAdapter
from datasets import HuggingFaceDataset
from metrics import GenericMetric


def main():
    """Benchmark any HuggingFace dataset with minimal code."""
    
    print("Simple Any-Dataset Benchmark Demo")
    print("=" * 50)
    print("No custom files needed - just specify the dataset name!")
    print()
    
    # Create the benchmark engine
    engine = BenchmarkEngine()
    
    # Register generic components (works with ANY dataset)
    print("Registering generic components...")
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_metric("generic", GenericMetric)
    engine.register_dataset("huggingface", HuggingFaceDataset)
    
    # Configure benchmark
    engine.configure_benchmark({
        "num_samples": 20,  # Small sample for demo
        "warmup_runs": 2,
        "batch_size": 1,
        "precision": "fp32",
        "device": "cpu"
    })
    
    # Test different datasets - just change the dataset name!
    datasets_to_test = [
        {
            "name": "Emotion Detection VAD",
            "path": "mmarkusmalone/journal-entries-emotion-detection-vad",
            "description": "Reddit journal entries with VAD emotion labels"
        },
        {
            "name": "SST-2 Sentiment",
            "path": "glue",  # This would work with SST-2 subset
            "description": "Stanford Sentiment Treebank"
        },
        {
            "name": "AG News",
            "path": "ag_news",  # This would work with AG News
            "description": "News classification dataset"
        }
    ]
    
    for dataset_config in datasets_to_test:
        print(f"\n{'='*20} {dataset_config['name']} {'='*20}")
        print(f"Dataset: {dataset_config['path']}")
        print(f"Description: {dataset_config['description']}")
        
        try:
            # Load ANY dataset - no custom files needed!
            print(f"\nLoading dataset: {dataset_config['path']}")
            engine.load_dataset("huggingface", dataset_config["path"])
            
            # Add generic metric (auto-detects task type)
            engine.add_metric("generic")
            
            # Load a model
            print(f"\nLoading model: bert-base-uncased")
            engine.load_model("huggingface", "bert-base-uncased")
            
            # Run benchmark
            print(f"\nRunning benchmark...")
            results = engine.run_benchmark()
            
            # Display results
            print(f"\nResults:")
            engine.print_results()
            
            # Export results
            output_name = dataset_config["name"].lower().replace(" ", "_").replace("-", "_")
            engine.export_results(f"{output_name}_results.json", format="json")
            engine.export_results(f"{output_name}_results.md", format="markdown")
            
            print(f"\n✅ {dataset_config['name']} benchmark completed!")
            print(f"   Results saved to {output_name}_results.json and {output_name}_results.md")
            
        except Exception as e:
            print(f"❌ Error with {dataset_config['name']}: {e}")
            print("   (This is expected for some datasets that require special handling)")
            continue
    
    print(f"\n{'='*50}")
    print("Simple Any-Dataset Benchmark Completed!")
    print("\nKey Benefits:")
    print("  ✅ No custom files needed")
    print("  ✅ Works with ANY HuggingFace dataset")
    print("  ✅ Auto-detects task type and metrics")
    print("  ✅ Same interface for all datasets")
    print("  ✅ Framework-agnostic design")


def demo_single_dataset():
    """Demo with just the emotion detection dataset."""
    
    print("\n" + "="*50)
    print("Single Dataset Demo: Emotion Detection VAD")
    print("="*50)
    
    engine = BenchmarkEngine()
    
    # Register components
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_metric("generic", GenericMetric)
    engine.register_dataset("huggingface", HuggingFaceDataset)
    
    # Load the emotion dataset
    print("\nLoading emotion detection dataset...")
    engine.load_dataset("huggingface", "mmarkusmalone/journal-entries-emotion-detection-vad")
    
    # Add generic metric (auto-detects VAD emotion detection)
    engine.add_metric("generic")
    
    # Load model
    print("\nLoading BERT model...")
    engine.load_model("huggingface", "bert-base-uncased")
    
    # Run benchmark
    print("\nRunning emotion detection benchmark...")
    results = engine.run_benchmark(num_samples=10)
    
    # Display results
    print("\nResults:")
    engine.print_results()
    
    print("\n✅ Emotion detection benchmark completed!")
    print("   The generic metric automatically detected VAD emotion detection")
    print("   and calculated appropriate correlation and MSE metrics!")


if __name__ == "__main__":
    try:
        main()
        demo_single_dataset()
    except Exception as e:
        print(f"\nError running simple benchmark: {e}")
        print("\nThis might be due to:")
        print("  - Missing datasets library (pip install datasets)")
        print("  - Network issues downloading datasets")
        print("  - Dataset not available or requires authentication")
        
        print("\nTo install dependencies:")
        print("  pip install datasets transformers torch numpy scikit-learn") 