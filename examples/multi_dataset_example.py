"""
Multi-Dataset Benchmark Example

This example demonstrates how the framework provides standardized test suites
for different ML tasks. Just like different subjects have different exams,
different ML tasks have different standardized datasets.

For emotion classification, we test the model on multiple emotion datasets
to ensure comprehensive evaluation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BenchmarkEngine, TaskType
from plugins import TensorFlowLiteAdapter, HuggingFaceAdapter
from metrics import TemplateAccuracyMetric, TemplateMultiLabelMetric
from benchmark_datasets import TemplateDataset


def benchmark_emotion_classification():
    """
    Benchmark emotion classification models across multiple datasets.
    
    """
    
    print("EMOTION CLASSIFICATION BENCHMARK SUITE")
    
    # Initialize the framework (the "exam administrator")
    engine = BenchmarkEngine()
    
    # Register components (the "tools" students can use)
    engine.register_adapter("tflite", TensorFlowLiteAdapter)
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_metric("template_accuracy", TemplateAccuracyMetric)
    engine.register_metric("template_multilabel", TemplateMultiLabelMetric)
    engine.register_dataset("template", TemplateDataset)
    
    # Configure the benchmark (exam settings)
    engine.configure_benchmark({
        "num_samples": 500,  # Test on 500 samples per dataset
        "warmup_runs": 2,
        "batch_size": 1,
        "precision": "fp32",
        "device": "cpu"
    })
    
    # Test with TFLite model
    print("TensorFlow Lite Model")
    print("Model: notQuantizedModel.tflite")

    
    model_path = "models/notQuantizedModel.tflite"
    model_config = {
        "task_type": "emotion-detection",
        "max_length": 512
    }
    
    # Run multi-dataset benchmark for emotion classification
    results = engine.run_multi_dataset_benchmark(
        task_type=TaskType.EMOTION_CLASSIFICATION,
        model_path=model_path,
        adapter_name="tflite",
        model_config=model_config
    )
    
    # Display comprehensive results
    engine.print_multi_dataset_results(results)
    
    return results


def benchmark_sentiment_analysis():
    """
    Benchmark sentiment analysis models across multiple datasets.
    
    """
    
    print("SENTIMENT ANALYSIS BENCHMARK SUITE")
    
    
    # Initialize framework
    engine = BenchmarkEngine()
    
    # Register components
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_metric("template_accuracy", TemplateAccuracyMetric)
    engine.register_dataset("template", TemplateDataset)
    
    # Configure benchmark
    engine.configure_benchmark({
        "num_samples": 300,
        "warmup_runs": 2,
        "batch_size": 1,
        "precision": "fp32",
        "device": "cpu"
    })
    
    # Test with HuggingFace model
    print("HuggingFace Model")
    print("Model: distilbert-base-uncased-finetuned-sst-2-english")
    
    model_path = "distilbert-base-uncased-finetuned-sst-2-english"
    model_config = {
        "task_type": "sentiment-analysis",
        "max_length": 512
    }
    
    # Run multi-dataset benchmark for sentiment analysis
    results = engine.run_multi_dataset_benchmark(
        task_type=TaskType.SENTIMENT_ANALYSIS,
        model_path=model_path,
        adapter_name="huggingface",
        model_config=model_config
    )
    
    # Display results
    engine.print_multi_dataset_results(results)
    
    return results


def demonstrate_framework_capabilities():
    """
    Demonstrate the framework's capabilities as a standardized testing system.
    """
    
    print("FRAMEWORK CAPABILITIES DEMONSTRATION")
    
    
    # Initialize framework
    engine = BenchmarkEngine()
    
    
    # Show available task types
    print(f"\nSUPPORTED TASK TYPES:")
    for task_type in engine.dataset_registry.get_task_types():
        datasets = engine.dataset_registry.get_datasets_for_task(task_type)
        print(f"   - {task_type.value}: {len(datasets)} datasets")
    

def main():
    """Main function demonstrating multi-dataset benchmarking."""
    
    print("BENCHMARK ENGINE: STANDARDIZED ML TESTING FRAMEWORK")

    
    # Check if required files exist
    model_path = "models/notQuantizedModel.tflite"
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("   Skipping TFLite benchmark...")
        return
    
    # Demonstrate framework capabilities
    demonstrate_framework_capabilities()
    
    # Run emotion classification benchmark
    try:
        emotion_results = benchmark_emotion_classification()
        print(f"\nEmotion classification benchmark completed!")
    except Exception as e:
        print(f"\nEmotion classification benchmark failed: {e}")
    
    # Run sentiment analysis benchmark (if HuggingFace available)
    try:
        sentiment_results = benchmark_sentiment_analysis()
        print(f"\nSentiment analysis benchmark completed!")
    except Exception as e:
        print(f"\nSentiment analysis benchmark failed: {e}")
        print("   (This is expected if HuggingFace models aren't available)")


if __name__ == "__main__":
    main()
