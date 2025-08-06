"""
Working Template Benchmark with Real Model

This example demonstrates the template pattern with:
- Real TFLite model: models/notQuantizedModel.tflite
- Real dataset: benchmark_datasets/localTestSets/2018-E-c-En-test-gold.txt
- Explicit configuration - no auto-detection!
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BenchmarkEngine, DataType, OutputType
from plugins import TensorFlowLiteAdapter
from metrics import TemplateAccuracyMetric, TemplateMultiLabelMetric
from benchmark_datasets import TemplateDataset


def benchmark_with_real_model():
    """
    Demonstrate template pattern with real TFLite model and dataset.
    """
    print("=" * 60)
    print("WORKING TEMPLATE BENCHMARK WITH REAL MODEL")

    
    # User declares everything in main function
    engine = BenchmarkEngine()
    
    # 1. Register components
    engine.register_adapter("tflite", TensorFlowLiteAdapter)
    engine.register_metric("template_accuracy", TemplateAccuracyMetric)
    engine.register_metric("template_multilabel", TemplateMultiLabelMetric)
    engine.register_dataset("template", TemplateDataset)
    
    # 2. Configure benchmark
    engine.configure_benchmark({
        "num_samples": 50,  # Smaller sample for demo
        "warmup_runs": 2,
        "batch_size": 1,
        "precision": "fp32",
        "device": "cpu"
    })
    
    # 3. Load dataset with EXPLICIT configuration
    dataset_path = "benchmark_datasets/localTestSets/2018-E-c-En-test-gold.txt"
    
    # User explicitly specifies dataset configuration
    dataset_config = {
        "file_format": "tsv",
        "text_column": "Tweet",
        "label_columns": ["anger", "anticipation", "disgust", "fear", "joy", 
                         "love", "optimism", "pessimism", "sadness", "surprise", "trust"],
        "task_type": "multi-label",
        "max_length": 512
    }
    
    success = engine.load_dataset("template", dataset_path, dataset_config)
    if not success:
        print("Failed to load dataset")
        return None
    
    # 4. Load TFLite model with EXPLICIT configuration
    model_path = "models/notQuantizedModel.tflite"
    model_config = {
        "task_type": "emotion-detection",
        "max_length": 512,
    }
    
    success = engine.load_model("tflite", model_path, model_config)
    if not success:
        print("Failed to load model")
        return None
    
    # 5. Add metrics with EXPLICIT configuration
    # Create metrics with explicit parameters
    accuracy_metric = TemplateAccuracyMetric(input_type="probabilities", threshold=0.5)
    multilabel_metric = TemplateMultiLabelMetric(metric_type="accuracy", threshold=0.5)
    
    engine.add_metric("template_accuracy", accuracy_metric)
    engine.add_metric("template_multilabel", multilabel_metric)
    
    # 6. Run benchmark
    try:
        results = engine.run_benchmark()
        
        # 7. Display results
        print("\nBenchmark Results:")
        print("-" * 40)
        engine.print_results()
        
        return results
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return None



def main():
    """Main function demonstrating template pattern with real model."""
    
    # Check if files exist
    model_path = "models/notQuantizedModel.tflite"
    dataset_path = "benchmark_datasets/localTestSets/2018-E-c-En-test-gold.txt"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        return
    
    print("All required files found")
    print(f"Model: {model_path} ({os.path.getsize(model_path) / (1024*1024):.1f} MB)")
    print(f"Dataset: {dataset_path}")
    print()
    
    # Run the benchmark
    results = benchmark_with_real_model()


if __name__ == "__main__":
    main() 