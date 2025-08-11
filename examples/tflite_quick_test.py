"""
Quick TFLite Emotion Classification Test

This script quickly tests the TFLite emotion classifier with the local dataset.
No user input required - runs automatically.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import BenchmarkEngine
from plugins import TensorFlowLiteAdapter
from benchmark_datasets import TemplateDataset
from metrics.template_metric import TemplateMultiLabelMetric


def run_quick_tflite_test():
    """Run a quick test of the TFLite emotion classifier."""
    print("QUICK TFLITE EMOTION CLASSIFIER TEST")
    
    # Create engine
    engine = BenchmarkEngine()
    engine.register_adapter("tflite", TensorFlowLiteAdapter)
    engine.register_dataset("template", TemplateDataset)
    
    # Load TFLite model
    model_path = "models/notQuantizedModel.tflite"
    model_config = {
        "device": "cpu",
        "precision": "fp32",
        "max_length": 128,
        "input_type": "text",
        "output_type": "probabilities",
        "task_type": "multi-label",
        "is_multi_label": True
    }
    
    success = engine.load_model("tflite", model_path, model_config)
    if not success:
        print("Failed to load TFLite model")
        print("Note: This requires TensorFlow to be installed")
        print("Install with: pip install tensorflow")
        return False
    
    print("TFLite model loaded successfully")
    
    # Load local dataset
    dataset_config = {
        "file_format": "tsv",
        "text_column": "Tweet",
        "label_columns": ["anger", "anticipation", "disgust", "fear", "joy", 
                         "love", "optimism", "pessimism", "sadness", "surprise", "trust"],
        "task_type": "multi-label",
        "max_length": 128
    }
    
    dataset_path = "benchmark_datasets/localTestSets/2018-E-c-En-test-gold.txt"
    success = engine.load_dataset("template", dataset_path, dataset_config)
    if not success:
        print("Failed to load dataset")
        return False
    
    print("Dataset loaded successfully")
    
    # Add metric
    metric = TemplateMultiLabelMetric(metric_type="accuracy", threshold=0.5)
    engine.metrics = [metric]
    
    # Validate setup
    success = engine.validate_setup()
    if not success:
        print("Setup validation failed")
        return False
    
    print("Setup validation passed")
    print(f"  - Dataset outputs: {engine.dataset.output_type.value}")
    print(f"  - Model expects: {engine.model_adapter.input_type.value}")
    print(f"  - Model outputs: {engine.model_adapter.output_type.value}")
    print(f"  - Metric expects: {engine.metrics[0].expected_input_type.value}")
    
    # Run quick benchmark
    try:
        engine.configure_benchmark({
            "num_samples": 20,  # Quick test with 20 samples
            "warmup_runs": 1,
            "batch_size": 1,
            "precision": "fp32",
            "device": "cpu"
        })
        
        results = engine.run_benchmark()
        
        print("\nBENCHMARK RESULTS")
        engine.print_results()
        
        # Extract accuracy for summary
        if "metrics" in results:
            for metric_name, metric_values in results["metrics"].items():
                if isinstance(metric_values, dict) and "accuracy" in metric_values:
                    accuracy = metric_values["accuracy"]
                    print(f"\nFinal Accuracy: {accuracy:.4f}")
                    break
        
        print("\nQuick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return False


def main():
    """Main test function."""
    print("TFLITE EMOTION CLASSIFIER QUICK TEST")
    print("Testing TFLite emotion classifier with local dataset")
    
    # Check if TensorFlow is available
    try:
        import tensorflow as tf
        print(f"TensorFlow available: {tf.__version__}")
    except ImportError:
        print("TensorFlow not available. Please install: pip install tensorflow")
        return False
    
    # Check if TFLite model exists
    model_path = "models/notQuantizedModel.tflite"
    if not os.path.exists(model_path):
        print(f"TFLite model not found: {model_path}")
        return False
    
    print(f"TFLite model found: {model_path}")
    
    # Run the test
    success = run_quick_tflite_test()
    
    if success:
        print("\nTest passed! TFLite emotion classifier is working correctly.")
    else:
        print("\nTest failed. Check the error messages above.")


if __name__ == "__main__":
    main()

