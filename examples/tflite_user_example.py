"""
Simple TFLite Emotion Classifier Example

This example shows how a user would use the BenchmarkEngine to test their TFLite emotion classifier.
Simple, straightforward usage - just load model, load dataset, and run benchmark.
"""

import os
import sys

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import BenchmarkEngine
from plugins.tflite_adapter import TensorFlowLiteAdapter
from benchmark_datasets.template_dataset import TemplateDataset
from metrics.template_metric import TemplateAccuracyMetric


def main():
    """Simple example of using TFLite emotion classifier."""
    
    # Create the benchmark engine
    engine = BenchmarkEngine()
    
    # Register what we need
    engine.register_adapter("tflite", TensorFlowLiteAdapter)
    engine.register_dataset("template", TemplateDataset)
    
    # Load your TFLite model
    model_path = "models/notQuantizedModel.tflite"
    model_config = {
        "device": "cpu",
        "precision": "fp32",
        "max_length": 128,
        "input_type": "text",
        "output_type": "probabilities",  # Changed to probabilities for multi-label
        "task_type": "multi-label",      # Changed to multi-label
        "is_multi_label": True           # Enable multi-label mode
    }
    
    if not engine.load_model("tflite", model_path, model_config):
        print("Failed to load TFLite model")
        return False
    
    # Load dataset
    dataset_path = "benchmark_datasets/localTestSets/2018-E-c-En-test-gold.txt"
    
    # Configure dataset for 2018 emotion classification (11 emotions, multi-label)
    dataset_config = {
        "file_format": "tsv",
        "text_column": "Tweet",  # The actual tweet text column
        "label_columns": ["anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness", "surprise", "trust"],
        "task_type": "multi-label",
        "max_length": 128,
        "delimiter": "\t",
        "skip_header": True,  # 2018 dataset has header
        "id_column": "ID"     # Specify ID column to ignore
    }
    
    if not engine.load_dataset("template", dataset_path, dataset_config):
        print("Failed to load dataset")
        return False
    
    # Add evaluation metrics
    from metrics.template_metric import TemplateMultiLabelMetric
    multi_label_metric = TemplateMultiLabelMetric(metric_type="accuracy", threshold=0.5)
    engine.add_metric("multi_label_accuracy", multi_label_metric)
    
    # Run the benchmark
    results = engine.run_benchmark(num_samples=1000)  # Test on 1000 samples
    
    if results:
        # Export results
        export_file = engine.export_results("my_emotion_classifier_results.json")
        print(f"\nResults saved to: {export_file}")
        
        return True
    else:
        print("Benchmark failed")
        return False


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\nAll done!")
    else:
        print("\nSomething went wrong. Check the error messages above.")
        sys.exit(1)
