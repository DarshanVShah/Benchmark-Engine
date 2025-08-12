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
    
    # Check if TFLite model exists
    model_path = "models/notQuantizedModel.tflite"
    if not os.path.exists(model_path):
        print(f"TFLite model not found: {model_path}")
        print("Please put your TFLite emotion classifier in the models/ directory.")
        return False
    
    print(f"Found TFLite model: {model_path}")
    
    # Step 1: Create the benchmark engine
    engine = BenchmarkEngine()
    
    # Step 2: Register what we need
    engine.register_adapter("tflite", TensorFlowLiteAdapter)
    engine.register_dataset("template", TemplateDataset)
    
    # Step 3: Load your TFLite model
    model_config = {
        "device": "cpu",
        "precision": "fp32",
        "max_length": 128,
        "input_type": "text",
        "output_type": "class_id",
        "task_type": "single-label",
        "is_multi_label": False
    }
    
    if not engine.load_model("tflite", model_path, model_config):
        print("Failed to load TFLite model")
        return False
    
    print("TFLite model loaded successfully!")
    
    # Step 4: Load a test dataset
    dataset_path = "benchmark_datasets/localTestSets/goemotions_test.tsv"
    
    # If GoEmotions dataset doesn't exist locally, the engine will download it
    if not os.path.exists(dataset_path):
        print("GoEmotions dataset not found locally - will download automatically")
    
    dataset_config = {
        "file_format": "tsv",
        "text_column": "text",
        "label_columns": ["label"],
        "task_type": "single-label",
        "max_length": 128,
        "delimiter": "\t",
        "skip_header": False
    }
    
    if not engine.load_dataset("template", dataset_path, dataset_config):
        print("Failed to load dataset")
        return False
    
    print("Dataset loaded successfully!")
    
    # Step 5: Add evaluation metrics
    accuracy_metric = TemplateAccuracyMetric(input_type="class_id")
    engine.add_metric("accuracy", accuracy_metric)
    
    # Step 6: Run the benchmark!
    results = engine.run_benchmark(num_samples=100)  # Test on 100 samples
    
    if results:
        print(f"Tested on {results['benchmark_config']['num_samples']} samples")
        print(f"Total time: {results['timing']['total_time']:.2f} seconds")
        print(f"Throughput: {results['timing']['throughput']:.1f} samples/second")
        
        # Show accuracy results
        if "TemplateAccuracyMetric" in results["metrics"]:
            accuracy = results["metrics"]["TemplateAccuracyMetric"]
            if isinstance(accuracy, dict) and "accuracy" in accuracy:
                accuracy_value = accuracy["accuracy"]
            else:
                accuracy_value = accuracy
                
            print(f"Accuracy: {accuracy_value:.2%}")
            
            # Simple performance assessment
            if accuracy_value >= 0.8:
                print("Excellent performance!")
            elif accuracy_value >= 0.6:
                print("Good performance!")
            elif accuracy_value >= 0.4:
                print("Acceptable performance")
            else:
                print("Room for improvement")
        else:
            print("Metrics available:", list(results["metrics"].keys()))
        
        # Export results
        export_file = engine.export_results("my_emotion_classifier_results.json")
        print(f"\nResults saved to: {export_file}")
        
        return True
    else:
        print("Benchmark failed")
        return False


if __name__ == "__main__":
    print("Starting TFLite emotion classifier benchmark...")
    success = main()
    
    if success:
        print("\nAll done! Check your results file for detailed analysis.")
    else:
        print("\nSomething went wrong. Check the error messages above.")
        sys.exit(1)
