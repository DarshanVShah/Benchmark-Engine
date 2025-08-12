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
        "output_type": "probabilities",  # Changed to probabilities for multi-label
        "task_type": "multi-label",      # Changed to multi-label
        "is_multi_label": True           # Enable multi-label mode
    }
    
    if not engine.load_model("tflite", model_path, model_config):
        print("Failed to load TFLite model")
        return False
    
    print("TFLite model loaded successfully!")
    print(f"   Task type: {engine.model_adapter.task_type}")
    print(f"   Multi-label: {engine.model_adapter.is_multi_label}")
    print(f"   Output type: {engine.model_adapter.output_type.value}")
    
    # User guidance about TFLite preprocessing
    print("\nTFLite Adapter Configuration Notes:")
    print(f"   - Model expects input shape: {engine.model_adapter.input_shape}")
    print("   - Input format: [attention_mask, input_ids, token_type_ids]")
    print("   - The adapter handles preprocessing automatically")
    print("   - If you get errors, check the model's expected input format")
    print("   - Consider using HuggingFace models for easier text processing")
    
    # Step 4: Load a test dataset
    dataset_path = "benchmark_datasets/localTestSets/2018-E-c-En-test-gold.txt"
    
    # Check if dataset exists locally
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return False
    
    print(f"Found local dataset: {dataset_path}")
    
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
    
    print("Dataset loaded successfully!")
    
    # Check model-dataset compatibility
    print("\nModel-Dataset Compatibility Check:")
    print(f"  Model input shape: {engine.model_adapter.input_shape}")
    print(f"  Model output shape: {engine.model_adapter.output_shape}")
    print(f"  Dataset labels: {len(engine.dataset.label_columns)} columns")
    print(f"  2018 dataset has 11 emotion classes")
    
    # Check if model output matches dataset
    if engine.model_adapter.output_shape[1] == 11:
        print("Model output matches dataset! (11 classes)")
        print("   This should work much better than the 27-class GoEmotions dataset")
    else:
        print(f"Model outputs {engine.model_adapter.output_shape[1]} classes but dataset has 11 emotions")
        print("   This may still cause compatibility issues")
    
    # Step 5: Add evaluation metrics
    # Use multi-label metric since 2018 dataset is multi-label
    from metrics.template_metric import TemplateMultiLabelMetric
    multi_label_metric = TemplateMultiLabelMetric(metric_type="accuracy", threshold=0.5)
    engine.add_metric("multi_label_accuracy", multi_label_metric)
    
    # Step 6: Run the benchmark!
    
    results = engine.run_benchmark(num_samples=1000)  # Test on 100 samples
    
    if results:
        print(f"Tested on {results['benchmark_config']['num_samples']} samples")
        print(f"Total time: {results['timing']['total_time']:.2f} seconds")
        print(f"Throughput: {results['timing']['throughput']:.1f} samples/second")
        
        # Show accuracy results
        if "TemplateMultiLabelMetric" in results["metrics"]:
            accuracy = results["metrics"]["TemplateMultiLabelMetric"]
            if isinstance(accuracy, dict) and "accuracy" in accuracy:
                accuracy_value = accuracy["accuracy"]
                total_correct = accuracy.get("total_correct", 0)
                total_predictions = accuracy.get("total_predictions", 0)
                
                print(f"Multi-label Accuracy: {accuracy_value:.2%}")
                print(f"Correct predictions: {total_correct}/{total_predictions}")
                
                # Explain results and provide solutions
                
                # Simple performance assessment
                if accuracy_value >= 0.8:
                    print("Excellent performance!")
                elif accuracy_value >= 0.6:
                    print("Good performance!")
                elif accuracy_value >= 0.4:
                    print("Acceptable performance")
                else:
                    print("Room for improvement")
                    
                print(f"\nYour TFLite model achieved {accuracy_value:.1%} accuracy.")
                
        else:
            print("Metrics available:", list(results["metrics"].keys()))
            print("   Expected: TemplateMultiLabelMetric")
        
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
