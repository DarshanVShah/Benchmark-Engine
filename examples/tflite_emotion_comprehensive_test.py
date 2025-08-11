"""
Comprehensive TFLite Emotion Classification Test Suite

This script tests the TFLite emotion classifier against all available emotion datasets.
It demonstrates how to use the BenchmarkEngine with multiple datasets and metrics.
"""

import os
import sys

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import BenchmarkEngine
from plugins.tflite_adapter import TensorFlowLiteAdapter
from benchmark_datasets.template_dataset import TemplateDataset
from metrics.template_metric import TemplateAccuracyMetric
from metrics.template_metric import TemplateMultiLabelMetric
from core.dataset_registry import DatasetRegistry, TaskType


def test_tflite_emotion_classifier():
    """Test TFLite emotion classifier against all available datasets."""
    print("🚀 TFLITE EMOTION CLASSIFICATION COMPREHENSIVE TEST")
    print("=" * 70)
    
    # Check if TFLite model exists
    model_path = "models/notQuantizedModel.tflite"
    if not os.path.exists(model_path):
        print(f"❌ TFLite model not found: {model_path}")
        print("Please ensure your TFLite model is in the models/ directory.")
        return False
    
    print(f"✓ TFLite model found: {model_path}")
    
    # Get available datasets
    registry = DatasetRegistry()
    emotion_datasets = registry.get_datasets_for_task(TaskType.EMOTION_CLASSIFICATION)
    
    if not emotion_datasets:
        print("❌ No emotion datasets found in registry")
        print("Please add datasets to the registry or place them in benchmark_datasets/localTestSets/")
        return False
    
    print(f"✓ Found {len(emotion_datasets)} emotion dataset(s)")
    
    # Test each dataset
    results = {}
    datasets_tested = 0
    
    for dataset_config in emotion_datasets:
        print(f"\n{'='*60}")
        print(f"TESTING DATASET: {dataset_config.name}")
        print(f"Description: {dataset_config.description}")
        print(f"Task Type: {dataset_config.config.get('task_type', 'unknown')}")
        if dataset_config.expected_accuracy_range:
            print(f"Expected Accuracy: {dataset_config.expected_accuracy_range}")
        print(f"{'='*60}")
        
        # Check dataset availability
        if not registry.ensure_dataset_available(dataset_config):
            print(f"❌ Dataset {dataset_config.name} not available")
            continue
        
        try:
            # Create engine for this dataset
            engine = BenchmarkEngine()
            engine.register_adapter("tflite", TensorFlowLiteAdapter)
            engine.register_dataset("template", TemplateDataset)
            
            # Configure model for this dataset
            is_multi_label = dataset_config.config.get('task_type') == 'multi-label'
            model_config = {
                'device': 'cpu',
                'precision': 'fp32',
                'max_length': 128,
                'input_type': 'text',
                'output_type': 'probabilities' if is_multi_label else 'class_id',
                'task_type': dataset_config.config.get('task_type', 'single-label'),
                'is_multi_label': is_multi_label
            }
            
            # Load model
            if not engine.load_model("tflite", model_path, model_config):
                print(f"❌ Failed to load model for {dataset_config.name}")
                continue
            
            # Load dataset
            if not engine.load_dataset("template", dataset_config.path, dataset_config.config):
                print(f"❌ Failed to load dataset {dataset_config.name}")
                continue
            
            # Add appropriate metric
            if is_multi_label:
                metric = TemplateMultiLabelMetric(metric_type="accuracy", threshold=0.5)
                print(f"✓ Using multi-label accuracy metric")
            else:
                metric = TemplateAccuracyMetric(input_type="class_id")
                print(f"✓ Using single-label accuracy metric")
            
            engine.add_metric("template", metric)
            
            # Validate setup
            if not engine.validate_setup():
                print(f"❌ Setup validation failed for {dataset_config.name}")
                continue
            
            print(f"✓ Setup validation passed")
            print(f"  - Dataset outputs: {engine.dataset.output_type.value}")
            print(f"  - Model expects: {engine.model_adapter.input_type.value}")
            print(f"  - Model outputs: {engine.model_adapter.output_type.value}")
            print(f"  - Metric expects: {engine.metrics[0].expected_input_type.value}")
            
            # Run benchmark with limited samples for faster testing
            print(f"Running benchmark with 50 samples...")
            benchmark_results = engine.run_benchmark(num_samples=50)
            
            if benchmark_results and "metrics" in benchmark_results:
                # Extract accuracy from results
                accuracy = None
                for metric_name, metric_values in benchmark_results["metrics"].items():
                    if isinstance(metric_values, dict) and "accuracy" in metric_values:
                        accuracy = metric_values["accuracy"]
                        break
                
                if accuracy is not None:
                    results[dataset_config.name] = accuracy
                    datasets_tested += 1
                    
                    print(f"🎯 Accuracy: {accuracy:.4f}")
                    
                    # Validate against expected range
                    if dataset_config.expected_accuracy_range:
                        min_acc, max_acc = dataset_config.expected_accuracy_range
                        if min_acc <= accuracy <= max_acc:
                            print(f"✅ PASS: Accuracy within expected range ({min_acc:.2f}-{max_acc:.2f})")
                        else:
                            print(f"⚠️  WARNING: Accuracy outside expected range ({min_acc:.2f}-{max_acc:.2f})")
                else:
                    print(f"❌ No accuracy found in results for {dataset_config.name}")
            else:
                print(f"❌ No results obtained for {dataset_config.name}")
                
        except Exception as e:
            print(f"❌ Error testing {dataset_config.name}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print("COMPREHENSIVE TEST RESULTS")
    print(f"{'='*70}")
    
    if results:
        for dataset_name, accuracy in results.items():
            dataset_config = registry.get_dataset_by_name(dataset_name)
            if dataset_config and dataset_config.expected_accuracy_range:
                min_acc, max_acc = dataset_config.expected_accuracy_range
                status = "✅ PASS" if min_acc <= accuracy <= max_acc else "❌ FAIL"
                print(f"{status} {dataset_name}: {accuracy:.4f} (expected: {min_acc:.2f}-{max_acc:.2f})")
            else:
                print(f"✅ PASS {dataset_name}: {accuracy:.4f}")
        
        print(f"\nSummary: {datasets_tested}/{len(emotion_datasets)} datasets tested successfully")
        
        if datasets_tested == len(emotion_datasets):
            print("🎉 All tests passed! TFLite emotion classifier is working correctly.")
            return True
        else:
            print("⚠️  Some tests failed. Check the error messages above.")
            return False
    else:
        print("❌ No datasets were tested successfully")
        print("\nTo add your own datasets:")
        print("1. Place dataset files in benchmark_datasets/localTestSets/")
        print("2. Update the dataset registry in core/dataset_registry.py")
        print("3. Or use the registry.add_dataset() method programmatically")
        return False


def main():
    """Main function to run the comprehensive test."""
    try:
        success = test_tflite_emotion_classifier()
        if success:
            print("\n✅ Comprehensive test completed successfully!")
        else:
            print("\n❌ Some tests failed. Check the error messages above.")
        return success
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

