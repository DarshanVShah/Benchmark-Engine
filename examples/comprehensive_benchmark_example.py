"""
Comprehensive Benchmark Example

This example demonstrates the BenchmarkEngine acting as an "exam administrator"
providing comprehensive evaluation with JSON export functionality.
"""

import os
import sys

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import BenchmarkEngine
from core.dataset_registry import TaskType
from plugins.huggingface_adapter import HuggingFaceAdapter
from plugins.tflite_adapter import TensorFlowLiteAdapter
from benchmark_datasets.template_dataset import TemplateDataset
from metrics.template_metric import TemplateAccuracyMetric
from metrics.template_metric import TemplateMultiLabelMetric


def run_huggingface_comprehensive_benchmark():
    """Run comprehensive benchmark with HuggingFace emotion classifier."""
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK: HUGGINGFACE EMOTION CLASSIFIER")
    print("=" * 80)
    print("Role: Exam Administrator - Providing standardized testing infrastructure")
    print("=" * 80)
    
    # Create benchmark engine (administrator)
    engine = BenchmarkEngine()
    
    # Register components
    engine.register_adapter("huggingface", HuggingFaceAdapter)
    engine.register_dataset("template", TemplateDataset)
    
    # Model configuration
    model_config = {
        "model_name": "j-hartmann/emotion-english-distilroberta-base",
        "device": "cpu",
        "max_length": 128,
        "input_type": "text",
        "output_type": "class_id",
        "task_type": "single-label",
        "is_multi_label": False
    }
    
    # Load model
    print("Loading HuggingFace emotion classifier...")
    if not engine.load_model("huggingface", "j-hartmann/emotion-english-distilroberta-base", model_config):
        print("Failed to load HuggingFace model")
        return False
    
    # Add metrics
    accuracy_metric = TemplateAccuracyMetric(input_type="class_id")
    engine.add_metric("accuracy", accuracy_metric)
    
    # Run comprehensive benchmark
    print("\nRunning comprehensive benchmark across all emotion datasets...")
    results = engine.run_comprehensive_benchmark(
        task_type=TaskType.EMOTION_CLASSIFICATION,
        num_samples=50  # Limit samples for faster testing
    )
    
    if results and results["datasets_tested"]:
        print(f"\nComprehensive benchmark completed!")
        print(f"Datasets tested: {len(results['datasets_tested'])}")
        print(f"Overall assessment: {results['overall_assessment']}")
        
        # Export comprehensive results
        if engine.last_results:
            export_file = engine.export_results("huggingface_comprehensive_results.json")
            print(f"Comprehensive results exported to: {export_file}")
        
        return True
    else:
        print("Comprehensive benchmark failed")
        return False


def run_tflite_comprehensive_benchmark():
    """Run comprehensive benchmark with TFLite emotion classifier."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BENCHMARK: TFLITE EMOTION CLASSIFIER")
    print("=" * 80)
    
    # Check if TFLite model exists
    model_path = "models/notQuantizedModel.tflite"
    if not os.path.exists(model_path):
        print(f"TFLite model not found: {model_path}")
        print("Please ensure your TFLite model is in the models/ directory.")
        return False
    
    # Create benchmark engine (administrator)
    engine = BenchmarkEngine()
    
    # Register components
    engine.register_adapter("tflite", TensorFlowLiteAdapter)
    engine.register_dataset("template", TemplateDataset)
    
    # Model configuration
    model_config = {
        "device": "cpu",
        "precision": "fp32",
        "max_length": 128,
        "input_type": "text",
        "output_type": "class_id",
        "task_type": "single-label",
        "is_multi_label": False
    }
    
    # Load model
    print("Loading TFLite emotion classifier...")
    if not engine.load_model("tflite", model_path, model_config):
        print("Failed to load TFLite model")
        return False
    
    # Add metrics
    accuracy_metric = TemplateAccuracyMetric(input_type="class_id")
    engine.add_metric("accuracy", accuracy_metric)
    
    # Run comprehensive benchmark
    print("\nRunning comprehensive benchmark across all emotion datasets...")
    results = engine.run_comprehensive_benchmark(
        task_type=TaskType.EMOTION_CLASSIFICATION,
        num_samples=50  # Limit samples for faster testing
    )
    
    if results and results["datasets_tested"]:
        print(f"\nComprehensive benchmark completed!")
        print(f"Datasets tested: {len(results['datasets_tested'])}")
        print(f"Overall assessment: {results['overall_assessment']}")
        
        # Export comprehensive results
        if engine.last_results:
            export_file = engine.export_results("tflite_comprehensive_results.json")
            print(f"Comprehensive results exported to: {export_file}")
        
        return True
    else:
        print("Comprehensive benchmark failed")
        return False


def demonstrate_administrator_features():
    """Demonstrate key administrator features."""
    print("\n" + "=" * 80)
    print("ADMINISTRATOR FEATURES DEMONSTRATION")
    print("=" * 80)
    
    # Create engine
    engine = BenchmarkEngine()
    
    # Show available datasets
    print("Available datasets by task type:")
    available_datasets = engine.get_available_datasets()
    for task_type, datasets in available_datasets.items():
        print(f"\n{task_type}:")
        for dataset in datasets:
            print(f"  - {dataset}")
    
    # Show emotion datasets specifically
    print(f"\nEmotion classification datasets:")
    emotion_datasets = engine.get_available_datasets(TaskType.EMOTION_CLASSIFICATION)
    for dataset in emotion_datasets["datasets"]:
        print(f"  - {dataset['name']}: {dataset['description']}")
        print(f"    Expected accuracy: {dataset['expected_accuracy_range']}")
    
    print("\nAdministrator features:")
    print("✓ Standard dataset registry")
    print("✓ Comprehensive result reporting")
    print("✓ JSON export functionality")
    print("✓ Performance assessment and grading")
    print("✓ Cross-dataset evaluation")


def main():
    """Main function demonstrating comprehensive benchmarking."""
    print("🚀 COMPREHENSIVE BENCHMARK EXAMPLE")
    print("=" * 80)
    print("This example demonstrates the BenchmarkEngine as an 'exam administrator':")
    print("1. Providing standardized testing infrastructure")
    print("2. Running comprehensive benchmarks across multiple datasets")
    print("3. Generating detailed administrator reports")
    print("4. Exporting results to JSON files")
    print("5. Providing performance assessment and grading")
    print("=" * 80)
    
    try:
        # Demonstrate administrator features
        demonstrate_administrator_features()
        
        # Run comprehensive benchmarks
        print("\n" + "=" * 80)
        print("RUNNING COMPREHENSIVE BENCHMARKS")
        print("=" * 80)
        
        # HuggingFace benchmark
        hf_success = run_huggingface_comprehensive_benchmark()
        
        # TFLite benchmark (if model available)
        tflite_success = run_tflite_comprehensive_benchmark()
        
        # Summary
        print("\n" + "=" * 80)
        print("COMPREHENSIVE BENCHMARK SUMMARY")
        print("=" * 80)
        
        if hf_success:
            print("✅ HuggingFace comprehensive benchmark: PASSED")
        else:
            print("❌ HuggingFace comprehensive benchmark: FAILED")
        
        if tflite_success:
            print("✅ TFLite comprehensive benchmark: PASSED")
        else:
            print("❌ TFLite comprehensive benchmark: FAILED")
        
        print("\nAdministrator role completed!")
        print("All results have been exported to JSON files in the benchmark_results/ directory")
        print("Check the generated reports for detailed analysis and recommendations")
        
        return hf_success or tflite_success
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
