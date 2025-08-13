"""
Comprehensive Benchmark Example

This example demonstrates the BenchmarkEngine acting as an "exam administrator"
providing comprehensive evaluation with JSON export functionality.
"""

import os
import sys

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_datasets.template_dataset import TemplateDataset
from core.dataset_registry import TaskType
from core.engine import BenchmarkEngine
from metrics.template_metric import TemplateAccuracyMetric
from plugins.huggingface_adapter import HuggingFaceAdapter
from plugins.tflite_adapter import TensorFlowLiteAdapter


def run_huggingface_comprehensive_benchmark():
    """Run comprehensive benchmark with HuggingFace emotion classifier."""

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
        "is_multi_label": False,
    }

    # Load model
    if not engine.load_model(
        "huggingface", "j-hartmann/emotion-english-distilroberta-base", model_config
    ):
        print("Failed to load HuggingFace model")
        return False

    # Add metrics
    accuracy_metric = TemplateAccuracyMetric(input_type="class_id")
    engine.add_metric("accuracy", accuracy_metric)

    # Run comprehensive benchmark
    results = engine.run_comprehensive_benchmark(
        task_type=TaskType.EMOTION_CLASSIFICATION,
        num_samples=50,  # Limit samples for faster testing
    )

    if results and results["datasets_tested"]:
        print("\nComprehensive benchmark completed!")
        print(f"Datasets tested: {len(results['datasets_tested'])}")
        print(f"Overall assessment: {results['overall_assessment']}")

        # Export comprehensive results
        if engine.last_results:
            export_file = engine.export_results(
                "huggingface_comprehensive_results.json"
            )
            print(f"Comprehensive results exported to: {export_file}")

        return True
    else:
        print("Comprehensive benchmark failed")
        return False


def run_tflite_comprehensive_benchmark():
    """Run comprehensive benchmark with TFLite emotion classifier."""

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
        "is_multi_label": False,
    }

    # Load model
    if not engine.load_model("tflite", model_path, model_config):
        print("Failed to load TFLite model")
        return False

    # Add metrics
    accuracy_metric = TemplateAccuracyMetric(input_type="class_id")
    engine.add_metric("accuracy", accuracy_metric)

    # Run comprehensive benchmark
    results = engine.run_comprehensive_benchmark(
        task_type=TaskType.EMOTION_CLASSIFICATION,
        num_samples=50,  # Limit samples for faster testing
    )

    if results and results["datasets_tested"]:
        print("\nComprehensive benchmark completed!")
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


def main():
    """Main function demonstrating comprehensive benchmarking."""

    try:

        # HuggingFace benchmark
        hf_success = run_huggingface_comprehensive_benchmark()

        # TFLite benchmark (if model available)
        tflite_success = run_tflite_comprehensive_benchmark()

        if hf_success:
            print("HuggingFace comprehensive benchmark: PASSED")
        else:
            print("HuggingFace comprehensive benchmark: FAILED")

        if tflite_success:
            print("TFLite comprehensive benchmark: PASSED")
        else:
            print("TFLite comprehensive benchmark: FAILED")

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
